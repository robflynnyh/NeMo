'''
Heavily relies on code from: https://github.com/lucidrains/conformer
'''

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange
from nemo.collections.asr.parts.submodules.x_transformers_local import CrossAttender, Attention as xAttention


# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)

# helper classes

class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()

class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()

class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups = chan_in)

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)

# attention, feedforward, and conv module

class Scale(nn.Module):
    def __init__(self, scale, fn):
        super().__init__()
        self.fn = fn
        self.scale = scale

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        max_pos_emb = 512
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads= heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.max_pos_emb = max_pos_emb
        self.rel_pos_emb = nn.Embedding(2 * max_pos_emb + 1, dim_head)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context = None, mask = None, context_mask = None):
        n, device, h, max_pos_emb, has_context = x.shape[-2], x.device, self.heads, self.max_pos_emb, exists(context)
        context = default(context, x) 

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1)) 
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v)) 

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # shaw's relative positional embedding
        seq = torch.arange(n, device = device)
        dist = rearrange(seq, 'i -> i ()') - rearrange(seq, 'j -> () j')
        dist = dist.clamp(-max_pos_emb, max_pos_emb) + max_pos_emb
        rel_pos_emb = self.rel_pos_emb(dist).to(q)
        pos_attn = einsum('b h n d, n r d -> b h n r', q, rel_pos_emb) * self.scale
        dots = dots + pos_attn

        if exists(mask) or exists(context_mask):
            mask = default(mask, lambda: torch.ones(*x.shape[:2], device = device))
            context_mask = default(context_mask, mask) if not has_context else default(context_mask, lambda: torch.ones(*context.shape[:2], device = device))
            mask_value = -torch.finfo(dots.dtype).max
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(context_mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)

        attn = dots.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return self.dropout(out)

class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        mult = 4,
        dropout = 0.
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class ConformerConvModule(nn.Module):
    def __init__(
        self,
        dim,
        causal = False,
        expansion_factor = 2,
        kernel_size = 31,
        dropout = 0.):
        super().__init__()

        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n c -> b c n'),
            nn.Conv1d(dim, inner_dim * 2, 1),
            GLU(dim=1),
            DepthWiseConv1d(inner_dim, inner_dim, kernel_size = kernel_size, padding = padding),
            nn.BatchNorm1d(inner_dim) if not causal else nn.Identity(),
            Swish(),
            nn.Conv1d(inner_dim, dim, 1),
            Rearrange('b c n -> b n c'),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class GroupedConformerBlock(nn.Module):
    '''
    Uses idea from: https://arxiv.org/abs/2109.01163
    '''
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0.,
        grouped_attn_size = 2,
    ):
        super().__init__()
        self.ff1 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
        
        self.grouped_attn_size = grouped_attn_size

        self.attn = xAttention( 
            dim=dim * self.grouped_attn_size,
            dim_head=dim_head,
            heads=heads,
            dropout=attn_dropout,
            return_intermediates=False
        )

        self.conv = ConformerConvModule(dim = dim, causal = False, expansion_factor = conv_expansion_factor, kernel_size = conv_kernel_size, dropout = conv_dropout)
        self.ff2 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)

        self.attn = PreNorm(dim * self.grouped_attn_size, self.attn)
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))

        self.post_norm = nn.LayerNorm(dim)

    def group_x(self, x, lengths):
        '''
        Reshape x into x / attn_groups
        if x is not divisible by attn_groups then padding must be added
        additionally a new mask needs to be created for the new sequence length
        x: (b, n, c)
        lengths: length of each element in the batch excluding padding (used to create the new mask)
        '''
        b, n, c = x.shape
        attn_groups = self.grouped_attn_size
        padding_to_add = (attn_groups - n % attn_groups) % attn_groups # nice job co-pilot
        grouped_x = torch.cat([x, torch.zeros((b, padding_to_add, c), device = x.device)], dim = 1)
        grouped_x = grouped_x.view(b, torch.div(grouped_x.shape[1], attn_groups).int(), attn_groups * c)

        new_mask = torch.zeros((b, grouped_x.shape[1]), device = x.device, dtype = torch.bool)
        new_mask[:, :torch.div(lengths, attn_groups).ceil().int()] = 1
        return grouped_x, new_mask


    def ungroup_x(self, x, seq_len):
        '''
        Reshape x to be equal to previous shape, slice to the correct sequence length to account for any added padding
        x: (b, n, c)
        seq_len: original sequence length of x (including padding)
        '''
        return x.view(x.shape[0], -1, x.shape[2] * self.grouped_attn_size)[:, :seq_len, :]

    def forward(self, x, lengths, mask = None):
        x = self.ff1(x) + x

        grouped_x, grouped_mask = self.group_x(x, lengths)
        x = self.ungroup_x(self.attn(grouped_x, mask = grouped_mask), seq_len=x.shape[1]) + x

        x = self.conv(x) + x
        x = self.ff2(x) + x
        x = self.post_norm(x)
        return x


# Conformer Block

class ConformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0.
    ):
        super().__init__()
        self.ff1 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
        self.attn = Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)
        self.conv = ConformerConvModule(dim = dim, causal = False, expansion_factor = conv_expansion_factor, kernel_size = conv_kernel_size, dropout = conv_dropout)
        self.ff2 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)

        self.attn = PreNorm(dim, self.attn)
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))

        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x, mask = None):
        x = self.ff1(x) + x
        x = self.attn(x, mask = mask) + x
        x = self.conv(x) + x
        x = self.ff2(x) + x
        x = self.post_norm(x)
        return x


class FunnelConformerBlock(nn.Module):
    '''
    Similar to this: https://arxiv.org/pdf/2111.00127.pdf
    '''
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0.,
        residual = False
    ):
        super().__init__()
        self.ff1Q = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
        self.ff1KV = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)

        self.residual = residual

        self.attn = xAttention( 
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            dropout=attn_dropout,
            return_intermediates=False
        )

        self.convQ = ConformerConvModule(dim = dim, causal = False, expansion_factor = conv_expansion_factor, kernel_size = conv_kernel_size, dropout = conv_dropout)
        self.convKV = ConformerConvModule(dim = dim, causal = False, expansion_factor = conv_expansion_factor, kernel_size = conv_kernel_size, dropout = conv_dropout)

        self.ff2 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)

        self.attn = PreNorm(dim, self.attn) 

        self.ff1Q = Scale(0.5, PreNorm(dim, self.ff1Q))
        self.ff1KV = Scale(0.5, PreNorm(dim, self.ff1KV))

        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))

        self.post_norm = nn.LayerNorm(dim)


    def forward(self, Qs, KVs, mask = None, context_mask = None):
        # Initial Feedforwards scaled by 0.5
        Qs = self.ff1Q(Qs) + Qs
        KVs = self.ff1KV(KVs) + KVs
        Qs = self.convQ(Qs) + Qs
        KVs = self.convKV(KVs) + KVs

        x = self.attn(x=Qs, context=KVs, mask=mask, context_mask=context_mask) 
        if self.residual:
            x = x + Qs
            
        x = self.ff2(x) + x
        x = self.post_norm(x)
        
        return x


class IntegrationConformerBlock(nn.Module):
    '''
    Similar to this: https://arxiv.org/pdf/2111.00127.pdf
    '''
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0.,
        gating_method = 'Sigmoid'
    ):
        super().__init__()
        self.ff1Q = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
        self.ff1KV = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
        
        self.gating_method = gating_method
        assert gating_method in ['FiLM', 'Sigmoid', 'None'], 'Gating method must be one of FiLM, Conv, or None'
        
        self.gating_fn = lambda q, context: context # i.e. no gating
        if gating_method == 'FiLM': # doesn't work well as is
            self.init_FiLM(d_model=dim)
            self.gating_fn = self.apply_FiLM

        elif gating_method == 'Sigmoid':
            self.init_sigmoid_gating(d_model=dim)
            self.gating_fn = self.apply_sigmoid_gating

        #self.attn = Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)
        self.attn1 = xAttention( 
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            dropout=attn_dropout,
            return_intermediates=False
        )

        self.attn2 = xAttention( 
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            dropout=attn_dropout,
            return_intermediates=False
        )

        self.convQ = ConformerConvModule(dim = dim, causal = False, expansion_factor = conv_expansion_factor, kernel_size = conv_kernel_size, dropout = conv_dropout)
        self.convKV = ConformerConvModule(dim = dim, causal = False, expansion_factor = conv_expansion_factor, kernel_size = conv_kernel_size, dropout = conv_dropout)

        self.ff2 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)

        self.attn1 = PreNorm(dim, self.attn1)
        self.attn2 = PreNorm(dim, self.attn2)

        self.ff1Q = Scale(0.5, PreNorm(dim, self.ff1Q))
        self.ff1KV = Scale(0.5, PreNorm(dim, self.ff1KV))

        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))

        self.post_norm = nn.LayerNorm(dim)

    def init_sigmoid_gating(self, d_model):
        self.sigmoid_gating_w1 = nn.Linear(d_model, d_model, bias=False)
        self.sigmoid_gating_w2 = nn.Linear(d_model, d_model, bias=False)
        self.sigmoid_gating_b = nn.Parameter(torch.zeros(1, d_model))
        self.sigmoid = nn.Sigmoid()

    def init_FiLM(self, d_model):
        '''
        https://arxiv.org/pdf/1709.07871.pdf
        '''
        self.FiLM_r = nn.Linear(d_model, d_model)
        self.FiLM_h = nn.Linear(d_model, d_model)

    def apply_FiLM(self, q, x_context):
        r = self.FiLM_r(x_context)
        h = self.FiLM_h(x_context)
        return q * r + h

    def apply_sigmoid_gating(self, q, x_context):
        W1o = self.sigmoid_gating_w1(x_context)
        W2o = self.sigmoid_gating_w2(q)
        gate = self.sigmoid(W1o + W2o + self.sigmoid_gating_b)
        return q * gate + x_context * (1 - gate)

    def forward(self, Qs, KVs, mask = None, context_mask = None):
        Qs = self.ff1Q(Qs) + Qs
        KVs = self.ff1KV(KVs) + KVs

        Qs = self.convQ(Qs) + Qs
        KVs = self.convKV(KVs) + KVs

        x = self.attn1(x=Qs, context=KVs, mask=mask, context_mask=context_mask) + Qs
        x = self.gating_fn(Qs, x)

        x = self.attn2(x=Qs, context=x, mask=mask, context_mask=mask) + Qs # x.shape -> q.shape because of previous attention layer so we use the Q mask for both inputs here

        x = self.ff2(x) + x
        x = self.post_norm(x)
        
        return x
    