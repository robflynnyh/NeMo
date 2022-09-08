'''
Heavily relies on code from: https://github.com/lucidrains/conformer
'''
from math import pi, log
import torch
from torch import nn, einsum
import torch.nn.functional as F
import numpy as np
import os

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from nemo.collections.asr.parts.submodules.x_transformers_local import CrossAttender, Attention as xAttention
from inspect import isfunction

from batchrenorm import BatchRenorm1d

# helper functions

def exists(val):
    return val is not None

def notexists(val):
    return not exists(val)

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

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


# attention, feedforward, and conv module

# Weight Standardization for 1D Conv layers
class WSConv1d(nn.Conv1d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(WSConv1d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv1d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding, weight_standardize=False):
        super().__init__()
        self.padding = padding
        convtype = WSConv1d if weight_standardize else nn.Conv1d
        self.conv = convtype(chan_in, chan_out, kernel_size, groups = chan_in)

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)

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


class LocalAttentionMasking(nn.Module):
    def __init__(self, pattern:str='1, 4, 8, *'):
        super().__init__()
        self.pattern = [int(el) if el != '*' else '*' for el in [x.strip() for x in pattern.split(',')]]
        self.cache = None # cache mask so can reuse for the same input (if the batch size is the same)

    def create_local_mask(self, mem_vecs:int=10, batch_size:int=32, look_backward:int=1, causal:bool=False) -> torch.Tensor:
        """
        Creates mask for naive local attention (doesn't save any memory)
        """
        look_forward = 0 if causal else look_backward
        mask = torch.arange(0, batch_size).unsqueeze(0).repeat(batch_size, 1).repeat_interleave(mem_vecs, 1) - torch.arange(0, batch_size).unsqueeze(1)
        mask = mask.unsqueeze(1).repeat(1, mem_vecs, 1)
        mask = torch.where( (mask < look_forward + 1) & (mask > -look_backward - 1), torch.tensor(1), torch.tensor(0))
        return mask.bool()

    def create_head_specific_masks(self, mem_vecs:int=10, batch_size:int=32):
        """
        Creates mask for naive local attention specific to each head
        pattern determines lookahead and lookback values for each head (* means no masking)
        """
        masks = []
        for ptn in self.pattern:
            if ptn == '*':
                masks.append(torch.ones(batch_size, mem_vecs, mem_vecs*batch_size).bool())
            else:
                look_backward = ptn
                mask = self.create_local_mask(mem_vecs, batch_size, look_backward, causal=False)
                masks.append(mask)
        return torch.stack(masks, dim=1)

    def forward(self, x):
        '''
        apply mask to x (x is post-dot product but pre-softmax)
        '''
        if exists(self.cache) and self.cache.shape == x.shape[1:]:
            mask = self.cache
        else:
            mask = self.create_head_specific_masks(mem_vecs=x.shape[-2], batch_size=x.shape[0])
            self.cache = mask
        mask_val = -torch.finfo(x.dtype).max
        mask = mask.to(x.device)
        return x.masked_fill(~mask, mask_val)

def save_tensor(x, path='./attns'):
    indir = os.listdir(path)
    indir = [el for el in indir if el.endswith('.npy')]
    snum = 0
    if len(indir) > 0:
        snum = max([int(x.split('.')[0]) for x in indir]) + 1
    np.save(os.path.join(path, f'{snum}.npy'), x.detach().cpu().numpy())



class MappingAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        sparse_topk = None,
        num_bands = 32,
        max_freq = 60
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads= heads
        self.scale = dim_head ** -0.5

        self.num_bands = num_bands
        self.max_freq = max_freq

        dim_w_pos = dim + num_bands * 2 + 1
        self.to_q = nn.Linear(dim_w_pos, inner_dim, bias = True)
        self.to_k = nn.Linear(dim_w_pos, inner_dim, bias = True)
        self.to_v = nn.Linear(dim, inner_dim, bias = True)
        self.to_out = nn.Linear(inner_dim, dim)

        self.sparse_topk = sparse_topk
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def fourier_encode(x, max_freq=60, num_bands = 32):
        x = x.unsqueeze(-1)
        device, dtype, orig_x = x.device, x.dtype, x

        scales = torch.linspace(1., max_freq / 2, num_bands, device = device, dtype = dtype)
        scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

        x = x * scales * pi
        x = torch.cat([x.sin(), x.cos()], dim = -1)
        x = torch.cat((x, orig_x), dim = -1)
        return x


    def forward(self, x, context = None, mask = None, context_mask = None):
        n, device, h, has_context = x.shape[-2], x.device, self.heads, exists(context)
        context = default(context, x)

        qb, qn, qd, qdtype = *x.shape, x.dtype
        kb, kn, kd, kdtype = *context.shape, context.dtype
        
        axis_pos_q = rearrange(torch.linspace(-1., 1., steps=qn, device=device, dtype=qdtype), 'n -> n ()')
        axis_pos_k = rearrange(torch.linspace(-1., 1., steps=kn, device=device, dtype=kdtype), 'n -> n ()')
        
        enc_pos_q = repeat(rearrange(self.fourier_encode(axis_pos_q, max_freq=self.max_freq, num_bands=self.num_bands), 'n () d -> n d'), '... -> b ...', b = qb)
        enc_pos_k = repeat(rearrange(self.fourier_encode(axis_pos_k, max_freq=self.max_freq, num_bands=self.num_bands), 'n () d -> n d'), '... -> b ...', b = kb)

        for_q = torch.cat((x, enc_pos_q), dim=-1)
        for_k = torch.cat((context, enc_pos_k), dim=-1)
        for_v = context



        q, k, v = self.to_q(for_q), self.to_k(for_k), self.to_v(for_v) 
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v)) 

    
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if any(map(exists, (mask, context_mask))):
            q_mask = default(mask, lambda: torch.ones(*x.shape[:2], device = device).bool())
            k_mask = q_mask if not exists(context) else context_mask
            k_mask = default(k_mask, lambda: torch.ones(*context.shape[:2], device = device).bool())
            q_mask = rearrange(q_mask, 'b i -> b 1 i 1')
            k_mask = rearrange(k_mask, 'b j -> b 1 1 j')
            input_mask = q_mask * k_mask
            dots.masked_fill_(~input_mask, -torch.finfo(dots.dtype).max)

        if exists(self.sparse_topk) and self.sparse_topk < dots.shape[-1]:
            top, _ = dots.topk(self.sparse_topk, dim = -1)
            vk = top[..., -1].unsqueeze(-1).expand_as(dots)
            mask = dots < vk
            dots.masked_fill_(mask, -torch.finfo(dots.dtype).max)
            del mask

        attn = dots.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return self.dropout(out)




class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        sparse_topk = None,
        max_pos_emb = 256,
        rel_pos_emb = None,
        local_attn = False,
        local_attn_pattern = '1, 4, 8, *'
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads= heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, inner_dim, bias = True)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = True)
        self.to_out = nn.Linear(inner_dim, dim)

        self.sparse_topk = sparse_topk
        self.max_pos_emb = max_pos_emb
        self.rel_pos_emb = rel_pos_emb
        self.dropout = nn.Dropout(dropout)

        self.local_attn = local_attn
        if local_attn:
            assert len(local_attn_pattern.split(',')) == self.heads, f'local_attn_pattern must have {self.heads} heads'
            self.local_attn_mask = LocalAttentionMasking(local_attn_pattern)


    def forward(
            self, 
            x, 
            context = None, 
            mask = None, 
            context_mask = None, 
            gating_mask=None,
            return_attn = False
        ):
        n, device, h, max_pos_emb, has_context = x.shape[-2], x.device, self.heads, self.max_pos_emb, exists(context)
        context = default(context, x) 
      
        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1)) 
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v)) 

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if exists(self.rel_pos_emb):
            # shaw's relative positional embedding
            seq = torch.arange(n, device = device)
            dist = rearrange(seq, 'i -> i ()') - rearrange(seq, 'j -> () j') + max_pos_emb
            dist = dist.clamp(0, max_pos_emb - 1)
            rel_pos_emb = self.rel_pos_emb(dist).to(q)
            pos_attn = einsum('b h n d, n r d -> b h n r', q, rel_pos_emb) * self.scale
            dots = dots + pos_attn

        if any(map(exists, (mask, context_mask))):
            q_mask = default(mask, lambda: torch.ones(*x.shape[:2], device = device).bool())
            k_mask = q_mask if not exists(context) else context_mask
            k_mask = default(k_mask, lambda: torch.ones(*context.shape[:2], device = device).bool())
            q_mask = rearrange(q_mask, 'b i -> b 1 i 1')
            k_mask = rearrange(k_mask, 'b j -> b 1 1 j')
            input_mask = q_mask * k_mask
            dots.masked_fill_(~input_mask, -torch.finfo(dots.dtype).max)

        if exists(self.sparse_topk) and self.sparse_topk < dots.shape[-1]:
            top, _ = dots.topk(self.sparse_topk, dim = -1)
            vk = top[..., -1].unsqueeze(-1).expand_as(dots)
            mask = dots < vk
            dots.masked_fill_(mask, -torch.finfo(dots.dtype).max)
            del mask

        if self.local_attn:
            dots = self.local_attn_mask(dots)

        attn = dots.softmax(dim = -1)

        #save_tensor(attn, path='./attns')

        attn = attn * gating_mask if exists(gating_mask) else attn

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return (self.dropout(out), attn) if return_attn else self.dropout(out)


class GatingAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 44,
        cross_attn_pos_enc=None
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads= heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, inner_dim, bias = True)
        self.to_k = nn.Linear(dim, inner_dim, bias = True)
        self.cross_attn_pos_enc = cross_attn_pos_enc

    def forward(self, x, context = None, mask = None, context_mask = None, apply_pos_emb=False):
        n, device, h, has_context = x.shape[-2], x.device, self.heads, exists(context)
        context = default(context, x)

        if apply_pos_emb:
            context = self.cross_attn_pos_enc(context) if exists(self.cross_attn_pos_enc) else context

        q, k = (self.to_q(x), self.to_k(context)) 
        q, k = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k)) 

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if exists(mask) or exists(context_mask):
            mask = default(mask, lambda: torch.ones(*x.shape[:2], device = device))
            context_mask = default(context_mask, mask) if not has_context else default(context_mask, lambda: torch.ones(*context.shape[:2], device = device))
            mask_value = -torch.finfo(dots.dtype).max
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(context_mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)

        return dots.sigmoid()


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

def layernorm_fn(dim):
    return nn.Sequential( # have to transpose before layernorm
        Rearrange('b c n -> b n c'),
        nn.LayerNorm(dim),
        Rearrange('b n c -> b c n')
    )

def groupnorm_fn(dim, groups=32):
    return nn.GroupNorm(num_groups=groups, num_channels=dim)

    
class ConformerConvModule(nn.Module):
    def __init__(
        self,
        dim,
        causal = False,
        expansion_factor = 2,
        kernel_size = 31,
        dropout = 0.,
        conv_norm_type = 'batch_norm',
        weight_standardize = False
        ):
        super().__init__()

        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)

        norm_type = nn.BatchNorm1d 
        if conv_norm_type == 'layer_norm':
            norm_type = layernorm_fn
        elif conv_norm_type == 'group_norm':
            norm_type = groupnorm_fn
        elif conv_norm_type == 'batch_renorm':
            norm_type = BatchRenorm1d

        self.net1 = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n c -> b c n'),
            nn.Conv1d(dim, inner_dim * 2, 1),
            GLU(dim=1)
        )
        self.net2 = nn.Sequential(
            DepthWiseConv1d(inner_dim, inner_dim, kernel_size = kernel_size, padding = padding, weight_standardize=weight_standardize),
            norm_type(inner_dim) if not causal else nn.Identity(),
            Swish(),
            nn.Conv1d(inner_dim, dim, 1),
            Rearrange('b c n -> b n c'),
            nn.Dropout(dropout)
        )

    @staticmethod
    def mask_padding(x, pad_mask):
        if pad_mask is not None:
            x.masked_fill_(pad_mask.unsqueeze(1), 0)
        return x

    def forward(self, x, pad_mask=None):
        x = self.net1(x)
        x = self.mask_padding(x, pad_mask)
        x = self.net2(x)
        return x

class GroupedConformerBlock(nn.Module):
    '''
    Uses idea grouped attention idea
    from: https://arxiv.org/abs/2109.01163
    '''
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 1,
        conv_kernel_size = 31,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0.,
        grouped_attn_size = 2,
        pad_mask_fn = None, # function to create padding mask from list of lengths and sequence length i.e pad_mask_fn(lengths, seq_len) -> mask
        constant_dim = True, # if true uses projection layers to keep the dimension constant
        experimental_settings=False,
        talking_heads = True,
        num_mem_tokens = 16,
        pre_scale = False,
        qk_norm=False,
        qk_norm_groups=1,
        qk_norm_scale=1,
        sparse_topk=None,
    ):
        super().__init__()
        self.ff1 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
        
        self.constant_dim = constant_dim
        self.pad_mask_fn = pad_mask_fn
        self.pre_scale = pre_scale

        if self.pre_scale and self.constant_dim == False:
            raise ValueError('pre_scale is only supported when constant_dim is True')


        assert exists(pad_mask_fn) or grouped_attn_size == 1, 'pad_mask_fn must be defined if grouped_attn_size > 1'

        self.grouped_attn_size = grouped_attn_size
        
        dim_size = dim * self.grouped_attn_size if self.pre_scale == False else dim
        dim_head = dim_size // heads if self.constant_dim == False else dim // heads

        if self.pre_scale:
            self.scale_down = nn.Linear(dim_size * self.grouped_attn_size, dim_size)
            self.scale_up = nn.Linear(dim_size, dim_size * self.grouped_attn_size)

        #self.attn = Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)
        
        if experimental_settings == False:
            self.attn = xAttention( 
                dim=dim_size,
                dim_head=dim_head,
                heads=heads,
                dropout=attn_dropout,
                return_intermediates=False,
                talking_heads=talking_heads,
                num_mem_kv=num_mem_tokens,
                qk_norm = qk_norm,
                qk_norm_groups = qk_norm_groups,
                qk_norm_scale = qk_norm_scale,
                sparse_topk = sparse_topk
            )
        else:
            self.attn = xAttention( # possibly also add residual attention
                dim=dim_size,
                dim_head=dim_head,
                heads=heads,
                dropout=attn_dropout,
                return_intermediates=False,
                num_mem_kv=32,
                talking_heads = True, 
                #sparse_topk=8
            )   #        
        

        self.conv = ConformerConvModule(dim = dim, causal = False, expansion_factor = conv_expansion_factor, kernel_size = conv_kernel_size, dropout = conv_dropout)
        self.ff2 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)


        self.attn = PreNorm(dim_size, self.attn)
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))

        self.post_norm = nn.LayerNorm(dim)

    def group_x(self, x, lengths, mask):
        '''
        Reshape x into x / attn_groups
        if x is not divisible by attn_groups then padding must be added
        additionally a new mask needs to be created for the new sequence length
        x: (b, n, c)
        lengths: length of each element in the batch excluding padding (used to create the new mask)
        mask: this is a cache of the mask to prevent recomputation for each layer, first layer in each mini-batch mask == None so it is computed
        '''
        if self.grouped_attn_size == 1:
            return x, mask

        b, n, c = x.shape
        attn_groups = self.grouped_attn_size
        padding_to_add = (attn_groups - n % attn_groups) % attn_groups 
        grouped_x = torch.cat([x, torch.zeros((b, padding_to_add, c), device = x.device)], dim = 1)
        grouped_x = grouped_x.reshape(b, torch.div(grouped_x.shape[1], attn_groups).int(), attn_groups * c)

        if exists(mask):
            new_mask = mask
        else:
            new_lengths = torch.div(lengths, attn_groups).ceil().long()
            new_mask = self.pad_mask_fn(grouped_x.shape[1], new_lengths)

        if self.pre_scale:
            grouped_x = self.scale_down(grouped_x)

        return grouped_x, new_mask


    def ungroup_x(self, x, seq_len):
        '''
        Reshape x to be equal to previous shape, slice to the correct sequence length to account for any added padding
        x: (b, n, c)
        seq_len: original sequence length of x (including padding)
        '''
        if self.grouped_attn_size == 1:
            return x

        if self.pre_scale:
            x = self.scale_up(x)

        return x.reshape(x.shape[0], -1, x.shape[2] // self.grouped_attn_size)[:, :seq_len, :]

    def forward(self, x, lengths, mask=None, rel_pos=None, rotary_pos_emb=None):
        x = self.ff1(x) + x

        grouped_x, grouped_mask = self.group_x(x, lengths, mask)
        
        #check if rotary positional embedding is formed or is is a instance of a class (created in the first layer)
        if isinstance(rotary_pos_emb, nn.Module):
            max_rotary_emb_length = grouped_x.shape[1]
            rotary_pos_emb = rotary_pos_emb(max_rotary_emb_length, grouped_x.device)

        x = self.ungroup_x(self.attn(
            grouped_x, 
            mask = grouped_mask,
            rel_pos = rel_pos,
            rotary_pos_emb = rotary_pos_emb
            ), seq_len=x.shape[1]) + x


        x = self.conv(x) + x
        x = self.ff2(x) + x
        x = self.post_norm(x)
        return x, grouped_mask, rotary_pos_emb



class CtxCrossConformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 1,
        conv_kernel_size = 3,
        convKV_kernel_size = 10,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0.,
        sparse_topk = None,
        conv_norm_type = 'group_norm',
        local_attn = False,
        use_conv = True,
        weight_standardization = False,
    ):
        super().__init__()
        self.ff1 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
        self.ff1KV = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)

        self.attn = Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout, sparse_topk = sparse_topk, local_attn=local_attn)
        
        self.use_conv = use_conv
        if use_conv:
            self.conv = ConformerConvModule(dim = dim, causal = False, expansion_factor = conv_expansion_factor, kernel_size = conv_kernel_size, dropout = conv_dropout, conv_norm_type = conv_norm_type, weight_standardize=weight_standardization)
            self.convKV = ConformerConvModule(dim = dim, causal = False, expansion_factor = conv_expansion_factor, kernel_size = convKV_kernel_size, dropout = conv_dropout, conv_norm_type = conv_norm_type, weight_standardize=weight_standardization)

        self.ff2 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)

        self.attn = PreNorm(dim, self.attn)
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        self.ff1KV = Scale(0.5, PreNorm(dim, self.ff1KV))

        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))

        self.post_norm = nn.LayerNorm(dim)


    def forward(self, x, context, mask = None, context_mask = None, return_attn = False):
        x = self.ff1(x) + x
        context = self.ff1KV(context) + context

        if self.use_conv:
            x = self.conv(x, pad_mask=mask) + x
            context = self.convKV(context, pad_mask=context_mask) + context

        out_attn = self.attn(x, context=context, mask = mask, context_mask = context_mask, return_attn = return_attn) 
        out_attn, attn = out_attn if return_attn else (out_attn, None)
        x = out_attn + x

        x = self.ff2(x) + x
        x = self.post_norm(x)
    
        return x if not return_attn else (x, attn)

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
        conv_dropout = 0.,
        rel_pos_embs = None,
        max_pos_emb = 256
    ):
        super().__init__()
        self.ff1 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
        self.attn = Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout, rel_pos_emb = rel_pos_embs, max_pos_emb = max_pos_emb)
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



class CCATIntegrationConformerBlock(nn.Module):
    
    #Similar to this: https://arxiv.org/pdf/2111.00127.pdf
   
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 1,
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
        if gating_method == 'FiLM': 
            self.init_FiLM(d_model=dim)
            self.gating_fn = self.apply_FiLM

        elif gating_method == 'Sigmoid':
            self.init_sigmoid_gating(d_model=dim)
            self.gating_fn = self.apply_sigmoid_gating
        
        
        self.attn1 = xAttention( 
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            dropout=attn_dropout,
            return_intermediates=False,
            use_bias=True
        )

        self.attn2 = xAttention( 
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            dropout=attn_dropout,
            return_intermediates=False,
            use_bias=True
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
    
        #https://arxiv.org/pdf/1709.07871.pdf
    
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

    def forward(self, Qs, KVs, mask = None, context_mask = None, rel_pos = None):
        Qs = self.ff1Q(Qs) + Qs
        KVs = self.ff1KV(KVs) + KVs

        Qs = self.convQ(Qs) + Qs
        KVs = self.convKV(KVs) + KVs

        x = self.attn1(x=Qs, context=KVs, mask=mask, context_mask=context_mask, rel_pos=rel_pos) 
        # concat x and q along the seq dim
        x = torch.cat([x, Qs], dim=1)
        # extend the mask to include x
        xcmask = torch.ones(Qs.size(0), Qs.size(1), dtype=torch.bool, device=x.device)
        concatmask = torch.cat([xcmask, mask], dim=1)

        x = self.attn2(x=Qs, context=x, mask=mask, context_mask=concatmask, rel_pos=rel_pos) + Qs 
        x = self.ff2(x) + x
        x = self.post_norm(x)
        
        return x
    

class AdaptiveCrossAttnHistoryPositionalEncoding(nn.Module):
    '''
    Works for memory vector lengths that are dependent on the input sequence length.
    (CrossAttnHistoryPositionalEncoding only works for fixed memory vector lengths)
    '''
    def __init__(
        self,
        pos_dim = 176,
        max_seq_len = 24,
        num_context_vectors = 10
    ):
        super().__init__()
        self.pos_dim = pos_dim
        self.max_seq_len = max_seq_len
        self.num_context_vectors = num_context_vectors

        if self.max_seq_len % 2 != 0: 
            self.max_seq_len += 1 # make it even

        self.halfpoint = self.max_seq_len // 2

        self.embedding = nn.Embedding(max_seq_len, pos_dim) 
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        
        self.pos_embs = None

    def clear(self):
        self.pos_embs = None
    
    def create_sequence_pos_matrix(self, sub_batch_size, max_sub_batch_size, mem_sizes, max_mem_seq_len):
        max_mem_seq_len, mem_sizes = max_mem_seq_len.cpu(), mem_sizes.cpu() # keep on cpu for now

        interleave_nums = torch.ones(max_sub_batch_size, dtype=torch.long)
        interleave_nums[:mem_sizes.size(0)] = mem_sizes
 
        interleave_padding_amount = max_mem_seq_len - interleave_nums.sum()       
        interleave_nums[-1] += max(interleave_padding_amount, 0) # add a bunch of repeats to the last item so all the sequences are the same length (this is masked out later)

        halfpoint = self.halfpoint
        pos_matrix = torch.arange(0, max_sub_batch_size).unsqueeze(0).repeat(sub_batch_size, 1) - torch.arange(0, sub_batch_size).unsqueeze(1) + halfpoint
        pos_matrix = pos_matrix.clamp(min=0, max=self.max_seq_len - 1).repeat_interleave(interleave_nums, dim=1).long()

        if interleave_padding_amount < 0:
            pos_matrix = pos_matrix[:, :max_mem_seq_len]

        return pos_matrix

    @staticmethod
    def mem_size_list_for_segment(sub_batch_sizes, mem_sizes):
        '''Gets a list of the memory sizes for a segment'''
        culm_SB_lengths = sub_batch_sizes.cumsum(dim=0)
        mem_size_list = []
        for i, seg_len in enumerate(sub_batch_sizes):
            from_index = 0 if i == 0 else culm_SB_lengths[i-1]
            to_index = culm_SB_lengths[i]
            mem_size_list.append(mem_sizes[from_index:to_index])


        return mem_size_list

    def init_batch_pos_embs(self, sub_batch_sizes, mem_sizes, device, mem_mask=None):
        mem_size_list = self.mem_size_list_for_segment(sub_batch_sizes, mem_sizes)
        max_mem_seq_len = max([el.sum() for el in mem_size_list])

        pos_matrices = [self.create_sequence_pos_matrix(sub_batch_size, max(sub_batch_sizes), mem_size_list[i], max_mem_seq_len) for i, sub_batch_size in enumerate(sub_batch_sizes)]
        pos_matrices = torch.cat(pos_matrices, dim=0).to(device)
        mem_mask = mem_mask.to(device) if mem_mask is not None else None
    
        pos_embs = self.embedding(pos_matrices)
        
        if mem_mask is not None:
            assert mem_mask.shape[0] == pos_embs.shape[0] and mem_mask.shape[1] == pos_embs.shape[1], 'mem_mask and pos_embs have different shapes'
            if len(mem_mask.shape) == 2:
                mem_mask = mem_mask.unsqueeze(-1)
            assert len(mem_mask.shape) == 3, 'mem_mask has wrong shape'
            pos_embs.masked_fill_(~mem_mask, 0)
        self.pos_embs = pos_embs

    def forward(self, x):
        assert self.pos_embs is not None, 'pos_embs is not initialized'
        return x + self.pos_embs



class CrossAttnHistoryPositionalEncoding(nn.Module):
    def __init__(
        self,
        pos_dim = 176,
        max_seq_len = 24,
        num_context_vectors = 10
    ):
        super().__init__()
        self.pos_dim = pos_dim
        self.max_seq_len = max_seq_len
        self.num_context_vectors = num_context_vectors

        if self.max_seq_len % 2 != 0: 
            self.max_seq_len += 1 # make it even

        self.halfpoint = self.max_seq_len // 2

        self.embedding = nn.Embedding(max_seq_len, pos_dim) 
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        
        self.pos_embs = None

    def clear(self):
        self.pos_embs = None
    
    def create_sequence_pos_matrix(self, segment_length, max_segment_length):
        halfpoint = self.halfpoint
        pos_matrix = torch.arange(0, max_segment_length).unsqueeze(0).repeat(segment_length, 1) - torch.arange(0, segment_length).unsqueeze(1) + halfpoint
        pos_matrix = pos_matrix.clamp(min=0, max=self.max_seq_len - 1).repeat_interleave(self.num_context_vectors, dim=1).long()
        return pos_matrix

    def init_batch_pos_embs(self, segment_lengths, device, mem_mask=None):
        pos_matrices = [self.create_sequence_pos_matrix(segment_length, max(segment_lengths)) for segment_length in segment_lengths]
        pos_matrices = torch.cat(pos_matrices, dim=0).to(device)
        pos_embs = self.embedding(pos_matrices)
        if mem_mask is not None:
            assert mem_mask.shape[0] == pos_embs.shape[0] and mem_mask.shape[1] == pos_embs.shape[1], 'mem_mask and pos_embs have different shapes !'
            if len(mem_mask.shape) == 2:
                mem_mask = mem_mask.unsqueeze(-1)
            assert len(mem_mask.shape) == 3, 'mem_mask has wrong shape !'
            pos_embs.masked_fill_(~mem_mask, 0)
        self.pos_embs = pos_embs

    def forward(self, x):
        assert self.pos_embs is not None, 'pos_embs is not initialized !'
        return x + self.pos_embs


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
        conv_expansion_factor = 1,
        conv_kernel_size = 31,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0.,
        num_context_vectors = 10,
        shared_attn_layer = None,
        shared_gating_layer = None,
        seq_max_pos = 24,
        rel_pos_embs = None,
        max_pos_emb = 256
    ):
        super().__init__()
        self.ff1Q = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
        self.ff1KV = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
        
        cross_attn_positional_encoding = CrossAttnHistoryPositionalEncoding(
            pos_dim = dim,
            max_seq_len = seq_max_pos,
            num_context_vectors = num_context_vectors
        ) if shared_attn_layer is None else None

        self.attn1 = Attention( 
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            dropout=attn_dropout,
            cross_attn_pos_enc=cross_attn_positional_encoding
        ) if shared_attn_layer is None else shared_attn_layer

        self.attn2 = Attention( 
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            dropout=attn_dropout,
            max_pos_emb=max_pos_emb,
            rel_pos_emb=rel_pos_embs,
        )

        self.gating_attention = GatingAttention(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            cross_attn_pos_enc=cross_attn_positional_encoding
        ) if shared_gating_layer is None else shared_gating_layer
        
        self.num_context_vectors = num_context_vectors

        self.convQ = ConformerConvModule(dim = dim, causal = False, expansion_factor = conv_expansion_factor, kernel_size = conv_kernel_size, dropout = conv_dropout)
        self.convKV = ConformerConvModule(dim = dim, causal = False, expansion_factor = conv_expansion_factor, kernel_size = conv_kernel_size, dropout = conv_dropout)

        self.ff2 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)

        self.attn1 = PreNorm(dim, self.attn1)
        self.attn2 = PreNorm(dim, self.attn2)

        self.ff1Q = Scale(0.5, PreNorm(dim, self.ff1Q))
        self.ff1KV = Scale(0.5, PreNorm(dim, self.ff1KV))

        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))

        self.post_norm = nn.LayerNorm(dim)

   

    def forward(self, Qs, KVs, mask = None, context_mask = None):
        Qs = self.ff1Q(Qs) + Qs
        KVs = self.ff1KV(KVs) + KVs

        Qs = self.convQ(Qs) + Qs
        KVs = self.convKV(KVs) + KVs

        local_context_vectors = Qs[:, :self.num_context_vectors, :]

        ctx_attn_gating_scores = self.gating_attention(x=local_context_vectors, context=KVs, mask=None, context_mask=None, apply_pos_emb=True)
        context_info = self.attn1(x=local_context_vectors, context=KVs, mask=None, context_mask=None, gating_mask=ctx_attn_gating_scores) + local_context_vectors

        uQs = Qs.clone() # so we don't break the compute graph
        uQs[:, :self.num_context_vectors, :] = context_info 

        attn_gating_scores = self.gating_attention(x=uQs, context=uQs, mask=mask, context_mask=mask)

        x = self.attn2(x=uQs, context=uQs, mask=mask, context_mask=mask, gating_mask=attn_gating_scores) + uQs
        x = self.ff2(x) + x
        x = self.post_norm(x)
        
        return x
    
class CrossConformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 1,
        conv_kernel_size = 31,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0.,
        sparse_topk = None,
        conv_norm_type = 'group_norm',
        local_attn = False
    ):
        super().__init__()
        self.ff1 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
        self.ff1KV = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)

        self.attn = Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout, sparse_topk = sparse_topk, local_attn=local_attn)
        self.conv = ConformerConvModule(dim = dim, causal = False, expansion_factor = conv_expansion_factor, kernel_size = conv_kernel_size, dropout = conv_dropout, conv_norm_type = conv_norm_type)
        self.convKV = ConformerConvModule(dim = dim, causal = False, expansion_factor = conv_expansion_factor, kernel_size = conv_kernel_size, dropout = conv_dropout, conv_norm_type = conv_norm_type)

        self.ff2 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)

        self.attn = PreNorm(dim, self.attn)
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        self.ff1KV = Scale(0.5, PreNorm(dim, self.ff1KV))

        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))

        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x, context):
        x = self.ff1(x) + x
        context = self.ff1KV(context) + context

        x = self.conv(x) + x
        context = self.convKV(context) + context

        x = self.attn(x, context=context) + x
        
        x = self.ff2(x) + x
        x = self.post_norm(x)
        return x