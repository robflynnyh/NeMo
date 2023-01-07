# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import torch
from torch import nn as nn
from torch.nn import LayerNorm

from torch.nn import functional as F
from einops import rearrange, repeat

from nemo.collections.asr.parts.submodules.multi_head_attention import (
    MultiHeadAttention,
    RelPositionMultiHeadAttention,
    RelPositionSinusoidalGAU,
    MyopicAttention,
    CosineAttention
)

from nemo.collections.asr.parts.utils.helpers import (
    exists,
    isfalse,
)

from nemo.collections.asr.parts.utils.activations import Swish
from nemo.core.classes.mixins import AccessMixin
from nemo.core.classes.mixins.adapter_mixins import AdapterModuleMixin
from nemo.utils import logging
from nemo.collections.asr.parts.utils.helpers import exists, isfalse

from batchrenorm import BatchRenorm1d

__all__ = ['ConformerConvolution', 'ConformerFeedForward', 'ConformerLayer', 'CrossConformerLayer']

def exists(x):
    return x is not None

class ConformerLayer(torch.nn.Module, AdapterModuleMixin, AccessMixin):
    """A single block of the Conformer encoder.

    Args:
        d_model (int): input dimension of MultiheadAttentionMechanism and PositionwiseFeedForward
        d_ff (int): hidden dimension of PositionwiseFeedForward
        n_heads (int): number of heads for multi-head attention
        conv_kernel_size (int): kernel size for depthwise convolution in convolution module
        dropout (float): dropout probabilities for linear layers
        dropout_att (float): dropout probabilities for attention distributions
    """

    def __init__(
        self,
        layer_idx,
        d_model,
        d_ff,
        self_attention_model='rel_pos',
        n_heads=4,
        conv_kernel_size=31,
        conv_norm_type='batch_norm',
        dropout=0.1,
        dropout_att=0.1,
        pos_bias_u=None,
        pos_bias_v=None,
        sparse_topk=None,
        weight_standardization=False,
        num_memory_vectors=None,
        GAU=False, # whether to use 1-headed Gated Attention Unit
        qk_dim_divisor=4, # for GAU
        max_keep_keys = 64, # myopic attention
        chunk_window = 8, # myopic attention
        talking_heads = 'pre', # only implemented with cosine attn atm 'pre' or 'post' or 'both' or 'none'
        hydra_weighting = False, # https://arxiv.org/pdf/2209.07484.pdf
        spatial_attention_dropout = False,
        bottleneck_initial_ff = False
    ):
        super(ConformerLayer, self).__init__()

        if self_attention_model == "flash": # flash attention https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/flash_attention.py
            from flash_attn.flash_attention import FlashMHA 

        assert isfalse(GAU) or self_attention_model == 'rel_pos', "GAU is only supported for relative positional attention at the moment"

        self.GAU = GAU
        self.self_attention_model = self_attention_model
        self.n_heads = n_heads
        self.fc_factor = 0.5

        self.hydra_weighting = hydra_weighting
        self.layer_idx = layer_idx
        self.sparse_topk = sparse_topk

        if isfalse(GAU):
            # first feed forward module
            self.norm_feed_forward1 = LayerNorm(d_model)
            ff1_d_ff = d_ff
            ff1_d0 = dropout
            if bottleneck_initial_ff:
                ff1_d_ff = d_model // 2
                ff1_d0 = 0.0
            self.feed_forward1 = ConformerFeedForward(d_model=d_model, d_ff=ff1_d_ff, dropout=ff1_d0)

        # convolution module
        self.norm_conv = LayerNorm(d_model)
        self.conv = ConformerConvolution(
            d_model=d_model, 
            kernel_size=conv_kernel_size, 
            norm_type=conv_norm_type, 
            weight_standardization=weight_standardization
        )

        self.num_memory_vectors = num_memory_vectors
        if self.num_memory_vectors:
            self.norm_mem_vecs_conv = LayerNorm(d_model)
            mem_kernel = self.num_memory_vectors // 3
            self.conv_mem_vecs = ConformerConvolution(d_model=d_model, kernel_size=mem_kernel, norm_type=conv_norm_type, weight_standardization=weight_standardization)

        # multi-headed self-attention module
        self.norm_self_att = LayerNorm(d_model)
        if self_attention_model == 'rel_pos':
            if isfalse(GAU):
                self.self_attn = RelPositionMultiHeadAttention(
                    n_head=n_heads, n_feat=d_model, dropout_rate=dropout_att, pos_bias_u=pos_bias_u, pos_bias_v=pos_bias_v, sparse_topk=self.sparse_topk
                )
            else:
                self.self_attn = RelPositionSinusoidalGAU(
                    n_feat=d_model, dropout_rate=dropout_att, pos_bias_u=pos_bias_u, pos_bias_v=pos_bias_v, sparse_topk=self.sparse_topk, qk_dim_divisor=qk_dim_divisor
                )
        elif self_attention_model == 'abs_pos':
            self.self_attn = MultiHeadAttention(n_head=n_heads, n_feat=d_model, dropout_rate=dropout_att, sparse_topk=self.sparse_topk)
        elif self_attention_model == 'flash':
            self.self_attn = FlashMHA(
                embed_dim = d_model,
                num_heads = n_heads,
                bias = True,
                attention_dropout = dropout_att,
                causal = False,
                use_rotary_emb = '1d',
            )
        elif self_attention_model == 'myopic':
            self.self_attn = MyopicAttention(
                n_feats = d_model,
                head_dim = d_model // n_heads,
                n_heads = n_heads,
                max_keep_keys = max_keep_keys,
                chunk_window = chunk_window,
            )
        elif self_attention_model == 'cosine':
            if hydra_weighting == False or layer_idx <= 4:
                self.self_attn = CosineAttention(
                    n_feats = d_model,
                    head_dim = max(32, d_model // n_heads),
                    n_heads = n_heads,
                    bias = False,
                    temperature = 15.5,
                    causal = False,
                    shared_kv = True,
                    talking_heads = talking_heads,
                    dropout = dropout_att,
                    spatial_attention_dropout = spatial_attention_dropout
                )
            else:
                self.self_attn = HydraAttention(d_model=d_model)
        else:
            raise ValueError(
                f"'{self_attention_model}' is not not a valid value for 'self_attention_model', "
                f"valid values can be from ['rel_pos', 'abs_pos']"
            )

        if isfalse(GAU):
            # second feed forward module
            self.norm_feed_forward2 = LayerNorm(d_model)
            self.feed_forward2 = ConformerFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)


        self.dropout = nn.Dropout(dropout)
        self.norm_out = LayerNorm(d_model)

    @staticmethod
    def slice_mem_tokens(x, num_mem_tokens):
        return x[:, :num_mem_tokens, :], x[:, num_mem_tokens:, :]

    def forward(self, x, att_mask=None, pos_emb=None, pad_mask=None, num_memory_vectors=None, mem_pos_emb=None, return_attentions=False):
        """
        Args:
            x (torch.Tensor): input signals (B, T, d_model)
            att_mask (torch.Tensor): attention masks(B, T, T)
            pos_emb (torch.Tensor): (L, 1, d_model)
            pad_mask (torch.tensor): padding mask
        Returns:
            x (torch.Tensor): (B, T, d_model)
        """
        residual = x
        if isfalse(self.GAU):
            x = self.norm_feed_forward1(x)
            x = self.feed_forward1(x)
            residual = residual + self.dropout(x) * self.fc_factor

        for i in range(1) if isfalse(self.GAU) else range(2):
            x = self.norm_self_att(residual)
            #print(x.shape, pos_emb.shape, 'attn stuff')
            if self.self_attention_model == 'rel_pos':
                qkv = {'query': x, 'key': x, 'value': x} if isfalse(self.GAU) else {'qkv': x}
                x = self.self_attn(**qkv, mask=att_mask, pos_emb=pos_emb, mem_pos_emb=mem_pos_emb, return_attentions=return_attentions)
            elif self.self_attention_model == 'abs_pos':
                x = self.self_attn(query=x, key=x, value=x, mask=att_mask, return_attentions=return_attentions)
            elif self.self_attention_model == 'flash':
                x, weights = self.self_attn(x=x, key_padding_mask=pad_mask, need_weights=return_attentions)
                x = x if return_attentions is False else (x, weights)
            elif self.self_attention_model == "myopic":
                x = self.self_attn(x=x, mask=pad_mask, return_attention=return_attentions)
            elif self.self_attention_model == "cosine":
                if self.hydra_weighting == False or self.layer_idx <= 4:
                    x = self.self_attn(x=x, pos_fn=pos_emb, mask=pad_mask)
                else:
                    x = self.self_attn(x=x)
            else:
                x = None
            x, attns = x if return_attentions else (x, None)
            residual = residual + self.dropout(x)

        conv_pad_mask = pad_mask
        if exists(num_memory_vectors): # slice out memory vectors from the input to the convolution layer
            mem_vecs, residual = self.slice_mem_tokens(residual, num_memory_vectors)
            conv_pad_mask = pad_mask[:, num_memory_vectors:]

        if exists(self.num_memory_vectors):
            i_mem = self.norm_mem_vecs_conv(mem_vecs)
            i_mem = self.conv_mem_vecs(i_mem)
            mem_vecs = self.dropout(i_mem) + mem_vecs

        x = self.norm_conv(residual)
        x = self.conv(x, conv_pad_mask)

        residual = residual + self.dropout(x)

        if exists(num_memory_vectors): # concatenate memory vectors to the output of the convolution layer
            residual = torch.cat([mem_vecs, residual], dim=1)

        if isfalse(self.GAU):
            x = self.norm_feed_forward2(residual)
            x = self.feed_forward2(x)
            residual = residual + self.dropout(x) * self.fc_factor
        x = self.norm_out(residual) 

        if self.is_adapter_available():
            # Call the adapters
            x = self.forward_enabled_adapters(x)

        if self.is_access_enabled():
            self.register_accessible_tensor(tensor=x)

        return x if not return_attentions else (x, attns)




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

class TwoWayHydraAttention(nn.Module):
    def __init__(self, d_model, heads=1):
        super(TwoWayHydraAttention, self).__init__()
        self.d_model = d_model
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.kv = nn.Linear(d_model, d_model * 2, bias=False)
        self.per_dim_scaler = nn.Parameter(torch.ones(1,1,d_model), requires_grad=True)
        self.per_dim_bias = nn.Parameter(torch.zeros(1,1,d_model), requires_grad=True)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.heads = heads

    def forward(self, x, mask=None):
        '''x: (B, T, D)'''
        h = self.heads
        kv = self.kv(x)
        k, v = rearrange(kv, 'b n (h d) -> b n h d ()', h=h).chunk(2, dim=-2)
        knormD = k / (k.norm(dim=-2, keepdim=True))
        xnorm = x / (x.norm(dim=-1, keepdim=True))
     
        kv = knormD.transpose(-2,-1).matmul(v)
        xnorm = rearrange(xnorm, 'b n d -> b n () d ()')
        xnormW = (xnorm * kv).sum(dim=2).squeeze()

        xnormW = xnormW * self.per_dim_scaler + self.per_dim_bias
        if mask is not None:
            mask = rearrange(mask, 'b n -> b n ()')
            xnormW = xnormW.masked_fill(mask, 0)

        q, kv = self.q(xnormW), self.kv(xnormW)
        remap = lambda x: rearrange(x, 'b n d -> b d n ()')
        q, kv = map(remap, (q, kv))
        k, v = kv.chunk(2, dim=1)
        knorm = k / (k.norm(dim=-2, keepdim=True))
        qnorm = q / (q.norm(dim=-2, keepdim=True))
        kv = knorm.transpose(-2,-1).matmul(v)
        out = qnorm * kv
        out = rearrange(out, 'b d n () -> b n d')
        out = self.out(out) 
        return out


class HydraAttention(nn.Module):
    def __init__(self, d_model, output_layer='linear'):
        '''
        output_layer: 'linear' | 'none'
        '''
        super(HydraAttention, self).__init__()
        self.d_model = d_model
        self.qkv = nn.Linear(d_model, d_model * 3)
        if output_layer == 'linear':
            self.out = nn.Linear(d_model, d_model)
        elif output_layer == 'none':
            self.out = nn.Identity()

    def forward(self, x):
        '''x: (B, T, D)'''
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q / q.norm(dim=-1, keepdim=True)
        k = k / k.norm(dim=-1, keepdim=True)
        kv = (k * v).sum(dim=-2, keepdim=True)
        out = q * kv
        return self.out(out)

class ConformerConvolution(nn.Module):
    """The convolution module for the Conformer model.
    Args:
        d_model (int): hidden dimension
        kernel_size (int): kernel size for depthwise convolution
    """

    def __init__(
            self, 
            d_model, 
            kernel_size, 
            norm_type='batch_norm', 
            weight_standardization=False,
            ):

        super(ConformerConvolution, self).__init__()
        assert (kernel_size - 1) % 2 == 0
        self.d_model = d_model
     

        self.pointwise_conv1 = nn.Conv1d(
            in_channels=d_model, out_channels=d_model * 2, kernel_size=1, stride=1, padding=0, bias=True
        )

        dw_conv = nn.Conv1d if weight_standardization == False else WSConv1d
        self.depthwise_conv = dw_conv(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=d_model,
            bias=True,
        )
        if norm_type == 'batch_norm':
            self.batch_norm = nn.BatchNorm1d(d_model)
        elif norm_type == 'layer_norm':
            self.batch_norm = nn.LayerNorm(d_model)
        elif norm_type == 'group_norm':
            self.batch_norm = nn.GroupNorm(num_groups=32, num_channels=d_model)
        elif norm_type == 'batch_renorm':
            self.batch_norm = BatchRenorm1d(d_model)
        else:
            raise ValueError(f"conv_norm_type={norm_type} is not valid!")

        self.activation = Swish()
        self.pointwise_conv2 = nn.Conv1d(
            in_channels=d_model, out_channels=d_model, kernel_size=1, stride=1, padding=0, bias=True
        )

    def forward(self, x, pad_mask=None):
  
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = nn.functional.glu(x, dim=1)

        if pad_mask is not None: 
            x = x.float().masked_fill(pad_mask.unsqueeze(1), 0.0)

        x = self.depthwise_conv(x)

        if isinstance(self.batch_norm, nn.LayerNorm):
            x = x.transpose(1, 2)
            x = self.batch_norm(x)
            x = x.transpose(1, 2)
        else:
            x = self.batch_norm(x)

        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = x.transpose(1, 2)
        return x


class ConformerFeedForward(nn.Module):
    """
    feed-forward module of Conformer model.
    """

    def __init__(self, d_model, d_ff, dropout, activation=Swish()):
        super(ConformerFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
