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

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
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

"""
Part of this code is adopted from https://github.com/espnet/espnet
"""

import math

import torch
import torch.nn.functional as F
from torch import nn, einsum
from nemo.collections.asr.parts.utils.activations import LaplacianAttnFn, ReLUSquared
from nemo.collections.asr.parts.utils.helpers import exists, isfalse
from einops import rearrange, repeat
from nemo.collections.asr.parts.submodules.dynamic_positions import DynamicPositionBias


__all__ = [
    'RelPositionMultiHeadAttention',
    'RelPositionalEncoding',
    'PositionalEncoding',
]


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention layer of Transformer.
    Args:
        n_head (int): number of heads
        n_feat (int): size of the features
        dropout_rate (float): dropout rate
    """

    def __init__(self, n_head, n_feat, dropout_rate, sparse_topk=None, activation='softmax'):
        """Construct an MultiHeadedAttention object."""
        super(MultiHeadAttention, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.s_d_k = math.sqrt(self.d_k)
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)

        self.linear_out = nn.Linear(n_feat, n_feat)

        self.dropout = nn.Dropout(p=dropout_rate)
        self.sparse_topk = sparse_topk
        assert activation in ['softmax', 'laplace', 'relusq'], f'Unsupported activation {activation}'
        if activation == 'softmax':
            self.activation = nn.Softmax(dim=-1)
        elif activation == 'laplace':
            self.activation = LaplacianAttnFn()
        elif activation == 'relusq':
            self.activation = ReLUSquared()
    

    def forward_qkv(self, query, key, value):
        """Transforms query, key and value.
        Args:
            query (torch.Tensor): (batch, time1, size)
            key (torch.Tensor): (batch, time2, size)
            value (torch.Tensor): (batch, time2, size)
        returns:
            q (torch.Tensor): (batch, head, time1, size)
            k (torch.Tensor): (batch, head, time2, size)
            v (torch.Tensor): (batch, head, time2, size)
        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        return q, k, v

    def forward_attention(self, value, scores, mask, return_attentions=False):
        """Compute attention context vector.
        Args:
            value (torch.Tensor): (batch, time2, size)
            scores(torch.Tensor): (batch, time1, time2)
            mask(torch.Tensor): (batch, time1, time2)
        returns:
            value (torch.Tensor): transformed `value` (batch, time2, d_model) weighted by the attention scores
        """
        if self.sparse_topk is not None and self.sparse_topk < scores.shape[-1]:
            top, _ = scores.topk(self.sparse_topk, dim = -1)
            vk = top[..., -1].unsqueeze(-1).expand_as(scores)
            tk_mask = scores < vk
            scores.masked_fill_(tk_mask, -torch.finfo(scores.dtype).max)
            del tk_mask

        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1)  # (batch, 1, time1, time2)
            scores = scores.masked_fill(mask, -10000.0) 
            attn = self.activation(scores).masked_fill(mask, 0.0)  # (batch, head, time1, time2)
        else:
            attn = self.activation(scores)  # (batch, head, time1, time2)

        
        #print('self std', attn[0][0][0].std())

        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).reshape(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)

        return self.linear_out(x) if not return_attentions else (self.linear_out(x), attn)

    def forward(self, query, key, value, mask, pos_emb=None, return_attentions=False):
        """Compute 'Scaled Dot Product Attention'.
        Args:
            query (torch.Tensor): (batch, time1, size)
            key (torch.Tensor): (batch, time2, size)
            value(torch.Tensor): (batch, time2, size)
            mask (torch.Tensor): (batch, time1, time2)
        returns:
            output (torch.Tensor): transformed `value` (batch, time1, d_model) weighted by the query dot key attention
        """
        q, k, v = self.forward_qkv(query, key, value)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.s_d_k
        return self.forward_attention(v, scores, mask, return_attentions=return_attentions)


class RelPositionMultiHeadAttention(MultiHeadAttention):
    """Multi-Head Attention layer of Transformer-XL with support of relative positional encoding.
    Paper: https://arxiv.org/abs/1901.02860
    Args:
        n_head (int): number of heads
        n_feat (int): size of the features
        dropout_rate (float): dropout rate
    """

    def __init__(self, n_head, n_feat, dropout_rate, pos_bias_u, pos_bias_v, sparse_topk=None, activation='softmax'):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate, sparse_topk, activation)
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # these two learnable biases are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        if pos_bias_u is None or pos_bias_v is None:
            self.pos_bias_u = nn.Parameter(torch.FloatTensor(self.h, self.d_k))
            self.pos_bias_v = nn.Parameter(torch.FloatTensor(self.h, self.d_k))
            # nn.init.normal_(self.pos_bias_u, 0.0, 0.02)
            # nn.init.normal_(self.pos_bias_v, 0.0, 0.02)
            nn.init.zeros_(self.pos_bias_u)
            nn.init.zeros_(self.pos_bias_v)
        else:
            self.pos_bias_u = pos_bias_u
            self.pos_bias_v = pos_bias_v

    def rel_shift(self, x):
        """Compute relative positional encoding.
        Args:
            x (torch.Tensor): (batch, nheads, time, 2*time-1)
        """
        b, h, qlen, pos_len = x.size()  # (b, h, t1, t2)
        # need to add a column of zeros on the left side of last dimension to perform the relative shifting
        x = torch.nn.functional.pad(x, pad=(1, 0))  # (b, h, t1, t2+1)
        x = x.view(b, h, -1, qlen)  # (b, h, t2+1, t1)
        # need to drop the first row 
        x = x[:, :, 1:].view(b, h, qlen, pos_len)  # (b, h, t1, t2) 
        return x

    def forward(self, query, key, value, mask, pos_emb, mem_pos_emb=None, return_attentions=False):
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.
        Args:
            query (torch.Tensor): (batch, time1, size)
            key (torch.Tensor): (batch, time2, size)
            value(torch.Tensor): (batch, time2, size)
            mask (torch.Tensor): (batch, time1, time2)
            pos_emb (torch.Tensor) : (batch, time1, size)
        Returns:
            output (torch.Tensor): transformed `value` (batch, time1, d_model) weighted by the query dot key attention
        """
        q, k, v = self.forward_qkv(query, key, value)
        if exists(mem_pos_emb):
            b, n, d = mem_pos_emb.size()
            mem_pos_emb = mem_pos_emb.view(b, self.h, n, d // self.h)
            q = q + mem_pos_emb
            k = k + mem_pos_emb
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)

        n_batch_pos = pos_emb.size(0)
        
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, time1, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)   
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        # compute matrix b and matrix d
        # (batch, head, time1, time2)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd) # ahh
        # drops extra elements in the matrix_bd to match the matrix_ac's size
        matrix_bd = matrix_bd[:, :, :, : matrix_ac.size(-1)]

        scores = (matrix_ac + matrix_bd) / self.s_d_k  # (batch, head, time1, time2)
       
        return self.forward_attention(v, scores, mask, return_attentions=return_attentions)

class OffsetScale(nn.Module):
    def __init__(self, dim, heads = 1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(heads, dim))
        self.beta = nn.Parameter(torch.zeros(heads, dim))
        nn.init.normal_(self.gamma, std = 0.02)

    def forward(self, x):
        out = einsum('... d, h d -> ... h d', x, self.gamma) + self.beta
        return out.unbind(dim = -2)

class RelPositionSinusoidalGAU(nn.Module):
    """
    Single headed attention https://arxiv.org/pdf/2202.10447.pdf with gating and relative positional encoding
    Implementation adapted from: https://github.com/lucidrains/FLASH-pytorch/blob/main/flash_pytorch/flash_pytorch.py

    Multi-Head Attention layer of Transformer-XL with support of relative positional encoding.
    Paper: https://arxiv.org/abs/1901.02860
    Args:
        n_feat (int): size of the features
        dropout_rate (float): dropout rate
    """

    def __init__(self, n_feat, dropout_rate, pos_bias_u, pos_bias_v, sparse_topk=None, activation='laplace', expansion_factor=2, qk_dim_divisor=4):
        """
            activation: can either be softmax, laplace, or relusq (relu squared)
            sparse_topk: if not None, then use sparse attention with topk selection
            expansion_factor: expansion factor for the gating network
            qk_dim_divisor: the divisor for the qk dimension i.e if n_feats is 512 and qk_dim_divisor is 2, then qk dimension is 256
        """
        super().__init__()
        self.h = 1 # GAU is single headed
        
        self.sparse_topk = sparse_topk

        self.dropout = nn.Dropout(dropout_rate)
        hidden_dim = int(n_feat * expansion_factor)
        self.qk_dim = n_feat // qk_dim_divisor

        assert activation in ['softmax', 'laplace', 'relusq']
        if activation == 'softmax':
            self.activation = nn.Softmax(dim=-1)
        elif activation == 'laplace':
            self.activation = LaplacianAttnFn()
        elif activation == 'relusq':
            self.activation = ReLUSquared()

        self.to_hidden = nn.Sequential( # used for gate and value
            nn.Linear(n_feat, hidden_dim * 2),
            nn.SiLU()
        )

        self.to_qk = nn.Sequential(
            nn.Linear(n_feat, self.qk_dim),
            nn.SiLU()
        )
        self.to_out = nn.Linear(hidden_dim, n_feat)
        
        self.offset_scale = OffsetScale(self.qk_dim, heads = 2)

        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # these two learnable biases are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        if pos_bias_u is None or pos_bias_v is None:
            self.pos_bias_u = nn.Parameter(torch.FloatTensor(self.h, self.qk_dim))
            self.pos_bias_v = nn.Parameter(torch.FloatTensor(self.h, self.qk_dim))
            # nn.init.normal_(self.pos_bias_u, 0.0, 0.02)
            # nn.init.normal_(self.pos_bias_v, 0.0, 0.02)
            nn.init.zeros_(self.pos_bias_u)
            nn.init.zeros_(self.pos_bias_v)
        else:
            self.pos_bias_u = pos_bias_u
            self.pos_bias_v = pos_bias_v

    def rel_shift(self, x):
        """Compute relative positional encoding.
        Args:
            x (torch.Tensor): (batch, nheads, time, 2*time-1)
        """
        b, qlen, pos_len = x.size()  # (b, t1, t2)
        x.unsqueeze_(1)  # (b, 1, t1, t2)
        # need to add a column of zeros on the left side of last dimension to perform the relative shifting
        x = torch.nn.functional.pad(x, pad=(1, 0))  # (b, h, t1, t2+1)
        x = x.view(b, 1, -1, qlen)  # (b, h, t2+1, t1)
        # need to drop the first row 
        x = x[:, :, 1:].view(b, 1, qlen, pos_len)  # (b, h, t1, t2) 
        return x.squeeze(1)  # (b, t1, t2)

    def forward(self, qkv, mask, pos_emb, mem_pos_emb=None, return_attentions=False):
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.
        Args:
            query (torch.Tensor): (batch, time1, size)
            key (torch.Tensor): (batch, time2, size)
            value(torch.Tensor): (batch, time2, size)
            mask (torch.Tensor): (batch, time1, time2)
            pos_emb (torch.Tensor) : (batch, time1, size)
        Returns:
            output (torch.Tensor): transformed `value` (batch, time1, d_model) weighted by the query dot key attention
        """
        #q, k, v = self.forward_qkv(query, key, value)
        qk = self.to_qk(qkv)
        v, gate = self.to_hidden(qkv).chunk(2, dim=-1)
        q, k = self.offset_scale(qk)

        if exists(mem_pos_emb):
            b, n, d = mem_pos_emb.size()
            mem_pos_emb = mem_pos_emb.view(b, self.h, n, d // self.h)
            q = q + mem_pos_emb
            k = k + mem_pos_emb
    

        n_batch_pos = pos_emb.size(0)
       
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.qk_dim)
        p = p.transpose(1, 2)  # (batch, head, time1, d_k)

        # (batch, head, time1, d_k)

        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)   
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = einsum(' b d n , b j d -> b n j', q_with_bias_u, k)
        #matrix_ac = torch.matmul(q_with_bias_u, k)

        # compute matrix b and matrix d
        # (batch, head, time1, time2)
        
        matrix_bd = torch.einsum('b d n, ... j d -> ... b n j', q_with_bias_v, p).squeeze()
        
        matrix_bd = self.rel_shift(matrix_bd) # ahh
        # drops extra elements in the matrix_bd to match the matrix_ac's size

        matrix_bd = matrix_bd[:, :, : matrix_ac.size(-1)]

        scores = (matrix_ac + matrix_bd) / math.sqrt(self.qk_dim)  # (batch, head, time1, time2)

        if self.sparse_topk is not None and self.sparse_topk < scores.shape[-1]:
            top, _ = scores.topk(self.sparse_topk, dim = -1)
            vk = top[..., -1].unsqueeze(-1).expand_as(scores)
            tk_mask = scores < vk
            scores.masked_fill_(tk_mask, -torch.finfo(scores.dtype).max)
            del tk_mask

        n_batch = v.size(0)
        if mask is not None:
            scores = scores.masked_fill(mask, -torch.finfo(scores.dtype).max)
            attn = self.activation(scores).masked_fill(mask, 0.0)  # (batch, head, time1, time2)
        else:
            attn = self.activation(scores)  # (batch, head, time1, time2)

        p_attn = self.dropout(attn)

        out = einsum("b i j, b j d -> b i d", p_attn, v)  # (batch, time1, d_model)
        out = out * gate
        out = self.to_out(out)

        return out if isfalse(return_attentions) else (out, attn)
        #return self.forward_attention(v, scores, mask, return_attentions=return_attentions)


class PositionalEncoding(torch.nn.Module):
    """Fixed sinusoidal positional encoding.
    Args:
        d_model (int): embedding dim
        dropout_rate (float): dropout rate
        max_len (int): maximum input length
        xscale (bool): whether to scale the input by sqrt(d_model)
        dropout_rate_emb (float): dropout rate for the positional embeddings
    """

    def __init__(self, d_model, dropout_rate, max_len=5000, xscale=None, dropout_rate_emb=0.0):
        """Construct an PositionalEncoding object."""
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.xscale = xscale
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.max_len = max_len
        if dropout_rate_emb > 0:
            self.dropout_emb = nn.Dropout(dropout_rate_emb)
        else:
            self.dropout_emb = None

    def create_pe(self, positions):
        pos_length = positions.size(0)
        pe = torch.zeros(pos_length, self.d_model, device=positions.device)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32, device=positions.device)
            * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        pe = pe.unsqueeze(0)
        if hasattr(self, 'pe'):
            self.pe = pe
        else:
            self.register_buffer('pe', pe, persistent=False)

    def extend_pe(self, length, device):
        """Reset and extend the positional encodings if needed."""
        if hasattr(self, 'pe') and self.pe.size(1) >= length:
            return
        positions = torch.arange(0, length, dtype=torch.float32, device=device).unsqueeze(1)
        self.create_pe(positions=positions)

    def forward(self, x: torch.Tensor):
        """Adds positional encoding.
        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, feature_size)
        Returns:
            x+pos_emb (torch.Tensor): Its shape is (batch, time, feature_size)
            pos_emb (torch.Tensor): Its shape is (1, time, feature_size)
        """
        if self.xscale:
            x = x * self.xscale
        pos_emb = self.pe[:, : x.size(1)]
        if self.dropout_emb:
            pos_emb = self.dropout_emb(pos_emb)
        x = x + pos_emb
        return self.dropout(x), pos_emb


class RelPositionalEncoding(PositionalEncoding):
    """Relative positional encoding for TransformerXL's layers
    See : Appendix B in https://arxiv.org/abs/1901.02860
    Args:
        d_model (int): embedding dim
        dropout_rate (float): dropout rate
        max_len (int): maximum input length
        xscale (bool): whether to scale the input by sqrt(d_model)
        dropout_rate_emb (float): dropout rate for the positional embeddings
    """

    def extend_pe(self, length, device):
        """Reset and extend the positional encodings if needed."""
        needed_size = 2 * length - 1
        if hasattr(self, 'pe') and self.pe.size(1) >= needed_size:
            return
        # positions would be from negative numbers to positive
        # positive positions would be used for left positions and negative for right positions
        positions = torch.arange(length - 1, -length, -1, dtype=torch.float32, device=device).unsqueeze(1)
        self.create_pe(positions=positions)
        self.center_pos = torch.tensor(self.pe.size(1) // 2 + 1, dtype=torch.int32, device=device)

    def forward(self, x):
        """Compute positional encoding.
        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, feature_size)
        Returns:
            x (torch.Tensor): Its shape is (batch, time, feature_size)
            pos_emb (torch.Tensor): Its shape is (1, time, feature_size)
        """

        if self.xscale:
            x = x * self.xscale

        # center_pos would be the index of position 0
        # negative positions would be used for right and positive for left tokens
        # for input of length L, 2*L-1 positions are needed, positions from (L-1) to -(L-1)
        start_pos = self.center_pos - x.size(1)
        end_pos = self.center_pos + x.size(1) - 1
     
        pos_emb = self.pe[:, start_pos:end_pos]
  
        if self.dropout_emb:
            pos_emb = self.dropout_emb(pos_emb)
        return self.dropout(x), pos_emb


class MyopicAttention(nn.Module):
    def __init__(
        self,
        n_feats,
        head_dim,
        n_heads,
        dropout=0.0,
        max_keep_keys=50,
        chunk_window=3,
        bias=True,
        return_attention=False,
    ):
        super().__init__()
        self.n_feats = n_feats
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.bias = bias
        self.return_attention = return_attention

        self.scale = head_dim ** -0.5

        self.max_keep_keys = max_keep_keys
        self.W = chunk_window

        self.positional_bias = DynamicPositionBias(
            dim = n_feats,
            heads = n_heads,
            depth = 2,
            log_distance = False,
            norm = False
        )

        self.qkv_proj = nn.Linear(n_feats, 3 * n_heads * head_dim, bias=bias)
        self.out_proj = nn.Linear(n_heads * head_dim, n_feats, bias=bias)

    def pad_to_window_size(self, x, window_size, axis=3, mask=None):
        """
        Pad the input on two sides to be divisible by `window_size`
        """
        QKV, batch_size, heads, sequence_length, hidden_size = x.shape
        padding_length = (window_size - sequence_length % window_size) % window_size
        padding = torch.zeros(QKV, batch_size, heads, padding_length, hidden_size,
            device=x.device,
            dtype=x.dtype,
        )
        mask = F.pad(mask, (0, padding_length), value=True) if exists(mask) else None
        return torch.cat([x, padding], axis=axis), padding_length, mask

    def unpad(self, x, padding_length):
        """
        Undo padding.
        """
        if padding_length > 0:
            return x[:, :-padding_length]
        return x

    def ChunkGrid(self, Total_Size, Block_Size):
        Psize = Total_Size // Block_Size
        chunk_grid = (torch.arange(0, Psize).repeat(Psize,1) - torch.arange(0, Psize).repeat(Psize,1).T ).repeat_interleave(Block_Size, dim=1).abs()
        #chunk_grid = 1 - (chunk_grid / chunk_grid.max(dim=-1)[0].unsqueeze(-1)) # don't normalize cus it'll stretch the distribution by sequence length
        return chunk_grid    

    def forward(self, x, mask=None, return_attention=False):
        B, N, C, H, D = *x.shape, self.n_heads, self.head_dim
        tokeep = min(self.max_keep_keys, N) if self.max_keep_keys != -1 else N
        W = min(self.W, N) if self.W != -1 else N
        qkv = rearrange(self.qkv_proj(x), "B N (H D QKV) -> QKV B H N D", QKV=3, H=H, D=D)

        qkv, pad_n, mask = self.pad_to_window_size(qkv, W, axis=3, mask=mask)
        q, kv = qkv[0], qkv[1:]
        # split q into windows/chunks of size W
        q = rearrange(q, "B H (N W) D -> B H N W D", W=W)
        if exists(mask):
            q_mask = rearrange(mask, "B (N W) -> B N W", W=W)
            q_mask = repeat(q_mask, "B N W -> B H N W", H=H)
            
        # duplicate k and v for total number of windows
        kv = repeat(kv, "KV B H N D -> KV B H NW N D", NW=q.shape[2])
        #print(q.shape, kv.shape)
        KV, B, H, NW, N, D = kv.shape

        chunkgrid = self.ChunkGrid(Total_Size=N, Block_Size=W).to(q.device)
        chunkgrid = repeat(chunkgrid, "W N -> B H W N", B=B, H=H).contiguous()

        SCALE = torch.tensor(3.0, device=q.device, dtype=q.dtype)
        ALPHA = torch.tensor(2.0, device=q.device, dtype=q.dtype)
        pareto_dist = torch.distributions.pareto.Pareto(SCALE, ALPHA).sample(chunkgrid.shape).to(q.device)
        chunkgrid = chunkgrid - pareto_dist

        chunkgrid = repeat(chunkgrid, "B H W N -> KV B H W N", KV=2)
       
        keep_indices = chunkgrid.topk(k=tokeep, dim=-1, sorted=False, largest=False).indices.sort(dim=-1).values
        KV, B, H, NW, N, D = kv.shape 
        kv = kv.gather(-2, repeat(keep_indices, "KV B H W N -> KV B H W N D", D=D))

        if exists(mask):
            kv_mask = repeat(mask, "B N -> B H NW N", H=H, NW=NW).gather(-1, keep_indices[0])

        k, v = kv
        # NW (number of windows) = P (in below einsum)
        dots = einsum("B H N P D, B H N Z D -> B H N P Z ", q, k) * self.scale # Z is number of chunks in Q, N is max sequence length after dropping

        ## positional stuff
        pos_bias = self.positional_bias(N, device=dots.device, dtype=dots.dtype)
        pos_bias = repeat(pos_bias, 'H I J -> B H I J', B = B)
        pos_bias = rearrange(pos_bias, 'B H (N W) J -> B H N W J', W = W)

        pos_bias = pos_bias.gather(-1, repeat(keep_indices, "KV B H NW N -> KV B H NW W N", W=W)[0])
        
        dots = dots + pos_bias

        if exists(mask):
            mask_val = -torch.finfo(dots.dtype).max
            qk_mask = rearrange(q_mask, "B H N W -> B H N W ()") * rearrange(kv_mask, "B H W N -> B H W () N")
            dots.masked_fill_(qk_mask, mask_val)

        attn = dots.softmax(dim=-1)
  
        out = einsum("B H N W Z, B H N Z D -> B H N W D", attn, v)
        out = rearrange(out, "B H N W D -> B (N W) (H D)")
        out = self.unpad(out, pad_n)
        out = self.out_proj(out)
        return out if not return_attention else (out, attn)

def l2norm(t, groups = 1, dim = -1):
    if groups == 1:
        return F.normalize(t, p = 2, dim = dim)
    t = rearrange(t, '... (g d) -> ... g d', g = groups)
    t = F.normalize(t, p = 2, dim = dim)
    return rearrange(t, '... g d -> ... (g d)')

class CosineAttention(nn.Module):
    def __init__(
        self,
        n_feats,
        head_dim,
        n_heads,
        dropout=0.1,
        bias=False,
        temperature=15.5,
        return_attention=False,
        causal=False,
        activation='softmax',
        **kwargs
    ):
        super().__init__()
        assert activation in ['relusq', 'softmax']
        self.shared_kv = kwargs.get('shared_kv', False)
        self.talking_heads = kwargs.get('talking_heads', False)

        self.n_feats, self.head_dim, self.n_heads = n_feats, head_dim, n_heads
        self.dropout = nn.Dropout(dropout)
        self.bias = bias
        self.return_attention = return_attention
        self.causal = causal

        if self.talking_heads:
            self._head_proj = nn.Conv2d(n_heads, n_heads, (1, 1))

        self.temperature = torch.nn.Parameter(torch.tensor(temperature), requires_grad=True) if isinstance(temperature, float) else temperature

        self.activation = ReLUSquared() if activation == 'relusq' else nn.Softmax(dim=-1)

        if not self.shared_kv:
            self.qkv_proj = nn.Linear(n_feats, 3 * n_heads * head_dim, bias=bias)
            self.qkv = lambda x: rearrange(self.qkv_proj(x), "b n (h d qkv) -> qkv b h n d", qkv=3, h=n_heads, d=head_dim)
        else:
            self.q_proj, self.kv_proj = [nn.Linear(n_feats, el, bias=bias) for el in [n_heads * head_dim, 2 * head_dim]]
            map_q, map_kv = lambda q: rearrange(q, 'b n (h d) -> b h n d', h=n_heads), lambda kv: rearrange(kv, 'b n (kv d) -> kv b () n d', kv=2, d=head_dim)
            self.qkv = lambda x: (map_q(self.q_proj(x)), *map_kv(self.kv_proj(x)))

        self.out_proj = nn.Linear(n_heads * head_dim, n_feats, bias=bias)
    
    def head_proj(self, dots):
        if not self.talking_heads:
            return dots
        dots = self._head_proj(dots)
        return dots      

    def attend(self, query, key, value, mask, pos_fn):
        query, key = map(l2norm, (query, key))

        dots = einsum('bhid,bhjd->bhij', query, key) * self.temperature
        dots = self.head_proj(dots)

        dots += pos_fn(dots.shape[-1], device=dots.device, dtype=dots.dtype)
        qkmask = ~mask
        attn_mask = ~(rearrange(qkmask, "b n -> b () n ()") * rearrange(qkmask, "b n -> b () () n"))
    
        if self.causal: # create a regular causal mask
            causal_mask = torch.ones(dots.shape[-2], dots.shape[-1], device=dots.device).triu(1).bool()
            attn_mask = torch.logical_or(attn_mask, causal_mask)
        
        dots.masked_fill_(attn_mask, -torch.finfo(dots.dtype).max)
    
        attn = self.activation(dots)
     
        attn = self.dropout(attn)
        return einsum("bhij,bhjd->bhid", attn, value)


    def forward(self, x, pos_fn, mask=None):
        assert pos_fn is not None, 'pls provide a position function'
        B, N, C, H, D = *x.shape, self.n_heads, self.head_dim
        #print(x.shape, mask.shape)
       
        if mask is None:
            mask = torch.zeros(B, N, device=x.device, dtype=torch.bool)

        q, k, v = self.qkv(x)
    
        out = self.attend(q, k, v, mask, pos_fn)

        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.out_proj(out)
        return out