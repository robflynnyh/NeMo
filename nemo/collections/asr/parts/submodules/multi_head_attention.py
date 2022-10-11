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
from torch import nn, einsum
from nemo.collections.asr.parts.utils.activations import LaplacianAttnFn, ReLUSquared
from nemo.collections.asr.parts.utils.helpers import exists, isfalse
from einops import rearrange

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
