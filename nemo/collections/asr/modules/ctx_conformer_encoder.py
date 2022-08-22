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

import math
from collections import OrderedDict
from typing import List, Optional

import torch
import torch.distributed
import torch.nn as nn
from omegaconf import DictConfig

from nemo.collections.asr.parts.submodules.conformer_modules import ConformerLayer
from nemo.collections.asr.parts.submodules.multi_head_attention import PositionalEncoding, RelPositionalEncoding
from nemo.collections.asr.parts.submodules.subsampling import ConvSubsampling, StackingSubsampling
from nemo.collections.asr.parts.utils import adapter_utils
from nemo.core.classes.common import typecheck
from nemo.core.classes.exportable import Exportable
from nemo.core.classes.mixins import adapter_mixins
from nemo.core.classes.module import NeuralModule
from nemo.core.neural_types import AcousticEncodedRepresentation, LengthsType, NeuralType, SpectrogramType

from torch.utils.checkpoint import checkpoint # gradient/activation checkpointing
from nemo.collections.asr.parts.submodules.lucid_conformer import CrossAttnHistoryPositionalEncoding, CtxCrossConformerBlock
from einops import rearrange

__all__ = ['CtxConformerEncoder']




class CtxConformerEncoder(NeuralModule, Exportable):
    """
    The encoder for ASR model of Conformer.
    Based on this paper:
    'Conformer: Convolution-augmented Transformer for Speech Recognition' by Anmol Gulati et al.
    https://arxiv.org/abs/2005.08100

    Args:
        feat_in (int): the size of feature channels
        n_layers (int): number of layers of ConformerBlock
        d_model (int): the hidden size of the model
        feat_out (int): the size of the output features
            Defaults to -1 (means feat_out is d_model)
        subsampling (str): the method of subsampling, choices=['vggnet', 'striding']
            Defaults to striding.
        subsampling_factor (int): the subsampling factor which should be power of 2
            Defaults to 4.
        subsampling_conv_channels (int): the size of the convolutions in the subsampling module
            Defaults to -1 which would set it to d_model.
        ff_expansion_factor (int): the expansion factor in feed forward layers
            Defaults to 4.
        self_attention_model (str): type of the attention layer and positional encoding
            'rel_pos': relative positional embedding and Transformer-XL
            'abs_pos': absolute positional embedding and Transformer
            default is rel_pos.
        pos_emb_max_len (int): the maximum length of positional embeddings
            Defaulst to 5000
        n_heads (int): number of heads in multi-headed attention layers
            Defaults to 4.
        xscaling (bool): enables scaling the inputs to the multi-headed attention layers by sqrt(d_model)
            Defaults to True.
        untie_biases (bool): whether to not share (untie) the bias weights between layers of Transformer-XL
            Defaults to True.
        conv_kernel_size (int): the size of the convolutions in the convolutional modules
            Defaults to 31.
        conv_norm_type (str): the type of the normalization in the convolutional modules
            Defaults to 'batch_norm'.
        dropout (float): the dropout rate used in all layers except the attention layers
            Defaults to 0.1.
        dropout_emb (float): the dropout rate used for the positional embeddings
            Defaults to 0.1.
        dropout_att (float): the dropout rate used for the attention layer
            Defaults to 0.0.
    """

    def input_example(self, max_batch=1, max_dim=256):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
        dev = next(self.parameters()).device
        input_example = torch.randn(max_batch, self._feat_in, max_dim).to(dev)
        input_example_length = torch.randint(1, max_dim, (max_batch,)).to(dev)
        return tuple([input_example, input_example_length])

    @property
    def input_types(self):
        """Returns definitions of module input ports."""
        return OrderedDict(
            {
                "audio_signal": NeuralType(('B', 'D', 'T'), SpectrogramType()),
                "length": NeuralType(tuple('B'), LengthsType()),
            }
        )

    @property
    def output_types(self):
        """Returns definitions of module output ports."""
        return OrderedDict(
            {
                "outputs": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
                "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
            }
        )

    def __init__(
        self,
        feat_in,
        n_layers,
        d_model,
        feat_out=-1,
        subsampling='striding',
        subsampling_factor=4,
        subsampling_conv_channels=-1,
        ff_expansion_factor=4,
        self_attention_model='rel_pos',
        n_heads=4,
        att_context_size=None,
        xscaling=True,
        untie_biases=True,
        pos_emb_max_len=5000,
        conv_kernel_size=31,
        conv_norm_type='batch_norm',
        dropout=0.1,
        dropout_emb=0.1,
        dropout_att=0.0,
        checkpoint_every_n_layers=1,
        num_memory_vectors=10,
        num_repeats = 6,
        max_seq_len = 40, # max batch size for cross-batch attention position encoding
        sparse_topk = None,
        memory_dropout = 0.0,
        cross_attn_dropout = 0.0,
        cross_post_attn_dropout = 0.1,
        local_attn=False, # whether to use a local attention pattern for the cross-attention
        weight_standardization=True,
    ):
        super().__init__()

        assert local_attn == False or cross_attn_dropout == 0.0, "local attention is not compatible with cross attention dropout (to do!)"
        if local_attn:
            print('Using local attention for cross-attention')

        d_ff = d_model * ff_expansion_factor
        self.d_model = d_model
        self._feat_in = feat_in
        self.scale = math.sqrt(self.d_model)
        if att_context_size:
            self.att_context_size = att_context_size
        else:
            self.att_context_size = [-1, -1]

        if xscaling:
            self.xscale = math.sqrt(d_model)
        else:
            self.xscale = None

        self.checkpoint_every_n_layers = checkpoint_every_n_layers

        if subsampling_conv_channels == -1:
            subsampling_conv_channels = d_model
        if subsampling and subsampling_factor > 1:
            if subsampling == 'stacking':
                self.pre_encode = StackingSubsampling(
                    subsampling_factor=subsampling_factor, feat_in=feat_in, feat_out=d_model
                )
            else:
                self.pre_encode = ConvSubsampling(
                    subsampling=subsampling,
                    subsampling_factor=subsampling_factor,
                    feat_in=feat_in,
                    feat_out=d_model,
                    conv_channels=subsampling_conv_channels,
                    activation=nn.ReLU(),
                )
        else:
            self.pre_encode = nn.Linear(feat_in, d_model)

        self._feat_out = d_model

        if not untie_biases and self_attention_model == "rel_pos":
            d_head = d_model // n_heads
            pos_bias_u = nn.Parameter(torch.Tensor(n_heads, d_head))
            pos_bias_v = nn.Parameter(torch.Tensor(n_heads, d_head))
            nn.init.zeros_(pos_bias_u)
            nn.init.zeros_(pos_bias_v)
        else:
            pos_bias_u = None
            pos_bias_v = None

        self.pos_emb_max_len = pos_emb_max_len
        if self_attention_model == "rel_pos":
            self.pos_enc = RelPositionalEncoding(
                d_model=d_model,
                dropout_rate=dropout,
                max_len=pos_emb_max_len,
                xscale=self.xscale,
                dropout_rate_emb=dropout_emb,
            )
        elif self_attention_model == "abs_pos":
            pos_bias_u = None
            pos_bias_v = None
            self.pos_enc = PositionalEncoding(
                d_model=d_model, dropout_rate=dropout, max_len=pos_emb_max_len, xscale=self.xscale
            )
        else:
            raise ValueError(f"Not valid self_attention_model: '{self_attention_model}'!")

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            layer = ConformerLayer(
                d_model=d_model,
                d_ff=d_ff,
                self_attention_model=self_attention_model,
                n_heads=n_heads,
                conv_kernel_size=conv_kernel_size,
                conv_norm_type=conv_norm_type,
                dropout=dropout,
                dropout_att=dropout_att,
                pos_bias_u=pos_bias_u,
                pos_bias_v=pos_bias_v,
                weight_standardization=weight_standardization,
            )
            self.layers.append(layer)

        self.repeated_conformer_layer = ConformerLayer( # full seq layer
                d_model=d_model,
                d_ff=d_ff,
                self_attention_model=self_attention_model,
                n_heads=n_heads,
                conv_kernel_size=conv_kernel_size,
                conv_norm_type=conv_norm_type,
                dropout=dropout,
                dropout_att=dropout_att,
                pos_bias_u=pos_bias_u,
                pos_bias_v=pos_bias_v,
                sparse_topk = sparse_topk,
                weight_standardization=weight_standardization,
        )
     
        self.cross_attn_pos_enc = CrossAttnHistoryPositionalEncoding( 
            pos_dim=d_model,
            max_seq_len=max_seq_len,
            num_context_vectors=num_memory_vectors
        )
     
        self.cross_conformer_layer = CtxCrossConformerBlock( # cross attention applied between local and global context
            dim = d_model,
            dim_head = d_model // n_heads,
            heads = n_heads,
            ff_mult = ff_expansion_factor,
            attn_dropout = cross_post_attn_dropout,
            ff_dropout = dropout,
            sparse_topk = sparse_topk,
            conv_norm_type = conv_norm_type,
            local_attn = local_attn,
            use_conv = False
        )

        self.num_repeats = num_repeats

        self.memory_dropout = nn.Dropout(memory_dropout)
        self.cross_attn_dropout = cross_attn_dropout

        self.num_memory_vectors = num_memory_vectors
        self.memory_vectors = nn.Parameter(torch.Tensor(num_memory_vectors, d_model))
        nn.init.uniform_(self.memory_vectors, -0.1, 0.1)

        if feat_out > 0 and feat_out != self._feat_out:
            self.out_proj = nn.Linear(self._feat_out, feat_out)
            self._feat_out = feat_out
        else:
            self.out_proj = None
            self._feat_out = d_model
        self.set_max_audio_length(self.pos_emb_max_len)
        self.use_pad_mask = True

    def set_max_audio_length(self, max_audio_length):
        """
        Sets maximum input length.
        Pre-calculates internal seq_range mask.
        """
        self.max_audio_length = max_audio_length
        device = next(self.parameters()).device
        seq_range = torch.arange(0, self.max_audio_length, device=device)
        if hasattr(self, 'seq_range'):
            self.seq_range = seq_range
        else:
            self.register_buffer('seq_range', seq_range, persistent=False)
        self.pos_enc.extend_pe(max_audio_length, device)

    @staticmethod
    def create_custom_forward(module): # for activation checkpointing 
        def custom_forward(*args, **kwargs):
            return module(*args, **kwargs)
        return custom_forward

    @typecheck()
    def forward(self, audio_signal, length=None):
        self.update_max_seq_length(seq_length=audio_signal.size(2), device=audio_signal.device)
        return self.forward_for_export(audio_signal=audio_signal, length=length)

    def apply_cross_attend_dropout(self, x):
        '''
        cross-attend dropout used here: https://arxiv.org/pdf/2202.07765.pdf
        this is basically a random selection of elements in the input tensor
        i.e if the input tensor is (B, N, D) and the cross attend dropout is
        0.25 then output will be (B, N*0.75, D)
        Wasn't sure how to implement this efficiently so used lucidrains implementation from the perceiver AR 
        https://github.com/lucidrains/perceiver-ar-pytorch/blob/main/perceiver_ar_pytorch/perceiver_ar_pytorch.py
        '''
        cross_attn_dropout = self.cross_attn_dropout
        # check if model is training or not
        if cross_attn_dropout > 0 and self.training:
            batch_size, num_frames, dim = x.size()
            rand = torch.zeros((batch_size, num_frames), device=x.device).uniform_()
            num_frames_to_keep = int(num_frames * (1 - cross_attn_dropout))
            keep_indices = rand.topk(num_frames_to_keep, dim=-1).indices
            keep_mask = torch.zeros_like(rand).scatter_(1, keep_indices, 1).bool()
            # dang thats clean
            return rearrange(x[keep_mask], '(b n) d -> b n d', b=batch_size)
        else:
            return x
            
            
            

    @typecheck()
    def forward_for_export(self, audio_signal, length):
        max_audio_length: int = audio_signal.size(-1)

        if max_audio_length > self.max_audio_length:
            self.set_max_audio_length(max_audio_length)

        if length is None:
            length = audio_signal.new_full(
                audio_signal.size(0), max_audio_length, dtype=torch.int32, device=self.seq_range.device
            )

        audio_signal = torch.transpose(audio_signal, 1, 2) 

        if isinstance(self.pre_encode, nn.Linear):
            audio_signal = self.pre_encode(audio_signal)
        else:
            audio_signal, length = self.pre_encode(audio_signal, length)

        # adjust size
        max_audio_length = audio_signal.size(1)
        max_audio_length += self.num_memory_vectors
        length += self.num_memory_vectors


        audio_signal = torch.cat([self.memory_vectors.unsqueeze(0).repeat([audio_signal.size(0), 1, 1]), audio_signal], dim=1)
        audio_signal, pos_emb = self.pos_enc(audio_signal)


        # Create the self-attention and padding masks

        pad_mask = self.make_pad_mask(max_audio_length, length)
        att_mask = pad_mask.unsqueeze(1).repeat([1, max_audio_length, 1])
        att_mask = torch.logical_and(att_mask, att_mask.transpose(1, 2))
        if self.att_context_size[0] >= 0:
            att_mask = att_mask.triu(diagonal=-self.att_context_size[0])
        if self.att_context_size[1] >= 0:
            att_mask = att_mask.tril(diagonal=self.att_context_size[1])
        att_mask = ~att_mask

        if self.use_pad_mask:
            pad_mask = ~pad_mask
        else:
            pad_mask = None

        for lth, layer in enumerate(self.layers):
            if self.checkpoint_every_n_layers > 0 and lth % self.checkpoint_every_n_layers == 0:
                audio_signal = checkpoint(self.create_custom_forward(layer), audio_signal, att_mask, pos_emb, pad_mask, self.num_memory_vectors)
            else:
                audio_signal = layer(x=audio_signal, att_mask=att_mask, pos_emb=pos_emb, pad_mask=pad_mask, num_memory_vectors=self.num_memory_vectors)
        
        # this assumes the batch is composed of utterances that are sampled contiguously from a discourse
        for repeat in range(self.num_repeats): 
            # memory vectors are just learnt embeddings that are prepended to the input prior to the first layer
            memory_vectors = audio_signal[:, :self.num_memory_vectors, :]
            # reshape to (1, num_memory_vectors * batch_size, d_model)
            memory_sequences = memory_vectors.reshape(1, -1, memory_vectors.size(-1))
            # duplicate along the batch dimension i.e (1, num_memory_vectors * batch_size, d_model) -> (batch_size, num_memory_vectors * batch_size, d_model)
            memory_sequences = memory_sequences.repeat(audio_signal.size(0), 1, 1)
            # apply relative cross attn positional encoding to the memory sequences
            memory_sequences = self.cross_attn_pos_enc(memory_sequences)
            # dropout of memory sequences (not usin here anymore) but kept for compatibility
            memory_sequences = self.memory_dropout(memory_sequences)
            # cross attend dropout
            memory_sequences = self.apply_cross_attend_dropout(memory_sequences)
            # now cross attention between memory vectors and memory sequences
            ctx_memories = checkpoint(self.create_custom_forward(self.cross_conformer_layer), memory_vectors, memory_sequences)
            # re-connect the memory vectors to the audio_signal, but clone so we don't break the computation graph
            audio_signal = audio_signal.clone()
            audio_signal[:, :self.num_memory_vectors, :] = ctx_memories
            # now for self attention over the audio_signal with the cross-utterance memory vectors
            if repeat != self.num_repeats - 1:
                audio_signal = checkpoint(self.create_custom_forward(self.repeated_conformer_layer), audio_signal, att_mask, pos_emb, pad_mask, self.num_memory_vectors)
            else: # don't checkpoint grad on last layer
                audio_signal = self.repeated_conformer_layer(x=audio_signal, att_mask=att_mask, pos_emb=pos_emb, pad_mask=pad_mask, num_memory_vectors=self.num_memory_vectors)
        
        #########################

        if self.out_proj is not None:
            audio_signal = self.out_proj(audio_signal) # if dim of decoder is not equal to dim of encoder, then we need to project the output

        out_signal = audio_signal[:, self.num_memory_vectors:, :]
        length -= self.num_memory_vectors

        out_signal = torch.transpose(out_signal, 1, 2) # (batch, seq_len, d_model) -> (batch, d_model, seq_len) 
     
        return out_signal, length


    def update_max_seq_length(self, seq_length: int, device):
        # Find global max audio length across all nodes
        if torch.distributed.is_initialized():
            global_max_len = torch.tensor([seq_length], dtype=torch.float32, device=device)

            # Update across all ranks in the distributed system
            torch.distributed.all_reduce(global_max_len, op=torch.distributed.ReduceOp.MAX)

            seq_length = global_max_len.int().item()

        if seq_length > self.max_audio_length:
            self.set_max_audio_length(seq_length)

    def make_pad_mask(self, max_audio_length, seq_lens):
        """Make masking for padding."""
        mask = self.seq_range[:max_audio_length].expand(seq_lens.size(0), -1) < seq_lens.unsqueeze(-1)
        return mask

    def enable_pad_mask(self, on=True):
        # On inference, user may chose to disable pad mask
        mask = self.use_pad_mask
        self.use_pad_mask = on
        return mask


class CtxConformerEncoderAdapter(CtxConformerEncoder, adapter_mixins.AdapterModuleMixin):

    # Higher level forwarding
    def add_adapter(self, name: str, cfg: dict):
        cfg = self._update_adapter_cfg_input_dim(cfg)
        for conformer_layer in self.layers:  # type: adapter_mixins.AdapterModuleMixin
            conformer_layer.add_adapter(name, cfg)

    def is_adapter_available(self) -> bool:
        return any([conformer_layer.is_adapter_available() for conformer_layer in self.layers])

    def set_enabled_adapters(self, name: Optional[str] = None, enabled: bool = True):
        for conformer_layer in self.layers:  # type: adapter_mixins.AdapterModuleMixin
            conformer_layer.set_enabled_adapters(name=name, enabled=enabled)

    def get_enabled_adapters(self) -> List[str]:
        names = set([])
        for conformer_layer in self.layers:  # type: adapter_mixins.AdapterModuleMixin
            names.update(conformer_layer.get_enabled_adapters())

        names = sorted(list(names))
        return names

    def _update_adapter_cfg_input_dim(self, cfg: DictConfig):
        cfg = adapter_utils.update_adapter_cfg_input_dim(self, cfg, module_dim=self.d_model)
        return cfg



"""
Register any additional information
"""
if adapter_mixins.get_registered_adapter(CtxConformerEncoder) is None:
    adapter_mixins.register_adapter(base_class=CtxConformerEncoder, adapter_class=CtxConformerEncoderAdapter)
