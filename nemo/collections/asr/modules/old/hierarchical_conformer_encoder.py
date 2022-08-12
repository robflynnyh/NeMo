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
from torch.utils.checkpoint import \
    checkpoint  # # gradient/activation checkpointing

from nemo.collections.asr.parts.submodules.conformer_modules import \
    ConformerLayer  # , CrossConformerLayer
from nemo.collections.asr.parts.submodules.lucid_conformer import (
    FunnelConformerBlock, IntegrationConformerBlock)
from nemo.collections.asr.parts.submodules.multi_head_attention import (
    PositionalEncoding, RelPositionalEncoding)
from nemo.collections.asr.parts.submodules.subsampling import (
    ConvSubsampling, StackingSubsampling)
from nemo.collections.asr.parts.submodules.x_transformers_local import \
    RelativePositionBias
from nemo.collections.asr.parts.utils import adapter_utils
from nemo.core.classes.common import typecheck
from nemo.core.classes.exportable import Exportable
from nemo.core.classes.mixins import adapter_mixins
from nemo.core.classes.module import NeuralModule
from nemo.core.neural_types import (AcousticEncodedRepresentation, LengthsType,
                                    NeuralType, SpectrogramType)

__all__ = ['HierarchicalConformerEncoder']




class HierarchicalConformerEncoder(NeuralModule, Exportable):
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
        pos_emb_max_len=256,
        conv_kernel_size=31,
        conv_norm_type='batch_norm',
        dropout=0.1,
        dropout_emb=0.1,
        dropout_att=0.0,
        downsampling_type='conv', #  'pooling' or 'conv'
        n_repeats=3, # number of repeats of the cross-conformer layers NOT USED HERE..
        checkpoint_last_layer=False, # turn grad checkpointing off for last layer of the last iteration
        checkpoint_every_n_layers=1,
        freeze_initial_encoder=False,
    ):
        super().__init__()

        
        self.checkpoint_every_n_layers = checkpoint_every_n_layers
        self.checkpoint_last_layer = checkpoint_last_layer

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


        self.pos_emb_max_len = pos_emb_max_len

        

        self.CrossConformerLayers = nn.ModuleList()

        self.num_context_vectors = 10 # add this to arguments
        shared_attn_param = None
        shared_gating_param = None
        pos_emb = torch.nn.Embedding(self.pos_emb_max_len, self.d_model // n_heads)

        for n in range(n_layers):
            layer = IntegrationConformerBlock(
                dim = d_model,
                heads = n_heads,
                dim_head = d_model // n_heads,
                ff_mult = ff_expansion_factor,
                conv_expansion_factor = 1,
                conv_kernel_size = conv_kernel_size,
                attn_dropout = dropout_att,
                ff_dropout = dropout,
                conv_dropout = dropout,
                num_context_vectors = self.num_context_vectors,
                shared_attn_layer = shared_attn_param,
                shared_gating_layer = shared_gating_param,
                seq_max_pos = 40, #24
                rel_pos_embs = pos_emb,
                max_pos_emb = self.pos_emb_max_len
            )
            self.CrossConformerLayers.append(layer)
            if shared_attn_param == None:
                shared_attn_param = layer.attn1
            if shared_gating_param == None:
                shared_gating_param = layer.gating_attention

        self.context_vectors = torch.nn.Parameter(torch.Tensor(self.num_context_vectors, d_model))
        torch.nn.init.xavier_uniform_(self.context_vectors) if len(self.context_vectors.size()) > 1 else torch.nn.init.uniform_(self.context_vectors, -0.1, 0.1)
        

        if feat_out > 0 and feat_out != self._feat_out:
            self.out_proj = nn.Linear(self._feat_out, feat_out)
            self._feat_out = feat_out
        else:
            self.out_proj = None
            self._feat_out = d_model
        self.set_max_audio_length(self.pos_emb_max_len)
        self.use_pad_mask = True

        self.n_repeats = n_repeats



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
        #self.pos_enc.extend_pe(max_audio_length, device)

    @typecheck()
    def forward(self, audio_signal, length=None):
        self.update_max_seq_length(seq_length=audio_signal.size(2), device=audio_signal.device)
        return self.forward_for_export(audio_signal=audio_signal, length=length)

    @staticmethod
    def create_custom_forward(module): # for activation checkpointing allow passing dictionary as the argument to the module
        def custom_forward(*args, **kwargs):
            return module(*args, **kwargs)
        return custom_forward

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
        max_audio_length = audio_signal.size(1) + self.num_context_vectors
        length += self.num_context_vectors
        # Create the self-attention and padding masks

        pad_mask = self.make_pad_mask(max_audio_length, length)
        
        # concat audio with context vectors along the sequence dimension
        audio_signal = torch.cat([self.context_vectors.unsqueeze(0).repeat([audio_signal.size(0), 1, 1]), audio_signal], dim=1)

        for repeat in range(len(self.CrossConformerLayers)):
            
            context_sequences = audio_signal[:, :self.num_context_vectors, :]
            # reshape to (1, num_context_seq * batch_size, dim)
            context_sequences = context_sequences.reshape(1, -1, context_sequences.size(-1))
            # duplicate along the batch dimension
            context_sequences = context_sequences.repeat(audio_signal.size(0), 1, 1)

            if repeat == len(self.CrossConformerLayers) - 1 and self.checkpoint_last_layer == False or self.checkpoint_every_n_layers == 0 or repeat % self.checkpoint_every_n_layers != 0:
                audio_signal = self.CrossConformerLayers[repeat](
                    Qs=audio_signal,
                    KVs=context_sequences,
                    mask=pad_mask,
                    context_mask=None,
                )
            else:
                audio_signal = checkpoint(self.create_custom_forward(self.CrossConformerLayers[repeat]), 
                    audio_signal, # Qs
                    context_sequences, # KVs
                    pad_mask,  # mask
                    None, # context_mask
                )


        if self.out_proj is not None:
            audio_signal = self.out_proj(audio_signal) # if dim of decoder is not equal to dim of encoder, then we need to project the output
     
        # trim to remove the context vectors
        out_signal = audio_signal[:, self.num_context_vectors:, :]
        length -= self.num_context_vectors

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
        mask = self.seq_range[:max_audio_length].expand(seq_lens.size(0), -1) < seq_lens.unsqueeze(-1) # boolean operater
        return mask

    def enable_pad_mask(self, on=True):
        # On inference, user may chose to disable pad mask
        mask = self.use_pad_mask
        self.use_pad_mask = on
        return mask


class HConformerEncoderAdapter(HierarchicalConformerEncoder, adapter_mixins.AdapterModuleMixin):

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
if adapter_mixins.get_registered_adapter(HierarchicalConformerEncoder) is None:
    adapter_mixins.register_adapter(base_class=HierarchicalConformerEncoder, adapter_class=HConformerEncoderAdapter)
