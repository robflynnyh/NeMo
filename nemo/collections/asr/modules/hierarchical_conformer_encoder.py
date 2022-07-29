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

from nemo.collections.asr.parts.submodules.conformer_modules import ConformerLayer, CrossConformerLayer
from nemo.collections.asr.parts.submodules.lucid_conformer import CrossConformerBlock

from nemo.collections.asr.parts.submodules.multi_head_attention import PositionalEncoding, RelPositionalEncoding
from nemo.collections.asr.parts.submodules.subsampling import ConvSubsampling, StackingSubsampling
from nemo.collections.asr.parts.utils import adapter_utils
from nemo.core.classes.common import typecheck
from nemo.core.classes.exportable import Exportable
from nemo.core.classes.mixins import adapter_mixins
from nemo.core.classes.module import NeuralModule
from nemo.core.neural_types import AcousticEncodedRepresentation, LengthsType, NeuralType, SpectrogramType

from torch.utils.checkpoint import checkpoint # # gradient/activation checkpointing

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
        pos_emb_max_len=5000,
        conv_kernel_size=31,
        conv_norm_type='batch_norm',
        dropout=0.1,
        dropout_emb=0.1,
        dropout_att=0.0,
        downsampling_type='pooling', #  'pooling' or 'conv'
        n_repeats=3, # number of repeats of the cross-conformer layers
    ):
        super().__init__()

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
        
        assert n_layers > 2, "Number of layers must be at least 3!"

        for i in range(n_layers - 3):
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
            )
            self.layers.append(layer)

        self.CrossConformerLayers = nn.ModuleList()
        for i in range(2):
            layer = CrossConformerBlock(
                dim=d_model,
                heads=n_heads,
                dim_head=d_model // n_heads,
                ff_mult=ff_expansion_factor,
                conv_expansion_factor = 2,
                conv_kernel_size=conv_kernel_size,
                attn_dropout=dropout_att,
                ff_dropout=dropout,
                conv_dropout=dropout,
            )
            self.CrossConformerLayers.append(layer)

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
        )
        self.CrossConformerLayers.append(layer) # final cross-conformer layer is standard self-attention for feature extraction
        # other cross-conformer layers use cross-attention
        
        self.downsampling_type = downsampling_type
        if downsampling_type == 'pooling':
            self.downsampling = nn.AvgPool1d(kernel_size=3, stride=3, count_include_pad=False) # don't count padding in average 
        elif downsampling_type == 'conv': # NOT IMPLEMENTED YET
            raise ValueError("Convolutional downsampling is not implemented yet!")
            self.downsampling = nn.Conv1d(d_model, d_model, kernel_size=3, stride=3)
        else:
            raise ValueError("Not valid downsampling type: '{}'!".format(downsampling_type))

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
        self.pos_enc.extend_pe(max_audio_length, device)

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

        audio_signal, pos_emb = self.pos_enc(audio_signal)
        # adjust size
        max_audio_length = audio_signal.size(1)
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
            #audio_signal = layer(x=audio_signal, att_mask=att_mask, pos_emb=pos_emb, pad_mask=pad_mask)
            audio_signal = checkpoint(self.create_custom_forward(layer), audio_signal, att_mask, pos_emb, pad_mask)
        

        for repeat in range(self.n_repeats):
            # downsample the sequence length using average pooling or convolution
            
            if self.downsampling_type == 'pooling': # this isn't pooling the right amount
                pooled_audio_signal = self.downsampling(audio_signal.transpose(1, 2)).transpose(1, 2) # transpose so that the sequence length is the first dimension
            
            
            # we need to create new masks and pos_embs for the downsampled sequence length
            max_downsampled_audio_length = pooled_audio_signal.size(1)
            # calculate new length of each element (without padding) in the downsampled sequence
            #downsampled_lengths = (length) // 3 # 3 is the downsampling factor # might need to do # this might cause some ISSUEs !!
            
            downsampled_lengths = length.div(3, rounding_mode='floor')

            downsampled_pad_mask = self.make_pad_mask(max_downsampled_audio_length, downsampled_lengths)

    
            downsampled_x = checkpoint(self.create_custom_forward(self.CrossConformerLayers[0]), 
                pooled_audio_signal, # Qs
                audio_signal, # Ks
                downsampled_pad_mask, # mask 
                pad_mask # Context mask
            )
            
            # maybe actually I should slice and rearange the padding so it's only at the end of the sequence.. maybe
            
            cat_downsampled_pad_mask = torch.empty(downsampled_pad_mask.shape[0], downsampled_pad_mask.shape[1]*3, dtype=torch.bool, device=downsampled_pad_mask.device)
            cat_downsampled_x = torch.empty((downsampled_x.shape[0], downsampled_x.shape[1]*3, downsampled_x.shape[2]), device=downsampled_x.device)
            padding_length = torch.tensor([0], dtype=torch.int32, device=downsampled_lengths.device) # padding items have a sequence length of 0
            cat_downsampled_lengths = torch.cat([padding_length,  downsampled_lengths, padding_length], dim=0)

            left_and_right_padding = torch.zeros_like(downsampled_x[0, :, :]).unsqueeze(0)
            downsampled_x = torch.cat((left_and_right_padding, downsampled_x, left_and_right_padding), dim=0) 
         
            # hopefully I can think of a nicer way to do this
            for i in range(1, downsampled_x.shape[0]-1): # skip the first and last element (the padding)
                position = i - 1 
                left_seq = downsampled_x[i-1, :cat_downsampled_lengths[i-1], :]
                left_seq_padding = downsampled_x[i-1, cat_downsampled_lengths[i-1]:, :]
                #
                right_seq = downsampled_x[i+1, :cat_downsampled_lengths[i+1], :]
                right_seq_padding = downsampled_x[i+1, cat_downsampled_lengths[i+1]:, :]
                #
                middle_seq = downsampled_x[i, :cat_downsampled_lengths[i], :]
                middle_seq_padding = downsampled_x[i, cat_downsampled_lengths[i]:, :]
                #
                # now concatenate all these things (clean this all up later future me)
                non_pad_seq = torch.cat((left_seq, middle_seq, right_seq), dim=0)
                pad_seq = torch.cat((left_seq_padding, middle_seq_padding, right_seq_padding), dim=0)

                full_seq = torch.cat((non_pad_seq, pad_seq), dim=0)
                #del pad_seq, left_seq, left_seq_padding, right_seq, right_seq_padding, middle_seq, middle_seq_padding
                # IDEA/TODO: maybe to avoid messing with the padding, I should just add the padding to the end of the non-padded sequence, and then pad the end of the full sequence

                cur_pad_mask = torch.ones(cat_downsampled_x[position, :, :].shape[0], dtype=torch.bool, device=downsampled_x.device)
                cur_pad_mask[:non_pad_seq.shape[0]] = False

                cat_downsampled_x[position, :, :] = full_seq.unsqueeze(0)
                cat_downsampled_pad_mask[position, :] = cur_pad_mask.reshape(-1)
            
     
            cross_x = checkpoint(self.create_custom_forward(self.CrossConformerLayers[1]), 
                audio_signal, # Qs
                cat_downsampled_x, # KVs
                pad_mask,  # mask
                cat_downsampled_pad_mask # context_mask
            )

            cross_x = audio_signal + cross_x # add the two together i.e a residual/skip connection
            # now standard conformer layer for feature extraction from cross_x

            if repeat == self.n_repeats - 1: # dont't checkpoint on the last repeat because we need it for the backward pass straight away (pretty much)
                audio_signal = self.CrossConformerLayers[2](
                    x=cross_x,
                    att_mask=att_mask,
                    pos_emb=pos_emb,
                    pad_mask=pad_mask
                )
            else: 
                audio_signal = checkpoint(self.create_custom_forward(self.CrossConformerLayers[2]),
                    cross_x, 
                    att_mask, 
                    pos_emb, 
                    pad_mask
                )

        if self.out_proj is not None:
            audio_signal = self.out_proj(audio_signal) # if dim of decoder is not equal to dim of encoder, then we need to project the output
     
        audio_signal = torch.transpose(audio_signal, 1, 2) # (batch, seq_len, d_model) -> (batch, d_model, seq_len) 
     
        return audio_signal, length

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
