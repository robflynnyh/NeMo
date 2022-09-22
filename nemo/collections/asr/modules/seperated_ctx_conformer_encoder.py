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
from nemo.core.neural_types import AcousticEncodedRepresentation, LengthsType, NeuralType, SpectrogramType, LogprobsType
from nemo.core.neural_types.elements import BoolType

from torch.utils.checkpoint import checkpoint # gradient/activation checkpointing
from nemo.collections.asr.parts.submodules.lucid_conformer import CrossAttnHistoryPositionalEncoding, CtxCrossConformerBlock, MappingAttention

from einops import rearrange

from torch.nn import functional as F


__all__ = ['CtxConformerEncoder']


def pad_mask_to_attn_mask(pad_mask):
    """Converts padding mask to attention mask."""
    return rearrange(pad_mask, 'b i -> b i ()') * rearrange(pad_mask, 'b j -> b () j')



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
                "decoder": NeuralType(None),
                "length": NeuralType(tuple('B'), LengthsType()),
                "segment_lens": NeuralType(tuple('S'), LengthsType(), optional=True),
                "return_cross_utterance_attention": NeuralType(tuple(), BoolType(), optional=True),
            }
        )

    @property
    def output_types(self):
        """Returns definitions of module output ports."""
        return OrderedDict(
            {
                "outputs": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
                "iterim_posteriors": NeuralType(('H', 'B', 'D', 'T'), LogprobsType()),
                "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
                "additional_outputs": NeuralType(None, optional=True),
            }
        )

    def __init__(
        self,
        feat_in,
        n_layers,
        d_model,
        feat_out=-1,
        n_ds_layers=4,
        n_full_folded_layers=3,
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
        conv_norm_type='group_norm',
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
        self_condition=True
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
        for _ in range(n_layers):
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
                weight_standardization=weight_standardization
            )
            self.layers.append(layer)

        self.repeated_conformer_layers = nn.ModuleList()

        for _ in range(n_full_folded_layers):
            repeated_conformer_layer = ConformerLayer( # full seq layer
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
                    weight_standardization=weight_standardization
            )
            self.repeated_conformer_layers.append(repeated_conformer_layer)

        self.ds_sequence_conformer_layers = nn.ModuleList()
        for _ in range(n_ds_layers):
            ds_sequence_conformer_layer = ConformerLayer( # ds seq layer
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
                weight_standardization=weight_standardization
            )
            self.ds_sequence_conformer_layers.append(ds_sequence_conformer_layer)

        self.down_and_up_module = upsample_downsample_module(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=4, 
            stride=2, 
            padding=1, 
            num_repeats=3, 
            ceil_mode=True
        )


        self.self_condition = self_condition

        self.num_repeats = num_repeats

  

        if feat_out > 0 and feat_out != self._feat_out:
            self.out_proj = nn.Linear(self._feat_out, feat_out)
            self._feat_out = feat_out
        else:
            self.out_proj = None
            self._feat_out = d_model
        self.set_max_audio_length(self.pos_emb_max_len)
        self.use_pad_mask = True

        self.pad_vector = torch.zeros(1, d_model)

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


  
            
    def selfconditioning(self, audio_signal, decoder):
        interim_logits = decoder(encoder_output=audio_signal.transpose(1, 2), logits=True)
        interim_posterior = F.softmax(interim_logits, dim=-1)
        interim_log_posterior = torch.log(interim_posterior)

        if self.self_condition:
            audio_signal = decoder.integrate_projections(audio_signal, decoder.project_back(interim_posterior))
        
        return audio_signal, interim_log_posterior


    @typecheck()
    def forward(self, audio_signal, decoder, length=None, segment_lens=None, return_cross_utterance_attention=False):
        self.update_max_seq_length(seq_length=audio_signal.size(2), device=audio_signal.device)

        return self.forward_for_export(
            audio_signal=audio_signal, 
            decoder=decoder, 
            length=length, 
            segment_lens=segment_lens, 
            return_cross_utterance_attention=return_cross_utterance_attention
        )


    @typecheck()
    def forward_for_export(self, audio_signal, decoder, length, segment_lens, return_cross_utterance_attention):
        max_audio_length: int = audio_signal.size(-1)

        if segment_lens == None:
            segment_lens = torch.as_tensor([audio_signal.shape[0]], dtype=torch.int32) 

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
                audio_signal = checkpoint(self.create_custom_forward(layer), audio_signal, att_mask, pos_emb, pad_mask)
            else:
                audio_signal = layer(x=audio_signal, att_mask=att_mask, pos_emb=pos_emb, pad_mask=pad_mask)
        

        interim_posteriors = []

        audio_signal, interim_log_posterior = self.selfconditioning(audio_signal, decoder)
        interim_posteriors.append(interim_log_posterior)

        pad_manager = padding_manager(sub_batch_sizes=segment_lens)

       
        ds_pos_emb = None

        for repeat in range(self.num_repeats): 
            ds_audio, downsampled_lens, upsample_fn = self.down_and_up_module(audio_signal, length)
            #ds_audio = ds_audio[:, :downsampled_lens.max(), :]
           
            ds_sb_audio, ds_sb_masks = pad_manager.convert_to_subbatches(ds=ds_audio, ds_lengths=downsampled_lens)
            
            ds_sb_attn_mask = pad_mask_to_attn_mask(ds_sb_masks)
            
            # update max_lens
            max_ds_audio_length = ds_sb_audio.size(1)
            if max_ds_audio_length > self.max_audio_length:
                self.set_max_audio_length(max_ds_audio_length)

            if ds_pos_emb == None:
                ds_sb_audio, ds_pos_emb = self.pos_enc(ds_sb_audio)

            return_attentions = False if return_cross_utterance_attention == False or repeat < self.num_repeats - 1 else True
            for lth, ds_sequence_conformer_layer in enumerate(self.ds_sequence_conformer_layers):
                ds_sb_audio = checkpoint(
                    self.create_custom_forward(ds_sequence_conformer_layer),
                        ds_sb_audio, # x
                        ds_sb_attn_mask, # attn_mask
                        ds_pos_emb, # pos_emb
                        ds_sb_masks, # pad_mask
                        None,
                        None,
                        return_attentions
                )
                ds_sb_audio, cross_attns = ds_sb_audio if return_attentions else (ds_sb_audio, None) # should add cross_attns to list so it's not overwritten :{

            ds_reverted_audio = pad_manager.revert_from_subbatches(ds_sb_audio)
            reverted_audio = upsample_fn(ds_reverted_audio)
            audio_signal = reverted_audio

            for lth, repeated_conformer_layer in enumerate(self.repeated_conformer_layers):
                if repeat != self.num_repeats - 1 or lth != len(self.repeated_conformer_layers) - 1:
                    audio_signal = checkpoint(self.create_custom_forward(repeated_conformer_layer), audio_signal, att_mask, pos_emb, pad_mask)
                    audio_signal, interim_log_posterior = self.selfconditioning(audio_signal, decoder) # SC-CTC
                    interim_posteriors.append(interim_log_posterior)
                else: # don't checkpoint grad on last layer
                    audio_signal = repeated_conformer_layer(x=audio_signal, att_mask=att_mask, pos_emb=pos_emb, pad_mask=pad_mask)
        
        #########################

        if self.out_proj is not None:
            audio_signal = self.out_proj(audio_signal) # if dim of decoder is not equal to dim of encoder, then we need to project the output

        out_signal = audio_signal

        out_signal = torch.transpose(out_signal, 1, 2) # (batch, seq_len, d_model) -> (batch, d_model, seq_len) 

        # stack the posteriors along the first dimension (height, batch, d_model, seq_len)
        interim_posteriors = torch.stack(interim_posteriors, dim=0)
     
        main_outputs = (out_signal, interim_posteriors, length)
        attn_map = cross_attns
        additional_outputs = {'attn_map': attn_map, 'ds_lengths': downsampled_lens}
        return main_outputs if attn_map is None else main_outputs + (additional_outputs,)


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


def calc_length(lengths, padding, kernel_size, stride, ceil_mode, repeat_num=1):
    """ Calculates the output length of a Tensor passed through a convolution or max pooling layer"""
    add_pad: float = (padding * 2) - kernel_size
    one: float = 1.0
    for i in range(repeat_num):
        lengths = torch.div(lengths.to(dtype=torch.float) + add_pad, stride) + one
        if ceil_mode:
            lengths = torch.ceil(lengths)
        else:
            lengths = torch.floor(lengths)
    return lengths.to(dtype=torch.long)


class upsample_downsample_module(nn.Module):
    '''
    Performs downsampling then returns a function that can be used to upsample the output
    '''
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, num_repeats=3, ceil_mode=True):
        super(upsample_downsample_module, self).__init__()
        self.downsample = downsample_layer(in_channels, out_channels, kernel_size, stride, padding, num_repeats, ceil_mode)
        self.upsample = upsample_layer(out_channels, in_channels, kernel_size, stride, padding, num_repeats, ceil_mode)

    def forward(self, x, length):
        x, masks, residuals, pad_n, ds_lengths = self.downsample(x, length)
        return x, ds_lengths, lambda z: self.upsample(z, masks, residuals, pad_n)
        

class downsample_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, num_repeats=3, ceil_mode=True):
        super(downsample_layer, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.activation = nn.ReLU()
        self.num_repeats = num_repeats
        self.ceil_mode = ceil_mode
        self.stride = stride

    @staticmethod
    def padder(x, divisible):
        '''
        Pads so that the length is divisible by the given number
        '''
        pad_n = (divisible - x.shape[1] % divisible) % divisible
        return torch.cat([x, torch.zeros(x.shape[0], pad_n, x.shape[2], device=x.device)], dim=1), pad_n

    def forward(self, x, length):
        to_be_divisible = self.stride ** self.num_repeats
        b, n, c = x.shape
        inp_mask = torch.arange(n, device=x.device).expand(b, n) < length.unsqueeze(1)
        masks = [inp_mask]
        residuals = [x]
        ds_lengths = length
        x, pad_n = self.padder(x, to_be_divisible)
        
        for i in range(self.num_repeats):
            x = self.conv(x.transpose(1, 2)).transpose(1, 2)
            x = self.activation(x)
            ds_lengths = calc_length(ds_lengths, self.conv.padding[0], self.conv.kernel_size[0], self.conv.stride[0], self.ceil_mode)
            mask = torch.arange(x.shape[1], device=x.device).expand(b, x.shape[1]) >= ds_lengths.unsqueeze(1)
            masks.append(mask)
            x = x.masked_fill(mask.unsqueeze(-1), 0)
            residuals.append(x)
        
        # remove last item and reverse
        masks = masks[:-1][::-1]
        residuals = residuals[:-1][::-1]

        return x, masks, residuals, pad_n, ds_lengths
        
class upsample_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, num_repeats=3, ceil_mode=True):
        super(upsample_layer, self).__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.residual_merge_layer = nn.Linear(out_channels*2, out_channels)
        self.merge_residual_fn = lambda x, residual: self.residual_merge_layer(torch.cat([x, residual], dim=-1))
        self.activation = nn.ReLU()
        self.num_repeats = num_repeats
        self.ceil_mode = ceil_mode
        self.stride = stride

    def forward(self, x, masks, residuals, pad_n):
        for i in range(self.num_repeats):
            x = self.conv(x.transpose(1, 2)).transpose(1, 2)
            x = self.activation(x)
            if i == self.num_repeats - 1:
                x = x[:, :(x.shape[1] - pad_n), :]
            x = x.masked_fill(masks[i].unsqueeze(-1), 0)
            #x = x + residuals[i]
            x = self.merge_residual_fn(x, residuals[i])
        return x

'''
class _downsample_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=10, padding=0, expansion_factor=2):
        super(downsample_layer, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels*expansion_factor, kernel_size, stride, padding)
        self.relu = nn.ReLU()
        self.ffn = nn.Linear(out_channels*expansion_factor, out_channels)
        self.swish = lambda x: x * torch.sigmoid(x)

    @staticmethod
    def autopad(x, stride):

        topad = ((torch.tensor((x.size(1)/stride)).ceil() - x.size(1)/stride)*stride).ceil().long()
        if topad > 0:
            x = torch.cat([x, torch.zeros(x.size(0), topad, x.size(-1)).to(x.device)], dim=1)
        return x

    def forward(self, x, mask=None):
        x = self.conv(x.transpose(1,2)).transpose(1,2)
        x = self.relu(x)
        x = self.ffn(x)
        x = self.swish(x)
        return x.masked_fill(mask, 0.0) if mask is not None else x


def upsample_fn(x, scale_factor):
    return x.repeat_interleave(scale_factor, dim=1)
'''


class padding_manager():
    def __init__(self, sub_batch_sizes):
        self.sb_indices = None
        self.sb_masks = None
        self.revert_indices = None
        self.sub_batch_sizes = sub_batch_sizes

        self.ds_lengths = None
        self.ds_total_sb_lengths = None
        self.max_ds_total_sb_lengths = None
      
    def convert_to_subbatches(self, ds, ds_lengths):
        self.ds_lengths = ds_lengths
        b, t, d, sub_batch_sizes = *ds.shape, self.sub_batch_sizes
        flat_sequences = rearrange(ds, 'b t d -> (b t) d')
        padding_index = torch.zeros(1, d).to(ds.device)
        flat_sequence_with_padding_index = torch.cat([flat_sequences, padding_index], dim=0)
        if self.sb_indices != None:
            return flat_sequence_with_padding_index[self.sb_indices], self.sb_masks

        ds_sb_lengths = torch.split(ds_lengths, sub_batch_sizes.tolist()) # list of utterance lengths within each sub-batch
        max_ds_lengths = torch.full((b,), ds_lengths.max(), dtype=torch.long, device=ds.device) # max length of utterance in each sub-batch i.e with the padding
        culm_max_ds_lengths = torch.cumsum(max_ds_lengths, dim=0) # culmulative sum of max lengths of utterances in each sub-batch
        culm_max_ds_lengths_start = culm_max_ds_lengths - ds_lengths.max() # start of each utterance in the flat sequence
        # split these into sub-batches
        culm_max_ds_lengths_start_sb = torch.split(culm_max_ds_lengths_start, sub_batch_sizes.tolist()) # list of start indices of each utterance in each sub-batch within the flat sequence
        culm_ds_lengths_end_sb = torch.split(culm_max_ds_lengths_start + ds_lengths, sub_batch_sizes.tolist()) # list of end indices of each utterance in each sub-batch within the flat sequence
        ds_total_sb_lengths = [el.sum().item() for el in torch.split(ds_lengths, sub_batch_sizes.tolist())] # list of total lengths of each sub-batch
        self.ds_total_sb_lengths = ds_total_sb_lengths
        max_ds_total_sb_lengths = max(ds_total_sb_lengths) # max total length of all sub-batches
        self.max_ds_total_sb_lengths = max_ds_total_sb_lengths
        # now get the indices
        all_sb_indices = []
        all_sb_pad_masks = []
        for culm_max_ds_length_start_sb, culm_ds_length_end_sb in zip(culm_max_ds_lengths_start_sb, culm_ds_lengths_end_sb):
            sb_indices = []
            for start, end in zip(culm_max_ds_length_start_sb, culm_ds_length_end_sb):
                sb_indices.append(torch.arange(start, end))
            sb_indices = torch.cat(sb_indices)
            diff_from_max_length = max_ds_total_sb_lengths - sb_indices.shape[0]
            # pad with -1 which will be used to grab the padding from the flat sequence
            sb_indices = torch.cat([sb_indices, torch.full((diff_from_max_length,), -1, dtype=torch.long)])
            cur_pad_mask = torch.zeros(sb_indices.shape[0], dtype=torch.bool) # create a mask for the padding
            cur_pad_mask[sb_indices == -1] = True # set the padding mask to True where the padding is ;)
            all_sb_pad_masks.append(cur_pad_mask)
            all_sb_indices.append(sb_indices)
        all_sb_pad_masks = torch.stack(all_sb_pad_masks)
        all_sb_indices = torch.stack(all_sb_indices)
        # okay now we have the indices, we can grab the flat sequences
        all_sb_pad_masks = all_sb_pad_masks.to(ds.device)
        self.sb_indices = all_sb_indices
        self.sb_masks = all_sb_pad_masks
        
        return flat_sequence_with_padding_index[all_sb_indices], all_sb_pad_masks


    def revert_from_subbatches(self, ds_sb_sequences):
        assert self.sb_indices != None, "You need to call convert_to_subbatches first"
        b, t, d, sub_batch_sizes, ds_lengths, max_ds_total_sb_lengths, ds_total_sb_lengths = *ds_sb_sequences.shape, self.sub_batch_sizes, self.ds_lengths, self.max_ds_total_sb_lengths, self.ds_total_sb_lengths
        reflat_sequences = rearrange(ds_sb_sequences, 'b t d -> (b t) d')
        padding_index = torch.zeros(1, d).to(ds_sb_sequences.device)
        reflat_sequence_with_padding_index = torch.cat([reflat_sequences, padding_index], dim=0)
        
        if self.revert_indices != None:
            return reflat_sequence_with_padding_index[self.revert_indices]

        ds_sb_lengths = torch.split(ds_lengths, sub_batch_sizes.tolist())
        all_revert_indices = []
        pos = 0

        for ds_sb_length, ds_total_sb_length in zip(ds_sb_lengths, ds_total_sb_lengths):
            cur_revert_indices = []
            for utt_length in ds_sb_length:
                cur_indices = torch.arange(pos, pos+utt_length)
                diff_from_max_length = ds_lengths.max() - cur_indices.shape[0]
                cur_indices = torch.cat([cur_indices, torch.full((diff_from_max_length,), -1, dtype=torch.long)])
                cur_revert_indices.append(cur_indices)
                pos += utt_length
            pos += max_ds_total_sb_lengths - ds_total_sb_length
            all_revert_indices.append(torch.stack(cur_revert_indices))
        all_revert_indices = torch.cat(all_revert_indices)

        self.revert_indices = all_revert_indices
        return reflat_sequence_with_padding_index[all_revert_indices]