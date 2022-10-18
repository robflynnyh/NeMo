import torch, numpy as np, torch.nn as nn, torch.nn.functional as F
from typing import List, Optional, Union, Dict, Any, Tuple, Callable
import inspect
from nemo.collections.asr.parts.submodules.multi_head_attention import RelPositionMultiHeadAttention
from nemo.collections.asr.parts.utils.helpers import (
    exists,
    isfalse
)

class BertIntermediate(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = F.gelu

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout: float=0.1,
    ):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(
        self,
        n_head,
        n_feat,
        dropout_rate,
        pos_bias_u=None,
        pos_bias_v=None
    ):
        super().__init__()
        self.self = RelPositionMultiHeadAttention(
            n_head,
            n_feat,
            dropout_rate,
            pos_bias_u=pos_bias_u,
            pos_bias_v=pos_bias_v
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.LayerNorm = nn.LayerNorm(n_feat)

    def forward(self, x, mask, return_attentions=False, pos_emb=None):
        attn_out = self.self(
            query=x, 
            key=x, 
            value=x, 
            mask=mask, 
            pos_emb=pos_emb,
            return_attentions=return_attentions
        )
        attention_out, maps = attn_out if return_attentions else (attn_out, None)
        x = self.dropout(attention_out) + x
        x = self.LayerNorm(x)
        return (x, maps) if return_attentions else x



class BertLayer(nn.Module):
    def __init__(
            self, 
            hidden_size: int,
            num_attention_heads: int,
            dropout_rate_attn: float,
            pos_bias_u,
            pos_bias_v,
        ):
        super().__init__()

        self.attention = BertAttention(
            n_head=num_attention_heads,
            n_feat=hidden_size,
            dropout_rate=dropout_rate_attn,
            pos_bias_u=pos_bias_u,
            pos_bias_v=pos_bias_v
        )
       
        self.intermediate = BertIntermediate(
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 4,
        )
        
        self.output = BertOutput(
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 4,
            dropout=0.1,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        pos_emb: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:

        self_attention_outputs = self.attention(
            x=hidden_states,
            mask=attention_mask,
            return_attentions=output_attentions,
            pos_emb=pos_emb
        )
        attention_output, map = self_attention_outputs if output_attentions else (self_attention_outputs, None)
        layer_output = self.feed_forward_chunk(attention_output)

        return layer_output if not output_attentions else (layer_output, map)

    def freeze_params(self, freeze: bool = True):
        for p in self.parameters():
            p.requires_grad = not freeze

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output



class BertEncoderCTC(nn.Module):
    def __init__(
        self,
        num_encoder_layers: int,
        hidden_size: int,
        num_attention_heads: int,
        dropout_rate_attn: float,
        ctc_vocab_size: int,
        acoustic_hidden_size: int,
        pos_bias_u,
        pos_bias_v,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.dropout_rate_attn = dropout_rate_attn
        self.ctc_vocab_size = ctc_vocab_size
        self.acoustic_hidden_size = acoustic_hidden_size

        self.project_pos_emb = nn.Linear(acoustic_hidden_size, hidden_size)

        self.layers = nn.ModuleList([])
        for _ in range(num_encoder_layers):
            layer = BertLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                dropout_rate_attn=dropout_rate_attn,
                pos_bias_u=pos_bias_u,
                pos_bias_v=pos_bias_v,
            )
            self.layers.append(layer)

        self.fuse_inputs = nn.Linear(hidden_size + ctc_vocab_size, hidden_size)
    
        self.project_to_acoustic = nn.Linear(hidden_size, acoustic_hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        layer_num: int,
        pos_emb: torch.Tensor,
        residual: torch.Tensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
    ):
        assert layer_num >= 0 and layer_num < len(self.layers), f"layer_num must be between 0 and {len(self.layers)}"
        layer = self.layers[layer_num]
        b, n, c = x.shape
        residual = residual if exists(residual) else torch.zeros(b, n, self.hidden_size, device=x.device)

        x = torch.cat((x, residual), dim=-1) # concat along feature dim
        x = self.fuse_inputs(x)
        
        pos_emb = self.project_pos_emb(pos_emb) # convert to bert hidden size

        x = layer(x, attention_mask=attention_mask, pos_emb=pos_emb)
        
        new_residual = x
        x = self.project_to_acoustic(x)
        return x, new_residual

    @staticmethod
    def model_surgery(roberta_state_dict):
        def tx_transforms(tx:str):
            tx = tx.replace("roberta.encoder.layer.", "layers.")
            tx = tx.replace('.self.query.', '.self.linear_q.')
            tx = tx.replace('.self.key.', '.self.linear_k.')
            tx = tx.replace('.self.value.', '.self.linear_v.')
            tx = tx.replace('.attention.output.dense.', '.attention.self.linear_out.')
            tx = tx.replace('.attention.output.LayerNorm.', '.attention.LayerNorm.')
            return tx
        '''
        Changes the name of robertas keys to match my model
        '''
        new_state_dict = {}
        for k, v in roberta_state_dict.items():
            new_k = tx_transforms(k)
            new_state_dict[new_k] = v
        return new_state_dict

    def load_roberta(self, roberta_path, map_location='cpu'):
        roberta_state_dict = torch.load(roberta_path, map_location=map_location)
        roberta_state_dict = self.model_surgery(roberta_state_dict)
        self.load_state_dict(roberta_state_dict, strict=False)

    def freeze_BERT(self, freeze: bool = True):
        '''
        Use this for freezing the BERT layers, but not the other FF layers
        '''
        for layer in self.layers:
            layer.freeze_params(freeze=freeze)