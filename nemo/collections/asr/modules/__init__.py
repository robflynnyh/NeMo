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

from nemo.collections.asr.modules.audio_preprocessing import (
    AudioToMelSpectrogramPreprocessor,
    AudioToMFCCPreprocessor,
    CropOrPadSpectrogramAugmentation,
    MaskedPatchAugmentation,
    SpectrogramAugmentation,
)
from nemo.collections.asr.modules.beam_search_decoder import BeamSearchDecoderWithLM
from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder, ConformerEncoderAdapter

from nemo.collections.asr.modules.scconformer_encoder import SelfConditionedConformerEncoder, SelfConditionedConformerEncoderAdapter
from nemo.collections.asr.modules.folded_scconformer_encoder import FoldedSelfConditionedConformerEncoder, FoldedSelfConditionedConformerEncoderAdapter 

#from nemo.collections.asr.modules.compositional_scconformer_encoder import CompositionalSelfConditionedConformerEncoder, CompositionalSelfConditionedConformerEncoderAdapter
#from nemo.collections.asr.modules.exp3_ctx_compositional_scconformer_encoder import experiment3
#from nemo.collections.asr.modules.hierarchical_conformer_encoder import HierarchicalConformerEncoder

from nemo.collections.asr.modules.ctx_conformer_encoder import CtxConformerEncoder
from nemo.collections.asr.modules.seperated_ctx_conformer_encoder import CtxConformerEncoder as SeparatedCtxConformerEncoder

from nemo.collections.asr.modules.adaptive_seperated_ctx_conformer_encoder import CtxConformerEncoder as AdaptiveSeparatedCtxConformerEncoder

from nemo.collections.asr.modules.conformer_encoder_grouped_attention import GroupedConformerEncoder

from nemo.collections.asr.modules.scconformer_encoder_BERT import SelfConditionedConformerEncoder as SelfConditionedConformerEncoderBERT

from nemo.collections.asr.modules.sconformer_xl import SelfConditionedConformerXL

from nemo.collections.asr.modules.conv_asr import (
    ConvASRDecoder,
    ConvASRSelfConditioningDecoder,
    ConvASRDecoderClassification,
    ConvASRDecoderReconstruction,
    ConvASREncoder,
    ConvASREncoderAdapter,
    ECAPAEncoder,
    ParallelConvASREncoder,
    SpeakerDecoder,
)
from nemo.collections.asr.modules.graph_decoder import ViterbiDecoderWithGraph
from nemo.collections.asr.modules.lstm_decoder import LSTMDecoder
from nemo.collections.asr.modules.rnn_encoder import RNNEncoder
from nemo.collections.asr.modules.rnnt import RNNTDecoder, RNNTDecoderJointSSL, RNNTJoint
