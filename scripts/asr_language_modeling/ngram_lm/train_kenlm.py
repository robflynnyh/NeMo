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


# This script would train an N-gram language model with KenLM library (https://github.com/kpu/kenlm) which can be used
# with the beam search decoders on top of the ASR models. This script supports both character level and BPE level
# encodings and models which is detected automatically from the type of the model.
# After the N-gram model is trained, and stored in the binary format, you may use
# 'scripts/ngram_lm/eval_beamsearch_ngram.py' to evaluate it on an ASR model.
#
# You need to install the KenLM library and also the beam search decoders to use this feature. Please refer
# to 'scripts/ngram_lm/install_beamsearch_decoders.sh' on how to install them.
#
# USAGE: python train_kenlm.py --nemo_model_file <path to the .nemo file of the model> \
#                              --train_file <path to the training text or JSON manifest file \
#                              --kenlm_bin_path <path to the bin folder of KenLM library> \
#                              --kenlm_model_file <path to store the binary KenLM model> \
#                              --ngram_length <order of N-gram model>
#
# After training is done, the binary LM model is stored at the path specified by '--kenlm_model_file'.
# You may find more info on how to use this script at:
# https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/asr_language_modeling.html

import argparse
import logging
import os
import subprocess
import sys

import kenlm_utils
import torch


from nemo.utils import logging
from nemo.collections import nlp as nemo_nlp
"""
NeMo's beam search decoders only support char-level encodings. In order to make it work with BPE-level encodings, we
use a trick to encode the sub-word tokens of the training data as unicode characters and train a char-level KenLM. 
TOKEN_OFFSET is the offset in the unicode table to be used to encode the BPE sub-words. This encoding scheme reduces 
the required memory significantly, and the LM and its binary blob format require less storage space.
"""
TOKEN_OFFSET = 100

CHUNK_SIZE = 8192
CHUNK_BUFFER_SIZE = 512

def load_tokenizer(model_path):
    tokenizer_spe = nemo_nlp.modules.get_tokenizer(tokenizer_name="sentencepiece", tokenizer_model=model_path)
    return tokenizer_spe

def main():
    parser = argparse.ArgumentParser(
        description='Train an N-gram language model with KenLM to be used with beam search decoder of ASR models.'
    )
    parser.add_argument(
        "--train_file",
        required=True,
        type=str,
        help="Path to the training file, it can be a text file or JSON manifest",
    )

    parser.add_argument(
        '-tokenizer','--tokenizer',
        required=True,
        type=str,
        help="Path to the tokenizer model file",
    )
    parser.add_argument(
        "--kenlm_model_file", required=True, type=str, help="The path to store the KenLM binary model file"
    )
    parser.add_argument("--ngram_length", required=True, type=int, help="The order of N-gram LM")
    parser.add_argument("--kenlm_bin_path", required=True, type=str, help="The path to the bin folder of KenLM")
    parser.add_argument(
        "--do_lowercase", action='store_true', help="Whether to apply lower case conversion on the training text"
    )
    args = parser.parse_args()

 
    encoding_level = 'subword'
    tokenizer = load_tokenizer(model_path=args.tokenizer)

    """ DATASET SETUP """
    logging.info(f"Encoding the train file '{args.train_file}' ...")
    dataset = kenlm_utils.read_train_file(args.train_file, lowercase=args.do_lowercase)
    encoded_train_file = f"{args.kenlm_model_file}.tmp.txt"
    if encoding_level == "subword":
        kenlm_utils.tokenize_text(
            dataset,
            tokenizer,
            path=encoded_train_file,
            chunk_size=CHUNK_SIZE,
            buffer_size=CHUNK_BUFFER_SIZE,
            token_offset=TOKEN_OFFSET,
        )
        # --discount_fallback is needed for training KenLM for BPE-based models
        discount_arg = "--discount_fallback"
    else:
        with open(encoded_train_file, 'w', encoding='utf-8') as f:
            for line in dataset:
                f.write(f"{line}\n")

        discount_arg = ""

    arpa_file = f"{args.kenlm_model_file}.tmp.arpa"
    """ LMPLZ ARGUMENT SETUP """
    kenlm_args = [
        os.path.join(args.kenlm_bin_path, 'lmplz'),
        "-o",
        f"{args.ngram_length}",
        "--text",
        encoded_train_file,
        "--arpa",
        arpa_file,
        discount_arg,
    ]

    ret = subprocess.run(kenlm_args, capture_output=False, text=True, stdout=sys.stdout, stderr=sys.stderr)
    if ret.returncode != 0:
        raise RuntimeError("Training KenLM was not successful!")
    """ BINARY BUILD """
    logging.info(f"Running binary_build command \n\n{' '.join(kenlm_args)}\n\n")
    kenlm_args = [
        os.path.join(args.kenlm_bin_path, "build_binary"),
        "trie",
        arpa_file,
        args.kenlm_model_file,
    ]
    ret = subprocess.run(kenlm_args, capture_output=False, text=True, stdout=sys.stdout, stderr=sys.stderr)

    if ret.returncode != 0:
        raise RuntimeError("Training KenLM was not successful!")

    os.remove(encoded_train_file)
    logging.info(f"Deleted the temporary encoded training file '{encoded_train_file}'.")
    os.remove(arpa_file)
    logging.info(f"Deleted the arpa file '{arpa_file}'.")


if __name__ == '__main__':
    main()
