#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Command line tool that produces embeddings for a large documents base based on the pretrained ctx & question encoders
 Supposed to be used in a 'sharded' way to speed up the process.
"""
import os
import pathlib
import gzip, pickle

import argparse
import csv
import logging
import pickle
from typing import List, Tuple

import numpy as np
import torch
from torch import nn

from dpr.models import init_biencoder_components
from dpr.options import add_encoder_params, setup_args_gpu, print_args, set_encoder_params_from_state, \
    add_tokenizer_params, add_cuda_params
from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import setup_for_distributed_mode, get_model_obj, load_states_from_checkpoint,move_to_device
import json
from tqdm import tqdm, trange

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if (logger.hasHandlers()):
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)

def gen_ctx_vectors(ctx_rows: List[Tuple[object, str, str]], model: nn.Module, tensorizer: Tensorizer,
                    insert_title: bool = True) -> List[Tuple[object, np.array]]:
    n = len(ctx_rows)
    print(f"total row:{n}")
    bsz = args.batch_size
    total = 0
    results = []
    for j, batch_start in enumerate(trange(0, n, bsz)):

        batch_token_tensors = [tensorizer.text_to_tensor(ctx[1], title=ctx[2] if insert_title else None) for ctx in
                               ctx_rows[batch_start:batch_start + bsz]]

        ctx_ids_batch = move_to_device(torch.stack(batch_token_tensors, dim=0),args.device)
        ctx_seg_batch = move_to_device(torch.zeros_like(ctx_ids_batch),args.device)
        ctx_attn_mask = move_to_device(tensorizer.get_attn_mask(ctx_ids_batch),args.device)
        with torch.no_grad():
            _, out, _ = model(ctx_ids_batch, ctx_seg_batch, ctx_attn_mask)
        out = out.cpu()

        ctx_ids = [r[0] for r in ctx_rows[batch_start:batch_start + bsz]]

        assert len(ctx_ids) == out.size(0)

        total += len(ctx_ids)

        results.extend([
            (ctx_ids[i], out[i].view(-1).numpy())
            for i in range(out.size(0))
        ])

        if total % 10 == 0:
            logger.info('Encoded passages %d', total)

    return results


def main(args):
    saved_state = load_states_from_checkpoint(args.model_file)
    set_encoder_params_from_state(saved_state.encoder_params, args)
    print_args(args)
    
    tensorizer, encoder, _ = init_biencoder_components(args.encoder_model_type, args, inference_only=True)

    encoder = encoder.ctx_model

    encoder, _ = setup_for_distributed_mode(encoder, None, args.device, args.n_gpu,
                                            args.local_rank,
                                            args.fp16,
                                            args.fp16_opt_level)
    encoder.eval()

    # load weights from the model file
    model_to_load = get_model_obj(encoder)
    logger.info('Loading saved model state ...')
    logger.debug('saved model keys =%s', saved_state.model_dict.keys())
    if  args.code_to_text:
        logger.info('Code to Text')
    else:
        logger.debug('Text to Code')



    prefix_len = len('ctx_model.')
    ctx_state = {key[prefix_len:]: value for (key, value) in saved_state.model_dict.items() if
                 key.startswith('ctx_model.')}
    model_to_load.load_state_dict(ctx_state)

    logger.info('reading data from file=%s', args.ctx_file)

    rows = []
    idx=0
    if args.ctx_file.endswith('gz'):
        with gzip.open(args.ctx_file, 'r') as pf:
            data = pf.readlines()
            for idx, d in enumerate(data):
                line_a = json.loads(str(d, encoding='utf-8'))
                doc_token = ' '.join(line_a['docstring_tokens'])
                code_token = ' '.join(line_a['code_tokens'])
                rows.extend([(args.ctx_file + "_" + str(idx), code_token, None)])
    elif args.ctx_file.endswith('pkl'):
        definitions = pickle.load(open(args.ctx_file, 'rb'))
        for idx, d in enumerate(definitions):
            if not args.code_to_text:
                rows.extend([(args.ctx_file + "_" + str(idx), ' '.join(d["function_tokens"]), None)])
            else:
                rows.extend([(args.ctx_file + "_" + str(idx), ' '.join(d["docstring_tokens"]), None)])
    elif args.ctx_file.endswith('deduplicated.summaries.txt'):
        with open(args.ctx_file) as f:
            for idx, line in enumerate(f):
                rows.extend([(args.ctx_file + "_" + str(idx), line, None)])
    else:
        with open(args.ctx_file) as tsvfile:
            # reader = csv.reader(tsvfile, delimiter='\t')
            # # file format: doc_id, doc_text, title
            # rows.extend([(row[0], row[1], row[2]) for row in reader if row[0] != 'id'])

            prompt_counts=0
            for line in tsvfile:
                if args.CSNET_ADV:
                    prompt_counts+=1
                    js=json.loads(line)
                    rows.extend([(js["idx"], ' '.join(js["function_tokens"]) , None )])
                    if prompt_counts<5:
                        print('encoding function tokens: ', ' '.join(js["function_tokens"]))
                elif args.dataset == 'conala':
                    js=json.loads(line)
                    rows.extend([(js["id"], ' '.join(js["snippet"]), None)])
                elif args.WEBQUERY:
                    prompt_counts+=1
                    js=json.loads(line)
                    raw_code = js["code"]
                    rows.extend([(js["idx"], ' '.join(js["function_tokens"]) , None )])
                    if prompt_counts<5:
                        print('encoding function tokens: ', ' '.join(js["code"]))
                elif args.dataset == "csnet_candidates":
                    prompt_counts += 1
                    js = json.loads(line)
                    raw_code = js["code"]
                    rows.extend([(js["url"], ' '.join(js["code_tokens"]), None)])
                    if prompt_counts < 5:
                        print('encoding function tokens: ', ' '.join(js["code_tokens"]))
                elif args.dataset == "CONCODE":
                    js = json.loads(line)
                    rows.extend([(args.ctx_file + "_" + str(idx), js["code"], None)])
                elif args.dataset=="KP20k":
                    js = json.loads(line)
                    text = js["title"] + ' </s> ' + js["abstract"]
                    rows.extend([(str(idx), line, None)])
                elif args.ctx_file.endswith("txt"):
                    line = line.strip().split('<CODESPLIT>')
                    target_str = 4
                    code = line[target_str]
                    rows.extend([(args.ctx_file + "_" + str(idx), code, None)])
                    if idx<5:
                        logger.info("Encoding Code: %s", code)
                else:
                    rows.extend([(args.ctx_file + "_" + str(idx), line, None)])
                idx+=1



    shard_size = int(len(rows) / args.num_shards)
    start_idx = args.shard_id * shard_size
    end_idx = start_idx + shard_size

    logger.info('Producing encodings for passages range: %d to %d (out of total %d)', start_idx, end_idx, len(rows))
    rows = rows[start_idx:end_idx]

    # data = gen_ctx_vectors(rows, encoder, tensorizer, True)
    data = gen_ctx_vectors(rows, encoder, tensorizer, False)

    file = args.out_file + '_' + str(args.shard_id) + '.pkl'
    pathlib.Path(os.path.dirname(file)).mkdir(parents=True, exist_ok=True)
    logger.info('Writing results to %s' % file)
    with open(file, mode='wb') as f:
        pickle.dump(data, f)

    logger.info('Total passages processed %d. Written to %s', len(data), file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    add_encoder_params(parser)
    add_tokenizer_params(parser)
    add_cuda_params(parser)

    parser.add_argument('--ctx_file', type=str, default=None, help='Path to passages set .tsv file')
    parser.add_argument('--out_file', required=True, type=str, default=None,
                        help='output file path to write results to')

    parser.add_argument('--shard_id', type=int, default=0, help="Number(0-based) of data shard to process")
    parser.add_argument('--num_shards', type=int, default=1, help="Total amount of data shards")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for the passage encoder forward pass")
    parser.add_argument("--CSNET_ADV", action='store_true',
                        help="Whether to parse CSNET_ADV.")
    parser.add_argument("--WEBQUERY", action='store_true',
                        help="Whether to parse CSNET_ADV.")
    parser.add_argument("--concode_with_code", action='store_true',
                        help="Whether to parse CSNET_ADV.")
    parser.add_argument("--code_to_text", action='store_true',
                        help="Whether to parse CSNET_ADV.")
    parser.add_argument('--dataset', type=str, default=None,
                        help=' to build correct dataset parser ')

    args = parser.parse_args()

    assert args.model_file, 'Please specify --model_file checkpoint to init model weights'
    print('--------------------------')
    print(args.pretrained_model_cfg)
    print('--------------------------')

    setup_args_gpu(args)

    
    main(args)
