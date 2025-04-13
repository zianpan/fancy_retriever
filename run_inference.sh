TOP_K=200
RETRIEVAL_RESULT_FILE={output file path like OUTPUT_DIR_PATH+split+_+${TOP_K}+_+{code_totext or text_to_code}+.json e.g., ../redcoder_data/retriever_output/csnet_text_to_code/with_comments/python_csnet_pos_only_retrieval_dedup_test_30.json}
CHECKPOINT={retirver best checkpint from first step e.g., ../redcoder_data/checkpint/codexglue_csnet_java or python_scoder_text_to_code.cp}
CANDIDATE_FILE={java/python_dedupe_definitions_v2.pkl file path this file reasled with official CSNET e.g., ../redcoder_data/retrieval_database/java or python_dedupe_definitions_v2.pkl}
ENCODDING_CANDIDATE_PREFIX = {OUTPUT DIR/encoddings_${candidate_file}}
PRETRAINED_MODEL_PATH={local file path of [hf_graphcoebert/hf_codebert] as discussed above}
FILE_FOR_WHICH_TO_RETIRVE={each of train/dev/test filepath e.g., CodeSearchNet/java or python/split.jsonl}

CUDA_VISIBLE_DEVICES=${SINGLE_GPU_DEVICE_IS_ENOUGH} python {dense_retriever.py | dense_retriever_with_comments.py}
            --model_file ${CHECKPOINT}
            --ctx_file  ${CANDIDATE_FILE}
            --qa_file ${FILE_FOR_WHICH_TO_RETIRVE}
            --encoded_ctx_file {encoded document files glob expression e.g., ENCODDING_CANDIDATE_PREFIX}
            --out_file  ${RETRIEVAL_RESULT_FILE}
            --n-docs  ${TOP_K}
            --sequence_length 256 
            --save_or_load_index 