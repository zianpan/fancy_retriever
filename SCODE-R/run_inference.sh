TOP_K=200
RETRIEVAL_RESULT_FILE="/mnt/new_volume/outputs_ret/retrieval_valid_full_top500_result.json"
CHECKPOINT="/mnt/new_volume/ckpts/REDCODER-main.cp"
CANDIDATE_FILE="/mnt/new_volume/database/python_dedupe_definitions_v2-001.pkl"

CANDIDATE_FILENAME=$(basename "$CANDIDATE_FILE")
# ENCODDING_CANDIDATE_PREFIX="/mnt/new_volume/outputs_emb/encoddings_${CANDIDATE_FILENAME}"
ENCODDING_CANDIDATE_PREFIX="/mnt/new_volume/outputs_emb/encoddings_python_dedupe_definitions_v2-001.pkl_0.pkl"

PRETRAINED_MODEL_PATH="/mnt/new_volume/graphcodebert-base"
# FILE_FOR_WHICH_TO_RETIRVE="/mnt/new_volume/codesearchnet/codesearchnet_validation.jsonl"
# FILE_FOR_WHICH_TO_RETIRVE="/mnt/new_volume/codesearchnet/codesearchnet_train_sub200.jsonl"
FILE_FOR_WHICH_TO_RETIRVE="/mnt/new_volume/codesearchnet/codesearchnet_validation_filtered_full.jsonl"
SINGLE_GPU_DEVICE_IS_ENOUGH=0

for i in $(seq 1 24); do
    FILE_FOR_WHICH_TO_RETIRVE=/mnt/new_volume/codesearchnet/ds_validation_clean_chunk_${i}.jsonl 
    RETRIEVAL_RESULT_FILE="/mnt/new_volume/outputs_ret/retrieval_chunk_${i}_top${TOP_K}_result.json"

    echo "Processing chunk $i ..."
    
    CUDA_VISIBLE_DEVICES=${SINGLE_GPU_DEVICE_IS_ENOUGH} python dense_retriever.py \
        --model_file ${CHECKPOINT} \
        --ctx_file  ${CANDIDATE_FILE} \
        --qa_file ${FILE_FOR_WHICH_TO_RETIRVE} \
        --encoded_ctx_file ${ENCODDING_CANDIDATE_PREFIX} \
        --out_file  ${RETRIEVAL_RESULT_FILE} \
        --n-docs  ${TOP_K} \
        --sequence_length 256 \
        --save_or_load_index

    echo "Finished chunk $i ✅"
done

# CUDA_VISIBLE_DEVICES=${SINGLE_GPU_DEVICE_IS_ENOUGH} python dense_retriever.py \
#             --model_file ${CHECKPOINT} \
#             --ctx_file  ${CANDIDATE_FILE} \
#             --qa_file ${FILE_FOR_WHICH_TO_RETIRVE} \
#             --encoded_ctx_file ${ENCODDING_CANDIDATE_PREFIX} \
#             --out_file  ${RETRIEVAL_RESULT_FILE} \
#             --n-docs  ${TOP_K} \
#             --sequence_length 256 \
#             --save_or_load_index \