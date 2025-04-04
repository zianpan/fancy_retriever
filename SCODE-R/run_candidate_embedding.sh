# CHECKPOINT={/opt/dlami/nvme/pretrained/ckpts/REDCODER-main.cp}
# CANDIDATE_FILE={/opt/dlami/nvme/pretrained/database/python_dedupe_definitions_v2-001.pkl}
# DEVICES={0}
# NUM_DEVICES={1}
# ENCODDING_CANDIDATE_PREFIX={/opt/dlami/nvme/outputs_emb/encoddings_${candidate_file}}
# PRETRAINED_MODEL_PATH={/opt/dlami/nvme/hgface_models}
          

# CUDA_VISIBLE_DEVICES=${DEVICES} python  -m torch.distributed.launch 
#           --nproc_per_node=${NUM_DEVICES} generate_dense_embeddings.py 
#           --model_file  ${CHECKPOINT}  
#           --encoder_model_type hf_roberta
#           --pretrained_model_cfg  ${PRETRAINED_MODEL_PATH}
#           --batch_size 512 
#           --ctx_file  ${CANDIDATE_FILE}
#           --shard_id 0 
#           --num_shards 1 
#           --out_file  ENCODDING_CANDIDATE_PREFIX

#!/bin/bash

CHECKPOINT="/mnt/new_volume/ckpts/REDCODER-main.cp"
CANDIDATE_FILE="/mnt/new_volume/database/python_dedupe_definitions_v2-001.pkl"
DEVICES=0
NUM_DEVICES=1
PRETRAINED_MODEL_PATH="/mnt/new_volume/graphcodebert-base"

# Get filename only from candidate path
CANDIDATE_FILENAME=$(basename "$CANDIDATE_FILE")
ENCODDING_CANDIDATE_PREFIX="/mnt/new_volume/outputs_emb/encoddings_${CANDIDATE_FILENAME}"

# CUDA_VISIBLE_DEVICES=$DEVICES python -m torch.distributed.launch \
#   --nproc_per_node=$NUM_DEVICES generate_dense_embeddings.py \
#   --model_file "$CHECKPOINT" \
#   --encoder_model_type hf_roberta \
#   --pretrained_model_cfg "$PRETRAINED_MODEL_PATH" \
#   --batch_size 512 \
#   --ctx_file "$CANDIDATE_FILE" \
#   --shard_id 0 \
#   --num_shards 1 \
#   --out_file "$ENCODDING_CANDIDATE_PREFIX"

CUDA_VISIBLE_DEVICES=$DEVICES python generate_dense_embeddings.py \
  --model_file "$CHECKPOINT" \
  --encoder_model_type hf_roberta \
  --pretrained_model_cfg "$PRETRAINED_MODEL_PATH" \
  --batch_size 1024 \
  --ctx_file "$CANDIDATE_FILE" \
  --shard_id 0 \
  --num_shards 1 \
  --out_file "$ENCODDING_CANDIDATE_PREFIX"
