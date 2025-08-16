#!/bin/sh
#PBS -q rt_HG
#PBS -N bbh
#PBS -l select=1
#PBS -l walltime=2:00:00
#PBS -j oe
#PBS -m n
#PBS -v USE_SSH=1
#PBS -koed
#PBS -V
#PBS -o outputs/bbh_cot_fewshot/
#PBS -P gag51395

cd $PBS_O_WORKDIR

set -e

export TMP="/groups/gag51395/fujii/tmp"
export TMP_DIR="/groups/gag51395/fujii/tmp"
export HF_HOME="/groups/gag51395/fujii/hf_cache"

model_name="/groups/gcg51558/fujii/checkpoints/megatron-to-hf/Llama-3.1-8b/swallow-math/exp5/iter_0010000"
tensor_parallel_size=1
data_parallel_size=1

export CUDA_VISIBLE_DEVICES="0"

source .venv/bin/activate

lm_eval --model vllm \
  --model_args pretrained=${model_name},tensor_parallel_size=${tensor_parallel_size},dtype=auto,gpu_memory_utilization=0.9,data_parallel_size=${data_parallel_size} \
  --tasks bbh_cot_fewshot \
  --batch_size auto
