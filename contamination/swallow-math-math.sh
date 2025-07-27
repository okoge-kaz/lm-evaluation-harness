#!/bin/bash
#PBS -q rt_HF
#PBS -N contamination
#PBS -l select=1:ncpus=192:ngpus=8
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -m n
#PBS -v USE_SSH=1
#PBS -koed
#PBS -V
#PBS -o outputs/contamination/swallow-math/

set -e
cd $PBS_O_WORKDIR

# environment variables
export TMP="/groups/gag51395/fujii/tmp"
export TMP_DIR="/groups/gag51395/fujii/tmp"
export HF_HOME="/groups/gag51395/fujii/hf_cache"

source .venv/bin/activate

export NUMEXPR_MAX_THREADS=384

echo "Checking contamination for Swallow Math dataset against Minerva Math dataset..."
# MATH: https://huggingface.co/datasets/svc-huggingface/minerva-math
python contamination/check_contamination.py \
  --reference-type "hf" \
  --hf-dataset "svc-huggingface/minerva-math" \
  --hf-split "test" \
  --question-key "problem" \
  --answer-key "solution" \
  --input-jsonl "/groups/gag51395/datasets/raw/pretrain/swallow-math/train-00001-of-00002.jsonl,/groups/gag51395/datasets/raw/pretrain/swallow-math/train-00002-of-00002.jsonl" \
  --num-processes 384 \
  --jaccard-threshold 0.8 \
  --output "/groups/gag51395/fujii/src/lm-evaluation-harness/contamination/results/swallow-math-minerva-math.jsonl"
