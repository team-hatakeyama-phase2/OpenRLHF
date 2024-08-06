#!/bin/bash
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --nodelist slurm0-a3-ghpc-[15]
#SBATCH --job-name=single-rlhf        # name

#SBATCH --cpus-per-task=8            # number of cores per tasks
#SBATCH --gres=gpu:8                 # number of gpus
#SBATCH --mem-per-gpu=128G                 # number of gpus
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
CONDA_ENV_NAME="dev23-openrlhf"
# YOUR_HF_LOGIN_TOKEN="xxxxxxxx"
# huggingface-cli login

source ~/.bashrc && wandb login $WANDB_API_KEY && huggingface-cli login  --token $HF_TOKEN 

echo "starting-env.."  
source ~/miniconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV_NAME 
echo "set-env:"$(which python)
# export GPUS_PER_NODE=8
# export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
# export MASTER_PORT=9901

export OPENRLHF_LOAD_MODEL_NAME_OR_PATH="/storage5/someya/outputs/sftlab-experiments/8B/someya-sft_011-zero1"
export OPENRLHF_SAVE_MODEL_PATH="/storage5/saito/ckpt_openrlhf"
export OPENRLHF_DATA_LIST="team-hatakeyama-phase2/synth-dpo-calm22b-250k"

# NOTE: eval_steps -1 means one evaluation per epoch

srun bash -c 'python ../train_dpo.py \
     --save_path  $OPENRLHF_SAVE_MODEL_PATH \
     --ckpt_path  $OPENRLHF_SAVE_MODEL_PATH/checkpoints_dpo \
     --save_steps 50 \
     --logging_steps 1 \
     --eval_steps -1 \
     --micro_train_batch_size 1 \
     --train_batch_size 512 \
     --micro_train_batch_size 1 \
     --pretrain $OPENRLHF_LOAD_MODEL_NAME_OR_PATH \
     --bf16 \
     --max_epochs 1 \
     --max_len 2048 \
     --zero_stage 1 \
     --beta 0.1 \
     --learning_rate 5e-7 \
     --dataset $OPENRLHF_DATA_LIST \
     --dataset_probs 1.0 \
     --flash_attn \
     --gradient_checkpointing \
     --ref_offload \
     --chosen_key chosen --rejected_key rejected \
     --apply_chat_template
     '
