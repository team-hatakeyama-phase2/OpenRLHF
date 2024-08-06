#!/bin/bash
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --nodelist slurm0-a3-ghpc-[5,14]

#SBATCH --job-name=mn-rlhf        # name
#SBATCH --nodes=2                    # nodes
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=8            # number of cores per tasks
#SBATCH --gres=gpu:8                 # number of gpus
#SBATCH --mem-per-gpu=128G                 # number of gpus
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
CONDA_ENV_NAME="dev23-openrlhf"
# YOUR_HF_LOGIN_TOKEN="xxxxxxxx"
# huggingface-cli login

echo "starting-env.."  
source ~/miniconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV_NAME 
echo "set-env:"$(which python)
export GPUS_PER_NODE=2
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9901

export OPENRLHF_LOAD_MODEL_NAME_OR_PATH="/storage5/someya/outputs/sftlab-experiments/8B_iter126000/someya-exp007-zero1/checkpoint-2000"
export OPENRLHF_LOAD_TOKENIZER_REPO="team-hatakeyama-phase2/tanuki-tokenizer-without-space"
export OPENRLHF_SAVE_MODEL_PATH="/storage5/saito/dev-ckpt_openrlhf"
export OPENRLHF_DATA_LIST="team-hatakeyama-phase2/mt-dpo-small"

# NOTE: eval_steps -1 means one evaluation per epoch

srun --jobid $SLURM_JOBID bash -c 'python -m torch.distributed.run \
 --nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
 --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
 ../train_dpo.py \
     --save_path  $OPENRLHF_SAVE_MODEL_PATH \
     --ckpt_path  $OPENRLHF_SAVE_MODEL_PATH/checkpoints_dpo \
     --save_steps 2 \
     --logging_steps 1 \
     --eval_steps -1 \
     --micro_train_batch_size 1 \
     --train_batch_size 128 \
     --micro_train_batch_size 1 \
     --pretrain $OPENRLHF_LOAD_MODEL_NAME_OR_PATH \
     --tokenizer $OPENRLHF_LOAD_TOKENIZER_REPO \
     --bf16 \
     --max_epochs 1 \
     --max_len 2048 \
     --zero_stage 2 \
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
