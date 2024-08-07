#!/bin/bash
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --nodelist slurm0-a3-ghpc-[5,6,7,8,9,10,11,12]

#SBATCH --job-name=MnMoeDpo        # name
#SBATCH --nodes=4                    # nodes
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=8            # number of cores per tasks
#SBATCH --gres=gpu:8                 # number of gpus
#SBATCH --mem-per-gpu=180G                 # number of gpus
#SBATCH --time 24:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name

CONDA_ENV_NAME="dev23-openrlhf"
# YOUR_HF_LOGIN_TOKEN="xxxxxxxx"
# huggingface-cli login

echo "starting-env.."  
source ~/miniconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV_NAME 
echo "set-env:"$(which python)
export GPUS_PER_NODE=8
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9901

#export OPENRLHF_LOAD_MODEL_NAME_OR_PATH="/storage5/shared/Nishijima/Llama-3-8b-MoE/5th_tonyu_iter_7200"
# "/storage5/someya/outputs/sftlab-experiments/8B_iter126000/someya-exp007-zero1/checkpoint-2000"
#export OPENRLHF_SAVE_MODEL_PATH="/storage5/saito/ckpt_moe_openrlhf"
#export OPENRLHF_DATA_LIST="team-hatakeyama-phase2/applied-chat-templace-dpo-120k"
# "team-hatakeyama-phase2/synth-dpo-calm22b-250k"

export OPENRLHF_LOAD_MODEL_NAME_OR_PATH="/storage5/someya/outputs/sftlab-experiments/8x8B/someya-sft_013-zero3_multi_node_no_offload"
# "/storage5/shared/Nishijima/Llama-3-8b-MoE/9th_tonyu_iter_0001200"
# "/storage5/someya/outputs/sftlab-experiments/8x8B/someya-sft_004-zero3_multi_node"
# "/storage5/shared/Nishijima/Llama-3-8b-MoE/5th_tonyu_iter_7200"
# "/storage5/someya/outputs/sftlab-experiments/8B/someya-sft_011-zero1"
export OPENRLHF_SAVE_MODEL_PATH="/storage5/saito/04_4node_ckpt_moe_openrlhf"
export OPENRLHF_DATA_LIST="team-hatakeyama-phase2/dpo-oai-format-mixed-95k"
# "team-hatakeyama-phase2/dpo-oai-format-mixed-224k"
# "team-hataKeyama-phase2/dpo-nemotron-math-coding-reasoning-14k,team-hatakeyama-phase2/dpo-multiturn-rand-genre-60k-len-80,team-hatakeyama-phase2/dpo-singleturn-rand-genre-140k"
# ",team-hatakeyama-phase2/dpo-singleturn-rand-genre-140k"
export OPENRLHF_DATA_PROB="1.0"
# "team-hatakeyama-phase2/applied-chat-templace-dpo-small"
# "team-hatakeyama-phase2/synth-dpo-calm22b-250k"


# NOTE: eval_steps -1 means one evaluation per epoch

srun --jobid $SLURM_JOBID bash -c 'python -m torch.distributed.run \
 --nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
 --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
 ../train_dpo.py \
     --save_path  $OPENRLHF_SAVE_MODEL_PATH \
     --ckpt_path  $OPENRLHF_SAVE_MODEL_PATH/checkpoints_dpo \
     --save_steps 50 \
     --logging_steps 1 \
     --eval_steps 50 \
     --micro_train_batch_size 1 \
     --train_batch_size 1024 \
     --pretrain $OPENRLHF_LOAD_MODEL_NAME_OR_PATH \
     --bf16 \
     --max_epochs 1 \
     --max_len 2048 \
     --zero_stage 3 \
     --beta 0.1 \
     --learning_rate 1e-6 \
     --dataset $OPENRLHF_DATA_LIST \
     --dataset_probs $OPENRLHF_DATA_PROB \
     --flash_attn \
     --gradient_checkpointing \
     --adam_offload \
     --chosen_key chosen --rejected_key rejected \
     --apply_chat_template
     '

# sleep 60
