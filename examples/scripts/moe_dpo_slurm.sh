#!/bin/bash
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --nodelist slurm0-a3-ghpc-[2,5-6,8-12]
#SBATCH --nodes=8                    # nodes

#SBATCH --job-name=MnDpo        # name
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
export TRITON_CACHE_DIR="/storage5/.triton"

export OPENRLHF_WANDB_NAME="weblab-geniac1"
export OPENRLHF_WANDB_RUN="exp_"$(date '+%Y%m%d%H%S')
export OPENRLHF_WANDB_PROJECT="openrlhf_train_dpo"

# export OPENRLHF_LOAD_MODEL_NAME_OR_PATH="/storage5/someya/outputs/sftlab-experiments/8x8B/someya-sft_019-zero3_multi_node_no_offload/checkpoint-215"
export OPENRLHF_LOAD_MODEL_NAME_OR_PATH="/storage5/someya/outputs/sftlab-experiments/8x8B/someya-sft_022-zero3_multi_node_no_offload/checkpoint-136"
export OPENRLHF_TOKENIZER_REPO="team-hatakeyama-phase2/tanuki-tokenizer-without-space"
export OPENRLHF_SAVE_MODEL_PATH="/storage5/saito/01_08nodes_ckpt_moe_openrlhf/"$OPENRLHF_WANDB_RUN
export OPENRLHF_DATA_LIST="team-hatakeyama-phase2/dpo-oai-format-satoshi-ayaja-randgenre-80k"
export OPENRLHF_DATA_PROB="1.0"

mkdir -p $OPENRLHF_SAVE_MODEL_PATH
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
     --micro_train_batch_size 4\
     --train_batch_size 1024 \
     --pretrain $OPENRLHF_LOAD_MODEL_NAME_OR_PATH \
     --tokenizer $OPENRLHF_TOKENIZER_REPO\
     --bf16 \
     --max_epochs 1 \
     --max_len 2048 \
     --zero_stage 3 \
     --beta 0.5 \
     --label_smoothing 0.01 \
     --learning_rate 1e-6 \
     --dataset $OPENRLHF_DATA_LIST \
     --dataset_probs $OPENRLHF_DATA_PROB \
     --flash_attn \
     --gradient_checkpointing \
     --adam_offload \
     --chosen_key chosen --rejected_key rejected \
     --apply_chat_template \
     --use_wandb True \
     --wandb_org $OPENRLHF_WANDB_NAME \
     --wandb_project $OPENRLHF_WANDB_PROJECT \
     --wandb_run_name $OPENRLHF_WANDB_RUN 
     '

# sleep 60
