set -x 

read -r -d '' training_commands <<EOF
../train_dpo.py \
     --save_path ./ckpt/7b_llama \
     --save_steps -1 \
     --logging_steps 1 \
     --eval_steps -1 \
     --train_batch_size 128 \
     --micro_train_batch_size 1 \
     --pretrain /storage5/hf_private/model/hatakeyama-llm-team/Tanuki-8B-Instruct-without-DPO \
     --bf16 \
     --max_epochs 1 \
     --max_len 2048 \
     --zero_stage 3 \
     --beta 0.1 \
     --learning_rate 5e-7 \
     --dataset Anthropic/hh-rlhf,tasksource/oasst1_pairwise_rlhf_reward,openai/webgpt_comparisons \
     --dataset_probs 0.72,0.08,0.2 \
     --flash_attn \
     --gradient_checkpointing \
     --ref_offload
     --lora_rank 8
     --lora_alpha 16
     --lora_dropout 0.05
     --use_wandb true
     --wandb_org weblab-geniac1
     --wandb_project OpenRLHF-test
     --wandb_group test
     --wandb_run_name test
EOF
     # --wandb [WANDB_TOKENS] or True (use wandb login command)
     # --ipo [for IPO]
     # --label_smoothing 0.1 [for cDPO]


if [[ ${1} != "slurm" ]]; then
    deepspeed $training_commands
fi
