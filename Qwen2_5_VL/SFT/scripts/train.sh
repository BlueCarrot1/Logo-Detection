accelerate launch \
    --num_processes 1 \
    --main_process_port 25001 \
    --config_file /root/autodl-tmp/textvqa_grounding_task_qwen2.5-vl-ft-main/configs/deepspeed_bf16_zero2.yaml \
    /root/autodl-tmp/textvqa_grounding_task_qwen2.5-vl-ft-main/sft.py \
    --config /root/autodl-tmp/textvqa_grounding_task_qwen2.5-vl-ft-main/configs/SFT_v1.yaml
