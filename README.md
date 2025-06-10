# Logo-Detection
use florence model for few shot special logo detection;  iou reward training with grpo for VLM: Qwen2.5 VL 

# Train
## Florence SFT
train lora
```nohup python lora.py > demo_lora_epoch_50_no_frozen.out &```

## Qwen2.5 VL SFT + GRPO

for sft training:
```bash Qwen2.5VL/SFT/scripts/train.sh```

for GRPO training:
```bash Qwen2.5VL/GRPO/grpo.py```
reward design
- IOU reward: matched boxes's performance, using mIOU
- match reward: non match boxes 's performance, using F1 score

# results


