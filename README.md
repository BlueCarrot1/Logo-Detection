# Logo-Detection
use florence model for few shot special logo detection;  iou reward training with grpo for florence and VLMs 

# train frozen visual parameter with other full parameter

GPU: 31238MB

# train lora

## train with frozen visual encoder

```nohup python lora.py > demo_lora_epoch_50_no_frozen.out &```

lora setting: lora_rank = 4 , lora_alpha = 8, lora_dropout = 0.05 

trainable params: 3,334,500 || all params: 826,028,388 || trainable%: 0.4037

GPU: 16000MB

## train with non frozen visual encoder


# inference & evaluate

inference time:  4 seconds / 10 pictures

evaluate: evaluate.py

# RL training

- IOU reward: matched boxes's performance, using mIOU
- match reward: non match boxes 's performance, using F1 score

train with RL ...


