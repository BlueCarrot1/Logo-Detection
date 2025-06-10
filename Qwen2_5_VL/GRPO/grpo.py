import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration
from scipy.optimize import linear_sum_assignment

from math_verify import parse, verify
from open_r1.trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

import json

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy'.
    """
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )

def extract_bbox(response):
    start_tag = "<answer>"
    end_tag = "</answer>"
    input_str = response
    if start_tag in input_str:
        start_idx = input_str.find(start_tag) + len(start_tag)
        end_idx = input_str.find(end_tag)
        if end_idx == -1:
            end_idx = len(input_str)
        content_str = input_str[start_idx:end_idx]
        if not content_str.endswith("]"):
            content_str = content_str.rsplit("},", 1)[0] + "}]"
        content_str_corrected = content_str.replace("'", '"')
        try:
            bbox_list = json.loads(content_str_corrected)
        except json.JSONDecodeError:
            bbox_list = None
    else:
        bbox_list = None
    return bbox_list

def calculate_iou(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    xi1 = max(x1, x1_2)
    yi1 = max(y1, y1_2)
    xi2 = min(x2, x2_2)
    yi2 = min(y2, y2_2)
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    intersection_area = (xi2 - xi1) * (yi2 - yi1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - intersection_area
    iou = intersection_area / union_area
    return iou

def nms(boxes, iou_threshold):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = np.argsort(areas)[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    return boxes[keep].tolist()

def hungarian_matching(true_boxes, pred_boxes, iou_threshold):
    if len(true_boxes) == 0 or len(pred_boxes) == 0:
        return [], list(range(len(true_boxes))), list(range(len(pred_boxes)))
    cost_matrix = np.zeros((len(true_boxes), len(pred_boxes)))
    for i, true_box in enumerate(true_boxes):
        for j, pred_box in enumerate(pred_boxes):
            iou = calculate_iou(true_box, pred_box)
            cost_matrix[i, j] = -iou
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = []
    for r, c in zip(row_ind, col_ind):
        iou = -cost_matrix[r, c]
        if iou >= iou_threshold:
            matches.append((r, c, iou))
    unmatched_true = [i for i in range(len(true_boxes)) if i not in row_ind]
    unmatched_pred = [j for j in range(len(pred_boxes)) if j not in col_ind]
    return matches, unmatched_true, unmatched_pred

def reward_calculate(true_boxes_list, pred_boxes_list, iou_threshold=0.5):
    """
    true_boxes_list: 真实box列表 [[x1,y1,x2,y2], ...]
    pred_boxes_list: 预测box列表 [[x1,y1,x2,y2], ...]
    param iou_threshold: IoU阈值
    return: 标量奖励值
    """
    true_boxes_list = nms(true_boxes_list, iou_threshold=0.9)
    matches, unmatched_true, unmatched_pred = hungarian_matching(true_boxes_list, pred_boxes_list, iou_threshold)
    num_true, num_pred, num_matches = len(true_boxes_list), len(pred_boxes_list), len(matches)
    mean_iou = np.mean([m[2] for m in matches]) if num_matches > 0 else 0
    recall = num_matches / num_true if num_true > 0 else 0
    precision = num_matches / num_pred if num_pred > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    final_reward = 0.6 * mean_iou + 0.4 * f1_score
    return final_reward

def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using the reward_calculate function."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        reward = 0.0
        try:
            ground_truth = sol.strip()
            ground_truth_bbox = extract_bbox(ground_truth)
            if ground_truth_bbox is None:
                raise ValueError("Failed to extract ground truth bbox")
            
            content_match = re.search(r'<answer>(.*?)</answer>', content)
            student_answer = content_match.group(1).strip() if content_match else content.strip()
            student_answer = '<answer>' + student_answer + '</answer>'
            student_answer = student_answer.replace("[[", '[').replace("]]", ']').replace("\n", '')
            student_answer_bbox = extract_bbox(student_answer)
            if student_answer_bbox is None:
                raise ValueError("Failed to extract student answer bbox")
            
            true_boxes = [bbox['Position'] for bbox in ground_truth_bbox]
            pred_boxes = [bbox['Position'] for bbox in student_answer_bbox]
            reward = reward_calculate(true_boxes, pred_boxes)
        except Exception as e:
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                with open(log_path, "a") as f:
                    f.write(f"------------- {current_time} Error in accuracy_reward: {str(e)} -------------\n")
                    f.write(f"content: {content}\n")
                    f.write(f"sol: {sol}\n")
        rewards.append(reward)
    return rewards

reward_funcs_registry = {
    "accuracy": accuracy_reward,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

def main(script_args, training_args, model_args):
    script_args.reward_funcs = ['accuracy']
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    from datasets import DatasetDict
    dataset = DatasetDict.load_from_disk(script_args.dataset_name)

    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    def make_conversation_image(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": example["problem"]},
                    ],
                },
            ],
        }

    if "image" in dataset[script_args.dataset_train_split].features:
        print("has image in dataset")
        dataset = dataset.map(make_conversation_image)
    else:
        print("no image in dataset")
        dataset = dataset.map(make_conversation)
        dataset = dataset.remove_columns("messages")

    trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainer
    print("using: ", trainer_cls)

    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
