import os
from collections import defaultdict
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches 
import numpy as np
from tqdm import tqdm
import requests
import torch
from scipy.optimize import linear_sum_assignment
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForCausalLM, AutoProcessor
from util import url_image


class FlorenceModel:
    # load model
    def __init__(
        self, model_id="/root/autodl-tmp/model/florence", device="cuda:0" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = (
            AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype="auto").eval().to(device)
        )
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    def run_example(self, image, task_prompt, text_input=None):
        if text_input is None:
            prompt = task_prompt
        else:
            prompt = task_prompt + text_input
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device, torch.float16)
        input_ids=inputs["input_ids"].to(self.device)
        pixel_values=inputs["pixel_values"].to(self.device)
        generated_ids = self.model.generate(
            input_ids=input_ids, # inputs["input_ids"].to(self.device),
            pixel_values=pixel_values, # inputs["pixel_values"].to(self.device),
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3,
            early_stopping=False,
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text, task=task_prompt, image_size=(image.width, image.height)
        )
        torch.cuda.empty_cache()
        return parsed_answer

    def logo_box(self, image, text):
        """
        image 示例: image = url_image(image_path).convert("RGB")
        """
        task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>" # 后续模型训练好之后，会改成ANKER
        ans = self.run_example(image, task_prompt, text)[task_prompt]
        return ans["bboxes"]


def calculate_iou(box1, box2):
    """
    计算两个bounding box之间的IoU
    :param box1: [x1, y1, x2, y2]
    :param box2: [x1, y1, x2, y2]
    :return: IoU
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area > 0 else 0
    return iou


def hungarian_matching(true_boxes, pred_boxes):
    """
    使用匈牙利算法进行box匹配
    :param true_boxes: 真实box列表 [[x1,y1,x2,y2], ...]
    :param pred_boxes: 预测box列表 [[x1,y1,x2,y2], ...]
    :return: 匹配列表 [(true_idx, pred_idx)], 未匹配的真实框, 未匹配的预测框
    """
    num_true = len(true_boxes)
    num_pred = len(pred_boxes)
    cost_matrix = np.zeros((num_true, num_pred)) # 构建代价矩阵 (1-IoU)
    for i in range(num_true):
        for j in range(num_pred):
            iou = calculate_iou(true_boxes[i], pred_boxes[j])
            cost_matrix[i, j] = 1 - iou  # 转换为最小化问题
    # 执行匈牙利算法
    true_indices, pred_indices = linear_sum_assignment(cost_matrix)
    matches = []
    for i in range(len(true_indices)):
        true_idx = true_indices[i]
        pred_idx = pred_indices[i]
        if calculate_iou(true_boxes[true_idx], pred_boxes[pred_idx]) > 0:
            matches.append((true_idx, pred_idx, calculate_iou(true_boxes[true_idx], pred_boxes[pred_idx])))
    matched_true = set([m[0] for m in matches])
    matched_pred = set([m[1] for m in matches])
    unmatched_true = [i for i in range(num_true) if i not in matched_true]
    unmatched_pred = [i for i in range(num_pred) if i not in matched_pred]
    return matches, unmatched_true, unmatched_pred


def evaluate_matching(true_boxes_list, pred_boxes_list, iou_threshold=0.5):
    """
    评估匹配结果并计算各项指标
    :param true_boxes: 真实box列表 [[x1,y1,x2,y2], ...]
    :param pred_boxes: 预测box列表 [[x1,y1,x2,y2], ...]
    :param iou_threshold: IoU阈值
    :return: 评估结果字典
    """
    # hungarian_matching
    matches, unmatched_true, unmatched_pred = hungarian_matching(true_boxes_list, pred_boxes_list, iou_threshold)

    num_true, num_pred = len(true_boxes), len(pred_boxes)
    num_matches = len(matches)
    
    # mIoU: only calculate for matched box pairs
    # mean_iou = np.mean([calculate_iou(true_boxes[m[0]], pred_boxes[m[1]]) for m in matches]) if num_matches > 0 else 0 
    mean_iou = np.mean([m[2] for m in matches]) if num_matches > 0 else 0
    # 召回率 = TP / (TP + FN) = 匹配的真实框数 / 总真实框数
    recall = num_matches / num_true if num_true > 0 else 0
    # 查准率 = TP / (TP + FP) = 匹配的预测框数 / 总预测框数
    precision = num_matches / num_pred if num_pred > 0 else 0
    # F1分数
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    results = {
        'num_true_boxes': num_true,
        'num_pred_boxes': num_pred,
        'num_matches': num_matches,
        'mean_iou': mean_iou,
        'recall': recall,
        'precision': precision,
        'f1_score': f1_score,
        'unmatched_true': unmatched_true,
        'unmatched_pred': unmatched_pred,
        'matches': matches  # 每个匹配项包含 (true_idx, pred_idx, iou)
    }
    return results


