from ast import Break
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
from peft import PeftModel, PeftConfig
from util import url_image,plot_all_logo


class FlorenceModel:
    # load model
    def __init__(self, model_id="/root/autodl-tmp/model/florence", device="cuda:0" if torch.cuda.is_available() else "cpu"):
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
        #print(generated_text)
        #print(parsed_answer)
        return parsed_answer

    def logo_box(self, image,task_prompt, text):
        """
        image 示例: image = url_image(image_path).convert("RGB")
        """
        # task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>" # 后续模型训练好之后，会改成ANKER
        ans = self.run_example(image, task_prompt, text)[task_prompt]
        #print(ans)
        return ans["bboxes"]

class FlorenceModel2:
    # load model
    def __init__(self, model_id="/root/autodl-tmp/model/florence",lora_model_path=None, device="cuda:0" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = (
            AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype="auto").eval().to(device)
        )
        # 如果提供了LoRA路径，则加载LoRA权重
        if lora_model_path:
            self.model = PeftModel.from_pretrained(self.model, lora_model_path)
            print(f"已加载LoRA权重: {lora_model_path}")
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
        #print(generated_text)
        #print(parsed_answer)
        return parsed_answer

    def run_example_2(self, image, task_prompt, text_input=None):
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
        #print(generated_text)
        #print(parsed_answer)
        return parsed_answer,image.width, image.height

    def logo_box(self, image,task_prompt, text):
        """
        image 示例: image = url_image(image_path).convert("RGB")
        """
        # task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>" # 后续模型训练好之后，会改成ANKER
        ans = self.run_example(image, task_prompt, text)[task_prompt]
        #print(ans)
        return ans["bboxes"]
    
    def logo_box_2(self, image,task_prompt, text):
        ans,width,height = self.run_example_2(image, task_prompt, text)
        return ans,width,height


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


def nms(boxes, iou_threshold=0.9):
    """
    非极大值抑制（NMS）去重
    - boxes: [[x1,y1,x2,y2], ...]
    - iou_threshold: IoU threshold，超过该阈值的框会被认为是重复框。
    """
    if len(boxes) == 0:
        return [], []
    boxes = np.array(boxes)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    indices = np.argsort(areas)
    keep = []
    while len(indices) > 0:
        last = len(indices) - 1
        i = indices[last]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[indices[:last]])
        yy1 = np.maximum(y1[i], y1[indices[:last]])
        xx2 = np.minimum(x2[i], x2[indices[:last]])
        yy2 = np.minimum(y2[i], y2[indices[:last]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[indices[:last]] - inter)
        indices = np.delete(indices, np.concatenate(([last], np.where(iou > iou_threshold)[0])))
    new_boxes = boxes[keep].tolist()
    return new_boxes


def hungarian_matching(true_boxes, pred_boxes, iou_threshold):
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
        if calculate_iou(true_boxes[true_idx], pred_boxes[pred_idx]) > iou_threshold:
            matches.append((int(true_idx), int(pred_idx), calculate_iou(true_boxes[true_idx], pred_boxes[pred_idx])))
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
    # nms drop duplicates
    true_boxes_list = nms(true_boxes_list, iou_threshold=0.9)
    # hungarian_matching
    matches, unmatched_true, unmatched_pred = hungarian_matching(true_boxes_list, pred_boxes_list, iou_threshold)

    num_true, num_pred = len(true_boxes_list), len(pred_boxes_list)
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

def save_results(results, save_path):
    """保存结果为JSON文件"""
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)

def merge_json_files(folder_path):
    true_answers_dict = {}
    
    # 遍历文件夹中的所有文件
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        
        # 检查文件是否是 JSON 文件
        if os.path.isfile(file_path) and file_name.lower().endswith('.json'):
            with open(file_path, 'r') as f:
                data = json.load(f)
                # 将 JSON 文件内容合并到字典中
                for answer in data:
                    if 'image' in answer and 'bounding boxes' in answer:
                        true_answers_dict[answer['image']] = answer['bounding boxes']
    
    return true_answers_dict

def get_jpg_paths(root_folder):
    jpg_file_paths = []
    
    for folder_path, _, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith('.jpg'):
                full_path = os.path.join(folder_path, file)
                jpg_file_paths.append(full_path)
    
    return jpg_file_paths

def get_jpg_paths_from_list(folder_paths):
    jpg_file_paths = []
    for folder_path in folder_paths:
        if os.path.isdir(folder_path):  # 确保路径是一个文件夹
            for file in os.listdir(folder_path):
                full_path = os.path.join(folder_path, file)
                if os.path.isfile(full_path) and file.lower().endswith('.jpg'):
                    jpg_file_paths.append(full_path)
    return jpg_file_paths

import re

def parse_result_to_list(result_str):
    """从结果字符串中解析出坐标列表"""
    # 提取所有 <loc_XXX> 格式的数字
    loc_matches = re.findall(r'<loc_(\d+)>', result_str)
    # 将字符串数字转换为整数列表
    box_list = [int(loc) for loc in loc_matches]
    # 将列表分组为 [x1, y1, x2, y2] 的形式
    return [box_list[i:i+4] for i in range(0, len(box_list), 4)]

def inverse_convert_box_format(box_list, width, height, scaled_size):
    """将放缩后的坐标还原为原始图像的坐标"""
    original_box_list = []
    for box in box_list:
        x1, y1, x2, y2 = box
        original_box_list.append([
            int((x1 / scaled_size) * width),
            int((y1 / scaled_size) * height),
            int((x2 / scaled_size) * width),
            int((y2 / scaled_size) * height)
        ])
    return original_box_list

if __name__ == '__main__':

    scaled_size = 1000

    # 读模型文件
    model_id = './model/Florence-2-large-ft'
    epoch_path = './output_0601/huawei_huawei_logo_none_500_2/epoch_31'
    save_name = epoch_path.split('/')[-2]
    print(save_name)
    # model = FlorenceModel(model_id = model_id, device = 'cuda:0')
    model2 = FlorenceModel2(model_id=model_id, lora_model_path=epoch_path, device='cuda:0')
    
    # 读图片文件
    img_folder_dir = ['./huawei_and_mcdonald',]
    jpg_file_paths = get_jpg_paths_from_list(img_folder_dir)
    print(len(jpg_file_paths))

    # 读正确答案
    true_answers_floder = './test_data/0602test_records'
    true_answers_dict = merge_json_files(true_answers_floder)
    print(len(true_answers_dict))

    # 确保输出目录存在
    output_dir = f'./test_data/0602test_result/{save_name}_mix_after_SFT/'
    output_dir_2 = './test_data/0602test_result/'
    os.makedirs(output_dir, exist_ok=True)

    results = []
    # 遍历所有文件夹
    for image_path in jpg_file_paths:
        try:
            folder_path, file_name = os.path.split(image_path)
            #task_prompt = f"<CAPTION_TO_PHRASE_GROUNDING>"
            #text = f"logo"
            task_prompt = '<huawei_logo>'
            text = ''
            image = url_image(image_path)
            #r = model2.logo_box(image, task_prompt, text)
            r,width,height = model2.logo_box_2(image, task_prompt, text)
            res_str = str(r)
            res_list = parse_result_to_list(res_str)
            original_res_list = inverse_convert_box_format(res_list, width, height, scaled_size)

            pred_boxes_list = original_res_list
            save_path = os.path.join(output_dir, file_name)
            #print(save_path)
            plot_all_logo(image, pred_boxes_list, save_path)
            true_boxes_list = true_answers_dict.get(file_name, [])
            rr = evaluate_matching(true_boxes_list, pred_boxes_list, iou_threshold=0.5)
            results.append({
                "image": file_name,
                "bounding boxes": pred_boxes_list,
                "true_boxes":true_boxes_list,
                "evaluation": rr
            })
            #print(results)
        
        except Exception as e:
            print('fail:',str(e))        
    # 保存结果
    save_path = os.path.join(output_dir_2, f"{save_name}_mix_after_SFT.json")
    save_results(results, save_path)
    print(f"Results saved to {save_path}")