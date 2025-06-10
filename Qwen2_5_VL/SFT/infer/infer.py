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
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from util import url_image
import ast
from qwen_vl_utils import process_vision_info
from transformers import modeling_utils
if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none","colwise",'rowwise']
device1 = "cuda:0" if torch.cuda.is_available() else "cpu"
import pandas as pd

class Qwen_2_5_Model:
    # load model
    def __init__(
        self, model_id="/root/autodl-tmp/model/qwen2.5-vl-3b", device="cuda:0" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = (    Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, 
                                                        device_map="cuda:0", torch_dtype=torch.bfloat16, 
                                                        trust_remote_code=True).to(device1).eval()
        )
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    def Qwen2_vl_predict(self, image_path, GROUNDING_PROMPT ):
        messages = [{"role": "user","content": [{"type": "image","image": image_path},
                {"type": "text", "text": GROUNDING_PROMPT},],}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device1)
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        try:
            boxes = ast.literal_eval(output_text)
            # Case 1: If it's a list of lists (already correct format)
            if isinstance(boxes, list) and boxes and isinstance(boxes[0], list):
                # Case 3: Check for [[]] and convert to []
                if boxes == [[]]:
                    boxes = []
            # Case 2: If it's a single list (only one bracket pair), wrap it
            elif isinstance(boxes, list):
                boxes = [boxes]
            else:
                boxes = []  # Not a list at all, treat as empty
        except Exception as e:
            print(f"解析 box_text 出错: {e}")
            print("output_text:")
            print(output_text)
            boxes = []
        return boxes


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

if __name__ == "__main__":
    evaluate_class = Qwen_2_5_Model()
    Logo_name = "复旦大学圆形校徽水印"
    Logo_path_name = "fudan_pic"

    GROUNDING_PROMPT = f"""请你检测图片当中纯蓝色的"{Logo_name}"文字Logo的目标，输出目标的像素位置坐标。
    你要输出bounding box，输出要求是:
    1、每个bounding box的格式是[x1,y1,x2,y2],其中(x1,y1)表示{Logo_name}标签的左上角像素坐标，(x2,y2)表示{Logo_name}标签的右下角像素坐标。表示的都是图片的绝对像素点位置。
    2、如果图片中有多个Logo，那么请以列表的形式输出，不要有任何多余的字符，即[[x1,y1,x2,y2],[x1',y1',x2',y2'], ...
    3、仅输出一行列表，不要输出其他格式例如json格式，不要输出任何冗余的信息，也不要输出重复的box。
    
    示例1：
    若图片中{Logo_name} Logo只有一个，且其Logo左上角像素坐标为（24，78），右下角像素坐标为（345，89）
    输出：
    [[24,78,345,89]]

    示例2：
    若图片中{Logo_name} Logo有三个，第一个Logo左上角像素坐标为（389，128），右下角像素坐标为（476，289），第二个Logo左上角像素坐标为（222，258），右下角像素坐标为（333，369）, 第三个Logo左上角像素坐标为（721，798），右下角像素坐标为（834，890）
    输出：
    [[389,128,476,289], [222,258,333,369], [721，798，834，890]]
    
    result = {}
    all_true_box=[]
    all_pred_box=[]
    ## 分别
    for logo_num in [1,3,5]:
        print("logo_num:", logo_num)
        
        json_file_path = f"data/task_1_3_5/generate_new/records/records_{Logo_path_name}_{logo_num}.json"
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
        # 处理每个条目
        iou =[]
        precision = []
        recall = []

        for item in data:
            true_boxes_list = []
            print("process",item["image"])
            jpg_name = item["image"]
            img_path = f"data/task_1_3_5/generate_new/{Logo_path_name}_{logo_num}/"+jpg_name
            for bbox in item['bounding boxes']:
                true_boxes_list.append(bbox)
            pred_boxes = evaluate_class.Qwen2_vl_predict(img_path,GROUNDING_PROMPT)
            if pred_boxes == [[]]:
                pred_boxes = []
            print("true boxes:")
            print(true_boxes_list)
            print("pred boxes:")
            print(pred_boxes)
            result_dict= evaluate_matching(true_boxes_list, pred_boxes)
            iou.append(result_dict["mean_iou"])
            precision.append(result_dict["precision"])
            recall.append(result_dict["recall"])

        

        result[str(logo_num)] = [np.mean(iou),np.mean(precision),np.mean(recall)]
        print(result[str(logo_num)])
    
    df= pd.DataFrame(result,index=["iou","precision","recall"])
    df["整体"] = df.mean(axis=1)
    df=df[["整体","1","3","5"]]
    df.round(4).to_csv(f"result_{Logo_name}.csv",encoding="gbk")



