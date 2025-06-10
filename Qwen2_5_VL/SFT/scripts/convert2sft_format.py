"""
将数据集格式转换成多模态模型监督微调格式，格式如下所示，保存文件格式为jsonl格式：
{
    "image": "demo/COCO_train2014_000000580957.jpg",
    "conversations": [
        {
            "from": "human",
            "value": "<image>\nLocate house in this image and output the bbox coordinates in JSON format."
        },
        {
            "from": "gpt",
            "value": "{\n"bbox_2d": [135, 114, 1016, 672]\n}"
        }
    ]
}
该格式是参考qwen2.5-vl-finetune文件中提到的Grounding Example所示。

原数据集格式为：
{
    'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1024x681 at 0x7FA58E1DB340>, 
    'question': 'what is the name of the company on the card?', 
    'answer': ['blink', 'intergrative nutrition', 'blink', 'blink', 'blink', 'blink', 'blink', 'blink', 'blink', 'blink'], 
    'dataset_id': '36269', 
    'bbox': [712.0, 255.0, 64.0, 43.0]
}

"""


import json
import os
from tqdm import tqdm
from datasets import load_dataset
import math

"""
Qwen2.5-VL使用绝对协调为调整大小的图像。对于原始绝对坐标，应该乘以调整大小的高度和宽度，然后除以其原始高度和宽度。
具体代码官网给了，链接：https://github.com/QwenLM/Qwen2.5-VL/blob/main/qwen-vl-finetune/tools/process_bbox.ipynb
可以参考官方的链接
"""

# This is the resize function of Qwen2.5-VL
def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280
):
    """Rescales the image so that the following conditions are met:
    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def convert_to_qwen25vl_format(bbox, orig_height, orig_width, factor=28, min_pixels=56*56, max_pixels=14*14*4*1280):
    new_height, new_width = smart_resize(orig_height, orig_width, factor, min_pixels, max_pixels)
    scale_w = new_width / orig_width
    scale_h = new_height / orig_height
    
    x1, y1, x2, y2 = bbox
    x1_new = round(x1 * scale_w)
    y1_new = round(y1 * scale_h)
    x2_new = round(x2 * scale_w)
    y2_new = round(y2 * scale_h)
    
    x1_new = max(0, min(x1_new, new_width - 1))
    y1_new = max(0, min(y1_new, new_height - 1))
    x2_new = max(0, min(x2_new, new_width - 1))
    y2_new = max(0, min(y2_new, new_height - 1))
    
    return [x1_new, y1_new, x2_new, y2_new]


def convert_to_sft_format(data_path,save_path,type='train'):
    # 加载数据集
    dataset = load_dataset(data_path,split='train')

    # 每个数据保存到一个jsonl文件中，并且图片的话要另外放到一起
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 创建 JSONL 文件
    jsonl_file = os.path.join(save_path, f"{type}.jsonl")
    with open(jsonl_file, 'w', encoding='utf-8') as jsonl_out:
        # 遍历数据集并保存图片，其他的部分信息保存成jsonl文件
        for idx,sample in tqdm(enumerate(dataset),total=len(dataset)):
            if type == 'train':
                if idx >= 3000:  # 判断是否处理到3000条数据
                    break
            elif type == 'test':
                # 判断是否处理到3001到3100条数据
                if idx < 3000 or idx >= 3100:
                    continue
            # 保存图片
            image = sample['image']
            # 生成文件名（格式为 000001.jpg, 000002.jpg 等）
            filename = f"{idx + 1:06d}.jpg"  # 使用 6 位数字格式化文件名
            jpg_path = os.path.join(save_path, type)
            if not os.path.exists(jpg_path):
                os.makedirs(jpg_path)
            output_path = os.path.join(jpg_path, filename)
            # 保存图片
            image.save(output_path)

            # 保存其他信息
            # 坐标信息
            old_bbox = sample['bbox']
            #### 这里需要将坐标转换成Qwen2.5-VL的坐标格式   
            image_width, image_height = image.size
            x1, y1, w, h = old_bbox
            new_bboxes = [x1, y1, x1 + w, y1 + h]
            # 转换坐标
            qwen25_bboxes = convert_to_qwen25vl_format(new_bboxes, image_height, image_width)
            bbox_dict = {"bbox_2d": qwen25_bboxes}
            formatted_json = json.dumps(bbox_dict, indent=None)
            data = {
                "image":[output_path],
                "query":sample['question'],
                "response":formatted_json,
            }

            # 将数据写入 JSONL 文件
            # 将每条数据写入 JSONL 文件
            jsonl_out.write(json.dumps(data, ensure_ascii=False) + '\n')

    print(f"All images and data have been saved to {save_path} and {jsonl_file}")

# 示例调用
convert_to_sft_format(data_path='/home/lixinyu/data/textvqa_bbox', save_path='./data', type='test')