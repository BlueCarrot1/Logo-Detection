import json
import os
from PIL import Image
from torch.utils.data import Dataset
from typing import List, Dict, Any, Tuple
from util import url_image

SCALED_IMAGE_SIZE = 1000

def convert_box_format(img, box_list):
    '''box coordinate [x1,y1,x2,y2] scaled：x1 / width * Scaled image size '''
    width, height = img.size
    refined_box_list = []
    for box in box_list:
        x1, y1, x2, y2 = box
        refined_box_list.append([int(x1/width * SCALED_IMAGE_SIZE) , int(y1/height * SCALED_IMAGE_SIZE), 
                                 int(x2/width * SCALED_IMAGE_SIZE), int(y2/height * SCALED_IMAGE_SIZE)])
    return refined_box_list

def add_prefix(text = "logo", task = "<CAPTION_TO_PHRASE_GROUNDING>"):
    return task + text

def trans_answer(box_list, text = "logo"):
    text_format = text
    for box in box_list:
        text_format += "".join([f"<loc_{str(c)}>" for c in box])
    # text_format += "".join([f"<loc_{str(c)}>" for c in box_list])
    return text_format

class JSONLDataset:
    def __init__(self, jsonl_file_path: str, image_directory_path: str):
        self.jsonl_file_path = jsonl_file_path
        self.image_directory_path = image_directory_path
        self.entries = self._load_entries()

    def _load_entries(self) -> List[Dict[str, Any]]:
        # entries: box list
        entries = []
        with open(self.jsonl_file_path, 'r') as file:
            entries = json.load(file)
        return entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, Dict[str, Any]]:
        if idx < 0 or idx >= len(self.entries):
            raise IndexError("Index out of range")
        entry = self.entries[idx]
        image_path = os.path.join(self.image_directory_path, entry['image'])
        try:
            # image = url_image(image_path)
            image = url_image(image_path).convert("L").convert("RGB")
            return (image, entry)
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file {image_path} not found.")

class DetectionDataset(Dataset):
    def __init__(self, jsonl_file_path: str, image_directory_path: str, text: str, task: str):
        self.dataset = JSONLDataset(jsonl_file_path, image_directory_path)
        self.text = text
        self.task = task

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, info = self.dataset[idx]
        box_list = convert_box_format(image, info["bounding boxes"])
        query, answer = add_prefix(self.text,self.task), trans_answer(box_list)
        return query, answer, image

class Mix_DetectionDataset(Dataset):
    def __init__(self, jsonl_file_path1: str, image_directory_path1: str, text1: str, task1: str,
                 jsonl_file_path2: str, image_directory_path2: str, text2: str, task2: str):
        self.dataset1 = JSONLDataset(jsonl_file_path1, image_directory_path1)
        self.dataset2 = JSONLDataset(jsonl_file_path2, image_directory_path2)
        self.text1 = text1
        self.task1 = task1
        self.text2 = text2
        self.task2 = task2

    def __len__(self) -> int:
        return len(self.dataset1) + len(self.dataset2)

    def __getitem__(self, idx: int) -> Tuple[str, str, Image.Image]:
        if idx < len(self.dataset1):
            image, info = self.dataset1[idx]
            box_list = convert_box_format(image, info["bounding boxes"])
            query, answer = add_prefix(self.text1, self.task1), trans_answer(box_list)
        else:
            idx -= len(self.dataset1)
            image, info = self.dataset2[idx]
            box_list = convert_box_format(image, info["bounding boxes"])
            query, answer = add_prefix(self.text2, self.task2), trans_answer(box_list)
        return query, answer, image

