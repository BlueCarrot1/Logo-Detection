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
    def __init__(self, jsonl_file_path: str, image_directory_path: str):
        self.dataset = JSONLDataset(jsonl_file_path, image_directory_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, info = self.dataset[idx]
        box_list = convert_box_format(image, info["bounding box"])
        query, answer = add_prefix(), trans_answer(box_list)
        return query, answer, image

