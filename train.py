from transformers import AutoModelForCausalLM, AutoProcessor
import torch
from data import DetectionDataset
import os
from util import url_image
import json
from PIL import Image
from torch.utils.data import Dataset
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from transformers import get_scheduler
from torch.optim import AdamW
# import logging

from torch.utils.data import DataLoader


if __name__ == "__main__":
    batch_size = 6
    num_workers = 0
    epochs = 7
    save_path = './output/v1_demo'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model_id = "/root/autodl-tmp/model/florence"
    # logger.info(f"load model from {model_id}")
    print(f"load model from {model_id}")
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype="auto").eval().to(device)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    for param in model.vision_tower.parameters():
        param.is_trainable = False
    
    # logger.info(f"load model finish!")
    # logger.info(f"load data from /root/autodl-tmp/Logo-Detection/data/v1_records.json")
    print(f"load model finish!")
    print(f"load data from /root/autodl-tmp/Logo-Detection/data/v1_records.json")

    
    train_dataset = DetectionDataset("/root/autodl-tmp/Logo-Detection/data/v1_records.json", "/root/autodl-tmp/Logo-Detection/data/v1_image")
    val_dataset = DetectionDataset("/root/autodl-tmp/Logo-Detection/data/v1_records.json", "/root/autodl-tmp/Logo-Detection/data/v1_image")
    # logger.info(f"load data finish!")
    print(f"load data finish!")
    print("data format: ", train_dataset[2])
   
    def collate_fn(batch):
        questions, answers, images = zip(*batch)
        inputs = processor(text=list(questions), images=list(images), return_tensors="pt", padding=True).to(device, torch.float16)
        return inputs, answers
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              collate_fn=collate_fn, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                              collate_fn=collate_fn, num_workers=num_workers)
    
    optimizer = AdamW(model.parameters(), lr=1e-6)
    num_training_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler(name="linear", optimizer=optimizer,
                                  num_warmup_steps=0, num_training_steps=num_training_steps,)
    # logger.info(f"start training: total epoch {epoch}, batch_size:{batch_size}, model save path:{save_path}")
    print(f"start training: total epoch {epochs}, batch_size:{batch_size}, model save path:{save_path}")
    for epoch in tqdm(range(epochs)):
        model.train()
        train_loss = 0
        i = -1
        for inputs, answers in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
            i += 1
            input_ids = inputs["input_ids"].to(device)
            pixel_values = inputs["pixel_values"].to(device)
            labels = processor.tokenizer(text=answers, return_tensors="pt", padding=True, return_token_type_ids=False).input_ids.to(device)
            outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        print(f"Average Training Loss: {avg_train_loss}")
    
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}"):
                inputs, answers = batch
                input_ids = inputs["input_ids"].to(device)
                pixel_values = inputs["pixel_values"].to(device)
                labels = processor.tokenizer(text=answers, return_tensors="pt", padding=True, return_token_type_ids=False).input_ids.to(device)
                outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                val_loss += loss.item()
            print(val_loss / len(val_loader))
        output_dir = f"{save_path}/epoch_{epoch + 1}"
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir,use_safetensors=False)
        
    # logger.info(f"train finish!")
    print(f"train finish")
    
    

