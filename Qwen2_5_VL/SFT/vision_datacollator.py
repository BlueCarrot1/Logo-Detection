from typing import Optional, Tuple
import copy

import transformers
import torch

from PIL import Image
import json

IGNORE_INDEX = -100

# 缩放图像的大小，同时因为grounding任务，需要同时缩放坐标
def resize_with_max_side(image, max_side_length):
    # 获取原始尺寸
    width, height = image.size
    # 计算缩放比例
    scale = min(max_side_length / width, max_side_length / height)
    # 计算新的尺寸
    new_width = int(width * scale)
    new_height = int(height * scale)
    # 调整图像大小
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return resized_image, scale

def resize_bbox(bbox, scale):
    # 缩放矩形框坐标
    return [int(coord * scale) for coord in bbox]


class Qwen2_5VLCollator:

    def __init__(
        self, processor, max_seq_length=1024, max_img_side_length=1024, **kwargs
    ):
        self.processor = processor
        # to fix bug in Qwen2.5VL
        self.processor.tokenizer.chat_template =  "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
        self.max_seq_length = max_seq_length
        self.max_img_side_length = max_img_side_length

    def __call__(self, examples):
        batch_input_ids = []
        for example in examples:
            # 根据数据集格式来，数据集格式如下：
            """
            {"image": ["./data/train/000001.jpg"], "query": "what is the name of the company on the card?", "response": "{\n  \"bbox_2d\": [\n    712.0,\n    255.0,\n    64.0,\n    43.0\n  ]\n}"}
            """
            question = example["user"]
            answer = example["assistant"]
            # 需要读取图像，需要确保是RGB图像
            image_path = example['image'][0]
            image = Image.open(image_path)
            # 输出缩放后的图像以及缩放倍率
            image, scale = resize_with_max_side(
                image, max_side_length=self.max_img_side_length
            )
            # 缩放answer的坐标值
            # answer是一个json字符串，解析成字典
            answer = json.loads(answer)
            # answer = {"bbox_2d": resize_bbox(answer["bbox_2d"],scale)}
            answer = {"bbox_2d": [resize_bbox(box,scale) for box in answer["bbox_2d"]]}
            # 转化新的answer
            answer = json.dumps(answer, indent=None)
            # 这了不知道是否需要添加prompt
            # prompt = "Please enclose the corresponding positions using coordinate boxes. Examples of coordinate value formats: [x1,y1,x2,y2]"
            # question = question+prompt
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question},
                    ],
                }
            ]
            prompt = self.processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            answer = f"{answer}<|im_end|>\n"
            input_ids = self.processor(
                images=[image],
                text=prompt + answer,
                return_tensors="pt",
                max_length=self.max_seq_length,
                truncation=False,
                padding=False,
            )
            answer_ids = self.processor.tokenizer(
                answer, add_special_tokens=False, return_tensors="pt"
            )
            ignore_ids_len = len(input_ids["input_ids"][0]) - len(
                answer_ids["input_ids"][0]
            )
            input_ids["labels"] = torch.cat(
                [
                    torch.tensor([IGNORE_INDEX] * ignore_ids_len).unsqueeze(0),
                    answer_ids["input_ids"],
                ],
                dim=1,
            )
            # position_ids
            position_ids, _ = self.get_rope_index_2(
                self.processor.image_processor.merge_size,
                input_ids["input_ids"],
                input_ids["image_grid_thw"],
            )
            input_ids["position_ids"] = position_ids

            # padding
            if len(input_ids["labels"]) < self.max_seq_length:
                input_ids["input_ids"] = torch.cat(
                    [
                        input_ids["input_ids"],
                        torch.tensor(
                            [self.processor.tokenizer.pad_token_id]
                            * (self.max_seq_length - len(input_ids["input_ids"]))
                        ).unsqueeze(0),
                    ],
                    dim=1,
                )
                input_ids["labels"] = torch.cat(
                    [
                        input_ids["labels"],
                        torch.tensor(
                            [IGNORE_INDEX]
                            * (self.max_seq_length - len(input_ids["labels"]))
                        ).unsqueeze(0),
                    ],
                    dim=1,
                )
                input_ids["attention_mask"] = input_ids["input_ids"].ne(
                    self.processor.tokenizer.pad_token_id
                )
                # padding position_ids
                pad_length = self.max_seq_length - input_ids["position_ids"].shape[2]
                input_ids["position_ids"] = torch.nn.functional.pad(
                    input_ids["position_ids"], (0, pad_length), "constant", 1
                )

            # truncate
            if len(input_ids["input_ids"][0]) > self.max_seq_length:
                input_ids["input_ids"] = input_ids["input_ids"][
                    :, : self.max_seq_length
                ]
                input_ids["labels"] = input_ids["labels"][:, : self.max_seq_length]
                input_ids["attention_mask"] = input_ids["attention_mask"][
                    :, : self.max_seq_length
                ]
                input_ids["position_ids"] = input_ids["position_ids"][
                    :, : self.max_seq_length
                ]
            # batching
            batch_input_ids.append(input_ids)

        batch_input_ids = {
            "input_ids": torch.cat(
                [input_ids["input_ids"] for input_ids in batch_input_ids], dim=0
            ),
            "attention_mask": torch.cat(
                [input_ids["attention_mask"] for input_ids in batch_input_ids], dim=0
            ),
            "labels": torch.cat(
                [input_ids["labels"] for input_ids in batch_input_ids], dim=0
            ),
            "pixel_values": torch.cat(
                [input_ids["pixel_values"] for input_ids in batch_input_ids], dim=0
            ),
            "image_grid_thw": torch.cat(
                [input_ids["image_grid_thw"] for input_ids in batch_input_ids], dim=0
            ),
            "position_ids": torch.cat(
                [input_ids["position_ids"] for input_ids in batch_input_ids], dim=1
            ),
        }
        return batch_input_ids

    def get_rope_index_2(
        self,
        spatial_merge_size: Optional[int] = 2,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with mordern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embeddin for text part.
            Examples:
                Assume we have a video input with 3 temporal patches, 2 height patches and 2 width patches.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [3, 4, 5, 6, 7]
                text height position_ids: [3, 4, 5, 6, 7]
                text width position_ids: [3, 4, 5, 6, 7]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        image_token_id = 151655
        video_token_id = 151656
        vision_start_token_id = 151652
        mrope_position_deltas = []
        if input_ids is not None and (
            image_grid_thw is not None or video_grid_thw is not None
        ):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(
                    input_ids == vision_start_token_id
                ).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    )

                    t_index = (
                        torch.arange(llm_grid_t)
                        .view(-1, 1)
                        .expand(-1, llm_grid_h * llm_grid_w)
                        .flatten()
                    )
                    h_index = (
                        torch.arange(llm_grid_h)
                        .view(1, -1, 1)
                        .expand(llm_grid_t, -1, llm_grid_w)
                        .flatten()
                    )
                    w_index = (
                        torch.arange(llm_grid_w)
                        .view(1, 1, -1)
                        .expand(llm_grid_t, llm_grid_h, -1)
                        .flatten()
                    )
                    llm_pos_ids_list.append(
                        torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                    )
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    )

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(
                    position_ids.device
                )
                mrope_position_deltas.append(
                    llm_positions.max() + 1 - len(total_input_ids[i])
                )
            mrope_position_deltas = torch.tensor(
                mrope_position_deltas, device=input_ids.device
            ).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = (
                    position_ids.unsqueeze(0)
                    .expand(3, -1, -1)
                    .to(attention_mask.device)
                )
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(
                    -1, keepdim=True
                )[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas
        

################
# Data collator map
################
vision_data_collator_map = {
    "Qwen2_5VLCollator": Qwen2_5VLCollator,
}