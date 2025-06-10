import os
from typing import Optional, List
from dataclasses import dataclass, field
from functools import partial
import torch
import transformers
from transformers import (
    AutoProcessor,
    TrainingArguments,
    Trainer,
)
from transformers.trainer_utils import get_last_checkpoint
import datasets
from trl import (
    TrlParser,
)
from peft import get_peft_model, LoraConfig

from utils import set_seeds
from vision_datacollator import vision_data_collator_map

import swanlab
swanlab.login(api_key="***", save=True) # use your api key
device_type = "gpu" if torch.cuda.is_available() else "cpu"
@dataclass
class ModelArguments:
    auto_model_class: Optional[str] = field(
        default="AutoModelForCausalLM",
        metadata={
            "help": (
                "The auto model class to use for the model. Default is AutoModelForCausalLM."
            )
        },
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models."
        },
    )
    processor_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to pretrained processor or processor identifier from huggingface.co/models."
        },
    )
    trust_remote_code: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether to trust the remote code when loading the model and processor. default is True."
        },
    )
    torch_dtype: Optional[str] = field(
        default="bfloat16",
        metadata={"help": "The torch dtype to use for the model. Default is bfloat16."},
    )

    def __post_init__(self):
        if self.processor_name_or_path is None:
            self.processor_name_or_path = self.model_name_or_path


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the train dataset to use (via the datasets library)."},
    )
    test_dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the test dataset to use (via the datasets library)."},
    )
    data_collator: Optional[str] = field(
        default="vision_data_collator",
        metadata={
            "help": (
                "The data collator to use for the dataset. Default is vision_data_collator."
            )
        },
    )
    max_seq_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_image_side: Optional[int] = field(
        default=9000,
        metadata={
            "help": ("The size of the image to use for the dataset. Default is 224.")
        },
    )


@dataclass
class LoraArguments:
    use_lora: bool = False
    r: int = 2
    lora_alpha: int = 32
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias = "none"
    task_type: str = "CAUSAL_LM"
    lora_dropout: float = 0.05
    inference_mode: bool = False

def main(data_args, training_args, model_args, lora_args):
    output_dir = training_args.output_dir
    dir_path, model_name = os.path.split(output_dir)
    new_model_name = device_type + "_" + model_name
    training_args.output_dir = os.path.join(dir_path, new_model_name)
    training_args.run_name = new_model_name
    set_seeds(training_args.seed)

    processor = AutoProcessor.from_pretrained(
        pretrained_model_name_or_path=model_args.processor_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        local_files_only=True,
    )
    model_class = getattr(transformers, model_args.auto_model_class)  # 动态加载模型类
    if model_class is None:
        raise ValueError(f"Model class {model_args.auto_model_class} is not available.")
    model = model_class.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        torch_dtype=getattr(torch, model_args.torch_dtype),
        trust_remote_code=model_args.trust_remote_code,
        local_files_only=True,
    )
    if lora_args.use_lora:
        lora_config = LoraConfig(
            r=lora_args.r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.bias,
        )
        model = get_peft_model(model, lora_config)

    train_dataset = datasets.load_dataset("json", data_files=data_args.train_dataset_name)
    test_dataset = datasets.load_dataset("json", data_files=data_args.test_dataset_name)
    raw_dataset = datasets.DatasetDict({
        "train": train_dataset["train"],
        "test": test_dataset["train"]
    })
    print(raw_dataset)
    def preporocess_textvqa(example):
        return {
            "image": example["image"],
            "user": example["query"],
            "assistant": example["response"],
        }

    raw_dataset = raw_dataset.map(
        preporocess_textvqa,
        remove_columns=raw_dataset["train"].column_names,
        desc="Preprocessing textvqa dataset",
    )
    data_collator = vision_data_collator_map[data_args.data_collator](
        processor=processor,
        max_seq_length=data_args.max_seq_length,
        max_img_side_length=data_args.max_image_side,
    )

    
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        print(
            f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
            "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
        )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=raw_dataset["train"],
        eval_dataset=(
            raw_dataset["test"] if training_args.eval_strategy != "no" else None
        ),
        data_collator=data_collator,
    )
    trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    dataclass_types = (
        DataTrainingArguments,
        TrainingArguments,
        ModelArguments,
        LoraArguments,
    )
    parser = TrlParser(dataclass_types)
    data_args, training_args, model_args, lora_args = parser.parse_args_and_config()
    main(data_args, training_args, model_args, lora_args)
