import random
import numpy as np
import torch
import os
from typing import Optional, Union
from dataclasses import dataclass, field

import datasets
import accelerate
from accelerate import Accelerator

def set_seeds(seed: int, deterministic: bool = False) -> int:
    """
    智能设置随机种子：
    - 如果通过 `accelerate launch` 启动，使用 `accelerate.utils.set_seed`
    - 否则使用常规 PyTorch 种子设置

    Args:
        seed: 基础随机种子
        deterministic: 是否启用完全确定性模式（可能降低性能）

    Returns:
        实际使用的种子（多进程时会自动调整）
    """
    # 检查是否通过 accelerate launch 启动
    if hasattr(Accelerator, "_shared_state") and Accelerator._shared_state:
        accelerate.utils.set_seed(seed)  # Accelerate 会自动处理多进程种子
        if deterministic:
            torch.backends.cudnn.deterministic = True  # 如果使用会降低性能
            torch.backends.cudnn.benchmark = False
        return seed

    # 常规单机种子设置
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # GPU适配
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
    return seed