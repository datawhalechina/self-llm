"""
核心类型定义
"""

from enum import Enum


class Framework(str, Enum):
    """推理框架枚举"""
    MLX = "mlx"
    TRANSFORMERS = "transformers"
