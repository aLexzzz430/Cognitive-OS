"""
core/ — AGI_V2 唯一主循环

根据 specs/main_loop.md 实现，包含 10 个步骤。
禁止在 core/ 外存在第二条主循环。
"""

from .cognition.ablation_config import CausalLayerAblationConfig

__all__ = [
    'CausalLayerAblationConfig',
]
