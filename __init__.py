"""STRIVE Project - Simulation and Testing of Rare Events in Interactions for Validation and Evaluation

CVPR 2022 论文项目实现，用于自动驾驶场景的对抗性生成和测试。

Main Modules:
    - src: 核心源代码（模型、数据集、工具等）
    - longterm: 长期行为分析和风险评估模块
"""

# 这里不直接导入子模块，避免在导入时就加载所有内容
# 用户可以按需导入：
# from STRIVE.src.models import ...
# from STRIVE.longterm.agents import ...

__version__ = "1.0.0"
__author__ = "STRIVE Team"

__all__ = [
    "src",
    "longterm",
    "__version__",
    "__author__"
]
