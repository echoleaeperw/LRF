import os
import sys
import json
import torch
import textwrap
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

from longterm.core.llm_factory import BaseAgent
from langchain_core.messages import SystemMessage, HumanMessage


class LLMGenerateDescription(BaseAgent):

    def __init__(self,
                 temperature: float = 0.2,
                 provider: Optional[str] = "deepseek",
                 verbose: bool = False):

        super().__init__(temperature=temperature, provider=provider, verbose=verbose)
        print(f"LLMGenerateDescription initialized with provider: {provider}")

    def llmdescriptiongenerate(self,
                                      scenario_data: Dict) -> str:
        """
        使用LLM将一个详细的、结构化的场景数据字典（通常来自ScenarioExtractor）
        转换成一个单一、连贯的自然语言叙事段落。

        参数:
            scenario_data: 包含完整场景信息的字典，结构应与
                           ScenarioExtractor.extract_carla_scenario 的输出一致。

        返回:
            一个单一、连贯的段落，描述整个场景。
        """
        if self.llm is None:
            raise RuntimeError("LLM尚未初始化，请检查您的配置。")

        # 1. 将完整的JSON数据转换为格式化的字符串以便放入Prompt
        # 我们假设 scenario_data 本身就是JSON格式的字典
        structured_input = json.dumps(scenario_data, indent=2, ensure_ascii=False)

        # 2. 构建Prompt
        system_message_content = textwrap.dedent("""\
         你是一位世界级的自动驾驶场景分析师和交通故事讲述者。你的任务是将一份详细的、结构化的JSON格式的场景数据，转换成一个单一、连贯且生动的自然语言叙事段落。

         【输入数据结构解析】
         你将收到的JSON数据（在"场景描述"字段中）包含以下关键信息：
         - `map`: 场景发生的地图名称。
         - `dt`: 轨迹数据点之间的时间步长（秒）。
         - `vehicles`: 一个包含场景中所有车辆的列表。每个车辆对象都包含：
           - `id`: 车辆的唯一标识。
           - `is_ego`: 布尔值，标识是否为主角车辆（自车）。
           - `type`: 车辆类型，如 "car", "truck", "ego_vehicle"。
           - `length`, `width`: 车辆的物理尺寸。
           - `trajectory`: 一个描述该车辆完整运动轨迹的时间序列列表。每个时间点包含 `t` (时间, t=0是当前时刻), `x`, `y` (世界坐标), `heading` (朝向角度), 和 `velocity` (速度 m/s)。t<0是过去，t>=0是未来预测。

         【你的任务和生成要求】
         1.  **设定场景**: 首先，根据`map`信息和车辆的总体布局，用一两句话描述故事发生的宏观环境。例如：“在一个晴朗的午后，一辆特斯拉Model 3（自车）正行驶在多车道的城市快速路'Town04'上。”
         2.  **聚焦主角**: 找到`is_ego: true`的车辆，它是故事的主角。描述它在t=0时刻的状态（位置、速度、行驶意图）。
         3.  **描绘关键配角**: 描述周围1-3个关键车辆在t=0时刻的状态，并特别强调它们与主角车辆的空间关系（例如，“在它的右前方，一辆卡车正同向行驶”，“一辆摩托车从后方快速接近”）。
         4.  **讲述动态故事**: 这是最关键的一步。分析`trajectory`数据（特别是t>=0的未来部分），不要仅仅罗列坐标或速度，而是要将这些数据翻译成驾驶行为和事件。
             - **识别行为**: 描述加速、减速、变道、转弯、跟车等行为。例如：“在接下来的几秒内，后方的卡车突然加速，意图从右侧强行超车。”
             - **营造冲突**: 描述车辆之间正在形成的、有潜在风险的交互。例如：“与此同时，自车前方的车辆开始减速，使得卡车的超车行为变得极其危险，形成了一个潜在的夹击情境。”
             - **使用生动的语言**: 使用“迅速地”、“突然”、“平稳地”、“危险地靠近”等词语来增强叙事的动态感。
         5.  **输出格式**: 最终输出必须是一个**单一、流畅、连贯的自然语言段落**，而不是分点列表。整个段落应该像一个专业的交通场景观察者在描述一个正在发生的交通事件。

         (1) 动态轨迹：车辆在时间上的位置、速度和航向，转换为自然语言表达。 (2) 静态地图环境：道路几何形状、车道标记、交通信号灯和定义环境的障碍物。 (3) 关系提示：车辆间距离、速度差异和潜在冲突区域。
        """)

        user_prompt_content = textwrap.dedent(f"""
        请将以下结构化的场景数据综合成一个流畅、描述性的单一段落。

        ### 结构化场景数据 ###
        ```json
        {structured_input}
        ```
        
        ### 生成的叙事性描述 ###
        """)
        
        # 3. 组装LangChain消息
        messages = [
            SystemMessage(content=system_message_content),
            HumanMessage(content=user_prompt_content)
        ]

        # 4. 调用LLM并获取响应
        response = self.llm.invoke(messages)
        generated_text = response.content.strip()
        return generated_text
