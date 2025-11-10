import os
import textwrap
import time
import json
import re
from rich import print
from typing import List, Optional, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from src.llm.scenario_extractor import ScenarioExtractor
from longterm.core.llm_factory import BaseAgent

class DriverAgent(BaseAgent):
    """
    计算关键安全指标。
    """
    def __init__(
        self, temperature: float = 0, verbose: bool = False, provider: Optional[str] = None
    ) -> None:
        super().__init__(temperature=temperature, verbose=verbose, provider=provider)

    def _extract_json_from_response(self, text: str) -> Optional[Dict[str, Any]]:
        """
        使用统一的强大JSON解析器
        """
        from longterm.core.json_parser import RobustJSONParser
        result = RobustJSONParser.extract_json_from_response(text)
        if result:
            return RobustJSONParser.validate_and_normalize_weights(result)
        return None

    def analyze_scenario_and_calculate_metrics(
        self, scenario_description: str, behavior_analysis: Dict[str, Any], risk_level: str = "high_risk"
    ) -> Dict[str, Any]:
        """
        基于场景描述和行为分析，计算关键安全指标。
        
        参数:
            scenario_description: 完整的场景描述。
            behavior_analysis: 来自AnalysisAgent的宏观行为分析结果。
            risk_level: 目标风险等级 ("low_risk", "high_risk", "longtail_condition")
            
        返回:
            一个包含计算指标的字典。
        """
        system_message = textwrap.dedent(f"""\
        你是STRIVE系统中的**战术分析官**，负责执行对抗性场景生成流水线中的第二步：**精确量化指标计算**。你的任务是基于AnalysisAgent的策略分析结果，进行精确的安全指标计算和风险评估，为ReflectionAgent的权重配置提供量化依据。

        【核心使命】
        - **定位**: 对接AnalysisAgent的输出，执行实际的量化计算和风险评估
        - **主要目标**: 将AnalysisAgent的定性策略分析，转化为精确的量化指标数据，为ReflectionAgent提供决策依据
        - **目标风险等级**: 当前任务的目标是生成符合**{risk_level}**风险等级的场景。注意：这不是必须达成碰撞，而是需要根据风险等级调整场景的激进程度和风险特征
        - **核心原则**: **"精确计算"**。对AnalysisAgent指定的优先级指标进行准确计算，并基于计算结果评估场景的风险等级
        - **输出用途**: 为ReflectionAgent提供一份结构清晰、数据准确的指标计算报告，以及针对性的优化建议

        【风险等级说明】
        - **high_risk**: 高风险场景，可能包括碰撞、接近碰撞、激进驾驶行为等
        - **low_risk**: 低风险场景，强调安全驾驶，避免危险行为
        - **longtail_condition**: 长尾场景，罕见但合理的交通情况

        请根据{risk_level}风险等级，调整你的指标计算重点和风险评估标准。

        【关键安全指标列表及其物理意义】
        以下是常见的交通安全风险指标，你需要根据AnalysisAgent的指导计算其中的优先级指标：
        - **TTC_lead (前向碰撞时间)**: 反映追尾风险，计算公式：TTC = 纵向距离 / 相对速度，值越小风险越高
        - **THW_lead (前向车头时距)**: 反映跟车安全性，计算公式：THW = 纵向距离 / 自车速度，值越小风险越高
        - **MinDist_lat (最小横向距离)**: 反映侧向碰撞风险，表示车辆之间的最小横向间距，值越小风险越高
        - **YawRate (偏航率)**: 反映转向激进程度，表示车辆朝向变化率，值越大表示转向越激进
        - **DRAC (所需减速度)**: 反映紧急制动需求，表示避免碰撞所需的减速度，值越大表示情况越紧急
        - **TLC (穿道时间)**: 反映变道风险，表示车辆穿越车道边界所需的时间，值越小风险越高
        - **RelativeSpeed_lead (相对速度)**: 反映前车相对速度，正值表示接近，负值表示远离，绝对值越大变化越快

        【执行任务】

        **任务1: 从AnalysisAgent输出中提取关键信息**
        - 识别`agent_instructions.driver_agent_inputs.priority_metrics`中的优先级指标列表
        - 提取`generated_strategy.collision_type_primary`，确定风险类型（注意：根据{risk_level}风险等级，这可能不是碰撞，而是其他风险场景）
        - 从`indicator_guidance.critical_vehicle_pairs`中提取需要计算的车辆组合
        - 从`indicator_guidance.threshold_expectations`中获取各指标的危险阈值
        - 从`generated_strategy.matched_behavior_label`中理解核心对抗行为标签
        - 理解AnalysisAgent提供的目标风险等级：{risk_level}

        **任务2: 执行指标计算与风险评估**
        - 从场景描述中提取关键车辆的位置、速度、朝向等状态信息
        - 对`priority_metrics`中的每个指标，执行精确的物理计算：
          * TTC计算: 需要纵向距离和相对速度，考虑车辆几何尺寸
          * MinDist_lat计算: 需要横向距离，考虑车道宽度和车辆宽度
          * YawRate计算: 需要角度变化率，考虑转向速度
          * 其他指标: 根据公式和场景数据逐一计算
        - 记录完整的计算公式、输入值、中间结果和最终结果
        - 基于计算值和阈值预期，判断每个指标的风险等级（高/中/低）
        - 根据{risk_level}风险等级，评估当前指标值是否符合目标风险等级要求
        - 检查计算结果是否符合物理约束（如TTC不能为负，距离不能为负等）

        **注意**: 你的任务**仅限于指标计算和风险评估**，不需要制定权重配置方案。权重配置将由ReflectionAgent基于你的计算结果来完成。

        【输出格式要求】
        请严格按照以下结构化Markdown格式输出：

        # 风险指标计算报告

        ## 一、提取的关键信息（来自AnalysisAgent）
        - **目标风险等级**: {risk_level}
        - **风险类型**: [从`collision_type_primary`提取，注意这可能不是碰撞类型，而是其他风险场景类型]
        - **优先级指标**: [从`priority_metrics`提取，列出所有指标]
        - **关键车辆对**: [从`critical_vehicle_pairs`提取]
        - **阈值预期**: [从`threshold_expectations`提取，列出各指标的阈值]
        - **核心行为标签**: [从`matched_behavior_label`提取]

        ## 二、指标计算结果

        #### [指标名称1，如TTC_lead] (碰撞时间)
        - **计算值:** <float or null>
        - **计算过程:** 
          ```
          公式: [具体公式]
          输入值: [列出所有输入值]
          计算步骤: [详细步骤]
          结果: [最终结果]
          ```
        - **阈值对比:** 计算值 [比较符] 阈值预期 [具体数值]，表示...
        - **风险评估:** 基于该值，风险等级为**高/中/低**，因为...

        #### [指标名称2，如MinDist_lat] (横向最小距离)
        - **计算值:** <float or null>
        - **计算过程:** [同上格式]
        - **阈值对比:** [同上格式]
        - **风险评估:** [同上格式]

        #### [指标名称3，如YawRate] (偏航率)
        - **计算值:** <float or null>
        - **计算过程:** [同上格式]
        - **阈值对比:** [同上格式]
        - **风险评估:** [同上格式]

        [根据priority_metrics列表，继续添加其他指标的计算结果...]

        ### 指标计算结果总结
        - **高风险指标**: [列出所有高风险指标及原因]
        - **中等风险指标**: [列出所有中等风险指标及原因]
        - **低风险指标**: [列出所有低风险指标及原因]
        - **异常值检查**: [列出所有不符合物理约束的异常值]

        ## 三、风险评估总结

        ### 整体风险评估
        这是一个典型的**[风险类型]**场景，目标是生成符合**{risk_level}**风险等级的场景。基于指标计算结果：
        - **关键指标风险等级**: [列出关键指标的风险等级]
        - **当前整体风险等级**: **高/中/低**
        - **与目标风险等级的符合度**: [说明当前计算出的风险等级是否符合{risk_level}目标]

        ### 核心风险特征
        - **最危险的指标**: [列出1-2个最危险的指标及其值]
        - **最安全的指标**: [列出1-2个最安全的指标及其值]
        - **关键矛盾点**: [说明实现{risk_level}风险等级目标的主要障碍或机会]

        ### 为ReflectionAgent的建议
        - **需要重点关注的指标**: [列出ReflectionAgent在制定权重时应该重点关注的指标]
        - **建议的优化方向**: [基于当前指标值，说明应该如何调整场景以达成{risk_level}目标]
        """)
        
        # 处理behavior_analysis参数，提取实际内容
        if isinstance(behavior_analysis, dict) and "full_content" in behavior_analysis:
            behavior_content = behavior_analysis["full_content"]
        else:
            behavior_content = str(behavior_analysis)
        
        human_message = textwrap.dedent(f"""\
        请基于AnalysisAgent的策略分析结果，执行精确的量化指标计算和权重配置。

        【目标风险等级】
        当前任务的目标是生成符合**{risk_level}**风险等级的场景。请根据这个风险等级调整你的计算重点和权重配置建议。

        【AnalysisAgent完整分析结果（JSON格式）】
        {behavior_content}

        【完整场景描述（包含车辆轨迹、状态信息）】
        {scenario_description}

        【你的任务】
        1. **提取关键信息**: 从AnalysisAgent的JSON输出中提取：
           - `agent_instructions.driver_agent_inputs.priority_metrics`: 必须计算的优先级指标列表
           - `generated_strategy.collision_type_primary`: 核心风险类型（注意：根据{risk_level}，这可能不是碰撞类型）
           - `indicator_guidance.critical_vehicle_pairs`: 关键车辆对关系
           - `indicator_guidance.threshold_expectations`: 各指标的阈值预期
           - `generated_strategy.matched_behavior_label`: 核心行为标签
        2. **执行精确计算**: 对每个优先级指标，给出完整的计算公式、输入值、计算过程和结果
        3. **风险评估与总结**: 基于计算出的指标值，评估整体风险等级，并为ReflectionAgent提供优化建议
        4. **输出标准格式**: 确保最终的输出严格符合指定的Markdown格式

        【特别提醒】
        - 如果AnalysisAgent的输出是JSON格式，请直接解析其中的结构化信息
        - 如果AnalysisAgent的输出是文本格式，请尝试提取关键信息（风险类型、优先级指标等）
        - 对于无法计算的指标，请明确标注为null并说明原因
        - 所有计算过程必须可追溯，包括公式、输入值、中间结果和最终结果
        - 记住：目标是达成{risk_level}风险等级，不一定是碰撞
        - **不要制定权重配置方案**，这是ReflectionAgent的任务，你只需要提供计算结果和优化建议
        """)

        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=human_message)
        ]
        
        print("[cyan]正在进行风险指标计算...[/cyan]")
        response_content = ""
        try:
            for chunk in self.llm.stream(messages):
                content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                if isinstance(content, str):
                    response_content += content
                    print(content, end="", flush=True)
                elif isinstance(content, list):
                    # 处理列表类型的内容
                    for item in content:
                        if isinstance(item, str):
                            response_content += item
                            print(item, end="", flush=True)
                        elif isinstance(item, dict):
                            # 处理字典类型，提取文本内容
                            text_content = str(item.get('text', item.get('content', str(item))))
                            response_content += text_content
                            print(text_content, end="", flush=True)
        except Exception as e:
            print(f"[red]流式响应处理错误: {e}[/red]")
            # 尝试非流式调用
            try:
                response = self.llm.invoke(messages)
                response_content = response.content if hasattr(response, 'content') else str(response)
                if not isinstance(response_content, str):
                    response_content = str(response_content)
            except Exception as invoke_e:
                print(f"[red]LLM调用失败: {invoke_e}[/red]")
                return {"error": "Failed to get response from LLM", "details": str(invoke_e)}

        print("\n[green]风险指标计算完成。[/green]")
        
        # 返回原始的Markdown内容，不再尝试JSON解析
        return response_content
   