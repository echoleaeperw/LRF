import os
import json
import time
import textwrap
import re
from typing import Dict, Any, Optional

from rich import print

# 使用兼容的 LangChain 和核心库导入
from langchain_core.messages import HumanMessage, SystemMessage
from longterm.core.llm_factory import BaseAgent



class ReflectionAgent(BaseAgent):
    """
    反思前两个Agent的分析结果，并生成最终的损失函数权重。
    能够处理仅有行为分析的降级情况。
    """

    def __init__(
        self, temperature: float = 0, verbose: bool = False, provider: Optional[str] = None
    ) -> None:
        super().__init__(temperature=temperature, verbose=verbose, provider=provider)

    def _extract_json_from_response(self, text: str) -> Optional[Dict[str, Any]]:
        """
        使用统一的强大JSON解析器
        """
        from longterm.utils.json_parser import RobustJSONParser
        result = RobustJSONParser.extract_json_from_response(text)
        if result:
            return RobustJSONParser.validate_and_normalize_weights(result)
        return None

    def reflect_and_generate_weights(
        self, behavior_analysis: Dict[str, Any], risk_metrics: Optional[Dict[str, Any]] = None, risk_level: str = "high_risk"
    ) -> Dict[str, Any]:
        """
        基于行为分析和（可选的）风险指标，反思并生成损失函数权重。

        参数:
            behavior_analysis: AnalysisAgent生成的场景行为分析报告。
            risk_metrics: (可选) DriverAgent生成的风险指标计算结果。
            risk_level: 目标风险等级 ("low_risk", "high_risk", "longtail_condition")

        返回:
            一个包含最终权重和分析理由的字典。
        """
        
        # 根据是否有risk_metrics来动态生成提示
        if risk_metrics:
            # --- 标准模式：综合分析 ---
            system_message_content = textwrap.dedent(f"""\
            你是**对抗性场景生成流水线**中的第三环节专家，定位为**"最终决策者"**。你的核心任务是综合所有输入信息，生成一个**动态自适应**的、能够达成**{risk_level}**风险等级目标的最终权重配置。

            **CRITICAL OUTPUT FORMAT**: 你必须严格按照JSON格式输出，不要使用Markdown格式。输出必须是有效的JSON对象。

            【核心使命】
            - **主要目标**: 基于AnalysisAgent的策略分析和DriverAgent的指标计算结果，动态生成能达成**{risk_level}**风险等级目标的最终权重配置
            - **目标风险等级**: {risk_level}（注意：不一定是碰撞，而是根据风险等级调整场景的激进程度和风险特征）
            - **核心原则**: **"基于量化反馈的动态调节"**。你必须根据DriverAgent计算出的具体风险指标值，来动态调整权重的激进程度
            - **输出要求**: 生成一份格式正确的JSON权重配置

            【风险等级说明】
            - **high_risk**: 高风险场景，可能包括碰撞、接近碰撞、激进驾驶行为等。权重应该高度激进
            - **low_risk**: 低风险场景，强调安全驾驶，避免危险行为。权重应该保守，强调安全约束
            - **longtail_condition**: 长尾场景，罕见但合理的交通情况。权重应该平衡，既要有特殊性又要符合物理和交通常识

            【工作流程：基于量化反馈和风险等级的动态权重决策】

            **第一步：确立基准 (Establish Baseline)**
            1.  **识别风险类型**: 从AnalysisAgent的`generated_strategy.collision_type_primary`中提取风险类型
            2.  **理解目标风险等级**: 当前目标是{risk_level}，这决定了权重的整体激进程度
            3.  **设定基准值**: 根据风险类型和{risk_level}，为核心权重选择初始基准值
                - **high_risk场景模板**:
                  * 追尾碰撞: `L_AdversarialCrash: 50.0`, `L_TTC: 20.0`, `L_DeltaV: 12.0`
                  * 侧向切入: `L_AdversarialCrash: 55.0`, `L_MinDist_lat: 30.0`, `L_YawRate: 15.0`, `L_DeltaV: 10.0`
                  * 路口冲突: `L_AdversarialCrash: 60.0`, `L_MinDist_lat: 25.0`, `L_TTC: 18.0`
                - **low_risk场景模板**:
                  * 所有攻击性权重应该降低70-80%，强调约束权重（L_MotionBehavior, L_PathAdherence等）
                - **longtail_condition场景模板**:
                  * 攻击性权重降低40-50%，平衡约束和特殊性

            **第二步：基于指标值动态调节 (Dynamic Adjustment)**
            这是最关键的一步。你必须根据DriverAgent计算出的**核心指标的当前值**，应用**反比调节原则**来调整基准权重。
            *   **原则**: 对于high_risk场景，初始风险越大（指标值越危险），达成目标所需的额外"推力"（权重）就越小；初始风险越小（指标值越安全），就越需要巨大的"推力"来达成目标
            
            **具体调节规则（仅适用于high_risk场景）:**
            *   **对于 `MinDist_lat` (横向距离)**:
                *   IF `value` < 1.0m (极度危险): 最终权重 = 基准权重 * 1.0 (保持)
                *   IF 1.0m <= `value` < 2.5m (中度危险): 最终权重 = 基准权重 * 1.5 (增强)
                *   IF `value` >= 2.5m (低度危险): 最终权重 = 基准权重 * 2.5 (极度增强)
            *   **对于 `TTC_lead` (碰撞时间)**:
                *   IF `value` < 1.5s (极度危险): 最终权重 = 基准权重 * 1.0 (保持)
                *   IF 1.5s <= `value` < 3.0s (中度危险): 最终权重 = 基准权重 * 1.6 (增强)
                *   IF `value` >= 3.0s (低度危险): 最终权重 = 基准权重 * 2.2 (极度增强)
            *   **对于 `DeltaV` / `RelativeSpeed_lead` (相对速度)**:
                *   *此项为正比调节*：相对速度越大，风险越高，权重也应越高
                *   IF `value` > 10 m/s (高速接近): 最终权重 = 基准权重 * 1.5 (鼓励)
                *   IF 5 m/s < `value` <= 10 m/s (中速接近): 最终权重 = 基准权重 * 1.2 (轻微鼓励)
                *   IF `value` <= 5 m/s (低速接近): 最终权重 = 基准权重 * 1.0 (保持)

            **第三步：整合与输出**
            1.  将经过动态调节后的核心权重填入最终配置
            2.  根据{risk_level}调整其他权重：
                - high_risk: 约束权重设置为较低值（1.0-8.0），正则化项极低（< 1.0）
                - low_risk: 约束权重设置为较高值（15.0-30.0），正则化项中等（2.0-5.0）
                - longtail_condition: 约束权重设置为中等值（8.0-15.0），正则化项较低（1.0-3.0）
            3.  输出最终的JSON格式权重配置

            【键名白名单（必须严格使用以下键名，未列出的一律不要输出）】
            risk_weight_keys = [
              "L_AdversarialCrash", "L_MinDist_lat", "L_TTC", "L_YawRate",
              "L_VehicleCollision", "L_VehicleCollision_Planner", "L_EnvironmentCollision",
              "L_THW", "L_DeltaV", "L_TLC", "L_PathAdherence",
              "L_MotionBehavior", "L_SceneSimilarity",
              "L_YawRate_Ego", "L_YawRate_NonEgo"
            ]

            【输出格式 - 必须是有效的JSON】
            ```json
            {{
              "reasoning": "基于AnalysisAgent识别的[风险类型]和目标风险等级{risk_level}，我选择了[模板名称]作为基准。根据DriverAgent计算出的核心指标：[列出关键指标及其值]，我进行了以下调整：[说明调整逻辑]。",
              "risk_weights": {{
                "L_AdversarialCrash": 65.0,
                "L_MinDist_lat": 45.0,
                "L_YawRate": 20.0,
                "L_TTC": 5.0,
                "L_DeltaV": 15.0,
                "L_EnvironmentCollision": 6.0,
                "L_VehicleCollision": 4.0,
                "L_VehicleCollision_Planner": 2.0,
                "L_MotionBehavior": 0.3,
                "L_SceneSimilarity": 0.2,
                "L_YawRate_Ego": 0.5,
                "L_YawRate_NonEgo": 1.0
              }}
            }}
            ```

            **重要提醒**：
            1. **严格按照上述JSON格式输出**，输出必须是有效的JSON对象
            2. 必须包含"reasoning"和"risk_weights"两个字段
            3. 只允许输出键名白名单中的权重；不要输出任何未列出的权重
            4. 权重值必须是数字类型（float），不要使用字符串
            5. 权重值必须根据{risk_level}风险等级进行调整，不要盲目使用示例中的数值
            """)
            
            risk_metrics_json_str = json.dumps(risk_metrics, indent=2, ensure_ascii=False)
            human_message_content = textwrap.dedent(f"""\
            请基于AnalysisAgent的结构化分析和DriverAgent的精确指标计算，生成最终的权重配置。

            【目标风险等级】
            {risk_level}

            【上游Agent分析结果】
            {behavior_analysis.get("full_content", str(behavior_analysis)) if isinstance(behavior_analysis, dict) else str(behavior_analysis)}

            【重点关注】
            请特别关注以下关键信息：
            1. AnalysisAgent的`generated_strategy.collision_type_primary`: 确定风险类型和基准模板
            2. AnalysisAgent的`agent_instructions.reflection_agent_inputs`: 提供权重配置指导
            3. DriverAgent计算的关键指标数值: 用于动态调整权重
            4. DriverAgent的优化建议: 提供权重调整的方向性指导

            【你的任务】
            1. 理解目标风险等级{risk_level}，选择合适的权重模板和调整策略
            2. 基于DriverAgent的指标计算结果，动态调整权重值
            3. 确保最终权重配置能够达成{risk_level}风险等级目标
            4. 严格按照JSON格式输出权重配置，必须包含"reasoning"和"risk_weights"字段

            请严格按照流程进行权重配置生成，输出有效的JSON格式，确保达成{risk_level}风险等级目标。
            """)
            
            system_message = SystemMessage(content=system_message_content)
            human_message = HumanMessage(content=human_message_content)
        else:
            # 没有risk_metrics时，程序停止
            stop_reason = "no_risk_metrics"
            print(f"[red]错误: 缺少risk_metrics数据，无法生成权重配置[/red]")
            return {"error": "missing_risk_metrics", "stop_reason": stop_reason}
        
        # --- 公共部分：LLM调用和解析 ---
        messages = [system_message, human_message]

        print("[cyan]正在进行最终权重生成...[/cyan]")
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

        print("\n[green]权重生成完成。[/green]")
        
        # 现在直接返回Markdown格式的原始响应内容
        return response_content


