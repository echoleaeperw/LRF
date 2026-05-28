import re
from typing import Dict, Any, Optional, List
import json


class ContentProcessor:

    @staticmethod
    def extract_behavior_key_info(behavior_content: str) -> Dict[str, Any]:
        """
        Extract key information from the AnalysisAgent's Markdown output
        Use pattern matching to ensure no key fields are lost
        """
        result = {
            "full_content": behavior_content,
            "matched_behavior_label": "",
            "matching_confidence": 0.0,
            "scenario_type": "",
            "key_interaction": {},
            "driver_agent_inputs": {},
            "reflection_agent_inputs": {}
        }

        behavior_match = re.search(r'Behavior label[：:]\s*([A-Za-z_]+)', behavior_content)
        if behavior_match:
            result["matched_behavior_label"] = behavior_match.group(1)
            
        confidence_match = re.search(r'Confidence[：:]\s*([0-9.]+)', behavior_content)
        if confidence_match:
            result["matching_confidence"] = float(confidence_match.group(1))
            
        scenario_match = re.search(r'Scenario type[：:]\s*([a-z_]+)', behavior_content)
        if scenario_match:
            result["scenario_type"] = scenario_match.group(1)
                
        metrics_match = re.search(r'Priority metrics[：:].*?[\[\(]([^\]\)]+)[\]\)]', behavior_content)
        if metrics_match:
                metrics_str = metrics_match.group(1)
                metrics_list = [m.strip().strip('"\'') for m in re.split(r'[,，]', metrics_str)]
                result["driver_agent_inputs"] = {"priority_metrics": metrics_list}

        
        return result
    
    @staticmethod
    def extract_risk_weights(reflection_content: str) -> Dict[str, Any]:
        result = {
            "full_content": reflection_content,
            "reasoning": "",
            "risk_weights": {}
        }
        
        reasoning_match = re.search(r'Reasoning[：:](.+?)(?=\n\n|\n#|\n\*|$)', reflection_content, re.DOTALL)
        if reasoning_match:
            result["reasoning"] = reasoning_match.group(1).strip()    
        weight_patterns = [
            r'- \*\*L_(\w+):\*\* ([0-9.]+)',
            r'"L_(\w+)":\s*([0-9.]+)',
            r'L_(\w+)[：:]\s*([0-9.]+)',
            r'\*\*L_(\w+)\*\*[：:]\s*([0-9.]+)',
            r'- L_(\w+)[：:]\s*([0-9.]+)',
        ]
        for pattern in weight_patterns:
                matches = re.findall(pattern, reflection_content)
                for weight_name, weight_value in matches:
                    result["risk_weights"][f"L_{weight_name}"] = float(weight_value)
        
        return result
    
    @staticmethod
    def clean_content(content: str) -> str:
        if not content:
            return ""
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        content = content.strip()
        return content
    
    @staticmethod
    def extract_section_content(content: str, section_name: str) -> str:
        pattern = rf'#{1,3}\s*{re.escape(section_name)}.*?\n(.*?)(?=\n#{1,3}\s|\Z)'
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        pattern2 = rf'{re.escape(section_name)}[：:].*?\n(.*?)(?=\n\n|\Z)'
        match2 = re.search(pattern2, content, re.DOTALL | re.IGNORECASE)
        if match2:
            return match2.group(1).strip()
        return ""
    
    @staticmethod
    def format_for_next_agent(content: str, agent_name: str) -> str:
        formatted = f"""
                From {agent_name}'s analysis result

                {ContentProcessor.clean_content(content)}

                -----------------------------   -----------------------------
                This is the complete analysis content from {agent_name}, please base your subsequent analysis on this information.
                -----------------------------   -----------------------------
            """
        return formatted.strip()


def extract_first_content_from_string(text: str) -> Optional[Dict]:
    if not text or not isinstance(text, str):
        return None
    cleaned_content = ContentProcessor.clean_content(text)
    if not cleaned_content:
        return None
    return {
        "full_content": cleaned_content,
        "content_type": "markdown",
        "success": True
    }