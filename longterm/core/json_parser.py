import json
import re
from typing import Dict, Any, Optional
from rich import print


class RobustJSONParser:
    @staticmethod
    def extract_json_from_response(text: str) -> Optional[Dict[str, Any]]:
        
        if not text or not isinstance(text, str):
            return None
        cleaned_text = RobustJSONParser._preprocess_llm_response(text)
        json_result = RobustJSONParser._extract_from_code_block(cleaned_text)
        if json_result:
            return json_result
        json_result = RobustJSONParser._extract_bare_json(cleaned_text)
        if json_result:
            return json_result
        json_result = RobustJSONParser._fix_incomplete_json(cleaned_text)
        if json_result:
            return json_result
        json_result = RobustJSONParser._reconstruct_from_fields(cleaned_text)
        if json_result:
            return json_result
        return None
    
    @staticmethod
    def _preprocess_llm_response(text: str) -> str:
        text = text.strip()
        if text.startswith('```json') and text.endswith('```'):
            return text
        if text.startswith('json\n{') or text.startswith('json\n '):
            return text
        if text.startswith('{') and text.endswith('}'):
            try:
                json.loads(text)
                return text
            except json.JSONDecodeError:
                pass
        fixes = [
            (r'^[^{]*?("[\w_]+"\s*:\s*\{)', r'{\1'),
            (r'\}\s*[^}]+$', r'}'),
        ]
        
        for pattern, replacement in fixes:
            if re.search(pattern, text, flags=re.MULTILINE | re.DOTALL):
                text = re.sub(pattern, replacement, text, flags=re.MULTILINE | re.DOTALL)
        return text
    
    @staticmethod
    def _extract_from_code_block(text: str) -> Optional[Dict[str, Any]]:
        patterns = [
            r"```json\s*(\{.*?\})\s*```",
            r"```\s*(\{.*?\})\s*```",
            r"`(\{.*?\})`",
            r"json\s*\n\s*(\{.*?\})",
            r"json\s*(\{.*?\})",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                json_string = match.group(1).strip()
                return json.loads(json_string)

        return None
    
    @staticmethod
    def _extract_bare_json(text: str) -> Optional[Dict[str, Any]]:
            if text.strip().startswith('json'):
                json_prefix_end = text.find('{')
                if json_prefix_end != -1:
                    text = text[json_prefix_end:]
            
            json_start = text.find('{')
            json_end = text.rfind('}') + 1
            
            if json_start != -1 and json_end != 0 and json_end > json_start:
                json_string = text[json_start:json_end]
                return json.loads(json_string)
            
            if '": {' in text and '}' in text:
                quote_start = text.find('": {')
                if quote_start > 0:
                    field_start = text.rfind('"', 0, quote_start)
                    if field_start != -1:
                        json_end = text.rfind('}') + 1
                        json_string = '{' + text[field_start:json_end]
                        return json.loads(json_string)
            
            lines = text.split('\n')
            json_lines = []
            in_json = False
            brace_count = 0
            
            for line in lines:
                stripped = line.strip()
                if not in_json and (stripped.startswith('{') or stripped.startswith('"')):
                    in_json = True
                    if not stripped.startswith('{'):
                        json_lines.append('{')
                
                if in_json:
                    json_lines.append(line)
                    brace_count += line.count('{') - line.count('}')
                    
                    if brace_count <= 0 and '}' in line:
                        break
            
            if json_lines:
                json_string = '\n'.join(json_lines)
                return json.loads(json_string)
                    
            return None
    
    @staticmethod
    def _fix_incomplete_json(text: str) -> Optional[Dict[str, Any]]:
        patterns = [
            r'"reasoning":\s*"([^"]*)".*?"risk_weights":\s*(\{[^}]*\})',
            r'"risk_weights":\s*(\{[^}]*\})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                if len(match.groups()) == 2:
                    reasoning = match.group(1).strip()
                    risk_weights = json.loads(match.group(2))
                    return {
                        "reasoning": reasoning,
                        "risk_weights": risk_weights
                    }
                elif len(match.groups()) == 1:
                    risk_weights = json.loads(match.group(1))
                    return {
                        "reasoning": "Reconstructed JSON",
                        "risk_weights": risk_weights
                    }
        return None
    
    @staticmethod
    def _reconstruct_from_fields(text: str) -> Optional[Dict[str, Any]]:
        risk_weights = {}
        reasoning = ""
        
        reasoning_match = re.search(r'"reasoning":\s*"([^"]*)"', text, re.DOTALL)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
        
        weight_patterns = [
            r'"(L_\w+)":\s*([0-9]+(?:\.[0-9]+)?)',
            r'"(\w+Crash)":\s*([0-9]+(?:\.[0-9]+)?)',
            r'"(\w+Collision)":\s*([0-9]+(?:\.[0-9]+)?)',
        ]
        
        for pattern in weight_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                key, value = match
                risk_weights[key] = float(value)
        
        if risk_weights:
            return {
                "reasoning": reasoning or "Reconstructed from fields",
                "risk_weights": risk_weights
            }
        
        return None
    
    @staticmethod
    def validate_and_normalize_weights(weights_dict: Dict[str, Any]) -> Dict[str, Any]:

        if not isinstance(weights_dict, dict):
            return {"error": "Weights are not a dictionary"}
        
        if "risk_weights" not in weights_dict:
            if any(key.startswith("L_") for key in weights_dict.keys()):
                weights_dict = {"risk_weights": weights_dict}
            else:
                weights_dict["risk_weights"] = {}
        
        risk_weights = weights_dict.get("risk_weights", {})
        normalized_weights = {}
        
        for key, value in risk_weights.items():
            try:
                if isinstance(value, (int, float)):
                    normalized_weights[key] = float(value)
                elif isinstance(value, str) and value.replace('.', '').replace('-', '').isdigit():
                    normalized_weights[key] = float(value)
                else:
                    print(f"Skipping invalid weight: {key}={value}")
            except (ValueError, TypeError):
                print(f"Weight conversion failed: {key}={value}")
                continue
        
        weights_dict["risk_weights"] = normalized_weights
        return weights_dict


