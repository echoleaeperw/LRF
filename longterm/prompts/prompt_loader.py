"""
Prompt 文件加载器。

所有 Agent 的 prompt 模板存放在本目录下的 .md 文件中，
通过此模块统一加载、缓存和渲染（变量替换）。

使用方式：
    from longterm.prompts.prompt_loader import PromptLoader
    system = PromptLoader.render("analysis_system", risk_level="high_risk", ...)
    human  = PromptLoader.render("analysis_human",  scenario_json=..., ...)
"""

import os
import re
from functools import lru_cache
from typing import Dict

_PROMPTS_DIR = os.path.dirname(os.path.abspath(__file__))


@lru_cache(maxsize=None)
def _load_raw(name: str) -> str:
    """从文件读取原始模板字符串（带缓存）。"""
    path = os.path.join(_PROMPTS_DIR, f"{name}.md")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt template not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class PromptLoader:
    """Prompt 模板加载与渲染工具。"""

    @staticmethod
    def load(name: str) -> str:
        """返回未渲染的原始模板字符串。"""
        return _load_raw(name)

    @staticmethod
    def render(name: str, **kwargs) -> str:
        """
        加载模板并用 kwargs 替换 {variable} 占位符。

        模板中使用 Python str.format_map 语法：{variable_name}。
        若模板中存在字面花括号（如 JSON 示例），请用 {{ }} 转义。
        """
        template = _load_raw(name)
        if kwargs:
            # 用 format_map 而非 format，避免缺失 key 时报错
            template = template.format_map(_SafeDict(kwargs))
        return template

    @staticmethod
    def reload(name: str) -> str:
        """强制重新从磁盘读取（开发期使用）。"""
        _load_raw.cache_clear()
        return _load_raw(name)


class _SafeDict(dict):
    """format_map 时，未提供的 key 保留原始占位符而不报 KeyError。"""
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"
