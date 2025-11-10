"""
提供统一的日志记录接口，适配项目中使用的Logger类
"""
from utils.logger import Logger

# 重新导出Logger类，保持接口一致性
__all__ = ['Logger'] 