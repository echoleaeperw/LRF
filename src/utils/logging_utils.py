"""
Provide a unified logging interface for the Logger class used in the project
"""
from utils.logger import Logger

# Re-export the Logger class to maintain interface consistency
__all__ = ['Logger'] 