# __init__.py

# 导入日志相关模块
from .logger import logger

# 控制 from log import * 导入的内容
__all__ = ['logger']
