# STRIVE Long-term Module

## 项目结构

已完成 Python 包结构重构，所有目录都包含 `__init__.py` 文件：

```
STRIVE/
├── __init__.py                 # 项目根包
├── setup_env.sh               # 环境设置脚本
└── STRIVE/
    ├── __init__.py            # 主包
    ├── longterm/              # 长期分析模块
    │   ├── __init__.py        # ✅ 新增
    │   ├── agents/            # 智能代理
    │   │   ├── __init__.py    
    │   │   ├── analysis.py    # 分析代理
    │   │   ├── driver.py      # 驾驶代理
    │   │   ├── reflection.py  # 反思代理
    │   │   └── flow.py        # 流程控制
    │   ├── core/              # 核心工具
    │   │   └── __init__.py    
    │   ├── utils/             # 工具函数
    │   │   └── __init__.py    # ✅ 新增
    │   ├── corpus/            # 语料库
    │   │   └── __init__.py    # ✅ 新增
    │   ├── knowledge/         # 知识库
    │   │   └── __init__.py    # ✅ 新增
    │   └── config/            # 配置文件
    │       └── __init__.py    # ✅ 新增
    └── src/                   # 源代码
        └── llm/               # LLM 相关
            └── scenario_extractor.py
```

## 使用方法

### 方法 1：设置环境变量（推荐）

```bash
# 在项目根目录执行
source setup_env.sh

# 或手动设置
export PYTHONPATH=/home/wuyou/STRIVE:$PYTHONPATH
```

### 方法 2：在 IDE 中配置

在 VSCode/Cursor 的 `.env` 文件中添加：
```
PYTHONPATH=/home/wuyou/STRIVE
```

### 方法 3：在 Python 代码中导入

```python
# 现在可以直接导入，无需 sys.path 操作
from STRIVE.longterm.agents import AnalysisAgent, DriverAgent
from STRIVE.src.llm.scenario_extractor import ScenarioExtractor
```

## 已完成的修改

1. ✅ 为所有目录添加了 `__init__.py` 文件
2. ✅ 移除了所有文件中的 `sys.path.insert()` 代码：
   - `agents/analysis.py`
   - `agents/driver.py`
   - `agents/reflection.py`
   - `agents/flow.py`
   - `utils/behaviorcot.py`
3. ✅ 创建了环境设置脚本 `setup_env.sh`
4. ✅ 项目现在符合标准 Python 包结构

## 注意事项

- 运行代码前请先设置 PYTHONPATH 环境变量
- 所有导入语句保持不变，无需修改
- 项目结构更加规范，便于后续打包和发布
