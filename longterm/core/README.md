# 场景描述文件保存和读取机制

## 概述

为了确保场景描述的生成和读取使用相同的文件，我们实现了一个简单可靠的文件机制。该机制使用固定的文件名，确保每次生成和读取都操作同一个文件。

## 机制设计

### 文件结构
```
scenario_descriptions/
├── current_scenario_description.txt    # 当前场景描述（固定文件名）
└── scenario_metadata.json             # 元数据（时间戳、文件信息等）
```

### 核心特性
1. **固定文件名**：总是使用 `current_scenario_description.txt`，确保读写一致性
2. **自动覆盖**：每次保存都会覆盖之前的内容，保持文件唯一性
3. **元数据跟踪**：记录最后更新时间、文件大小等信息
4. **错误处理**：完整的异常处理和日志记录

## 使用方法

### 1. 生成并保存场景描述（在主系统中）

在 `src/llm/scenario_extractor.py` 中，`extract_scenario_description` 方法会自动保存：

```python
# 在主系统中（如weight_manager.py）
scenario_description = self.scenario_extractor.extract_scenario_description(
    scene_graph, map_env, map_idx, future_pred,
    auto_save=True  # 默认为True，自动保存
)
```

### 2. 手动保存场景描述

```python
from src.llm.scenario_extractor import ScenarioExtractor

extractor = ScenarioExtractor()
extractor.save_scenario_description(description_text)
```

### 3. 读取场景描述（在AnalysisAgent中）

```python
from src.llm.scenario_extractor import ScenarioExtractor

# 静态方法调用，无需创建实例
scenario_description = ScenarioExtractor.load_scenario_description()

if scenario_description:
    # 成功读取，使用场景描述
    result = behavior_agent.few_shot_decision(scenario_description)
else:
    # 读取失败，使用备用方案
    print("警告: 无法读取场景描述，使用默认值")
```

## 集成示例

### 在 weight_manager.py 中生成
```python
# 当权重管理器提取场景描述时，自动保存到文件
scenario_description = self.scenario_extractor.extract_scenario_description(
    scene_graph, map_env, map_idx, future_pred
)
# 文件会自动保存到 scenario_descriptions/current_scenario_description.txt
```

### 在 2_behavior_cot.py 中读取
```python
# AnalysisAgent从文件读取场景描述
scenario_description = ScenarioExtractor.load_scenario_description()

if not scenario_description:
    # 如果读取失败，使用示例描述作为后备
    scenario_description = "示例场景描述..."

# 使用场景描述进行分析
result, query = behavior_agent.few_shot_decision(scenario_description)
```

## API 文档

### ScenarioExtractor.save_scenario_description()

```python
def save_scenario_description(self, description: str, save_dir: str = "scenario_descriptions") -> str:
    """
    保存场景描述到固定文件
    
    参数:
        description: 场景描述文本
        save_dir: 保存目录（默认: "scenario_descriptions"）
        
    返回:
        保存的文件路径，失败时返回空字符串
    """
```

### ScenarioExtractor.load_scenario_description()

```python
@staticmethod
def load_scenario_description(save_dir: str = "scenario_descriptions") -> str:
    """
    从固定文件加载场景描述
    
    参数:
        save_dir: 保存目录（默认: "scenario_descriptions"）
        
    返回:
        场景描述文本，失败时返回空字符串
    """
```

## 文件同步保证

### 写入时机
- 主系统调用 `extract_scenario_description()` 时自动保存
- 手动调用 `save_scenario_description()` 时立即保存

### 读取时机
- AnalysisAgent 启动时从文件读取
- 任何模块都可以通过静态方法读取

### 一致性保证
1. **固定文件名**：始终使用 `current_scenario_description.txt`
2. **原子操作**：写入操作是原子的，避免读取不完整数据
3. **错误处理**：读取失败时提供清晰的错误信息

## 测试验证

运行测试脚本验证机制：

```bash
cd STRIVE/longterm/utils
python test_scenario_file_flow.py
```

测试内容包括：
1. 场景描述的保存和读取
2. 多次保存的覆盖行为
3. 元数据文件的生成
4. 跨模块使用的可行性

## 故障排除

### 问题：读取到空字符串
**原因**：文件不存在或读取权限问题
**解决**：检查 `scenario_descriptions/current_scenario_description.txt` 是否存在

### 问题：保存失败
**原因**：目录权限或磁盘空间问题
**解决**：确保有写入权限，检查磁盘空间

### 问题：内容不同步
**原因**：多进程同时写入
**解决**：确保写入操作的顺序性，避免并发写入

## 未来扩展

1. **版本控制**：保留历史版本的场景描述
2. **压缩存储**：对大型场景描述进行压缩
3. **网络同步**：支持分布式环境下的文件同步
4. **缓存机制**：提高频繁读取的性能

## 总结

这个机制确保了：
- ✅ 每次生成和读取使用相同的文件
- ✅ 简单可靠的文件操作
- ✅ 完整的错误处理
- ✅ 跨模块的一致性访问
- ✅ 自动化的保存流程 