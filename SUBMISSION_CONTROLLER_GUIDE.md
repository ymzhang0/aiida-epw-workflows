# AiiDA Submission Controller 使用指南

## 概述

AiiDA Submission Controller 是一个用于管理大量AiiDA工作流提交的工具库。它提供了智能的批量提交、并发控制和状态监控功能。

## 核心概念

### 1. Controller 类型

#### BaseSubmissionController
- 基础抽象类
- 需要手动实现 `get_all_extras_to_submit()` 和 `get_inputs_and_processclass_from_extras()`
- 适用于从预定义数据源生成工作流

#### FromGroupSubmissionController
- 继承自 BaseSubmissionController
- 自动从父组中的节点生成工作流
- 使用节点的 extras 作为唯一标识符
- 适用于你的 SuperCon 项目

### 2. 关键组件

```python
# 核心属性
group_label: str          # 存储工作流节点的组
max_concurrent: int       # 最大并发工作流数
unique_extra_keys: tuple  # 唯一标识工作流的extras键
parent_group_label: str   # 父组标签（FromGroupSubmissionController）

# 核心方法
submit_new_batch()        # 提交新批次的工作流
get_all_extras_to_submit() # 获取所有待提交的extras
get_inputs_and_processclass_from_extras() # 根据extras生成工作流输入
```

## 你的 SuperCon Controller 实现

### 1. 基本结构

```python
from aiida_submission_controller import FromGroupSubmissionController
from aiida_supercon.workflows import EpwSuperConWorkChain

class SuperConWorkChainController(FromGroupSubmissionController):
    # 配置参数
    pw_code: str
    ph_code: str
    q2r_code: str
    matdyn_code: str
    protocol: str = "moderate"
    overrides: Optional[dict] = None
    
    # 唯一标识符
    def get_extra_unique_keys():
        return ("formula_hill", "number_of_sites", "source_db", "source_id")
    
    def get_inputs_and_processclass_from_extras(self, extras_values):
        # 实现工作流构建逻辑
        pass
```

### 2. 完整实现

我们为你创建了一个完整的 `SuperConWorkChainController`，包含：

- **代码配置**: pw, ph, q2r, matdyn 代码
- **工作流参数**: protocol, overrides, electronic_type, spin_type
- **收敛参数**: convergence_threshold, interpolation_distances
- **资源配置**: 计算资源、队列设置
- **监控功能**: 状态摘要、连续监控

## 使用方法

### 1. 基本使用

```python
from aiida import load_profile, orm
from aiida_supercon.workflows.controllers.supercon_complete import create_supercon_controller

# 加载 AiiDA profile
load_profile()

# 创建 controller
controller = create_supercon_controller(
    parent_group_label="input_structures",
    group_label="supercon_workchains",
    max_concurrent=5,
    pw_code="pw@lumi",
    ph_code="ph@lumi",
    q2r_code="q2r@lumi",
    matdyn_code="matdyn@lumi",
    protocol="moderate"
)

# 提交工作流
controller.submit_new_batch(verbose=True)
```

### 2. 使用预定义配置

```python
from aiida_supercon.workflows.controllers.config import get_lumi_debug_controller

# 使用 LUMI debug 配置
controller = get_lumi_debug_controller(
    parent_group="input_structures",
    workchain_group="supercon_workchains"
)

# 连续监控提交
controller.submit_with_monitoring(verbose=True, sleep_interval=60)
```

### 3. 手动控制

```python
# 检查状态
summary = controller.get_status_summary()
print(f"Active: {summary['active_slots']}/{summary['max_concurrent']}")
print(f"Remaining: {summary['still_to_run']}")

# 手动提交批次
submitted = controller.submit_new_batch(verbose=True)
for extras, node in submitted.items():
    print(f"Submitted {extras} -> PK: {node.pk}")
```

### 4. 干运行（测试）

```python
# 查看将要提交什么，但不实际提交
would_submit = controller.submit_new_batch(dry_run=True, verbose=True)
print(f"Would submit {len(would_submit)} workchains")
```

## 配置管理

### 1. 环境配置

我们提供了不同环境的预定义配置：

- **LUMI**: 高性能计算集群
- **Eiger**: 中等规模集群
- **Local**: 本地测试环境

### 2. 协议配置

- **fast**: 快速计算，较低精度
- **moderate**: 平衡精度和速度
- **precise**: 高精度计算

### 3. 自定义配置

```python
# 创建自定义配置
custom_config = {
    "max_concurrent": 3,
    "protocol": "moderate",
    "pw_code": "pw@mycluster",
    "ph_code": "ph@mycluster",
    "q2r_code": "q2r@mycluster",
    "matdyn_code": "matdyn@mycluster",
    "max_wallclock_seconds": 3600,
    "account": "my_project",
    "queue_name": "normal"
}

controller = create_supercon_controller(
    parent_group_label="inputs",
    group_label="workchains",
    **custom_config
)
```

## 最佳实践

### 1. 组管理

```python
# 创建输入组
input_group, _ = orm.Group.collection.get_or_create("supercon_inputs")

# 添加结构到输入组
structure = orm.StructureData(...)
structure.store()
input_group.add_nodes([structure])

# 设置必要的 extras
structure.base.extras.set_many({
    "formula_hill": "Al",
    "number_of_sites": 1,
    "source_db": "test",
    "source_id": "001"
})
```

### 2. 监控和调试

```python
# 检查所有已提交的工作流
all_submitted = controller.get_all_submitted_processes()

for extras, workchain in all_submitted.items():
    status = workchain.process_state.value
    print(f"{extras}: {status}")
    
    if status == "finished":
        # 检查输出
        outputs = workchain.outputs
        print(f"  Outputs: {list(outputs.keys())}")
```

### 3. 错误处理

```python
# 检查失败的工作流
failed_workchains = controller.get_all_submitted_processes()
for extras, workchain in failed_workchains.items():
    if workchain.process_state.value == "excepted":
        print(f"Failed: {extras}")
        print(f"  Exit code: {workchain.exit_code}")
        print(f"  Exit message: {workchain.exit_message}")
```

### 4. 资源优化

```python
# 根据集群负载调整并发数
if cluster_load_high:
    controller.max_concurrent = 2
else:
    controller.max_concurrent = 5

# 动态调整资源
controller.max_wallclock_seconds = 7200  # 2 hours
controller.num_mpiprocs_per_machine = 16
```

## 常见问题

### 1. 工作流不提交

- 检查父组中是否有节点
- 确认节点的 extras 设置正确
- 验证 `max_concurrent` 设置

### 2. 资源不足

- 减少 `max_concurrent`
- 调整 `max_wallclock_seconds`
- 检查队列配额

### 3. 工作流失败

- 检查代码配置
- 验证输入结构
- 查看工作流日志

## 高级功能

### 1. 自定义过滤器

```python
controller = SuperConWorkChainController(
    parent_group_label="inputs",
    group_label="workchains",
    filters={"extras.source_db": "specific_db"},  # 只处理特定数据库的结构
    max_concurrent=3
)
```

### 2. 排序控制

```python
# 按特定顺序提交
submitted = controller.submit_new_batch(sort=True, verbose=True)
```

### 3. 自定义监控

```python
# 创建自定义监控循环
while controller.num_to_run > 0:
    # 自定义逻辑
    if some_condition:
        controller.max_concurrent = 1
    
    submitted = controller.submit_new_batch(verbose=True)
    time.sleep(30)
```

## 总结

AiiDA Submission Controller 为你的 SuperCon 项目提供了强大的工作流管理能力：

1. **自动化**: 自动从输入组生成工作流
2. **并发控制**: 智能管理同时运行的工作流数量
3. **状态监控**: 实时跟踪提交和运行状态
4. **配置灵活**: 支持不同环境和协议配置
5. **错误处理**: 内置错误检测和处理机制

通过合理使用这个工具，你可以高效地管理大量的超导计算工作流，提高计算资源的利用率，并确保计算的可靠性和可重现性。 