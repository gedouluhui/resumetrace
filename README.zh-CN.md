# ResumeTrace

**checkpoint 能不能真正撑过一次进程退出？**

不是换一个 Python 对象，而是实际启动新解释器，比较连续训练与“保存 → 退出进程 → 重建 → 恢复”的差异。训练数据由你的训练代码重新读取，工具不会预先缓存一批数据喂给两边。

**0.2.1** 修掉一个审查发现的真实契约漏洞：结构自洽但无效的 RNG 证据（例如 float32、空 shape 的 Torch 状态）现在会被拒绝，而不是被当成可比较状态；每种 RNG 的内部结构、dtype 与 shape 必须与当前解释器实际产出的一致。仍为预发行版，没有整仓独立终审。仅支持 CPU eager、小型确定性训练测试；不证明模型质量、收敛效果、未来步骤或未观测状态正确。无需 GPU、模型下载或 API key。

[免安装查看真实演示](https://gedouluhui.github.io/resumetrace/) · [验收与审查限制](docs/VALIDATION.md)

## 从源码运行

在仓库根目录、自己的 Python 3.10+ 虚拟环境中：

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install .
python -m resumetrace demo --fault none --json normal.json --html normal.html
python -m resumetrace demo --fault process-cache --json drift.json --html drift.html
```

第一条演示预期 `PASS`；第二条预期 `DRIFT`，退出码 1 表示发现差异，不是工具崩溃。直接用浏览器打开 HTML；不加载外部字体、脚本、CDN 或统计服务。不宣称已发布 PyPI，请从源码或实际发行 wheel 安装。

`process-cache` 使用真实小模型、Dropout、AdamW 和数据游标：故意把优化器状态放在模块全局字典，而不是 checkpoint。相同进程中重建对象可能成功，新进程则丢失这份状态。另有 `optimizer`、`rng`、`cursor`、`scheduler` 故障可选。

一个容易忽略的反例：scheduler 周期和断点对齐时，即使恢复错了，后面的 loss 仍可能一样。因此工具还要比较参数、优化器、随机数和显式声明的状态，不能只看 loss。

## 接入你自己的训练

准备可导入的 `module:function` 工厂，接收一个 JSON 字典，返回具有以下四个方法的对象：

- `step()`：真正取下一批数据并完成一次优化器更新，返回非空的有限数值指标字典。
- `observe()`：返回相关 CPU 张量、普通字典/列表/元组与有限标量；必须只读，不消耗随机数。
- `save(directory)`：使用你原本的 checkpoint 逻辑写文件。
- `load(directory)`：恢复刚构建的新对象；工具不会替你补回遗漏状态。

完整可运行示例见 [custom_training.py](examples/custom_training.py)。从仓库根目录执行：

```bash
python -m resumetrace audit examples.custom_training:make_session --steps 6 --split 3 --json custom.json --html custom.html
```

也可以在 unittest/pytest 中调用：

```python
from resumetrace import audit_resume
result = audit_resume('your_project.training:make_session', steps=12, split=5)
assert result['status'] == 'PASS', result['first_divergence'] or result['error']
```

必须是新解释器能导入的模块工厂；不支持 notebook 内局部函数、闭包或直接传对象。回调是你明确提供的可信代码，不是沙箱。请使用隔离的小型实验，不要接有生产写入副作用的任务。

## 可选自动状态观察器

优化器 checkpoint 的整数 ID 和数值完全一致，也可能在恢复时绑定到了错误参数。新 `TorchSession` 直接读取真实参数对象与注册名称，不把序列化 ID 当作参数身份：

```python
from resumetrace import TorchSession

def make_session(config):
    return TorchSession(YourExistingSession(config), gradient_policy="ignore")
```

`YourExistingSession` 是你自己的训练类，需要普通实例属性 `model`、`optimizer`，以及原有 step/save/load；可选 observe 提供 scheduler、数据游标和私有生成器等状态。自动采集包含注册参数、全部注册 buffer、模块 train/eval、requires_grad、真实优化器归属与状态，不调用自定义 state_dict，不替你修复训练。

直接运行完整 SGD 示例：

```bash
python -m resumetrace audit examples.guarded_training:make_session --steps 3 --split 1 --json guarded.json --html guarded.html
python -m resumetrace audit examples.guarded_training:make_session --config '{"fault":"observer"}' --steps 3 --split 1 --json observer.json --html observer.html
python -m resumetrace audit examples.guarded_training:make_session --config '{"fault":"save"}' --steps 3 --split 1 --json save.json --html save.html
```

正常例预期 PASS；后两例分别在观察时清零权重、写完合法 checkpoint 后修改权重，预期 INVALID，报告具体检查、步数与差异路径。**save 副作用违反的是自选的纯度契约，不等于保存文件必然无法恢复。**未包裹的旧接口不会被强加这些新规则。

梯度策略为：`ignore` 默认不参与跨重启比较，允许下一步开头清梯度的合法训练循环；`capture` 精确比较梯度；`require_none` 要求观察时本来就已清空。三种策略的副作用检查都观察梯度，工具不会代清梯度。

自动适配器首批只支持原生 SGD/AdamW 和 CPU eager；参数／优化器状态／梯度之间共享存储、内部重叠或无法证明不重叠的布局、共享模型注册、自定义 Tensor／optimizer、超预算会返回 INVALID。未声明的 scheduler、数据、隐藏属性和文件／网络副作用依然不在保证内。新增报告明确展示覆盖范围与梯度策略；正常检查增加全状态指纹开销，不宣称免费或生产级全覆盖。

## 如何判读

| 状态 | 含义 | CLI退出码 |
|---|---|---:|
| PASS | 确定性对照、保存前前缀、恢复边界和所测后续步骤都精确匹配 | 0 |
| DRIFT | 对照有效，但实际观测到差异；不能自动断言根因 | 1 |
| INVALID | 对照不稳定、缺文件、超时、不支持类型、超预算等，不能作出恢复结论 | 2 |

四个进程依次运行：连续基准、独立连续对照、运行前缀并保存退出、冷启动恢复。前一个保存进程成功退出后才启动恢复进程。每次观测必须覆盖你关注的参数、优化器、scheduler、训练模式、私有 Generator、采样游标及 Python 属性。

张量比较包含 shape、dtype 和逻辑值 SHA-256，不以范数相等冒充逐元素相等。报告不输出原始权重；明确返回的数值指标会展示。观察键名和指标仍需由使用者自行避免夹带秘密。默认不覆盖报告文件；需要替换时加 `--force`。

## 支持范围

只支持单进程 CPU eager、完整优化步边界；worker 内启用确定性算法并使用一个 intra-op 线程。不支持 CUDA、分布式、AMP、编译图、多进程 DataLoader 或用户后台线程，不保证跨机器/版本一致。父进程随机数与设置不改动。

短期观测一致不等于全部训练状态都正确；文件哈希也不是来源认证。默认每个 worker 超时为 120 秒。资源预算不足会明确返回 INVALID，而不是把局部比较包装成通过。

[pytest-torch](https://github.com/AKJ7/pytest-torch) 已提供恢复等价检查；本项目明确聚焦“真正的新进程、重新进入数据管线、独立连续对照和离线证据”，不宣称发明恢复测试。更完整的边界、使用方法与相关项目见 [英文 README](README.md)。

项目由 AI 辅助开发并通过实际执行与审查推进，不承诺 GitHub stars、外部用户采用或真实性能收益。MIT 许可。

## 快速开始与支持状态

### 问题
通过比较不间断执行与保存、退出、重建、冷恢复，测试训练能否在真实进程退出后正确续跑。

### 安装
```sh
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu && python -m pip install .
```

### 成功命令
```sh
python -m resumetrace demo --fault none --json normal.json --html normal.html
```
预期：冷进程比较有效且匹配；退出码 0。

### 故意失败命令
```sh
python -m resumetrace demo --fault process-cache --json drift.json --html drift.html
```
预期：丢失的进程全局优化器状态被报告为 DRIFT；退出码 1。

### 证据与覆盖边界
证据包括四个真实 Python 进程、重复连续对照、数据管线重新进入、优化器/调度器/RNG/游标观察、严格 RNG 格式校验和 101 项本地测试。精确相等仅限于观察到的有限域和确定性 CPU eager 模式，不是通用可复现性保证。

### 支持状态
版本 0.2.1 支持 Python 3.10+ 和 CPU PyTorch。它仍是预发布版本，没有独立全项目批准。文档路径是源码或发布 wheel，不是猜测的 PyPI 包。维护为尽力而为，不提供 SLA。

### 限制
此版本不支持 GPU、分布式训练、AMP、编译图、后台线程、多进程数据加载器、自定义优化器和许多别名/布局情况。回调是受信任代码，可能产生任意副作用；预算不是进程 RSS 上限。
