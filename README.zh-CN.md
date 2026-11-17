# ResumeTrace

**checkpoint 能不能真正撑过一次进程退出？**

不是换一个 Python 对象，而是实际启动新解释器，比较连续训练与“保存 → 退出进程 → 重建 → 恢复”的差异。训练数据由你的训练代码重新读取，工具不会预先缓存一批数据喂给两边。

这是实验性 v0.1：CPU eager、小型确定性训练测试；不证明模型质量、收敛效果、未来步骤或未观测状态正确。无需 GPU、模型下载或 API key。

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
