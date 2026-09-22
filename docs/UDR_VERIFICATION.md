# UDR v3.2 验证记录

日期：2026-09-22。恢复起点：`ca791ba`，原有未提交改动已备份并恢复。

## 已执行

环境：Windows，Python 环境 `C:\Users\Alex\.conda\envs\mamba`，PyTorch 2.7.1+cu118。检测到 RTX 3090，但该环境缺少 `mamba_ssm` 和 `selective_scan_cuda`。

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:\Users\Alex\.conda\envs\mamba\python.exe' scripts/udr/check_udr.py --cpu-reference
```

结果：全部通过。使用原有检查工具中的可微分 selective-scan 参考递推；仅隔离不可用的导入，不改变生产模型 CUDA 路径。

- 两个阶段的完整 `datasets` 内容与原 GTSS 配置一致；测试数据和指标设置一致。
- 严格 RGB transfer 拒绝缺失/多余/形状错误的权重；完整 UDR 加载拒绝纯 RGB 权重。
- alpha=0、匹配随机种子时，奇数尺寸/B=2 的 UDR 输出与 RGB baseline 逐元素完全一致。
- 改变 Depth 不改变任一 RGB stage 的输出，但可以改变最终 SR 输出。
- 检查 pre-Gumbel 空间熵、均匀/确定分布端点、ASSB4–6 聚合。
- U=0 或 C=0 时残差为零；alpha 有界；常量深度梯度为零；检查极小输入和错位拒绝。
- 使用真实模型/优化器方法完成 A/B 的 L1 backward 和 optimizer step；A 的所有 UDR 参数有有限非零梯度，RGB 权重和梯度冻结；B 的 RGB/Depth 均有梯度。
- 检查固定 LR 参数组、完整检查点序列化加载、同阶段 optimizer/scheduler 恢复和 A 阶段 eval 冻结行为。
- 检查 B=1/2 的奇数尺寸分块对齐/拼接、train/eval 状态恢复；用拒绝 forward 的包装器确认 rank-0 验证绕过 DDP 包装。
- 16-bit 合成深度图检查全图归一化、共同裁剪/增强、缺图/错位拒绝。
- 合成指标检查四模型 mean/sample std、配对差、成功和停止条件；这些合成值不是模型实验结果。
- 实例化生产配置：总参数 23,092,885；UDR 新参数 42,172。

参考小模型初始 SR 与 RGB 的 RMS 差约 `5.53e-7`；仅验证小扰动初始化，不用于推断生产模型增益。

新增 Python 代码同时通过 Python 3.8 语法检查；权重 key 的前缀处理兼容原项目 Python 3.8 环境。真实多进程 DDP 尚未测试。

## 尚未执行

真实 Mamba CUDA kernel、DDP 多进程训练、服务器数据/490k 权重文件检查，以及 Phase A/B 各 100k 正式训练和五库三次性能评估。本地没有对应服务器文件，且缺少 Mamba CUDA 扩展。

服务器执行 `python scripts/udr/check_udr.py --check-data` 后，再按 [完整训练测试命令](UDR_MambaSR_GUIDE.md)运行正式实验。不能用本地参考验证代替正式性能结论。
