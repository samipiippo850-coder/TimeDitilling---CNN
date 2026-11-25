# TimeDistilling + CNN

基于 TimeDistill 的时间序列知识蒸馏实验代码，包含 CNNGlobal 学生模型及五数据集训练准备脚本。

## 我们在原论文基础上做了什么

TimeDistill 原论文研究从较大的教师模型向轻量 MLP 学生迁移时序知识，使用预测和特征层面的频域及多尺度监督。这个仓库保留该框架，主要探索**将学生从 MLP 换成 CNN 后，如何对接原有蒸馏流程**，而不是声称重新发明了原论文的蒸馏损失。

1. **CNN 学生与蒸馏接口。** 新增 `CNNGlobal`：将每个变量的 720 步历史划分为 60 段、每段 12 步，先做线性嵌入，再用带残差连接的两层一维卷积提取时序特征，最后映射到 720 步预测。各变量共享模型参数，输入变量数量可变化；预测前减去历史均值，输出时加回。模型还输出经插值对齐维度的特征，供原有特征蒸馏接口使用。当前实现只支持 **720→720**，不是任意预测长度。
2. **沿用并适配教师监督。** 训练入口支持以 ModernTCN 为教师，并保留真实标签监督、预测层和特征层的频域及多尺度蒸馏；频域损失基于 FFT 幅值，并非显式相位监督。训练后学生独立预测，不需要教师参与推理。蒸馏效果仍需与同预算、无蒸馏 CNN 对照验证。
3. **五数据集训练准备。** 新增 `prepare_five.py`，为 ETTm2、Electricity（ECL）、Solar、Traffic 和 Weather 组织数据检查与教师、蒸馏、纯监督训练命令；默认只检查和打印命令，不会启动训练。虽然命令参数保留其他 horizon 选项，CNNGlobal 本身目前仅支持 720→720。
4. **更谨慎的实验流程。** 训练阶段默认不进行测试集评估；教师权重匹配不唯一时会报错，避免静默选择错误文件。数据加载器仍可能读取整个原始数据文件，因此这不等同于测试数据在训练期间从未被读取，也不代表测试集是密封的。

这些是**当前仓库代码能够核对的内容**。此前讨论过的动态教师／频段选择、相位头、强化学习或其他诊断实验，不应仅凭讨论就认为已经包含在这个发布版本中。仓库也没有附带经过独立核验的结果表、数据或权重，因此目前不宣称超过原论文或证明知识蒸馏有效。

## What we changed relative to TimeDistill

The original TimeDistill paper studies transferring time-series knowledge from larger teachers to a lightweight MLP student using frequency-domain and multi-scale supervision at prediction and feature levels. This repository keeps that framework and explores **using a CNN student with the existing distillation interface**; it does not claim the original distillation losses as a new contribution.

1. **CNN student and distillation interface.** `CNNGlobal` splits each variable's 720-step history into 60 segments of 12 steps, applies a linear embedding and two residual-connected 1D convolution layers, and maps the resulting representation to a 720-step forecast. Parameters are shared across variables, so the number of input variables can vary. The historical mean is subtracted from the input and added back to the output. An interpolated feature representation is provided for feature distillation. The current implementation supports **720→720 only**.
2. **Adaptation to teacher supervision.** The training entry point supports ModernTCN as teacher and retains ground-truth supervision plus prediction- and feature-level frequency and multi-scale distillation. The frequency losses use FFT magnitudes; they do not explicitly supervise phase. At inference, the trained student predicts without calling the teacher. Benefits must still be checked against a same-budget CNN trained without distillation.
3. **Five-dataset preparation.** `prepare_five.py` checks data and constructs teacher, distillation and supervised-training commands for ETTm2, Electricity (ECL), Solar, Traffic and Weather. It defaults to inspection and command printing, not training. Although other horizons remain available as command-line choices, CNNGlobal itself currently supports only 720→720.
4. **More cautious evaluation workflow.** Test evaluation is disabled by default during training, and ambiguous teacher-checkpoint matches raise an error instead of silently selecting one. The data loader may still read the complete raw data file; this is not a guarantee that test rows were never read during training or that the test set is sealed.

These points describe what can be checked in **this published code**. Previous discussions of dynamic teacher or frequency selection, phase heads, reinforcement learning and other diagnostic experiments do not by themselves establish that those methods are included here. No independently verified result table, dataset or checkpoints are included, so this repository does not currently claim to outperform the paper or establish distillation effectiveness.

## 目录

- `timedistill_cnn/`：TimeDistill 代码副本及 CNN 学生相关修改。原项目说明见其 [README](timedistill_cnn/README.md)，依赖见 `requirements.txt`。
- `timedistill_cnn/models/CNNGlobal.py`：CNN 学生模型。
- `timedistill_cnn/prepare_five.py`：ETTm2、Electricity、Solar、Traffic 和 Weather 的训练准备入口，默认仅检查并打印命令，`--execute` 才执行训练。

数据集、训练权重、缓存、预测结果及服务器凭据不包含在仓库内。使用前需按上游说明准备数据和教师权重；实验参数、数据划分与结果应单独核验。本仓库不以代码上传作为实验有效或达到论文指标的证明。

## 来源与许可证

`timedistill_cnn/` 包含上游 TimeDistill 代码及后续修改，保留该目录的 [Apache 2.0 许可证](timedistill_cnn/LICENSE)；仓库根目录原有许可证保留不变。上游代码与其修改部分的使用应遵循相应许可证及署名要求。
