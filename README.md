# TimeDistilling + CNN

基于 TimeDistill 的时间序列知识蒸馏实验代码，包含 CNNGlobal 学生模型及五数据集训练准备脚本。

## 目录

- `timedistill_cnn/`：TimeDistill 代码副本及 CNN 学生相关修改。原项目说明见其 [README](timedistill_cnn/README.md)，依赖见 `requirements.txt`。
- `timedistill_cnn/models/CNNGlobal.py`：CNN 学生模型。
- `timedistill_cnn/prepare_five.py`：ETTm2、Electricity、Solar、Traffic 和 Weather 的训练准备入口，默认仅检查并打印命令，`--execute` 才执行训练。

数据集、训练权重、缓存、预测结果及服务器凭据不包含在仓库内。使用前需按上游说明准备数据和教师权重；实验参数、数据划分与结果应单独核验。本仓库不以代码上传作为实验有效或达到论文指标的证明。

## 来源与许可证

`timedistill_cnn/` 包含上游 TimeDistill 代码及后续修改，保留该目录的 [Apache 2.0 许可证](timedistill_cnn/LICENSE)；仓库根目录原有许可证保留不变。上游代码与其修改部分的使用应遵循相应许可证及署名要求。
