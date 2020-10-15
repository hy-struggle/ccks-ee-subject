## Multi-Pointer

**模型:**

[Pre-train Model]+multi-pointer

**方法的优势:**

可以解决多标签的问题

**方法存在的问题:**

分词不够准确(无可靠约束)，泛化性差

**文件结构:**

数据预处理：`python preprocess.py`

训练：`python train.py`

预测：`python predict.py`

后处理：`python postprocess.py`

结果融合：`python ensemble.py`