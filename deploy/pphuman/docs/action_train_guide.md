# PPHuman 行为识别模型——从训练到使用

PP-Human中集成了基于骨骼点的行为识别模块。本文档介绍如何基于[PaddleVideo](https://github.com/PaddlePaddle/PaddleVideo/)，完成行为识别模型的训练流程。

## 行为识别模型训练
目前行为识别模型使用的是[ST-GCN](https://arxiv.org/abs/1801.07455)，并在[PaddleVideo训练流程](https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/recognition/stgcn.md)的基础上修改适配，完成模型训练。

### 准备训练数据
STGCN是一个基于骨骼点坐标序列进行预测的模型。在PaddleVideo中，训练数据为采用`.npy`格式存储的`Numpy`数据，标签则可以是`.npy`或`.pkl`格式存储的文件。对于序列数据的维度要求为`(N,C,T,V,M)`。

以我们在PPhuman中的模型为例，其中具体说明如下：
| 维度 | 大小 | 说明 |
| ---- | ---- | ---------- |
| N | 不定 | 数据集序列个数 |
| C | 2 | 关键点坐标维度，即(x, y) |
| T | 50 | 动作序列的时序维度（即持续帧数）|
| V | 17 | 每个人物关键点的个数，这里我们使用了`COCO`数据集的定义，具体可见[这里](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.4/docs/tutorials/PrepareKeypointDataSet_cn.md#COCO%E6%95%B0%E6%8D%AE%E9%9B%86) |
| M | 1 | 人物个数，这里我们每个动作序列只针对单人预测 |


#### 1. 获取序列的骨骼点坐标
对于一个待标注的序列（这里序列指一个动作片段，可以是视频或有顺序的图片集合）。可以通过模型预测或人工标注的方式获取骨骼点（也称为关键点）坐标。
- 模型预测：可以直接选用[PaddleDetection KeyPoint模型系列](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/keypoint) 模型库中的模型，并根据`3、训练与测试 - 部署预测 - 检测+keypoint top-down模型联合部署`中的步骤获取目标序列的17个关键点坐标。
- 人工标注：若对关键点的数量或是定义有其他需求，也可以直接人工标注各个关键点的坐标位置，注意对于被遮挡或较难标注的点，仍需要标注一个大致坐标，否则后续网络学习过程会受到影响。

在完成骨骼点坐标的获取后，建议根据各人物的检测框进行归一化处理，以消除人物位置、尺度的差异给网络带来的收敛难度，这一步可以参考[这里](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.4/deploy/pphuman/pipe_utils.py#L352-L363)。

#### 2. 统一序列的时序长度
由于实际数据中每个动作的长度不一，首先需要根据您的数据和实际场景预定时序长度（在PPHuman中我们采用50帧为一个动作序列），并对数据做以下处理：
- 实际长度超过预定长度的数据，随机截取一个50帧的片段
- 实际长度不足预定长度的数据：补0，直到满足50帧
- 恰好等于预定长度的数据： 无需处理

注意：在这一步完成后，请严格确认处理后的数据仍然包含了一个完整的行为动作，不会产生预测上的歧义，建议通过可视化数据的方式进行确认。

#### 3. 保存为PaddleVideo可用的文件格式
在经过前两步处理后，我们得到了每个人物动作片段的标注，此时我们已有一个列表`all_kpts`，这个列表中包含多个关键点序列片段，其中每一个片段形状为(T, V, C) （在我们的例子中即(50, 17, 2)), 下面进一步将其转化为PaddleVideo可用的格式。
- 调整维度顺序： 可通过`np.transpose`和`np.expand_dims`将每一个片段的维度转化为(C, T, V, M)的格式。
- 将所有片段组合并保存为一个文件

- 可参考下列代码片段。

```python
all_annos = []
for kpt_anno in all_kpts:
    # kpt_anno's shape is (T, V, C)
    kpt_anno = np.transpose(kpt_anno, (2,0,1))
    # now kpt_anno's shape is (C, T, V)
    kpt_anno = np.expand_dims(kpt_anno, -1)
    # kpt_anno's shape is (C, T, V, M) and here M=1
    all_annos.append(kpt_anno)
all_annos = np.array(all_annos)
data = np.stack(all_annos, 0)
np.save("data.npy", data)
```
- 对于标签文件的处理，以`.pkl`为例，可参考以下代码片段。

```python
import pickle

"""
 Label is a List with length 2, includes 2 sub-list.
 Label[0] is the list of original label file name，not used practically.
 Label[1] is the list of class_id for each data.
"""

Label = [[],[]]
for class_id in all_labels:
    Label[0].append(str(video_name))  # Could be any string since it not used practically.
    Label[1].append(class_id)
pickle.dump(Label, open("label.pkl", "wb"))
```
注意：这里的`class_id`是`int`类型，与其他分类任务类似。例如`0：摔倒， 1：其他`。


至此，我们得到了可用的训练数据（`.npy`）和对应的标注文件（`.pkl`）。

### 修改PaddleVideo以实现自定义数据训练
#### 1. 配置PaddleVideo环境
依照下列步骤，配置PaddleVideo套件的训练环境，更详细的步骤请见[安装说明](https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/install.md)
```bash
   # Clone PaddleVideo套件代码
   git clone https://github.com/PaddlePaddle/PaddleVideo.git

   # 安装依赖
   cd PaddleVideo
   pip install -r requirements.txt

   # 安装paddlevideo
   python setup.py install
```

#### 2. 修改配置文件
在这一步中，我们对PaddleVideo的[STGCN-NTU配置文件](https://github.com/PaddlePaddle/PaddleVideo/blob/develop/configs/recognition/stgcn/stgcn_ntucs.yaml)进行修改。

- 在`Model`部分，做以下修改：
```yaml
MODEL:
    framework: "RecognizerGCN"
    backbone:
        name: "STGCN"
        in_channels: 2  # 与数据(N, C, T, V, M)中C维的尺度一致。
        dropout: 0.5
        layout: 'our_dataset' #名称自定义，与下一步中的代码一致即可
    head:
        name: "STGCNHead"
        num_classes: 2  # 与实际的行为类别数一致
```

- 在`DATASET`部分，分别将`train`/`valid`/`test`部分的`file_path`, `label_path`修改为我们前面整理的数据文件路径。

#### 3. 修改网络代码
在这一步中，由于我们定义的骨骼点结构、数量等与原始的`NTU-RGB+D`有很大不同，因此需要对网络做一些修改。

- 对文件[paddlevideo/modeling/backbones/stgcn.py](https://github.com/PaddlePaddle/PaddleVideo/blob/develop/paddlevideo/modeling/backbones/stgcn.py)中的`Graph`类做出如下修改：

```python
class Graph():
    def __init__(self,
    ...

    def get_edge(self, layout):
        # edge is a list of [child, parent] paris

        if layout == 'fsd10':
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(1, 8), (0, 1), (15, 0), (17, 15), (16, 0),
                             (18, 16), (5, 1), (6, 5), (7, 6), (2, 1), (3, 2),
                             (4, 3), (9, 8), (10, 9), (11, 10), (24, 11),
                             (22, 11), (23, 22), (12, 8), (13, 12), (14, 13),
                             (21, 14), (19, 14), (20, 19)]
            self.edge = self_link + neighbor_link
            self.center = 8
        elif layout == 'ntu-rgb+d':
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
                              (7, 6), (8, 7), (9, 21), (10, 9), (11, 10),
                              (12, 11), (13, 1), (14, 13), (15, 14), (16, 15),
                              (17, 1), (18, 17), (19, 18), (20, 19), (22, 23),
                              (23, 8), (24, 25), (25, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 21 - 1

        # 从这里开始新增自定义的数据任务
        elif layout == 'our_dataset':
            self.num_node = 17  # 基于COCO的关键点定义
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 8),
                 (7, 9), (8, 10), (5, 11), (6, 12), (11, 13), (12, 14),
                 (13, 15), (14, 16), (11, 12)]   # 基于COCO的骨骼定义
            neighbor_link = [(i, j) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 11  # COCO定义中没有一个完全意义上的中心点，我们选取身体偏中的一个点作为中心点即可。
        else:
            raise ValueError("Do Not Exist This Layout.")
```

- 当自定义的行为数量不足5类时，原评估中的`top5`会出错，需要对[`paddlevideo/modeling/heads/base.py`](https://github.com/PaddlePaddle/PaddleVideo/blob/develop/paddlevideo/modeling/heads/base.py#L158)做修改

```python
    def get_acc(self, scores, labels, valid_mode, if_top5=True):
        if if_top5:
            top1 = paddle.metric.accuracy(input=scores, label=labels, k=1)
            #top5 = paddle.metric.accuracy(input=scores, label=labels, k=5)
            top5 = paddle.metric.accuracy(input=scores, label=labels, k=2)  # 不要超过定义的行为数量即可
            _, world_size = get_dist_info()
            if world_size > 1 and valid_mode:
                top1 = paddle.distributed.all_reduce(
                    top1, op=paddle.distributed.ReduceOp.SUM) / world_size
                top5 = paddle.distributed.all_reduce(
                    top5, op=paddle.distributed.ReduceOp.SUM) / world_size

            return top1, top5
```

### 训练与测试
在PaddleVideo中，使用以下命令即可开始训练：
```bash
python main.py -c {your_config_file}

# 由于整个任务可能过拟合,也可以开启验证以保存最佳模型
python main.py --validate -c {your_config_file}
```

在训练完成后，采用以下命令进行预测：
```bash
python main.py --test -c {your_config_file}  -w {your_model_weights.params}
```

### 导出模型推理
- 在配置文件中增加以下导出部分设置：
```yaml
INFERENCE:
    name: 'STGCN_Inference_helper'
    num_channels: 2  # 同训练设置，与C维度一致
    window_size: 50  # 预定义的时序长度
    vertex_nums: 17  # 每人骨骼点数量
    person_nums: 1
```

- 在PaddleVideo中，通过以下命令实现模型的导出，得到模型结构文件`STGCN.pdmodel`和模型权重文件`STGCN.pdiparams`：
```python
python tools/export_model.py -c {your_config_file} \
                                -p {your_model_weights.params} \
                                -o inference/STGCN
```
- 通过以下命令重命名模型，以适配PPHuman的文件读取方法
```bash
# 为模型增加软链接
ln -s STGCN.pdiparams model.pdiparams
ln -s STGCN.pdiparams.info model.pdiparams.info
ln -s STGCN.pdmodel model.pdmodel

# 或直接重命名
mv STGCN.pdiparams model.pdiparams
mv STGCN.pdiparams.info model.pdiparams.info
mv STGCN.pdmodel model.pdmodel
```
完成后的导出模型目录结构如下：
```
STGCN
├── infer_cfg.yml
├── model.pdiparams -> STGCN.pdiparams
├── model.pdiparams.info -> STGCN.pdiparams.info
├── model.pdmodel -> STGCN.pdmodel
├── STGCN.pdiparams
├── STGCN.pdiparams.info
└── STGCN.pdmodel
```

至此，就可以使用PPHuman进行行为识别的推理了。
