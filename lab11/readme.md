# 实验任务书
## 基于 MediaPipe Pose 与骨架序列 Transformer 的羽毛球击球动作识别
课程实验时长：2 小时
任务类型：视频识别 / 序列建模 / Transformer 分类

提示：本实验不直接把原始视频像素送入大型 Video Transformer，而是先用 MediaPipe Pose 将视频转换为人体骨架时间序列，再用轻量 Transformer Encoder 完成动作分类。这一设计计算量低、可解释性强，适合课堂完成训练、测试与推理全流程。

## 一、实验背景
羽毛球击球动作具有明显的时间动态特征，例如引拍、挥拍、击球、收拍与回位等阶段。直接用原始视频训练 Transformer 需要大量视频标注和较高 GPU 显存；而人体骨架序列可以显著压缩视频信息，只保留与动作相关的人体关键点运动轨迹。

本实验将每一帧的人体姿态关键点看作一个时间 token，一段视频就是一个 token 序列。Transformer Encoder 负责学习这些骨架 token 随时间变化的模式，并输出击球动作类别。

## 二、数据集说明
本实验使用 Kaggle 数据集 badminton_storke_video。该数据集包含 6 类羽毛球击球动作视频片段，类别如下：

| 标签编号 | 英文类别 | 中文说明 |
| ---- | ---- | ---- |
| 0 | forehand drive | 正手平抽 / 正手驱动球 |
| 1 | forehand lift | 正手挑球 |
| 2 | forehand net shot | 正手网前球 |
| 3 | forehand clear | 正手高远球 |
| 4 | backhand drive | 反手平抽 / 反手驱动球 |
| 5 | backhand net shot | 反手网前球 |

数据集链接：https://www.kaggle.com/datasets/shenhuichang/badminton-storke-video

说明：Kaggle 页面中的数据集名称拼写为 “badminton_storke_video”，其中 storke 应理解为 stroke。

## 三、实验目标
- 理解视频动作识别任务如何转化为骨架时间序列分类任务。
- 掌握使用 MediaPipe Pose 从视频帧中提取人体 33 个关键点的方法。
- 掌握将每段视频统一转换为固定长度骨架序列的方法，例如 [60, 132]。
- 实现一个轻量级 Skeleton Transformer，用 Transformer Encoder 对动作序列分类。
- 完成模型训练、测试集评估与单个视频样本推理。
- 理解该方法在羽毛球商业化视频分析中的优势与局限。

## 四、任务总体流程
```
Kaggle 羽毛球视频数据集
        ↓
逐个读取视频片段
        ↓
MediaPipe Pose 提取每帧人体关键点
        ↓
每帧 33 个关键点 × 4 个特征 = 132 维
        ↓
每段视频重采样为固定 T 帧，例如 T = 30
        ↓
形成骨架序列 X_i ∈ R^(30×132)
        ↓
Transformer Encoder 进行动作分类
        ↓
训练、测试、推理与结果分析
```

## 五、骨架序列数据格式
MediaPipe Pose 对每一帧输出 33 个关键点。每个关键点包含 x、y、z 和 visibility 四个数值，因此每帧骨架特征维度为：
33 × 4 = 132

若每段视频统一采样为 T = 30 帧，则一个视频样本的数据形状为：
`X_i.shape = [30, 132]`

整个训练集保存为：
- `X_train.npy`  shape = [N_train, 30, 132]
- `y_train.npy`  shape = [N_train]
- `X_test.npy`   shape = [N_test, 30, 132]
- `y_test.npy`   shape = [N_test]
- `label_map.json`

提示：本实验中“token”不是单词，也不是图像 patch，而是一帧人体骨架状态。也就是说，第 1 帧骨架是 token 1，第 2 帧骨架是 token 2，依此类推。

## 六、预处理要求
学生需要编写或运行预处理程序，将 Kaggle 视频数据转换为骨架序列数据。建议处理步骤如下：
1. 遍历数据集中每个类别文件夹，读取所有 `.mp4` / `.avi` / `.mov` / `.mkv` 视频。
2. 使用 OpenCV 逐帧读取视频。
3. 使用 MediaPipe Pose 对每一帧提取人体关键点。
4. 将每一帧 33 个关键点的 x、y、z、visibility 展平成 132 维向量。
5. 将不同长度的视频统一重采样为 T = 30 帧。
6. 对骨架进行简单归一化，例如以左右髋部中心为原点，以肩宽进行尺度归一化。
7. 划分训练集和测试集，建议 test_size = 0.2，并保存为 `.npy` 文件。

## 七、模型设计要求
模型采用 Transformer Encoder 结构。推荐模型结构如下：
```
输入骨架序列 X: [B, T, 132]
        ↓
Linear Embedding: 132 → d_model
        ↓
Position Embedding: 加入时间位置信息
        ↓
Transformer Encoder × 2
        ↓
Mean Pooling 或 CLS Token
        ↓
MLP Classifier
        ↓
输出 6 类羽毛球击球动作 logits
```

| 参数 | 建议值 | 说明 |
| ---- | ---- | ---- |
| input_dim | 132 | 每帧骨架特征维度 |
| target_frames | 30 | 每段视频统一帧数 |
| d_model | 128 | Transformer 主维度 |
| nhead | 4 | 多头注意力 head 数 |
| num_layers | 2 | Transformer Encoder 层数 |
| dim_feedforward | 256 | FFN 中间层维度 |
| num_classes | 6 | Kaggle 数据集动作类别数 |
| dropout | 0.1 | 防止过拟合 |

## 八、训练与测试要求
训练阶段要求使用交叉熵损失函数和 Adam 优化器。建议配置如下：

| 项目 | 建议设置 | 说明 |
| ---- | ---- | ---- |
| loss function | CrossEntropyLoss | 多类别分类损失 |
| optimizer | Adam | 学习率建议 1e-3 |
| batch size | 16 或 32 | 根据 GPU/CPU 调整 |
| epochs | 20-50 | 课堂实验可先设 20 |
| evaluation | accuracy + confusion matrix | 测试分类效果 |

训练循环需包含以下步骤：
- `model.train()` 进入训练模式。
- 读取一个 batch 的 X 和 y。
- 前向传播得到 logits。
- 计算 CrossEntropyLoss。
- `loss.backward()` 反向传播。
- `optimizer.step()` 更新参数。
- 统计训练 loss 与训练 accuracy。

测试阶段要求使用 `model.eval()` 和 `torch.no_grad()`，输出测试集 accuracy，并尽量输出 confusion matrix 和 classification report。

## 九、推理任务要求
完成训练后，学生需要选取一个测试视频或 demo 视频，完成单样本推理。流程如下：
```
demo_video.mp4
    ↓
MediaPipe Pose 提取骨架序列
    ↓
重采样为 [60, 132]
    ↓
model(sample)
    ↓
softmax 得到各类别概率
    ↓
输出 predicted class 与 confidence
```

推理输出示例：
```
Predicted class: forehand clear
Confidence: 0.87
```

## 十、实验提交内容
每位学生或每个小组需提交以下内容：

| 序号 | 提交项 | 具体要求 |
| ---- | ---- | ---- |
| 1 | 预处理代码 | 包含视频读取、MediaPipe Pose 提取、重采样、归一化、保存 .npy。 |
| 2 | 训练代码 | 包含 Dataset、DataLoader、模型、训练循环。 |
| 3 | 测试与推理代码 | 输出测试准确率、混淆矩阵、单样本推理结果。 |
| 4 | 实验报告 | 说明方法、模型结构、训练曲线、测试结果和问题分析。 |
| 5 | 可选项 | 展示 1-2 个视频片段的骨架可视化或 attention 分析。 |
