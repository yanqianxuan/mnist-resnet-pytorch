## 1. 简介
本项目基于 **PyTorch** 框架构建了一个针对 MNIST 数据集优化的 **ResNet** 卷积神经网络模型。通过集成数据增强（Data Augmentation）、动态学习率调度和 GPU 加速技术，模型在极短的时间内达到了极高的识别精度。

## 2. 核心实验数据 (实测记录)
| 指标 | 数值 |
| :--- | :--- |
| **最高验证准确率** | **99.24%** (于第 8 轮达成) |
| **最终训练损失 (Loss)** | 0.0256 |
| **训练轮数 (Epochs)** | 10 |
| **硬件性能** | MacBook Air (Apple Silicon MPS 加速) |
| **单轮训练耗时** | 约 25-31 秒 |

## 3. 项目目录结构
* `resnet_model.py`：定义残差网络结构，包含 `BasicBlock` 和 `ResNetMNIST` 类。
* `dataset.py`：负责数据下载、预处理及 `RandomRotation` 数据增强。
* `config.py`：使用 `argparse` 管理超参数（如 `batch_size`, `lr`）。
* `train.py`：主训练脚本，包含验证逻辑、模型保存及日志输出。
* `test.py`：独立推理脚本，加载 `best_model.pt` 进行单张图片预测。
* `train.log`：完整的训练历史记录。
* `best_model.pt`：训练中自动保存的最佳模型权重文件。

## 4. 快速开始
### 训练模型
```bash
python train.py --epochs 10 --lr 0.001
