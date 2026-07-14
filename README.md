# BARU_Unet — 统一图像复原训练框架

以 U-Net 架构为核心的 SAR 图像复原模型对比框架。集成 5 个主流模型，统一训练入口。

| 模型 | 年份 | 核心思想 | Attention 类型 | 参数量 |
|---|---|---|---|---|
| **NAFNet** | ECCV 2022 | Activation-Free CNN | Simplified Channel Attention | 6~17M |
| **KBNet** | ICCV 2023 | Kernel Basis Network | Kernel Basis Attention | 16~18M |
| **MIRNetv2** | TPAMI 2022 | Multi-scale Feature Fusion | SKFF + 上下文 | 5.9M |
| **HINet** | CVPRW 2021 | Half Instance Normalization | 几乎无 | 8.7M |
| **SCUNet** | ICCV 2023 | Conv + Swin Transformer | Window Self-Attention | 12~18M |

---

## 目录

- [环境安装](#环境安装)
- [数据集放置](#数据集放置)
- [模型选择与参数](#模型选择与参数)
- [训练](#训练)
- [参数调节指南](#参数调节指南)
- [实验结果查看](#实验结果查看)
- [测试/推理](#测试推理)
- [配置文件结构](#配置文件结构)

---

## 环境安装

```bash
cd KBNet

# 安装依赖
pip install -r requirements.txt

# 安装框架（开发模式）
pip install -e . --no-build-isolation
```

`requirements.txt` 已包含所有必要依赖（opencv-python, scikit-image, PyYAML, lmdb, einops, timm 等）。

---

## 数据集放置

数据集按以下目录结构组织：

```text
data/
└── MixUpsample_Dataset_R2A2_256/
    ├── train/
    │   ├── H/       ← GT（高分辨率） *.png, 256×256, 灰度
    │   └── L/       ← LQ（低分辨率） *.png, 256×256, 灰度
    └── test/
        ├── H/       ← GT              *.png, 256×256, 灰度
        └── L/       ← LQ              *.png, 256×256, 灰度
```

### 配对方式

数据集文件名包含 `_H_` / `_L_` 后缀（例如 `Bangkok_1_r0001_p0001_H_TR.png` ↔ `Bangkok_1_r0001_p0001_L_TR.png`）。框架通过 `pair_by_index: true` 模式按排序索引配对，无需文件名一致。

> 如果使用文件名完全一致的标准数据集，设为 `pair_by_index: false` 或不写此项即可使用默认配对逻辑。

### 灰度图处理

灰度 PNG 通过 `cv2.IMREAD_COLOR` 自动转为 3 通道 BGR（值复制到三个通道），模型输入保持 `in_chn=3`，无需修改架构。

---

## 模型选择与参数

### HINet

```yaml
network_g:
  type: HINet
  in_chn: 3
  wf: 64                    # wf=32 为 0.5x 轻量版
  depth: 5
  hin_position_left: 0
  hin_position_right: 4
```

- 参数量：8.7M（wf=64）
- 显存：256² + batch=8 ≈ 2.5GB
- 特点：双阶段 U-Net + Half Instance Normalization

### KBNet_s

```yaml
network_g:
  type: KBNet_s
  img_channel: 3
  width: 64
  middle_blk_num: 12
  enc_blk_nums: [2, 2, 4, 8]
  dec_blk_nums: [2, 2, 2, 2]
```

- 参数量：~16M
- 显存：256² + batch=8 ≈ 3.0GB
- 特点：Kernel Basis Attention（核心创新）

### KBNet_l

```yaml
network_g:
  type: KBNet_l
  inp_channels: 3
  out_channels: 3
  dim: 48
  num_blocks: [4, 6, 6, 8]
  num_refinement_blocks: 4
  heads: [1, 2, 4, 8]
  ffn_expansion_factor: 1.5
  bias: false
  blockname: KBBlock_l
```

- 参数量：~18M（大容量版本）
- 显存占用较高，建议 batch=4

### MIRNetv2

```yaml
network_g:
  type: MIRNet_v2
  inp_channels: 3
  out_channels: 3
  n_feat: 80
  chan_factor: 1.5
  n_RRG: 4
  n_MRB: 2
  height: 3
  width: 2
  bias: false
```

- 参数量：~5.9M（最轻量）
- 显存：256² + batch=8 ≈ 1.5GB
- 特点：多尺度残差融合（SKFF + 全局上下文模块）

### NAFNet

```yaml
network_g:
  type: NAFNet
  img_channel: 3
  width: 32                    # wf=32 轻量版, wf=64 原版(~17M)
  middle_blk_num: 12
  enc_blk_nums: [2, 2, 4, 8]
  dec_blk_nums: [2, 2, 2, 2]
```

- 参数量：~6M（width=32）/ ~17M（width=64）
- 显存：256² + batch=8 ≈ 1.8GB（width=32）
- 特点：无激活函数（SimpleGate 替换），纯 CNN 基线

### SCUNet

```yaml
network_g:
  type: SCUNet
  in_nc: 3
  config: [4, 4, 4, 4, 4, 4, 4]
  dim: 64
```

- 参数量：~12M
- 显存：256² + batch=8 ≈ 3.5GB
- 特点：CNN + Swin Transformer 混合（7 尺度 U-Net）
- 依赖：`einops`, `timm`

---

## 训练

### 基础训练

所有模型通过同一入口训练，仅 YAML 配置不同：

```bash
cd KBNet

python basicsr/train.py -opt options/train/MixUpsample/HINet.yml
python basicsr/train.py -opt options/train/MixUpsample/KBNet_s.yml
python basicsr/train.py -opt options/train/MixUpsample/KBNet_l.yml
python basicsr/train.py -opt options/train/MixUpsample/MIRNetv2.yml
python basicsr/train.py -opt options/train/MixUpsample/NAFNet.yml
python basicsr/train.py -opt options/train/MixUpsample/SCUNet.yml
```

### 多 GPU 训练

```bash
python -m torch.distributed.launch --nproc_per_node=4 basicsr/train.py \
    -opt options/train/MixUpsample/HINet.yml --launcher pytorch
```

### 自动恢复训练

框架会自动检测 `experiments/实验名/training_states/` 下的 latest state 并恢复：

```bash
python basicsr/train.py -opt options/train/MixUpsample/HINet.yml
# 检测到 training_states/ 后自动 resume
```

### YAML 配置位置

所有训练配置位于：`KBNet/options/train/MixUpsample/`

| 配置文件 | 模型 |
|---|---|
| `HINet.yml` | HINet |
| `KBNet_s.yml` | KBNet_s |
| `KBNet_l.yml` | KBNet_l |
| `MIRNetv2.yml` | MIRNetv2 |
| `NAFNet.yml` | NAFNet |
| `SCUNet.yml` | SCUNet |

---

## 参数调节指南

### 通用训练设置

```yaml
train:
  total_iter: 300000
  optim_g:
    type: AdamW
    lr: !!float 2e-4               # 可调范围 1e-4 ~ 3e-4
    weight_decay: !!float 1e-4
    betas: [0.9, 0.999]
  pixel_opt:
    type: L1Loss                   # 可替换为 PSNRLoss / CharbonnierLoss
    loss_weight: 1
    reduction: mean
  scheduler:
    type: CosineAnnealingRestartCyclicLR
    periods: [300000]
    restart_weights: [1]
    eta_mins: [1e-7]
```

### 关键参数说明

| 参数 | 建议范围 | 说明 |
|---|---|---|
| `batch_size_per_gpu` | 4~16 | T4 5GB: 4~8; 16GB: 16 |
| `lr` | 1e-4 ~ 3e-4 | AdamW 默认 2e-4 |
| `pixel_opt.type` | L1Loss / PSNRLoss / CharbonnierLoss | L1 最稳定 |
| `total_iter` | 200000~400000 | 300K 为基线 |
| `geometric_augs` | true / false | 建议开启随机翻转+旋转 |
| `mixup` | true / false | 数据量少时可尝试开启 |
| `use_grad_clip` | true / false | 建议开启（0.01） |

### Progressive Learning（可选）

如果希望在训练中从 128×128 逐渐增长到 256×256 裁剪尺寸：

```yaml
datasets:
  train:
    mini_batch_sizes: [8, 5, 3, 2, 1]
    iters: [92000, 64000, 48000, 36000, 60000]
    gt_size: 384
    gt_sizes: [128, 160, 192, 256, 320]
```

---

## 实验结果查看

训练中间结果默认保存在：

```text
KBNet/experiments/
└── 实验名/
    ├── models/             ← 权重 *.pth
    ├── training_states/    ← 恢复用的训练状态 *.state
    ├── visual_val/         ← 验证集输出图片
    └── logs/               ← TensorBoard 日志
```

### TensorBoard 查看

```bash
tensorboard --logdir KBNet/experiments/实验名/logs
```

---

## 测试/推理

### 使用预训练权重进行单图推理

```python
import torch
from basicsr.models.archs import define_network

# 定义网络结构
opt = {
    'type': 'NAFNet',
    'img_channel': 3,
    'width': 32,
    'middle_blk_num': 12,
    'enc_blk_nums': [2, 2, 4, 8],
    'dec_blk_nums': [2, 2, 2, 2],
}
net = define_network(opt)

# 加载权重
checkpoint = torch.load('experiments/实验名/models/net_g_latest.pth')
net.load_state_dict(checkpoint['params'] if 'params' in checkpoint else checkpoint)
net.eval()

# 推理
with torch.no_grad():
    output = net(input_tensor)  # input: [1, 3, H, W], output: [1, 3, H, W]
```

---

## 配置文件结构

完整的 YAML 结构参考（以 HINet 为例）：

```yaml
name: HINet-MixUpsample
model_type: ImageCleanModel
scale: 1
num_gpu: 1
manual_seed: 42

datasets:
  train:
    name: MixUpsample_train
    type: Dataset_PairedImage
    dataroot_gt: ../../data/MixUpsample_Dataset_R2A2_256/train/H
    dataroot_lq: ../../data/MixUpsample_Dataset_R2A2_256/train/L
    io_backend:          { type: disk }
    pair_by_index:       true
    geometric_augs:      true
    gt_size:             256
    use_shuffle:         true
    num_worker_per_gpu:  4
    batch_size_per_gpu:  8
    mini_batch_sizes:    [8]
    iters:               [300000]
    gt_sizes:            [256]
    dataset_enlarge_ratio: 1
    prefetch_mode:       ~
  val:
    name: MixUpsample_test
    type: Dataset_PairedImage
    dataroot_gt: ../../data/MixUpsample_Dataset_R2A2_256/test/H
    dataroot_lq: ../../data/MixUpsample_Dataset_R2A2_256/test/L
    io_backend:          { type: disk }
    pair_by_index:       true

network_g:
  type: HINet
  in_chn: 3
  wf: 64

path:
  pretrain_network_g:   ~
  strict_load_g:        true
  resume_state:         ~

train:
  total_iter:           300000
  warmup_iter:          -1
  use_grad_clip:        true
  scheduler:
    type: CosineAnnealingRestartCyclicLR
    periods:            [300000]
    restart_weights:    [1]
    eta_mins:           [1e-7]
  mixing_augs:
    mixup:              false
  optim_g:
    type: AdamW
    lr:                 !!float 2e-4
    weight_decay:       !!float 1e-4
    betas:              [0.9, 0.999]
  pixel_opt:
    type: L1Loss
    loss_weight:        1
    reduction:          mean

val:
  val_freq:             !!float 5e3
  save_img:             false
  rgb2bgr:              true
  use_image:            false
  max_minibatch:        8
  metrics:
    psnr:
      type: calculate_psnr
      crop_border:      0
      test_y_channel:   false
    ssim:
      type: calculate_ssim
      crop_border:      0
      test_y_channel:   false

logger:
  print_freq:           200
  save_checkpoint_freq: !!float 5e3
  use_tb_logger:        true
  wandb:
    project:            ~
    resume_id:          ~

dist_params:
  backend: nccl
  port: 29500
```
