# BARU_Unet — 图像复原统一训练框架

以 U-Net 架构为核心的 SAR 图像复原模型对比框架。集成 5 个主流模型，统一 `training/` 入口。

---

## 目录结构

```
BARU_Unet/
├── training/                 ← 统一训练框架（基于 BasicSR）
│   ├── basicsr/
│   │   ├── train.py         ← 训练入口
│   │   ├── models/archs/   ← 5 个模型的网络架构
│   │   ├── data/           ← 数据集 + dataloader
│   │   ├── utils/          ← 工具函数
│   │   └── metrics/        ← 评估指标
│   ├── options/train/MixUpsample/  ← 训练 YAML 配置
│   ├── requirements.txt
│   ├── setup.py
│   └── VERSION
├── data/
│   └── MixUpsample_Dataset_R2A2_256/  ← 数据集
│       ├── train/H/   L/              ← 训练集 4704 对
│       └── test/H/    L/              ← 测试集 1188 对
├── z_paper/                 ← 论文 PDF
├── 模型区分.md               ← 模型对比资料
└── AGENTS.md                ← 仓库协作规则
```

---

## 环境安装

```bash
cd training

# 安装依赖
pip install -r requirements.txt

# 安装框架（开发模式）
pip install -e . --no-build-isolation
```

---

## 模型一览

| 模型 | 年份 | 参数量 | 显存 (256², bs=8) |
|---|---|---|---|
| **HINet** | CVPRW 2021 | 8.7M | ~2.5GB |
| **MIRNetv2** | TPAMI 2022 | 5.9M | ~1.5GB |
| **NAFNet** | ECCV 2022 | 6~17M | ~1.8GB |
| **KBNet** | ICCV 2023 | 16~18M | ~3.0GB |
| **SCUNet** | ICCV 2023 | 12~18M | ~3.5GB |

详细对比见 [`模型区分.md`](模型区分.md)。

---

## 数据集放置

```text
data/MixUpsample_Dataset_R2A2_256/
├── train/
│   ├── H/       ← GT（高分辨率） *.png, 256×256, 灰度
│   └── L/       ← LQ（低分辨率） *.png, 256×256, 灰度
└── test/
    ├── H/       ← GT              *.png, 256×256, 灰度
    └── L/       ← LQ              *.png, 256×256, 灰度
```

文件名包含 `_H_`/`_L_` 后缀配对（如 `xxx_H_TR.png` ↔ `xxx_L_TR.png`），框架通过 `pair_by_index: true` 按排序索引配对。

---

## 训练

```bash
cd training

python basicsr/train.py -opt options/train/MixUpsample/HINet.yml
python basicsr/train.py -opt options/train/MixUpsample/KBNet_s.yml
python basicsr/train.py -opt options/train/MixUpsample/KBNet_l.yml
python basicsr/train.py -opt options/train/MixUpsample/MIRNetv2.yml
python basicsr/train.py -opt options/train/MixUpsample/NAFNet.yml
python basicsr/train.py -opt options/train/MixUpsample/SCUNet.yml
```

### 多 GPU

```bash
python -m torch.distributed.launch --nproc_per_node=4 basicsr/train.py \
    -opt options/train/MixUpsample/HINet.yml --launcher pytorch
```

### 自动恢复

检测到 `experiments/实验名/training_states/` 后自动 resume，无需额外参数。

---

## 模型配置

### HINet

```yaml
network_g:
  type: HINet
  in_chn: 3
  wf: 64
  hin_position_left: 0
  hin_position_right: 4
```

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

### KBNet_l

```yaml
network_g:
  type: KBNet_l
  inp_channels: 3
  out_channels: 3
  dim: 48
  num_blocks: [4, 6, 6, 8]
  heads: [1, 2, 4, 8]
```

### MIRNetv2

```yaml
network_g:
  type: MIRNet_v2
  inp_channels: 3
  out_channels: 3
  n_feat: 80
  n_RRG: 4
  n_MRB: 2
```

### NAFNet

```yaml
network_g:
  type: NAFNet
  img_channel: 3
  width: 32
  middle_blk_num: 12
  enc_blk_nums: [2, 2, 4, 8]
  dec_blk_nums: [2, 2, 2, 2]
```

### SCUNet

```yaml
network_g:
  type: SCUNet
  in_nc: 3
  config: [4, 4, 4, 4, 4, 4, 4]
  dim: 64
```

---

## 参数调节

```yaml
train:
  total_iter:     300000
  optim_g:
    type:         AdamW
    lr:           !!float 2e-4    # 1e-4 ~ 3e-4
    weight_decay: !!float 1e-4
    betas:        [0.9, 0.999]
  pixel_opt:
    type:         L1Loss          # 也可用 PSNRLoss / CharbonnierLoss
    loss_weight:  1
    reduction:    mean
  scheduler:
    type: CosineAnnealingRestartCyclicLR
    periods:      [300000]
    eta_mins:     [1e-7]
```

| 参数 | 建议 | 说明 |
|---|---|---|
| `batch_size_per_gpu` | 4~16 | 根据显存调节 |
| `lr` | 1e-4~3e-4 | AdamW |
| `total_iter` | 200K~400K | 300K 基线 |
| `geometric_augs` | true | 随机翻转+旋转 |

---

## 结果查看

```
training/experiments/实验名/
├── models/             ← 权重 *.pth
├── training_states/    ← 恢复 *.state
├── visual_val/         ← 验证输出
└── logs/               ← TensorBoard
```

```bash
tensorboard --logdir training/experiments/实验名/logs
```

---

## 推理

```python
import torch
from basicsr.models.archs import define_network

opt = {
    'type': 'NAFNet', 'img_channel': 3, 'width': 32,
    'middle_blk_num': 12, 'enc_blk_nums': [2,2,4,8],
    'dec_blk_nums': [2,2,2,2],
}
net = define_network(opt)
net.load_state_dict(torch.load('experiments/实验名/models/net_g_latest.pth')['params'])
net.eval()

with torch.no_grad():
    out = net(inp)  # inp: [1,3,H,W], out: [1,3,H,W]
```

---

## YAML 配置位于

```
training/options/train/MixUpsample/
├── HINet.yml
├── KBNet_s.yml
├── KBNet_l.yml
├── MIRNetv2.yml
├── NAFNet.yml
└── SCUNet.yml
```
