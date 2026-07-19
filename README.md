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
│   ├── options/test/MixUpsample/   ← 测试 YAML 模板
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

python -m basicsr.train -opt options/train/MixUpsample/HINet.yml
python -m basicsr.train -opt options/train/MixUpsample/KBNet_s.yml
python -m basicsr.train -opt options/train/MixUpsample/MIRNetv2.yml
python -m basicsr.train -opt options/train/MixUpsample/NAFNet.yml
python -m basicsr.train -opt options/train/MixUpsample/SCUNet.yml

# 指定GPU
CUDA_VISIBLE_DEVICES=3 python -m basicsr.train -opt options/train/MixUpsample/KBNet_s.yml
```

### 多 GPU（单模型多卡）

```bash
python -m torch.distributed.launch --nproc_per_node=4 -m basicsr.train \
    -opt options/train/MixUpsample/HINet.yml --launcher pytorch
```

### 双卡跑两个不同模型

Kaggle 等双卡环境，每张卡各跑一个模型：`%%bash`魔法命令作用是让整个nb块都是bash命令输入

```bash
%%bash
CUDA_VISIBLE_DEVICES=0 python -m basicsr.train -opt options/train/MixUpsample/NAFNet.yml > nafnet_train.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -m basicsr.train -opt options/train/MixUpsample/SCUNet.yml > scunet_train.log 2>&1 &
wait
echo "All training tasks finished."
```

如需后台防断连：

```bash
nohup bash -c '
cd training
CUDA_VISIBLE_DEVICES=0 python -m basicsr.train -opt options/train/MixUpsample/NAFNet.yml &
CUDA_VISIBLE_DEVICES=1 python -m basicsr.train -opt options/train/MixUpsample/SCUNet.yml &
wait
' > train.log 2>&1 &
```

### 自动恢复

检测到 `experiments/实验名/training_states/` 后自动 resume，无需额外参数。

---

## 测试

使用 `basicsr/test.py` 在 MixUpsample 测试集上批量评估 PSNR / SSIM。

### 测试 YAML 模板

每个模型对应一个 YAML 模板，位于 `options/test/MixUpsample/`。其 `pretrain_network_g` 路径和 `name` 字段使用 `{SEED}` 占位符，运行时替换为实际 seed：

```text
options/test/MixUpsample/
├── HINet.yml
├── KBNet_s.yml
├── NAFNet.yml
├── SCUNet.yml
├── MIRNetv2.yml
└── run_all_seeds.sh          ← 批量遍历脚本
```

### 批量测试（推荐）

遍历 5 个模型 × 5 个 seed（42~46），自动输出汇总 CSV：

```bash
cd training
bash options/test/MixUpsample/run_all_seeds.sh
```

脚本执行流程：
1. 复制模板 YAML → 替换 `{SEED}` 为实际数值 → 生成临时 YAML
2. 调用 `python basicsr/test.py -opt 临时文件`
3. 从 `training/results/` 日志提取 PSNR / SSIM → 追加到汇总 CSV
4. 清理临时 YAML

汇总 CSV 生成在 `training/experiments/test_summary_日期.csv`，包含 `Status` 和
`Error` 字段。单项失败不会中断后续测试，但脚本最终会返回非零状态。

### 单模型单 seed 测试

手动替换 seed 后运行：

```bash
cd training

# 替换 {SEED} 为 42，生成临时 YAML
sed "s/{SEED}/42/g" options/test/MixUpsample/HINet.yml > temp_HINet_S42.yml

# 执行测试
python basicsr/test.py -opt temp_HINet_S42.yml

# 清理临时文件
rm temp_HINet_S42.yml
```

测试结果和框架日志输出到 `results/test_模型名-MixUpsample-Seed/`；批量脚本的
逐项控制台日志输出到 `results/batch_test_logs_日期/`。

### 测试配置参数

| 项 | 说明 |
|---|---|
| `manual_seed` | 固定 42（推理不依赖随机种子） |
| `val.save_img` | 设为 `true` 可保存复原结果图到 visualization 目录 |
| `val.metrics` | PSNR + SSIM，`crop_border: 0`, `test_y_channel: false` |

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

## 训练 YAML 配置

```
training/options/train/MixUpsample/
├── HINet.yml
├── KBNet_s.yml
├── KBNet_l.yml
├── MIRNetv2.yml
├── NAFNet.yml
└── SCUNet.yml
```

## 测试 YAML 配置

```
training/options/test/MixUpsample/
├── HINet.yml
├── KBNet_s.yml
├── NAFNet.yml
├── SCUNet.yml
├── MIRNetv2.yml
└── run_all_seeds.sh          ← 批量遍历脚本
```
