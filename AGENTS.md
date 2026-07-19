# 仓库协作规则

## 代码编写规则

1. 用户没有明确要求时，不主动编写或修改代码。
2. 输出代码时关键步骤、核心变量或不直观的设计意图加中文注释，不添加逐行注释噪声。
3. 每次改动优先 `git commit` 并用明确的中文说明改动内容。
4. 新建文件一律使用UTF-8编码

---

## 仓库结构

- **`training/`** — 统一训练框架入口。所有训练命令从该目录执行。
- **`data/`** — 数据集目录，与 `training/` 同级。
- **`z_paper/`** — 论文 PDF，参考用。
- **`模型区分.md`** — 模型对比表。
- **`training/Denoising/`、`training/Deraining/`、`training/Defocus_Deblurring/`** — 原始 KBNet 遗留目录，统一框架中不使用。

---

## 安装

```bash
cd training
pip install -r requirements.txt
pip install -e . --no_cuda_ext
```

必须带 `--no_cuda_ext` 跳过 CUDA kernel 编译（当前 5 个模型均不需要 deform_conv / fused_act 等自定义 CUDA op）。

---

## 训练

所有模型通过同一入口，YAML 切换：

```bash
cd training
python -m basicsr.train -opt options/train/MixUpsample/HINet.yml
```

YAML 位于 `training/options/train/MixUpsample/`，共 6 个（HINet / KBNet_s / KBNet_l / MIRNetv2 / NAFNet / SCUNet）。

关键细节：
- **Arch 自动注册**：`basicsr/models/archs/*_arch.py` 放进去即可，无需手动 import。
- **自动恢复**：检测到 `experiments/{name}/training_states/` 后自动 resume。
- **多 GPU**：加 `--launcher pytorch` 和 `torch.distributed.launch`。

---

## 数据集特殊性

- 文件名含 `_H_`/`_L_` 后缀（`xxx_H_TR.png` ↔ `xxx_L_TR.png`），YAML 中必须加 `pair_by_index: true`。
- 灰度 PNG 被 `cv2.IMREAD_COLOR` 自动转为 3 通道 BGR，模型均保持 `in_chn=3`。

---

## 新增模型

1. 将 `.py` 放入 `training/basicsr/models/archs/`，文件名必须以 `_arch.py` 结尾。
2. `training/options/train/MixUpsample/` 下新增对应 YAML。
3. 如引入新依赖，更新 `training/requirements.txt`。

## 运行环境

1. codex运行在windows，GPU 3050ti laptop 4GB
2. 实际训练环境是服务器4090 24GB