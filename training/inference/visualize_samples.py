"""
MixUpsample 测试集分层抽样推理可视化脚本

功能:
  1. 按场景(Bangkok/city1/city2/filed/port/SAR/suburb)分层抽样
  2. 对每张图运行模型推理
  3. 保存 H(左) | L(中) | result(右) 三联图，标注 PSNR/SSIM
  4. 每场景生成一张组合图 + 一张总览图

用法:
  cd training
  python inference/visualize_samples.py \
      --model HINet \
      --checkpoint experiments/HINet-MixUpsample-Controlled-S42-4090/models/net_g_latest.pth \
      --samples_per_scene 10 \
      --output_dir ../visualization/samples \
      --seed 42
"""

import argparse
import os
import random
import sys
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch

# ---------- 将 training/ 加入 path，以便导入 basicsr ----------
SCRIPT_DIR = Path(__file__).resolve().parent  # training/inference/
TRAINING_DIR = SCRIPT_DIR.parent  # training/
sys.path.insert(0, str(TRAINING_DIR))

import yaml
from basicsr.models.archs import define_network
from basicsr.metrics import calculate_psnr, calculate_ssim
from basicsr.utils.img_util import tensor2img


def parse_args():
    parser = argparse.ArgumentParser(
        description="MixUpsample 分层抽样推理可视化")
    parser.add_argument("--model", type=str, default="HINet",
                        choices=["HINet", "KBNet_s", "NAFNet", "SCUNet",
                                 "MIRNetv2", "MIRNet_v2"],
                        help="模型名称（对应 options/test/MixUpsample/ 下的 YAML）")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="权重路径，如 experiments/XXX/models/net_g_latest.pth")
    parser.add_argument("--samples_per_scene", type=int, default=10,
                        help="每场景抽样张数（默认 10）")
    parser.add_argument("--output_dir", type=str,
                        default="../visualization/samples",
                        help="可视化输出目录")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子（默认 42）")
    parser.add_argument("--test_dir", type=str,
                        default="../data/MixUpsample_Dataset_R2A2_256/test",
                        help="测试集目录")
    parser.add_argument("--device", type=str, default="cuda",
                        help="推理设备（cuda / cpu）")
    return parser.parse_args()


def load_network_from_yaml(model_name, checkpoint_path, device):
    """从测试 YAML 模板读取网络配置，构建网络并加载权重"""
    # MIRNet_v2 是历史 CLI 写法，配置文件统一使用 MIRNetv2.yml。
    config_name = "MIRNetv2" if model_name == "MIRNet_v2" else model_name
    yaml_path = TRAINING_DIR / "options/test/MixUpsample" / f"{config_name}.yml"
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"找不到模型 {model_name} 的测试配置: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        # 先读取纯文本，替换 {SEED} 占位符
        raw = f.read().replace("{SEED}", "0")  # 种子不影响网络结构

    opt = yaml.safe_load(raw)
    network_opt = opt["network_g"]

    # 构建网络
    net = define_network(network_opt)
    net.to(device)

    # 加载权重
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "params" in ckpt:
        net.load_state_dict(ckpt["params"], strict=True)
        print(f"  加载权重（'params' key）: {checkpoint_path}")
    elif "state_dict" in ckpt:
        net.load_state_dict(ckpt["state_dict"], strict=True)
        print(f"  加载权重（'state_dict' key）: {checkpoint_path}")
    else:
        net.load_state_dict(ckpt, strict=True)
        print(f"  加载权重（直接 state_dict）: {checkpoint_path}")

    net.eval()
    return net


def discover_scenes(test_dir):
    """扫描测试集 H 目录，按场景分类文件名列表"""
    h_dir = Path(test_dir) / "H"
    l_dir = Path(test_dir) / "L"

    scenes = defaultdict(list)
    for f in sorted(h_dir.iterdir()):
        if not f.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
            continue
        # 文件名: Bangkok_1_r0001_p0001_H_BL.png
        scene_name = f.stem.split("_")[0]
        # L 文件路径: 将 _H_ 替换为 _L_
        l_name = f.name.replace("_H_", "_L_")
        l_path = l_dir / l_name
        if l_path.exists():
            scenes[scene_name].append((str(f), str(l_path)))

    return scenes


def sample_files(scenes, samples_per_scene, seed):
    """每场景随机抽样"""
    random.seed(seed)
    sampled = {}
    for scene, file_pairs in sorted(scenes.items()):
        if len(file_pairs) <= samples_per_scene:
            sampled[scene] = file_pairs
            print(f"  {scene}: 共 {len(file_pairs)} 张，全部选用")
        else:
            sampled[scene] = random.sample(file_pairs, samples_per_scene)
            print(f"  {scene}: 共 {len(file_pairs)} 张，抽样 {samples_per_scene} 张")
    return sampled


def read_image(path):
    """读取图像 → BGR HWC uint8"""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取: {path}")
    return img  # HWC, BGR, uint8 [0,255]


def preprocess(img_bgr, device):
    """BGR uint8 → RGB float32 tensor [1,3,H,W]"""
    img = img_bgr.astype(np.float32) / 255.0  # [0,1]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR→RGB
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor


def postprocess(tensor):
    """RGB tensor [0,1] → BGR uint8 HWC"""
    img = tensor2img(tensor, rgb2bgr=True)  # 返回 uint8 [0,255], HWC, BGR
    return img


def make_triplet(h_img, l_img, result_img, psnr_val, ssim_val, scene_name, file_label):
    """拼接 H | L | Result 三联图，标注指标"""
    h1, w1 = h_img.shape[:2]
    h2, w2 = l_img.shape[:2]
    h3, w3 = result_img.shape[:2]

    # 统一高度
    max_h = max(h1, h2, h3)
    if h1 < max_h:
        pad = np.zeros((max_h - h1, w1, 3), dtype=np.uint8)
        h_img = np.vstack([h_img, pad])
    if h2 < max_h:
        pad = np.zeros((max_h - h2, w2, 3), dtype=np.uint8)
        l_img = np.vstack([l_img, pad])
    if h3 < max_h:
        pad = np.zeros((max_h - h3, w3, 3), dtype=np.uint8)
        result_img = np.vstack([result_img, pad])

    # 水平拼接
    gap = np.ones((max_h, 4, 3), dtype=np.uint8) * 255
    triplet = np.hstack([h_img, gap, l_img, gap, result_img])

    # 添加标注文字
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    color = (255, 255, 255)  # 白色文字

    # 顶部标签
    labels = ["GT (H)", "Input (L)", f"Output ({psnr_val:.2f}dB / {ssim_val:.4f})"]
    positions = [4, w1 + 8, w1 + w2 + 12]
    for label, x in zip(labels, positions):
        # 加黑色阴影提高可读性
        cv2.putText(triplet, label, (x + 1, 22), font, font_scale, (0, 0, 0), thickness + 1)
        cv2.putText(triplet, label, (x, 21), font, font_scale, color, thickness)

    # 底部文件名标注
    file_info = f"{scene_name} | {file_label}"
    cv2.putText(triplet, file_info, (5, max_h - 5), font, 0.45, (0, 0, 0), 3)
    cv2.putText(triplet, file_info, (5, max_h - 5), font, 0.45, (200, 200, 200), 1)

    return triplet


def build_scene_composite(rows, scene_name, max_row_width):
    """将一个场景的多行结果拼成一张大图"""
    if not rows:
        return None

    # 统一每行宽度（右边填充白色）
    padded_rows = []
    for row in rows:
        h, w = row.shape[:2]
        if w < max_row_width:
            pad = np.ones((h, max_row_width - w, 3), dtype=np.uint8) * 240
            row = np.hstack([row, pad])
        padded_rows.append(row)

    # 垂直拼接，行间加白色间隙
    gap_h = np.ones((6, max_row_width, 3), dtype=np.uint8) * 255
    composite = padded_rows[0]
    for row in padded_rows[1:]:
        composite = np.vstack([composite, gap_h, row])

    return composite


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    print(f"使用设备: {device}")
    print(f"模型: {args.model}")
    print(f"每场景抽样: {args.samples_per_scene} 张")

    # ---------- 1. 构建网络 ----------
    print("\n[1/5] 构建网络...")
    net = load_network_from_yaml(args.model, args.checkpoint, device)
    print(f"  网络: {args.model}")

    # ---------- 2. 扫描测试集场景 ----------
    print("\n[2/5] 扫描测试集场景...")
    test_dir = (TRAINING_DIR / args.test_dir).resolve()
    scenes = discover_scenes(str(test_dir))
    print(f"  发现 {len(scenes)} 个场景:")
    for s, pairs in sorted(scenes.items()):
        print(f"    {s}: {len(pairs)} 张")

    # ---------- 3. 分层抽样 ----------
    print("\n[3/5] 分层抽样...")
    sampled = sample_files(scenes, args.samples_per_scene, args.seed)
    total = sum(len(v) for v in sampled.values())
    print(f"  共抽样 {total} 张")

    # ---------- 4. 推理 ----------
    print("\n[4/5] 推理中...")
    output_dir = (TRAINING_DIR / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_scene_rows = {}  # scene -> list of triplet images
    for scene, file_pairs in sorted(sampled.items()):
        all_scene_rows[scene] = []
        for h_path, l_path in file_pairs:
            # 读取
            h_img = read_image(h_path)
            l_img = read_image(l_path)

            # 预处理
            h_tensor = preprocess(h_img, device)
            l_tensor = preprocess(l_img, device)

            # 推理
            with torch.no_grad():
                output_tensor = net(l_tensor)
                if isinstance(output_tensor, (list, tuple)):
                    output_tensor = output_tensor[-1]

            # 后处理
            result_img = postprocess(output_tensor)

            # 计算指标
            psnr_val = calculate_psnr(
                output_tensor, h_tensor,
                crop_border=0, test_y_channel=False)
            ssim_val = calculate_ssim(
                output_tensor, h_tensor,
                crop_border=0, test_y_channel=False)

            # 文件名标签（不含 _H_ 部分）
            base_name = Path(h_path).stem.replace("_H_", "_")

            # 构建三联图
            triplet = make_triplet(h_img, l_img, result_img,
                                   psnr_val, ssim_val, scene, base_name)
            all_scene_rows[scene].append(triplet)

            print(f"  {scene}/{base_name}: PSNR={psnr_val:.4f}  SSIM={ssim_val:.4f}")

    # ---------- 5. 保存可视化 ----------
    print("\n[5/5] 保存可视化...")

    # 5a. 每场景一张组合图
    scene_composites = []
    for scene, rows in sorted(all_scene_rows.items()):
        # 找该场景最大宽度
        max_w = max(r.shape[1] for r in rows) if rows else 0
        composite = build_scene_composite(rows, scene, max_w)
        if composite is not None:
            scene_path = output_dir / f"scene_{scene}.png"
            cv2.imwrite(str(scene_path), composite)
            scene_composites.append((scene, composite))
            print(f"  [场景] {scene}: {scene_path}")

    # 5b. 总览图：所有场景缩略图汇总
    # 将每个场景组合图缩放到统一宽度后垂直拼接
    if scene_composites:
        overview_width = 900
        overview_rows = []
        for scene, comp in scene_composites:
            h, w = comp.shape[:2]
            scale = overview_width / w
            new_h = int(h * scale)
            thumb = cv2.resize(comp, (overview_width, new_h),
                               interpolation=cv2.INTER_AREA)
            overview_rows.append(thumb)

        # 垂直拼接
        overview = overview_rows[0]
        for row in overview_rows[1:]:
            gap = np.ones((4, overview_width, 3), dtype=np.uint8) * 255
            overview = np.vstack([overview, gap, row])

        overview_path = output_dir / "overview.png"
        cv2.imwrite(str(overview_path), overview)
        print(f"  [总览] {overview_path}")

    print(f"\n✅ 完成！所有可视化保存至: {output_dir}")


if __name__ == "__main__":
    main()
