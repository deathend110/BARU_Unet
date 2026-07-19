#!/bin/bash
# ============================================================
# MixUpsample 批量测试 — 遍历 5 个模型 × 5 个 seed
# 使用方法：
#   从 training/ 目录运行：
#   bash options/test/MixUpsample/run_all_seeds.sh
# ============================================================

# 切换到 training/ 目录（脚本所在目录的 ../../..）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.." || { echo "无法切换到 training 目录"; exit 1; }
echo "当前目录: $(pwd)"

SEEDS=(42 43 44 45 46)
MODELS=("HINet" "KBNet_s" "NAFNet" "SCUNet" "MIRNetv2")
TEMPLATE_DIR="options/test/MixUpsample"

# 结果汇总
SUMMARY_FILE="experiments/test_summary_$(date +%Y%m%d_%H%M%S).csv"
echo "模型,Seed,PSNR,SSIM" > "$SUMMARY_FILE"

for MODEL in "${MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo ""
        echo "========== 测试 ${MODEL} - Seed ${SEED} =========="

        TEMP_YML="temp_${MODEL}_S${SEED}.yml"

        # 复制模板 → 替换 {SEED}
        cp "${TEMPLATE_DIR}/${MODEL}.yml" "${TEMPLATE_DIR}/${TEMP_YML}"
        sed -i "s/{SEED}/${SEED}/g" "${TEMPLATE_DIR}/${TEMP_YML}"

        # 运行测试
        python basicsr/test.py -opt "${TEMPLATE_DIR}/${TEMP_YML}"

        # 从输出日志提取 PSNR/SSIM
        EXP_NAME="test_${MODEL}-MixUpsample-${SEED}"
        LATEST_LOG=$(ls -t "experiments/${EXP_NAME}/log/" 2>/dev/null | head -1)
        PSNR=""
        SSIM=""
        if [ -n "$LATEST_LOG" ]; then
            PSNR=$(grep -oP 'PSNR: \K[\d.]+' "experiments/${EXP_NAME}/log/${LATEST_LOG}" | tail -1)
            SSIM=$(grep -oP 'SSIM: \K[\d.]+' "experiments/${EXP_NAME}/log/${LATEST_LOG}" | tail -1)
        fi
        echo "${MODEL},${SEED},${PSNR:-N/A},${SSIM:-N/A}" >> "$SUMMARY_FILE"

        # 清理临时 YAML
        rm -f "${TEMPLATE_DIR}/${TEMP_YML}"

        echo "  PSNR=${PSNR:-N/A}  SSIM=${SSIM:-N/A}"
        echo "========== ${MODEL} - Seed ${SEED} 完成 =========="
    done
done

echo ""
echo "========================================"
echo "全部测试完成！汇总文件: $SUMMARY_FILE"
echo "========================================"
column -t -s ',' "$SUMMARY_FILE"
