#!/bin/bash
# ============================================================
# MixUpsample 批量测试 — 遍历 5 个模型 × 5 个 seed
# 使用方法：
#   从 training/ 目录运行：
#   bash options/test/MixUpsample/run_all_seeds.sh
# ============================================================

set -o pipefail

# 切换到 training/ 目录（脚本所在目录的 ../../..）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.." || { echo "无法切换到 training 目录"; exit 1; }
echo "当前目录: $(pwd)"

SEEDS=(42 43 44 45 46)
MODELS=("HINet" "KBNet_s" "NAFNet" "SCUNet" "MIRNetv2")
TEMPLATE_DIR="options/test/MixUpsample"

# 结果汇总
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY_FILE="experiments/test_summary_${RUN_TIMESTAMP}.csv"
CONSOLE_LOG_DIR="results/batch_test_logs_${RUN_TIMESTAMP}"
mkdir -p "$CONSOLE_LOG_DIR"

TEMP_FILES=()
FAILED_COUNT=0

cleanup() {
    local temp_file
    for temp_file in "${TEMP_FILES[@]}"; do
        rm -f "$temp_file"
    done
}

trap cleanup EXIT
trap 'exit 130' INT TERM

# 所有字段统一加双引号，避免模型名或错误信息中的逗号破坏 CSV。
csv_escape() {
    local value="$1"
    value="${value//$'\r'/ }"
    value="${value//$'\n'/ }"
    value="${value//\"/\"\"}"
    printf '"%s"' "$value"
}

append_csv_row() {
    local first=1
    local value
    for value in "$@"; do
        if [ "$first" -eq 0 ]; then
            printf ',' >> "$SUMMARY_FILE"
        fi
        csv_escape "$value" >> "$SUMMARY_FILE"
        first=0
    done
    printf '\n' >> "$SUMMARY_FILE"
}

last_nonempty_line() {
    awk 'NF { line=$0 } END { print line }' "$1"
}

append_csv_row "模型" "Seed" "Status" "PSNR" "SSIM" "Error"

for MODEL in "${MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo ""
        echo "========== 测试 ${MODEL} - Seed ${SEED} =========="

        TEMP_YML="${TEMPLATE_DIR}/temp_${MODEL}_S${SEED}.yml"
        TEMP_FILES+=("$TEMP_YML")
        CONSOLE_LOG="${CONSOLE_LOG_DIR}/${MODEL}_S${SEED}.log"

        # 复制模板 → 替换 {SEED}
        cp "${TEMPLATE_DIR}/${MODEL}.yml" "$TEMP_YML"
        sed -i "s/{SEED}/${SEED}/g" "$TEMP_YML"

        # tee 保留实时输出；PIPESTATUS[0] 是 Python 的真实退出状态。
        python basicsr/test.py -opt "$TEMP_YML" 2>&1 | tee "$CONSOLE_LOG"
        PYTHON_STATUS=${PIPESTATUS[0]}

        # 从输出日志提取 PSNR/SSIM
        EXP_NAME="test_${MODEL}-MixUpsample-${SEED}"
        LATEST_LOG=$(ls -t "results/${EXP_NAME}"/test_*.log 2>/dev/null | head -1)
        PSNR=""
        SSIM=""
        STATUS="SUCCESS"
        ERROR=""

        if [ "$PYTHON_STATUS" -ne 0 ]; then
            STATUS="FAILED"
            ERROR="$(last_nonempty_line "$CONSOLE_LOG")"
            if [ -z "$ERROR" ]; then
                ERROR="测试进程退出码为 ${PYTHON_STATUS}，且没有控制台输出"
            fi
        elif [ -z "$LATEST_LOG" ]; then
            STATUS="FAILED"
            ERROR="未找到测试日志: results/${EXP_NAME}/test_*.log"
        else
            PSNR=$(sed -nE 's/.*# psnr: ([^[:space:]]+).*/\1/p' "$LATEST_LOG" | tail -1)
            SSIM=$(sed -nE 's/.*# ssim: ([^[:space:]]+).*/\1/p' "$LATEST_LOG" | tail -1)
            if [ -z "$PSNR" ] || [ -z "$SSIM" ]; then
                STATUS="FAILED"
                ERROR="测试日志中缺少 psnr 或 ssim 指标"
            fi
        fi

        if [ "$STATUS" = "FAILED" ]; then
            FAILED_COUNT=$((FAILED_COUNT + 1))
        fi
        append_csv_row "$MODEL" "$SEED" "$STATUS" "${PSNR:-N/A}" \
            "${SSIM:-N/A}" "$ERROR"

        # 清理临时 YAML
        rm -f "$TEMP_YML"

        echo "  Status=${STATUS}  PSNR=${PSNR:-N/A}  SSIM=${SSIM:-N/A}"
        if [ -n "$ERROR" ]; then
            echo "  Error=${ERROR}"
        fi
        echo "========== ${MODEL} - Seed ${SEED} 结束 =========="
    done
done

echo ""
echo "========================================"
echo "批量测试结束：成功 $(( ${#MODELS[@]} * ${#SEEDS[@]} - FAILED_COUNT ))，失败 ${FAILED_COUNT}"
echo "汇总文件: $SUMMARY_FILE"
echo "控制台日志: $CONSOLE_LOG_DIR"
echo "========================================"
if command -v column >/dev/null 2>&1; then
    column -t -s ',' "$SUMMARY_FILE"
else
    cat "$SUMMARY_FILE"
fi

if [ "$FAILED_COUNT" -ne 0 ]; then
    exit 1
fi
