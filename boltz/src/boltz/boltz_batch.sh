#!/bin/bash

# ================= 配置区域 =================
# boltz 可执行文件路径
BOLTZ_PATH="/home/fuxin/anaconda3/envs/boltz_xw/bin/boltz"
# 存放 yaml 文件的根目录 (输入)
RESULT_YAMLS="/home/fuxin/lab/xw/boltz/a_zcy_test/klk4"
# 输出目录
RESULTS_FINAL_DIR="/home/fuxin/lab/xw/boltz/a_zcy_test/klk4_results"

# 指定显卡
export CUDA_VISIBLE_DEVICES="3"

# ================= 脚本逻辑 =================

# 获取目录下所有的 .yaml 文件
all_yaml_files=("$RESULT_YAMLS"/*.yaml)
total_files=${#all_yaml_files[@]}

if [ "$total_files" -eq 0 ]; then
    echo "错误：在 $RESULT_YAMLS 中未找到任何 .yaml 文件。"
    exit 1
fi

skip_count=0
run_count=0

# 遍历每一个 yaml 文件
for yf in "${all_yaml_files[@]}"; do
    # 获取文件名（如: p53_AF2_1.yaml）
    base="$(basename "$yf")"
    # 获取不带后缀的文件名（如: p53_AF2_1），用作输出文件夹名称
    name="${base%.*}"
    
    # ==== 检查若目标 CIF 已存在则跳过 ====
    # 假设 Boltz 的输出路径结构为: out_dir/boltz_results_<name>/predictions/<name>/<name>_model_0.cif
    cif_path="$RESULTS_FINAL_DIR/boltz_results_${name}/predictions/${name}/${name}_model_0.cif"
    
    if [ -f "$cif_path" ]; then
        echo "[SKIP] 已存在结果：$base -> 跳过"
        ((skip_count++))
        continue
    fi
    # =======================================

    # 打印当前进度
    current_idx=$((skip_count + run_count + 1))
    echo "[RUNNING] ($current_idx / $total_files) 正在预测: $base"

    # 运行 Boltz
    "$BOLTZ_PATH" predict "$yf" \
        --out_dir "$RESULTS_FINAL_DIR" \
        --write_full_pde \
        --use_msa_server \
        # --seed "$name" # 如果需要指定种子可以取消注释，但通常不需要

    # 简单的错误检查
    if [ $? -eq 0 ]; then
        ((run_count++))
    else
        echo "[ERROR] 文件 $base 运行出错！"
    fi
done

echo "运行结束，总文件数: $total_files，跳过$skip_count 个文件，共处理 $run_count 个文件。"
