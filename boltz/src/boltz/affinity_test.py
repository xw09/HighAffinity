import os
import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import click
import torch
from pytorch_lightning import Trainer

import sys

print("Python module search paths:")
for p in sys.path:
    print(p)

from boltz.data.types import Manifest
from boltz.data.write.writer import BoltzAffinityWriter
from boltz.model.models.boltz2_test import Boltz2
from boltz.data.feature.featurizerv2 import Boltz2Featurizer
from boltz.data.tokenize.boltz2 import Boltz2Tokenizer

# 这是你刚才那份只保留 predict_dataloader 的 DataModule 定义所在文件
from boltz.data.module.trainingv2_test import (
    Boltz2InferenceDataModule,
    DataConfig,
    DatasetConfig,
    DefaultUniformSampler,  # 虽然预测不用 sampler，但字段需要一个占位
)

# 固定使用第二张 GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

@dataclass
class BoltzProcessedInput:
    """Processed input data."""

    manifest: Manifest
    targets_dir: Path
    msa_dir: Path
    constraints_dir: Optional[Path] = None
    template_dir: Optional[Path] = None
    extra_mols_dir: Optional[Path] = None

@dataclass
class Boltz2DiffusionParams:
    """Diffusion process parameters."""

    gamma_0: float = 0.8
    gamma_min: float = 1.0
    noise_scale: float = 1.003
    rho: float = 7
    step_scale: float = 1.5
    sigma_min: float = 0.0001
    sigma_max: float = 160.0
    sigma_data: float = 16.0
    P_mean: float = -1.2
    P_std: float = 1.5
    coordinate_augmentation: bool = True
    alignment_reverse_diff: bool = True
    synchronize_sigmas: bool = True

@dataclass
class PairformerArgsV2:
    """Pairformer arguments."""

    num_blocks: int = 64
    num_heads: int = 16
    dropout: float = 0.0
    activation_checkpointing: bool = False
    offload_to_cpu: bool = False
    v2: bool = True

@dataclass
class MSAModuleArgs:
    """MSA module arguments."""

    msa_s: int = 64
    msa_blocks: int = 4
    msa_dropout: float = 0.0
    z_dropout: float = 0.0
    use_paired_feature: bool = True
    pairwise_head_width: int = 32
    pairwise_num_heads: int = 4
    activation_checkpointing: bool = False
    offload_to_cpu: bool = False
    subsample_msa: bool = False
    num_subsampled_msa: int = 1024


def prepare_data(out_dir: str, cache: str, num_workers: int = 8) -> DataConfig:
    """
    读取 out_dir/processed 下的数据，构造用于预测的 DataConfig。
    - target_dir -> out_dir / predictions  (亲和力输出目录)
    - msa_dir    -> processed/msa
    - mol_dir    -> ~/.boltz/mols
    """
    cache = Path(cache).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    mol_dir = cache / "mols"

    out_dir = Path(out_dir).expanduser()
    processed_dir = out_dir / "processed"
    manifest_path = processed_dir / "manifest.json"

    ds_cfg = DatasetConfig(
        target_dir=str(out_dir / "predictions"),
        msa_dir=str(processed_dir / "msa"),
        mol_dir=str(mol_dir),
        num_workers=num_workers,
        sampler=DefaultUniformSampler(),  # 预测不会用到，但字段保留
        constraints_dir=(processed_dir / "constraints" if (processed_dir / "constraints").exists() else None),
        template_dir=(processed_dir / "templates" if (processed_dir / "templates").exists() else None),
        extra_mols_dir=(processed_dir / "mols" if (processed_dir / "mols").exists() else None),
        override_method=None,
        affinity=True,
        manifest_path=str(manifest_path),
        filters=None,
        split=None,  # 预测不再做 train/val 划分
    )

    data_cfg = DataConfig(
        datasets=[ds_cfg],
        filters=[],  # 若有 DynamicFilter，可加在这里；预测一般用不上
        featurizer=Boltz2Featurizer(),
        tokenizer=Boltz2Tokenizer(),
        val_batch_size=1,  # 在 DataModule 里会作为 predict batch size 使用
    )
    return data_cfg

def train_affinity_finetune(                 
    out_dir,
    cache,
    batch_size,
    pretrained_ckpt=None,
    devices=1,
    precision: str = "bf16-mixed",
    num_workers: int = 8,
    sampling_steps_affinity: int = 200,
    diffusion_samples_affinity: int = 1,
    subsample_msa: bool = False,
    num_subsampled_msa: int = 1024,
    affinity_mw_correction: Optional[bool] = False,
    accelerator: str = "gpu",
    **kwargs,  # 接收多余的旧参数（比如 max_epochs、ckpt_path、val_ratio），直接忽略
):
    # Print header
    click.echo("\nPredicting property: affinity\n")

    out_dir = Path(out_dir).expanduser()
    cache   = Path(cache).expanduser()
    processed_dir = out_dir / "processed"
    mol_dir = cache / "mols"

    # writer（可选，用于记录预测/评分）（训练阶段不需要写预测产物）
    pred_writer = BoltzAffinityWriter(
        data_dir=processed_dir / "structures",
        output_dir=out_dir / "predictions",
    )

    # Load manifest
    manifest = Manifest.load(out_dir / "processed" / "manifest.json")

    msg = f"Running affinity predict for {len(manifest.records)} input"
    msg += "s." if len(manifest.records) > 1 else "."
    click.echo(msg)

    data_cfg = prepare_data(out_dir=out_dir, cache=cache, num_workers=num_workers)
    data_module = Boltz2InferenceDataModule(data_cfg)
    data_module.batchsize = batch_size
    data_module.num_workers = num_workers

    "@@@结构/亲和力差别data_module:会加载带配体的结构预测结果；裁剪 pocket 区域，仅保留“有用的”配体和邻近残基;"
    "多的特征:deletion_mean_affinity, profile_affinity===结构预测的deletion_mean和profile,直接用就行"
    "affinity_mw:对输入复合物的规模做归一化"

    predict_affinity_args = {
        "recycling_steps": 5,
        "sampling_steps": sampling_steps_affinity,
        "diffusion_samples": diffusion_samples_affinity,
        "max_parallel_samples": 1,
        "write_confidence_summary": False,
        "write_full_pae": False,
        "write_full_pde": False,
    }

    # Load affinity model
    affinity_checkpoint = pretrained_ckpt  
    # affinity_checkpoint = cache / "boltz2_aff.ckpt"

    # Set up model parameters,boltz2
    diffusion_params = Boltz2DiffusionParams()
    pairformer_args = PairformerArgsV2()

    msa_args = MSAModuleArgs(
        subsample_msa=subsample_msa,
        num_subsampled_msa=num_subsampled_msa,
        use_paired_feature=True, 
    )

    model_module = Boltz2.load_from_checkpoint(
        affinity_checkpoint,
        strict=True,
        predict_args=predict_affinity_args,
        map_location="cpu",
        diffusion_process_args=asdict(diffusion_params),
        ema=False,
        pairformer_args=asdict(pairformer_args),
        msa_args=asdict(msa_args),
        steering_args={"fk_steering": False, "guidance_update": False},
        affinity_mw_correction=affinity_mw_correction,
        skip_run_structure=True,   # ★ 不再调用 structure_module.sample
    )

    # --- 补齐校验/训练用的 flag，避免 AttributeError ---
    if not hasattr(model_module, "validate_structure"):
        model_module.validate_structure = False
    if not hasattr(model_module, "validate_affinity"):
        model_module.validate_affinity = True
    if not hasattr(model_module, "training_affinity"):
        model_module.training_affinity = True   # 若你的 training_step 也用到
    # --------------------------------------------------

    model_module.eval()
    model_module.requires_grad_(False)

    # Set up trainer
    trainer = Trainer(
        default_root_dir=out_dir,
        strategy="auto",
        accelerator=accelerator,
        devices=devices,
        logger=False,            # 不再写 TensorBoard / CSV 日志
        precision=precision,  
    )

    trainer.callbacks[0] = pred_writer

    

    # 启动预测过程
    trainer.predict(
        model_module,
        datamodule=data_module,
        return_predictions=False,
    )


if __name__ == "__main__":
    # train_affinity_finetune(
    #     out_dir="/home/fuxin/lab/xw/boltz/a_zcy_test/MDM2_results/boltz_results_LHT_2",     # 结构 & processed 所在目录
    #     cache="/home/fuxin/.boltz",                             # ~/.boltz，含 mols & ckpt
    #     batch_size=1,
    #     # pretrained_ckpt="/home/fuxin/.boltz/boltz2_aff.ckpt",
    #     pretrained_ckpt="/home/fuxin/lab/xw/boltz/out_train_18429_3_1/checkpoints/epoch1-valloss0.2688.ckpt",   # 或你自己的 finetune ckpt
    #     devices=1,
    #     precision="bf16-mixed",
    #     num_workers=8,
    #     sampling_steps_affinity=200,
    #     diffusion_samples_affinity=1,
    #     subsample_msa=False,
    #     num_subsampled_msa=1024,
    #     affinity_mw_correction=False,
    #     # 下面这些旧参数传进来也不会报错（被 **kwargs 吃掉且忽略）
    #     ckpt_path="/home/fuxin/lab/xw/boltz/out_test",
    #     max_epochs=5,
    #     val_ratio=0.2,
    #     validation_only=False,
    # )
    # 1. 定义参数解析器
    parser = argparse.ArgumentParser(description="Run affinity test on a specific directory.")
    parser.add_argument(
        "--out_dir", 
        type=str, 
        required=True, 
        help="Path to the result directory (e.g., .../boltz_results_LHT_2)"
    )
    
    # 解析命令行参数
    args = parser.parse_args()

    # 2. 调用函数，将 out_dir 替换为 args.out_dir
    train_affinity_finetune(
        out_dir=args.out_dir,                                    # <--- 这里修改为动态接收
        cache="/home/fuxin/.boltz",
        batch_size=1,
        # pretrained_ckpt="/home/fuxin/.boltz/boltz2_aff.ckpt",
        pretrained_ckpt="/home/fuxin/lab/xw/boltz/out_train_18429_3_1/checkpoints/epoch1-valloss0.2688.ckpt",
        devices=1,
        precision="bf16-mixed",
        num_workers=8,
        sampling_steps_affinity=200,
        diffusion_samples_affinity=1,
        subsample_msa=False,
        num_subsampled_msa=1024,
        affinity_mw_correction=False,
        # 下面这些旧参数传进来也不会报错
        ckpt_path="/home/fuxin/lab/xw/boltz/out_test",
        max_epochs=5,
        val_ratio=0.2,
        validation_only=False,
    )
