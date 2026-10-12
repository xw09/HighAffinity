import multiprocessing
import os
import pickle
import platform
import tarfile
import urllib.request
import warnings
from dataclasses import asdict, dataclass
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Literal, Optional

import click
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from rdkit import Chem
from tqdm import tqdm

from boltz.data import const
from boltz.data.module.inference import BoltzInferenceDataModule
# from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
# from boltz.data.module.inferencev2_finetune import Boltz2InferenceDataModule
from boltz.data.types import MSA, Manifest, Record
from boltz.data.write.writer import BoltzAffinityWriter, BoltzWriter
# from boltz.model.models.boltz2 import Boltz2
from boltz.model.models.boltz2_finetune import Boltz2
import omegaconf
from omegaconf import OmegaConf, listconfig
from boltz.data.feature.featurizerv2 import Boltz2Featurizer
from boltz.data.tokenize.boltz2 import Boltz2Tokenizer

from boltz.data.module.trainingv2 import (
    Boltz2InferenceDataModule,
    DataConfig,
    DatasetConfig,
    DefaultUniformSampler,
)
import random
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

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


def prepare_data(
    out_dir: str,
    cache: str,
    val_ratio: float = 0.2,
    validation_ids_path: Optional[str] = None,
    train_ids_path: Optional[str] = None,
) -> DataConfig:
    """
    读取 out_dir/processed 下的数据，使用 trainingv2.py 的配置类拼装 DataConfig。
    - target_dir -> out_dir / predictions
    - msa_dir    -> processed/msa
    - mol_dir    -> ~/.boltz/mols
    """
    cache = Path(cache).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    mol_dir = cache / "mols"

    out_dir = Path(out_dir).expanduser()
    processed_dir = out_dir / "processed"
    manifest_path = processed_dir / "manifest.json"

    if validation_ids_path is not None:
        split_file = Path(validation_ids_path).expanduser().resolve()
        if not split_file.is_file():
            raise FileNotFoundError(f"Validation ID file not found: {split_file}")
    else:
        warnings.warn(
            "No fixed validation ID file was supplied; using the legacy random "
            "split. Revision experiments should always pass validation_ids_path.",
            stacklevel=2,
        )
        split_file = make_val_split_file(
            manifest_path, out_dir=out_dir, val_ratio=val_ratio, seed=42
        )

    train_split_file = None
    if train_ids_path is not None:
        train_split_file = Path(train_ids_path).expanduser().resolve()
        if not train_split_file.is_file():
            raise FileNotFoundError(f"Training ID file not found: {train_split_file}")

    # 注意：DataModule 内部会再次通过 manifest_path 加载 Manifest
    ds_cfg = DatasetConfig(
        target_dir=str(out_dir / "predictions"),
        msa_dir=str(processed_dir / "msa"),
        mol_dir=str(mol_dir),
        # num_workers=1,
        num_workers=8,
        sampler=DefaultUniformSampler(),
        constraints_dir=(processed_dir / "constraints" if (processed_dir / "constraints").exists() else None),
        template_dir=(processed_dir / "templates" if (processed_dir / "templates").exists() else None),
        extra_mols_dir=(processed_dir / "mols" if (processed_dir / "mols").exists() else None),
        override_method=None,
        affinity=True,
        manifest_path=str(manifest_path),
        filters=None,
        split=str(split_file),
        train_split=(str(train_split_file) if train_split_file else None),
    )

    data_cfg = DataConfig(
        datasets=[ds_cfg],
        filters=[],  # 如果有全局 DynamicFilter，可在此填入
        featurizer=Boltz2Featurizer(),
        tokenizer=Boltz2Tokenizer(),
        val_batch_size=1,
    )
    return data_cfg

def make_val_split_file(manifest_path: str, out_dir: str, val_ratio: float = 0.2, seed: int = 42) -> Path:
    """
    生成一个包含验证集 ID 的 txt 文件，以便 trainingv2 的 split 机制使用。
    """
    split_path = Path(out_dir).expanduser() / "processed" / "affinity_val_split.txt"
    # 如果文件已存在，直接返回，避免覆盖或竞争
    if split_path.exists():
        return split_path
    
    rng = random.Random(seed)
    manifest: Manifest = Manifest.load(manifest_path)
    ids = [r.id for r in manifest.records]
    rng.shuffle(ids)
    k = max(1, int(len(ids) * val_ratio))
    val_ids = ids[:k]
    split_path = Path(out_dir).expanduser() / "processed" / "affinity_val_split.txt"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with open(split_path, "w") as f:
        for x in val_ids:
            f.write(x + "\n")
    return split_path

def train_affinity_finetune(                 
    out_dir,
    ckpt_path,
    cache,
    max_epochs,
    batch_size,
    pretrained_ckpt=None,
    devices=1,
    precision="bf16-mixed",
    validation_only: bool = False,
    step_scale: Optional[float] = None,
    # num_workers: int = 2,
    num_workers: int = 8,
    sampling_steps_affinity: int = 200,
    diffusion_samples_affinity: int = 1,
    subsample_msa: bool = False,
    num_subsampled_msa: int = 1024,
    affinity_mw_correction: Optional[bool] = False,
    accelerator: str = "gpu",
    accumulate_steps: int = 4,            # 🔧 梯度累积步数（有效 batch = 1×accumulate_steps×devices）
    val_ratio: float = 0.2,            # ⭐ 新增：验证集比例
    validation_ids_path: Optional[str] = None,
    train_ids_path: Optional[str] = None,
    seed: int = 42,
    base_lr: float = 1e-4,
    regression_enabled: bool = True,
    classification_enabled: bool = True,
    regression_loss: Literal["huber", "mse"] = "huber",
    classification_loss: Literal["focal", "bce"] = "focal",
    lambda_reg: float = 1.0,
    lambda_cls: float = 1.0,
    huber_delta: float = 0.5,
    focal_gamma: float = 1.0,
    focal_alpha: Optional[float] = None,
    reset_affinity_on_load: bool = False,
    unfreeze_last_trunk_blocks: int = 0,
    early_stopping_patience: Optional[int] = None,
    resume_ckpt_path: Optional[str] = None,
):
    # Print header
    click.echo("\nTraining property: affinity\n")
    seed_everything(seed, workers=True)

    if pretrained_ckpt is None:
        raise ValueError("pretrained_ckpt is required for controlled fine-tuning.")
    if not regression_enabled and not classification_enabled:
        raise ValueError("At least one affinity task must be enabled.")
    if unfreeze_last_trunk_blocks < 0:
        raise ValueError("unfreeze_last_trunk_blocks must be non-negative.")

    out_dir = Path(out_dir).expanduser()
    cache   = Path(cache).expanduser()
    processed_dir = out_dir / "processed"
    mol_dir = cache / "mols"

    # Load manifest
    manifest = Manifest.load(out_dir / "processed" / "manifest.json")

    msg = f"Running affinity train for {len(manifest.records)} input"
    msg += "s." if len(manifest.records) > 1 else "."
    click.echo(msg)

    data_cfg = prepare_data(
        out_dir=out_dir,
        cache=cache,
        val_ratio=val_ratio,
        validation_ids_path=validation_ids_path,
        train_ids_path=train_ids_path,
    )
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
    affinity_checkpoint = pretrained_ckpt  # affinity_checkpoint = cache / "boltz2_aff.ckpt"

    # Set up model parameters,boltz2
    diffusion_params = Boltz2DiffusionParams()
    step_scale = 1.5 if step_scale is None else step_scale
    diffusion_params.step_scale = step_scale
    pairformer_args = PairformerArgsV2()

    msa_args = MSAModuleArgs(
        subsample_msa=subsample_msa,
        num_subsampled_msa=num_subsampled_msa,
        use_paired_feature=True, 
    )

    loss_args = {
        "regression_enabled": regression_enabled,
        "classification_enabled": classification_enabled,
        "regression_loss": regression_loss,
        "classification_loss": classification_loss,
        "lambda_reg": lambda_reg,
        "lambda_cls": lambda_cls,
        "huber_delta": huber_delta,
        "focal_gamma": focal_gamma,
        "focal_alpha": focal_alpha,
    }

    model_module = Boltz2.load_from_checkpoint(
        affinity_checkpoint,
        # strict=True,
        strict=False,
        predict_args=predict_affinity_args,
        map_location="cpu",
        diffusion_process_args=asdict(diffusion_params),
        ema=False,
        pairformer_args=asdict(pairformer_args),
        msa_args=asdict(msa_args),
        steering_args={"fk_steering": False, "guidance_update": False},
        affinity_mw_correction=affinity_mw_correction,
        skip_run_structure=True,   # ★ 关键：不再调用 structure_module.sample
        # affinity_ensemble=False,如果只用一个model的话需要重新训练，因为为affinity_model_args=None
        # 🔥🔥🔥 核心修改点 🔥🔥🔥
        # affinity_model_args=asdict(affinity_args),  # 传入亲和力网络结构参数
        affinity_prediction=True,                   # 强制告诉模型：我要构建亲和力模块！
        affinity_loss_args=loss_args,
        affinity_backbone_grad=(unfreeze_last_trunk_blocks > 0),
        reset_affinity_on_load=reset_affinity_on_load,
        # 🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥
    )
    # reset_affinity_on_load is only for the initial pretrained checkpoint.
    # A later Trainer resume must restore the learned affinity tensors.
    model_module.reset_affinity_on_load = False

    # --- 补齐校验/训练用的 flag，避免 AttributeError ---
    if not hasattr(model_module, "validate_structure"):
        model_module.validate_structure = False
    if not hasattr(model_module, "validate_affinity"):
        model_module.validate_affinity = True
    if not hasattr(model_module, "training_affinity"):
        model_module.training_affinity = True   # 若你的 training_step 也用到
    # ---------------------------------------------------

    # 在模型初始化后冻结非affinity模块的参数
    model_module.requires_grad_(False) 
    for name, param in model_module.named_parameters():
        if "affinity_module" in name:
            param.requires_grad = True

    if unfreeze_last_trunk_blocks:
        pairformer = model_module.pairformer_module
        if hasattr(pairformer, "_orig_mod"):
            pairformer = pairformer._orig_mod
        layers = pairformer.layers
        if unfreeze_last_trunk_blocks > len(layers):
            raise ValueError(
                f"Requested {unfreeze_last_trunk_blocks} trunk blocks, but the "
                f"model only has {len(layers)}."
            )
        for layer in layers[-unfreeze_last_trunk_blocks:]:
            layer.requires_grad_(True)

    trainable_names = [
        name for name, parameter in model_module.named_parameters()
        if parameter.requires_grad
    ]
    trainable_count = sum(
        parameter.numel() for parameter in model_module.parameters()
        if parameter.requires_grad
    )
    total_count = sum(parameter.numel() for parameter in model_module.parameters())
    click.echo(
        f"Trainable parameters: {trainable_count:,}/{total_count:,} "
        f"({100.0 * trainable_count / total_count:.3f}%), "
        f"across {len(trainable_names)} tensors."
    )
    if unfreeze_last_trunk_blocks:
        click.echo(
            f"Partial-trunk ablation enabled: last "
            f"{unfreeze_last_trunk_blocks} PairFormer blocks are trainable."
        )
    
    model_module.train()
    
    # === Callbacks & Logger ===
    logger_csv = CSVLogger(save_dir=str(ckpt_path), name="lightning_logs_csv")
    loggers = [logger_csv]
    try:
        logger_tb = TensorBoardLogger(
            save_dir=str(ckpt_path), name="lightning_logs"
        )
    except ModuleNotFoundError:
        click.echo(
            "TensorBoard is not installed; continuing with CSV logging only."
        )
    else:
        loggers.insert(0, logger_tb)

    ckpt_topk = ModelCheckpoint(
        dirpath=str(Path(ckpt_path) / "checkpoints"),
        save_last=True,
        save_top_k=1,
        monitor="val/loss_affinity",          # 监控验证集亲和力 loss
        mode="min",                           # 越小越好
        filename="epoch{epoch}-valloss{val/loss_affinity:.4f}",
        auto_insert_metric_name=False,
    )
    lr_cb = LearningRateMonitor(logging_interval="step")
    callbacks = [ckpt_topk, lr_cb]
    if early_stopping_patience is not None:
        callbacks.append(
            EarlyStopping(
                monitor="val/loss_affinity",
                mode="min",
                patience=early_stopping_patience,
            )
        )

    # Set up trainer
    trainer = Trainer(
        default_root_dir=ckpt_path,  # ⭐ 这个 ckpt_path 仅当作“工作目录/日志目录”用
        strategy="auto",
        # callbacks=[pred_writer],
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        max_epochs=max_epochs,   # ⭐ 用你传进来的 max_epochs
        accumulate_grad_batches=accumulate_steps,  # 🔧 梯度累积
        num_sanity_val_steps=2,       
        limit_val_batches=1.0,        # ⭐ 跑完整验证集
        check_val_every_n_epoch=1,    # ⭐ 每个 epoch 后都验证
        log_every_n_steps=1,             # 🔧 方便观测单步日志
        callbacks=callbacks,
        logger=loggers,
        deterministic=True,
    )

    # 启动预测过程
    # trainer.predict(
    #     model_module,
    #     datamodule=data_module,
    #     return_predictions=False,
    # )

    model_module.training_args.base_lr = base_lr
    model_module.training_args.max_lr = base_lr

    # 启动训练过程
    if validation_only:
        trainer.validate(
            model_module,
            datamodule=data_module
        )
    else:
        trainer.fit(
            model_module,  # 模型
            datamodule=data_module,  # 数据模块
            ckpt_path=resume_ckpt_path,
        )

if __name__ == "__main__":
    raise SystemExit(
        "Use revision/scripts/run_affinity_training.py --config <config.yaml>."
    )
