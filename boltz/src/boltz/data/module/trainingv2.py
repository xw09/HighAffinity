from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from dataclasses import asdict, dataclass, field

from boltz.data import const
from boltz.data.crop.affinity import AffinityCropper
from boltz.data.feature.featurizerv2 import Boltz2Featurizer
from boltz.data.filter.dynamic.filter import DynamicFilter
from boltz.data.sample.sampler import Sample, Sampler
from boltz.data.mol import load_canonicals, load_molecules
from boltz.data.pad import pad_to_max
from boltz.data.tokenize.boltz2 import Boltz2Tokenizer
from boltz.data.types import (
    MSA,
    Input,
    Manifest,
    Record,
    ResidueConstraints,
    StructureV2,
)

@dataclass
class DatasetConfig:
    """Dataset configuration."""
    target_dir: str
    msa_dir: str
    mol_dir: str
    num_workers: int 
    sampler: Sampler
    constraints_dir: Optional[Path] = None
    template_dir: Optional[Path] = None
    extra_mols_dir: Optional[Path] = None  
    override_method: Optional[str] = None
    affinity: bool = True
    manifest_path: Optional[str] = None
    filters: Optional[list] = None
    split: Optional[str] = None

@dataclass
class DataConfig:
    """Data configuration."""
    datasets: list[DatasetConfig]
    filters: list[DynamicFilter]
    featurizer: Boltz2Featurizer
    tokenizer: Boltz2Tokenizer
    val_batch_size: int = 1
    cropper = AffinityCropper

@dataclass
class Dataset:
    """Data holder."""
    target_dir: Path
    msa_dir: Path
    mol_dir: Path
    manifest: Manifest
    sampler: Sampler
    tokenizer: Boltz2Tokenizer
    featurizer: Boltz2Featurizer
    constraints_dir: Optional[Path] = None
    template_dir: Optional[Path] = None
    extra_mols_dir: Optional[Path] = None
    override_method: Optional[str] = None
    affinity: bool = True

    # 运行时补充
    canonicals: dict = field(default_factory=dict)

    # 固定：亲和力任务用的 cropper 类型
    cropper: type = AffinityCropper

@dataclass
class Sample:
    """A sample with optional chain and interface IDs.

    Attributes
    ----------
    record : Record
        The record.
    chain_id : Optional[int]
        The chain ID.
    interface_id : Optional[int]
        The interface ID.
    """

    record: Record
    chain_id: Optional[int] = None
    interface_id: Optional[int] = None

# 一个最简单的均匀随机采样器，满足 Sampler 接口：sample(records, rng) -> 迭代器
class DefaultUniformSampler(Sampler):
    def sample(self, records, rng):
        while True:
            if hasattr(rng, "integers"):           # Generator
                idx = int(rng.integers(0, len(records)))
            else:                                  # 模块 np.random
                idx = int(rng.randint(0, len(records)))
            yield Sample(record=records[idx])

def load_input(
    record: Record,
    target_dir: Path,
    msa_dir: Path,
    affinity: bool = True,
) -> Input:
    """Load the given input data.

    Parameters
    ----------
    record : Record
        The record to load.
    target_dir : Path
        The path to the data directory.
    msa_dir : Path
        The path to msa directory.
    affinity : bool
        Whether to load the affinity data.

    Returns
    -------
    Input
        The loaded input.

    """
    # Load the structure
    structure = StructureV2.load(target_dir / record.id / f"pre_affinity_{record.id}.npz")
        
    msas = {}
    for chain in record.chains:
        msa_id = chain.msa_id
        # Load the MSA for this chain, if any
        if msa_id != -1:
            msa = MSA.load(msa_dir / f"{msa_id}.npz")
            msas[chain.chain_id] = msa
    # Load templates
    templates = None

    # Load residue constraints
    residue_constraints = None
    # Load extra molecules
    extra_mols = {}

    return Input(
        structure,
        msas,
        record=record,
        residue_constraints=residue_constraints,
        templates=templates,
        extra_mols=extra_mols,
    )


def collate(data: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Collate the data.

    Parameters
    ----------
    data : List[Dict[str, Tensor]]
        The data to collate.

    Returns
    -------
    Dict[str, Tensor]
        The collated data.

    """
    # Get the keys
    keys = data[0].keys()

    # Collate the data
    collated = {}
    for key in keys:
        values = [d[key] for d in data]

        if key not in [
            "all_coords",
            "all_resolved_mask",
            "crop_to_all_atom_map",
            "chain_symmetries",
            "amino_acids_symmetries",
            "ligand_symmetries",
            "record",
            "affinity_mw",
        ]:
            # Check if all have the same shape
            shape = values[0].shape
            if not all(v.shape == shape for v in values):
                values, _ = pad_to_max(values, 0)
            else:
                values = torch.stack(values, dim=0)

        # Stack the values
        collated[key] = values

    return collated


class ValDataset(torch.utils.data.Dataset):
    """Base iterable dataset."""

    def __init__(
        self,
        manifest: Manifest,
        target_dir: Path,
        msa_dir: Path,
        mol_dir: Path,
        constraints_dir: Optional[Path] = None,
        template_dir: Optional[Path] = None,
        extra_mols_dir: Optional[Path] = None,
        override_method: Optional[str] = None,
        affinity: bool = True,
    ) -> None:
        """Initialize the validing dataset.
        """
        super().__init__()
        self.manifest = manifest
        self.target_dir = target_dir
        self.msa_dir = msa_dir
        self.mol_dir = mol_dir
        self.constraints_dir = constraints_dir
        self.template_dir = template_dir
        self.tokenizer = Boltz2Tokenizer()
        self.featurizer = Boltz2Featurizer()
        self.canonicals = load_canonicals(self.mol_dir)
        self.extra_mols_dir = extra_mols_dir
        self.override_method = override_method
        self.affinity = affinity
        self.cropper = AffinityCropper()

    def __getitem__(self, idx: int) -> dict:
        """Get an item from the dataset.

        Returns
        -------
        Dict[str, Tensor]
            The sampled data features.

        """
        # Get record
        record = self.manifest.records[idx]

        # Finalize input data
        input_data = load_input(
            record=record,
            target_dir=self.target_dir,
            msa_dir=self.msa_dir,
            affinity=True,
        )

        # Tokenize structure
        try:
            tokenized = self.tokenizer.tokenize(input_data)
        except Exception as e:  # noqa: BLE001
            print(  # noqa: T201
                f"Tokenizer failed on {record.id} with error {e}. Skipping."
            )
            return self.__getitem__(0)

        try:
            tokenized = self.cropper.crop(
                tokenized,
                max_tokens=256,
                max_atoms=2048,
            )
        except Exception as e:  # noqa: BLE001
            print(
                f"Cropper failed on {record.id} with error {e}. Skipping."
            )  # noqa: T201
            return self.__getitem__(0)

        # Load conformers
        try:
            molecules = {}
            molecules.update(self.canonicals)
            molecules.update(input_data.extra_mols)
            mol_names = set(tokenized.tokens["res_name"].tolist())
            mol_names = mol_names - set(molecules.keys())
            molecules.update(load_molecules(self.mol_dir, mol_names))
        except Exception as e:  # noqa: BLE001
            print(f"Molecule loading failed for {record.id} with error {e}. Skipping.")
            return self.__getitem__(0)

        # Inference specific options
        options = record.inference_options
        if options is None:
            pocket_constraints = None, None
        else:
            pocket_constraints = options.pocket_constraints

        # Get random seed
        seed = 42
        random = np.random.default_rng(seed)

        # Compute features
        try:
            features = self.featurizer.process(
                tokenized,
                molecules=molecules,
                random=random,
                training=False,
                max_atoms=None,
                max_tokens=None,
                max_seqs=const.max_msa_seqs,
                pad_to_max_seqs=False,
                single_sequence_prop=0.0,
                compute_frames=True,
                inference_pocket_constraints=pocket_constraints,
                compute_constraint_features=True,
                override_method=self.override_method,
                compute_affinity=True,
            )
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            print(
                f"Featurizer failed on {record.id} with error {e}. Skipping."
            )  # noqa: T201
            return self.__getitem__(0)

        # Add record
        features["record"] = record
        return features

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.

        """
        return len(self.manifest.records)


"!!TrainingDataset"
class TrainingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        datasets: list[Dataset],
    ) -> None:
        super().__init__()
        self.datasets = datasets
        self.samples = []
        for dataset in datasets:
            records = dataset.manifest.records
            iterator = dataset.sampler.sample(records, np.random)
            self.samples.append(iterator)

    def __getitem__(self, idx: int) -> dict:
        dataset_idx = np.random.choice(len(self.datasets))
        dataset = self.datasets[dataset_idx]

        # 从采样器取一个样本
        sample = next(self.samples[dataset_idx])

        # 准备输入（这里不要传入 load_input 不支持的关键字）
        input_data = load_input(
            record=sample.record,
            target_dir=dataset.target_dir,
            msa_dir=dataset.msa_dir,
            affinity=True,
        )

        # Tokenize
        try:
            tokenized = dataset.tokenizer.tokenize(input_data)
        except Exception as e:
            print(f"Tokenizer failed on {sample.record.id} with error {e}. Skipping.")
            return self.__getitem__(0)

        try:
            tokenized = dataset.cropper().crop(
                tokenized,
                max_tokens=256,
                max_atoms=2048,
            )
        except Exception as e:
            print(f"Cropper failed on {sample.record.id} with error {e}. Skipping.")
            return self.__getitem__(0)

        # 分子加载（修正 mol_dir 来源；额外分子来自 input_data.extra_mols）
        try:
            molecules = {}
            molecules.update(dataset.canonicals)
            molecules.update(input_data.extra_mols)
            mol_names = set(tokenized.tokens["res_name"].tolist())
            mol_names = mol_names - set(molecules.keys())
            molecules.update(load_molecules(dataset.mol_dir, mol_names))
        except Exception as e:
            print(f"Molecule loading failed for {sample.record.id} with error {e}. Skipping.")
            return self.__getitem__(0)

        options = sample.record.inference_options
        pocket_constraints = (None, None) if options is None else options.pocket_constraints

        seed = 42
        random = np.random.default_rng(seed)

        try:
            features = dataset.featurizer.process(
                tokenized,
                molecules=molecules,
                random=random,
                training=True,
                max_atoms=None,
                max_tokens=None,
                max_seqs=const.max_msa_seqs,
                pad_to_max_seqs=False,
                single_sequence_prop=0.0,
                compute_frames=True,
                inference_pocket_constraints=pocket_constraints,
                compute_constraint_features=True,
                override_method=dataset.override_method,
                compute_affinity=True,
            )
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"Featurizer failed on {sample.record.id} with error {e}. Skipping.")
            return self.__getitem__(0)

        features["record"] = sample.record
        return features

    def __len__(self) -> int:
        # 汇总全部训练记录数
        return sum(len(ds.manifest.records) for ds in self.datasets)


class Boltz2InferenceDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DataConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.batchsize = 5
        self.num_workers = 1

        assert self.cfg.val_batch_size == 1

        train: list[Dataset] = []
        val_sets: list[torch.utils.data.Dataset] = []

        for data_config in cfg.datasets:
            target_dir = Path(data_config.target_dir)
            msa_dir = Path(data_config.msa_dir)
            mol_dir = Path(data_config.mol_dir)

            # 载入 manifest
            path = Path(data_config.manifest_path)
            manifest: Manifest = Manifest.load(path)

            # 训练/验证划分
            if data_config.split is not None:
                with Path(data_config.split).open("r") as f:
                    split = {x.lower() for x in f.read().splitlines()}
                train_records, val_records = [], []
                for record in manifest.records:
                    (val_records if record.id.lower() in split else train_records).append(record)
            else:
                train_records = manifest.records
                val_records = []

            # 全局 + 数据集级筛选
            if self.cfg.filters:
                train_records = [r for r in train_records if all(f.filter(r) for f in self.cfg.filters)]
            if data_config.filters:
                train_records = [r for r in train_records if all(f.filter(r) for f in data_config.filters)]

            # 训练集 Dataset
            train_manifest = Manifest(train_records)
            ds = Dataset(
                target_dir=target_dir,
                msa_dir=msa_dir,
                mol_dir=mol_dir,
                manifest=train_manifest,
                sampler=data_config.sampler,
                tokenizer=cfg.tokenizer,
                featurizer=cfg.featurizer,
                constraints_dir=data_config.constraints_dir,
                template_dir=data_config.template_dir,
                extra_mols_dir=data_config.extra_mols_dir,
                override_method=data_config.override_method,
                affinity=True,
            )
            # 预载 canonical 分子
            ds.canonicals = load_canonicals(mol_dir)
            train.append(ds)

            # 验证集 Dataset（逐个构建）
            if val_records:
                val_manifest = Manifest(val_records)
                vset = ValDataset(
                    manifest=val_manifest,
                    target_dir=target_dir,
                    msa_dir=msa_dir,
                    mol_dir=mol_dir,
                    constraints_dir=data_config.constraints_dir,
                    template_dir=data_config.template_dir,
                    extra_mols_dir=data_config.extra_mols_dir,
                    override_method=data_config.override_method,
                    affinity=True,
                )
                val_sets.append(vset)

        # 训练/验证封装
        self._train_set = TrainingDataset(datasets=train)
        if len(val_sets) == 0:
            # 没有验证集就用第一个训练集的前若干条做一个轻量验证
            self._val_set = torch.utils.data.Subset(self._train_set, range(min(8, len(self._train_set))))
        elif len(val_sets) == 1:
            self._val_set = val_sets[0]
        else:
            self._val_set = torch.utils.data.ConcatDataset(val_sets)
        
    def setup(self, stage: Optional[str] = None) -> None:
        """Run the setup for the DataModule.

        Parameters
        ----------
        stage : str, optional
            The stage, one of 'fit', 'validate', 'test'.

        """
        return

    def train_dataloader(self) -> DataLoader:
        train_dl =  DataLoader(
            self._train_set,
            batch_size=self.batchsize,  # Adjust the batch size for training
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,  # Shuffle data for training
            collate_fn=collate,  # Assuming you have a collate function
        )
        # for batch_idx, batch in enumerate(train_dl):
        #     print(f"Batch {batch_idx}: {batch}")
        return train_dl

    def val_dataloader(self) -> DataLoader:
        val_dl = DataLoader(
            self._val_set,
            batch_size=self.batchsize,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            collate_fn=collate,
        )
        return val_dl

    def transfer_batch_to_device(
        self,
        batch: dict,
        device: torch.device,
        dataloader_idx: int,  # noqa: ARG002
    ) -> dict:
        """Transfer a batch to the given device.

        Parameters
        ----------
        batch : Dict
            The batch to transfer.
        device : torch.device
            The device to transfer to.
        dataloader_idx : int
            The dataloader index.

        Returns
        -------
        np.Any
            The transferred batch.

        """
        for key in batch:
            if key not in [
                "all_coords",
                "all_resolved_mask",
                "crop_to_all_atom_map",
                "chain_symmetries",
                "amino_acids_symmetries",
                "ligand_symmetries",
                "record",
                "affinity_mw",
            ]:
                batch[key] = batch[key].to(device)
        return batch
