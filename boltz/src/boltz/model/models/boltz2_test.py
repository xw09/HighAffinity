import gc
from typing import Any, Optional

import numpy as np
import torch
import torch._dynamo
from pytorch_lightning import Callback, LightningModule
from torch import Tensor, nn
from torchmetrics import MeanMetric

import boltz.model.layers.initialize as init
from boltz.data import const
from boltz.data.mol import (
    minimum_lddt_symmetry_coords,
)
from boltz.model.layers.pairformer import PairformerModule
from boltz.model.loss.bfactor import bfactor_loss_fn
from boltz.model.loss.confidencev2 import (
    confidence_loss,
)
from boltz.model.loss.distogramv2 import distogram_loss
from boltz.model.modules.affinity import AffinityModule
from boltz.model.modules.confidencev2 import ConfidenceModule
from boltz.model.modules.diffusion_conditioning import DiffusionConditioning
from boltz.model.modules.diffusionv2 import AtomDiffusion
from boltz.model.modules.encodersv2 import RelativePositionEncoder
from boltz.model.modules.trunkv2 import (
    BFactorModule,
    ContactConditioning,
    DistogramModule,
    InputEmbedder,
    MSAModule,
    TemplateModule,
    TemplateV2Module,
)
from boltz.model.optim.ema import EMA
from boltz.model.optim.scheduler import AlphaFoldLRScheduler

from boltz.model.loss.affinity_loss import affinity_loss

class Boltz2(LightningModule):
    """Boltz2 model."""

    def __init__(
        self,
        atom_s: int,
        atom_z: int,
        token_s: int,
        token_z: int,
        num_bins: int,
        training_args: dict[str, Any],
        validation_args: dict[str, Any],
        embedder_args: dict[str, Any],
        msa_args: dict[str, Any],
        pairformer_args: dict[str, Any],
        score_model_args: dict[str, Any],
        diffusion_process_args: dict[str, Any],
        diffusion_loss_args: dict[str, Any],
        confidence_model_args: Optional[dict[str, Any]] = None,
        affinity_model_args: Optional[dict[str, Any]] = None,
        affinity_model_args1: Optional[dict[str, Any]] = None,
        affinity_model_args2: Optional[dict[str, Any]] = None,
        validators: Any = None,
        num_val_datasets: int = 1,
        atom_feature_dim: int = 128,
        template_args: Optional[dict] = None,
        # confidence_prediction: bool = True,
        confidence_prediction: bool = False,
        affinity_prediction: bool = False,
        affinity_ensemble: bool = False,
        affinity_mw_correction: bool = True,
        run_trunk_and_structure: bool = True,
        skip_run_structure: bool = False,
        token_level_confidence: bool = True,
        alpha_pae: float = 0.0,
        # structure_prediction_training: bool = True,
        structure_prediction_training: bool = False,
        # validate_structure: bool = True,
        validate_structure: bool = False,
        atoms_per_window_queries: int = 32,
        atoms_per_window_keys: int = 128,
        compile_pairformer: bool = False,
        compile_structure: bool = False,
        compile_confidence: bool = False,
        compile_affinity: bool = False,
        compile_msa: bool = False,
        exclude_ions_from_lddt: bool = False,
        ema: bool = False,
        ema_decay: float = 0.999,
        min_dist: float = 2.0,
        max_dist: float = 22.0,
        predict_args: Optional[dict[str, Any]] = None,
        fix_sym_check: bool = False,
        cyclic_pos_enc: bool = False,
        aggregate_distogram: bool = True,
        bond_type_feature: bool = False,
        use_no_atom_char: bool = False,
        no_random_recycling_training: bool = False,
        use_atom_backbone_feat: bool = False,
        use_residue_feats_atoms: bool = False,
        conditioning_cutoff_min: float = 4.0,
        conditioning_cutoff_max: float = 20.0,
        steering_args: Optional[dict] = None,
        use_templates: bool = False,
        compile_templates: bool = False,
        predict_bfactor: bool = False,
        log_loss_every_steps: int = 50,
        checkpoint_diffusion_conditioning: bool = False,
        use_templates_v2: bool = False,
        use_kernels: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["validators"])

        # No random recycling
        self.no_random_recycling_training = no_random_recycling_training

        if validate_structure:
            # Late init at setup time
            self.val_group_mapper = (
                {}
            )  # maps a dataset index to a validation group name
            self.validator_mapper = {}  # maps a dataset index to a validator

            # Validators for each dataset keep track of all metrics,
            # compute validation, aggregate results and log
            self.validators = nn.ModuleList(validators)

        self.num_val_datasets = num_val_datasets
        self.log_loss_every_steps = log_loss_every_steps

        # EMA
        self.use_ema = ema
        self.ema_decay = ema_decay

        # Arguments
        self.training_args = training_args
        self.validation_args = validation_args
        self.diffusion_loss_args = diffusion_loss_args
        self.predict_args = predict_args
        self.steering_args = steering_args

        # Training metrics
        if validate_structure:
            self.train_confidence_loss_logger = MeanMetric()
            self.train_confidence_loss_dict_logger = nn.ModuleDict()
            for m in [
                "plddt_loss",
                "resolved_loss",
                "pde_loss",
                "pae_loss",
            ]:
                self.train_confidence_loss_dict_logger[m] = MeanMetric()

        self.exclude_ions_from_lddt = exclude_ions_from_lddt

        # Distogram
        self.num_bins = num_bins
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.aggregate_distogram = aggregate_distogram

        # Trunk
        self.is_pairformer_compiled = False
        self.is_msa_compiled = False
        self.is_template_compiled = False

        # Kernels
        self.use_kernels = use_kernels

        # Input embeddings
        full_embedder_args = {
            "atom_s": atom_s,
            "atom_z": atom_z,
            "token_s": token_s,
            "token_z": token_z,
            "atoms_per_window_queries": atoms_per_window_queries,
            "atoms_per_window_keys": atoms_per_window_keys,
            "atom_feature_dim": atom_feature_dim,
            "use_no_atom_char": use_no_atom_char,
            "use_atom_backbone_feat": use_atom_backbone_feat,
            "use_residue_feats_atoms": use_residue_feats_atoms,
            **embedder_args,
        }
        self.input_embedder = InputEmbedder(**full_embedder_args)

        self.s_init = nn.Linear(token_s, token_s, bias=False)
        self.z_init_1 = nn.Linear(token_s, token_z, bias=False)
        self.z_init_2 = nn.Linear(token_s, token_z, bias=False)

        self.rel_pos = RelativePositionEncoder(
            token_z, fix_sym_check=fix_sym_check, cyclic_pos_enc=cyclic_pos_enc
        )

        self.token_bonds = nn.Linear(1, token_z, bias=False)
        self.bond_type_feature = bond_type_feature
        if bond_type_feature:
            self.token_bonds_type = nn.Embedding(len(const.bond_types) + 1, token_z)

        self.contact_conditioning = ContactConditioning(
            token_z=token_z,
            cutoff_min=conditioning_cutoff_min,
            cutoff_max=conditioning_cutoff_max,
        )

        # Normalization layers
        self.s_norm = nn.LayerNorm(token_s)
        self.z_norm = nn.LayerNorm(token_z)

        # Recycling projections
        self.s_recycle = nn.Linear(token_s, token_s, bias=False)
        self.z_recycle = nn.Linear(token_z, token_z, bias=False)
        init.gating_init_(self.s_recycle.weight)
        init.gating_init_(self.z_recycle.weight)

        # Set compile rules
        # Big models hit the default cache limit (8)
        torch._dynamo.config.cache_size_limit = 512  # noqa: SLF001
        torch._dynamo.config.accumulated_cache_size_limit = 512  # noqa: SLF001

        # Pairwise stack
        self.use_templates = use_templates
        if use_templates:
            if use_templates_v2:
                self.template_module = TemplateV2Module(token_z, **template_args)
            else:
                self.template_module = TemplateModule(token_z, **template_args)
            if compile_templates:
                self.is_template_compiled = True
                self.template_module = torch.compile(
                    self.template_module,
                    dynamic=False,
                    fullgraph=False,
                )

        self.msa_module = MSAModule(
            token_z=token_z,
            token_s=token_s,
            **msa_args,
        )
        if compile_msa:
            self.is_msa_compiled = True
            self.msa_module = torch.compile(
                self.msa_module,
                dynamic=False,
                fullgraph=False,
            )
        self.pairformer_module = PairformerModule(token_s, token_z, **pairformer_args)
        if compile_pairformer:
            self.is_pairformer_compiled = True
            self.pairformer_module = torch.compile(
                self.pairformer_module,
                dynamic=False,
                fullgraph=False,
            )

        self.checkpoint_diffusion_conditioning = checkpoint_diffusion_conditioning
        self.diffusion_conditioning = DiffusionConditioning(
            token_s=token_s,
            token_z=token_z,
            atom_s=atom_s,
            atom_z=atom_z,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            atom_encoder_depth=score_model_args["atom_encoder_depth"],
            atom_encoder_heads=score_model_args["atom_encoder_heads"],
            token_transformer_depth=score_model_args["token_transformer_depth"],
            token_transformer_heads=score_model_args["token_transformer_heads"],
            atom_decoder_depth=score_model_args["atom_decoder_depth"],
            atom_decoder_heads=score_model_args["atom_decoder_heads"],
            atom_feature_dim=atom_feature_dim,
            conditioning_transition_layers=score_model_args[
                "conditioning_transition_layers"
            ],
            use_no_atom_char=use_no_atom_char,
            use_atom_backbone_feat=use_atom_backbone_feat,
            use_residue_feats_atoms=use_residue_feats_atoms,
        )

        # Output modules
        self.structure_module = AtomDiffusion(
            score_model_args={
                "token_s": token_s,
                "atom_s": atom_s,
                "atoms_per_window_queries": atoms_per_window_queries,
                "atoms_per_window_keys": atoms_per_window_keys,
                **score_model_args,
            },
            compile_score=compile_structure,
            **diffusion_process_args,
        )
        self.distogram_module = DistogramModule(
            token_z,
            num_bins,
        )
        self.predict_bfactor = predict_bfactor
        if predict_bfactor:
            self.bfactor_module = BFactorModule(token_s, num_bins)

        self.confidence_prediction = confidence_prediction
        self.affinity_prediction = affinity_prediction
        self.affinity_ensemble = affinity_ensemble
        self.affinity_mw_correction = affinity_mw_correction
        self.run_trunk_and_structure = run_trunk_and_structure
        self.skip_run_structure = skip_run_structure
        self.token_level_confidence = token_level_confidence
        self.alpha_pae = alpha_pae
        self.structure_prediction_training = structure_prediction_training

        if self.confidence_prediction:
            self.confidence_module = ConfidenceModule(
                token_s,
                token_z,
                token_level_confidence=token_level_confidence,
                bond_type_feature=bond_type_feature,
                fix_sym_check=fix_sym_check,
                cyclic_pos_enc=cyclic_pos_enc,
                conditioning_cutoff_min=conditioning_cutoff_min,
                conditioning_cutoff_max=conditioning_cutoff_max,
                **confidence_model_args,
            )
            if compile_confidence:
                self.confidence_module = torch.compile(
                    self.confidence_module, dynamic=False, fullgraph=False
                )

        if self.affinity_prediction:
            if self.affinity_ensemble:
                self.affinity_module1 = AffinityModule(
                    token_s,
                    token_z,
                    **affinity_model_args1,
                )
                self.affinity_module2 = AffinityModule(
                    token_s,
                    token_z,
                    **affinity_model_args2,
                )
                if compile_affinity:
                    self.affinity_module1 = torch.compile(
                        self.affinity_module1, dynamic=False, fullgraph=False
                    )
                    self.affinity_module2 = torch.compile(
                        self.affinity_module2, dynamic=False, fullgraph=False
                    )
            else:
                self.affinity_module = AffinityModule(
                    token_s,
                    token_z,
                    **affinity_model_args,
                )
                if compile_affinity:
                    self.affinity_module = torch.compile(
                        self.affinity_module, dynamic=False, fullgraph=False
                    )

        # # Remove grad from weights they are not trained for ddp
        # if not structure_prediction_training:
        #     for name, param in self.named_parameters():
        #         if (
        #             name.split(".")[0] not in ["confidence_module", "affinity_module"]
        #             and "out_token_feat_update" not in name
        #         ):
        #             param.requires_grad = False
        #⭐
        if not structure_prediction_training:
            allowed = {"confidence_module", "affinity_module", "affinity_module1", "affinity_module2"}
            for name, param in self.named_parameters():
                top = name.split(".")[0]
                if (top not in allowed) and ("out_token_feat_update" not in name):
                    param.requires_grad = False

    def setup(self, stage: str) -> None:
        """Set the model for training, validation and inference."""
        if stage == "predict" and not (
            torch.cuda.is_available()
            and torch.cuda.get_device_properties(torch.device("cuda")).major
            >= 8.0  # noqa: PLR2004
        ):
            self.use_kernels = False

        if (
            stage != "predict"
            and hasattr(self.trainer, "datamodule")
            and self.trainer.datamodule
            and self.validate_structure#验证结构
        ):
            self.val_group_mapper.update(self.trainer.datamodule.val_group_mapper)

            l1 = len(self.val_group_mapper)
            l2 = self.num_val_datasets
            msg = (
                f"Number of validation datasets num_val_datasets={l2} "
                f"does not match the number of val_group_mapper entries={l1}."
            )
            assert l1 == l2, msg

            # Map an index to a validator, and double check val names
            # match from datamodule
            all_validator_names = []
            for validator in self.validators:
                for val_name in validator.val_names:
                    msg = f"Validator {val_name} duplicated in validators."
                    assert val_name not in all_validator_names, msg
                    all_validator_names.append(val_name)
                    for val_idx, val_group in self.val_group_mapper.items():
                        if val_name == val_group["label"]:
                            self.validator_mapper[val_idx] = validator

            msg = "Mismatch between validator names and val_group_mapper values."
            assert set(all_validator_names) == {
                x["label"] for x in self.val_group_mapper.values()
            }, msg

    def forward(
        self,
        feats: dict[str, Tensor],
        recycling_steps: int = 0,
        num_sampling_steps: Optional[int] = None,
        multiplicity_diffusion_train: int = 1,
        diffusion_samples: int = 1,
        max_parallel_samples: Optional[int] = None,
        run_confidence_sequentially: bool = False,
    ) -> dict[str, Tensor]:
        with torch.set_grad_enabled(
            self.training and self.structure_prediction_training
        ):
            "feats.keys:dict_keys(['token_index', 'residue_index', 'asym_id', 'entity_id', 'sym_id', 'mol_type', 'res_type', 'disto_center', 'token_bonds', 'type_bonds',"
            "'token_pad_mask', 'token_resolved_mask', 'token_disto_mask', 'contact_conditioning', 'contact_threshold', 'method_feature', 'modified', 'cyclic_period', "
            "'affinity_token_mask', 'ref_pos', 'atom_resolved_mask', 'ref_atom_name_chars', 'ref_element', 'ref_charge', 'ref_chirality', 'atom_backbone_feat', "
            "'ref_space_uid', 'coords', 'atom_pad_mask', 'atom_to_token', 'token_to_rep_atom', 'r_set_to_rep_atom', 'token_to_center_atom', 'disto_target', "
            "'disto_coords_ensemble', 'bfactor', 'plddt', 'frames_idx', 'frame_resolved_mask', 'msa', 'msa_paired', 'deletion_value', 'has_deletion', "
            "'deletion_mean', 'profile', 'msa_mask', 'template_restype', 'template_frame_rot', 'template_frame_t', 'template_cb', 'template_ca', "
            "'template_mask_cb', 'template_mask_frame', 'template_mask', 'query_to_template', 'visibility_ids', 'ensemble_ref_idxs', "
            "'rdkit_bounds_index', 'rdkit_bounds_bond_mask', 'rdkit_bounds_angle_mask', 'rdkit_upper_bounds', 'rdkit_lower_bounds', "
            "'chiral_atom_index', 'chiral_reference_mask', 'chiral_atom_orientations', 'stereo_bond_index', 'stereo_reference_mask', "
            "'stereo_bond_orientations', 'planar_bond_index', 'planar_ring_5_index', 'planar_ring_6_index', 'connected_chain_index', "
            "'connected_atom_index', 'symmetric_chain_index', 'record'])"
            s_inputs = self.input_embedder(feats)

            # Initialize the sequence embeddings
            s_init = self.s_init(s_inputs)

            # Initialize pairwise embeddings
            z_init = (
                self.z_init_1(s_inputs)[:, :, None]
                + self.z_init_2(s_inputs)[:, None, :]
            )
            relative_position_encoding = self.rel_pos(feats)
            z_init = z_init + relative_position_encoding
            z_init = z_init + self.token_bonds(feats["token_bonds"].float())
            if self.bond_type_feature:
                z_init = z_init + self.token_bonds_type(feats["type_bonds"].long())
            z_init = z_init + self.contact_conditioning(feats)

            # Perform rounds of the pairwise stack
            s = torch.zeros_like(s_init)
            z = torch.zeros_like(z_init)

            # Compute pairwise mask
            mask = feats["token_pad_mask"].float()
            pair_mask = mask[:, :, None] * mask[:, None, :]
            if self.run_trunk_and_structure:
                for i in range(recycling_steps + 1):
                    with torch.set_grad_enabled(
                        self.training
                        and self.structure_prediction_training
                        and (i == recycling_steps)
                    ):
                        # Issue with unused parameters in autocast
                        if (
                            self.training
                            and (i == recycling_steps)
                            and torch.is_autocast_enabled()
                        ):
                            torch.clear_autocast_cache()

                        # Apply recycling
                        s = s_init + self.s_recycle(self.s_norm(s))
                        z = z_init + self.z_recycle(self.z_norm(z))

                        # Compute pairwise stack
                        if self.use_templates:
                            if self.is_template_compiled and not self.training:
                                template_module = (
                                    self.template_module._orig_mod
                                )  # noqa: SLF001
                            else:
                                template_module = self.template_module

                            z = z + template_module(
                                z, feats, pair_mask, use_kernels=self.use_kernels
                            )

                        if self.is_msa_compiled and not self.training:
                            msa_module = self.msa_module._orig_mod  # noqa: SLF001
                        else:
                            msa_module = self.msa_module

                        z = z + msa_module(
                            z, s_inputs, feats, use_kernels=self.use_kernels
                        )

                        # Revert to uncompiled version for validation
                        if self.is_pairformer_compiled and not self.training:
                            pairformer_module = (
                                self.pairformer_module._orig_mod
                            )  # noqa: SLF001
                        else:
                            pairformer_module = self.pairformer_module

                        s, z = pairformer_module(
                            s,
                            z,
                            mask=mask,
                            pair_mask=pair_mask,
                            use_kernels=self.use_kernels,
                        )

            pdistogram = self.distogram_module(z)
            dict_out = {"pdistogram": pdistogram}

            if (
                self.run_trunk_and_structure
                and ((not self.training) or self.confidence_prediction)
                and (not self.skip_run_structure)
            ):
                # if self.checkpoint_diffusion_conditioning and self.training:
                if self.checkpoint_diffusion_conditioning and self.training and self.structure_prediction_training:
                    # TODO decide whether this should be with bf16 or not
                    q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias = (
                        torch.utils.checkpoint.checkpoint(
                            self.diffusion_conditioning,
                            s,
                            z,
                            relative_position_encoding,
                            feats,
                        )
                    )
                else:
                    q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias = (
                        self.diffusion_conditioning(
                            s_trunk=s,
                            z_trunk=z,
                            relative_position_encoding=relative_position_encoding,
                            feats=feats,
                        )
                    )
                diffusion_conditioning = {
                    "q": q,
                    "c": c,
                    "to_keys": to_keys,
                    "atom_enc_bias": atom_enc_bias,
                    "atom_dec_bias": atom_dec_bias,
                    "token_trans_bias": token_trans_bias,
                }

                with torch.autocast("cuda", enabled=False):
                    struct_out = self.structure_module.sample(
                        s_trunk=s.float(),
                        s_inputs=s_inputs.float(),
                        feats=feats,
                        num_sampling_steps=num_sampling_steps,
                        atom_mask=feats["atom_pad_mask"].float(),
                        multiplicity=diffusion_samples,
                        max_parallel_samples=max_parallel_samples,
                        steering_args=self.steering_args,
                        diffusion_conditioning=diffusion_conditioning,
                    )
                    dict_out.update(struct_out)

                if self.predict_bfactor:
                    pbfactor = self.bfactor_module(s)
                    dict_out["pbfactor"] = pbfactor

            if self.training and self.confidence_prediction:
                assert len(feats["coords"].shape) == 4
                assert (
                    feats["coords"].shape[1] == 1
                ), "Only one conformation is supported for confidence"

            # Compute structure module
            if self.training and self.structure_prediction_training:
                atom_coords = feats["coords"]
                B, K, L = atom_coords.shape[0:3]
                assert K in (
                    multiplicity_diffusion_train,
                    1,
                )  # TODO make check somewhere else, expand to m % N == 0, m > N
                atom_coords = atom_coords.reshape(B * K, L, 3)
                atom_coords = atom_coords.repeat_interleave(
                    multiplicity_diffusion_train // K, 0
                )
                feats["coords"] = atom_coords  # (multiplicity, L, 3)
                assert len(feats["coords"].shape) == 3

                with torch.autocast("cuda", enabled=False):
                    struct_out = self.structure_module(
                        s_trunk=s.float(),
                        s_inputs=s_inputs.float(),
                        feats=feats,
                        multiplicity=multiplicity_diffusion_train,
                        diffusion_conditioning=diffusion_conditioning,
                    )
                    dict_out.update(struct_out)

            elif self.training:
                feats["coords"] = feats["coords"].squeeze(1)
                assert len(feats["coords"].shape) == 3

        if self.confidence_prediction:
            dict_out.update(
                self.confidence_module(
                    s_inputs=s_inputs.detach(),
                    s=s.detach(),
                    z=z.detach(),
                    x_pred=(
                        dict_out["sample_atom_coords"].detach()
                        if not self.skip_run_structure
                        else feats["coords"].repeat_interleave(diffusion_samples, 0)
                    ),
                    feats=feats,
                    pred_distogram_logits=(
                        dict_out["pdistogram"][
                            :, :, :, 0
                        ].detach()  # TODO only implemeted for 1 distogram
                    ),
                    multiplicity=diffusion_samples,
                    run_sequentially=run_confidence_sequentially,
                    use_kernels=self.use_kernels,
                )
            )

        if self.affinity_prediction:
            
            #⭐ ---------- 构造坐标：优先用结构采样，否则回退到输入 coords ----------
            sac = dict_out.get("sample_atom_coords", None)   # 结构采样得到的 [S,L,3] 或 [B,S,L,3]
            iptm = dict_out.get("iptm", None)                # 可能是 [S] 或 [B,S]

            if (not self.skip_run_structure) and (sac is not None):
                if sac.dim() == 3:
                    # [S, L, 3]（无 batch 维）
                    if iptm is not None and iptm.dim() == 1 and iptm.numel() == sac.shape[0]:
                        best_idx = torch.argmax(iptm).item()
                    else:
                        best_idx = 0
                    coords_affinity = sac[best_idx][None, None]          # -> [1,1,L,3]
                elif sac.dim() == 4:
                    # [B, S, L, 3]（有 batch 维）
                    B, S, L, _ = sac.shape
                    if iptm is not None and iptm.dim() == 2 and list(iptm.shape) == [B, S]:
                        best_idx = torch.argmax(iptm, dim=1)              # [B]
                    else:
                        best_idx = torch.zeros(B, dtype=torch.long, device=sac.device)
                    coords_affinity = sac[torch.arange(B, device=sac.device), best_idx]  # [B,L,3]
                    coords_affinity = coords_affinity[:, None, ...]        # -> [B,1,L,3]
                else:
                    raise RuntimeError(f"Unexpected shape for sample_atom_coords: {sac.shape}")
            else:
                # 回退到输入 coords
                coords_affinity = feats["coords"]                          # 可能是 [B,L,3] 或 [B,K,L,3] 或 [1,K,L,3]
                if coords_affinity.dim() == 3:
                    coords_affinity = coords_affinity[:, None, ...]        # -> [B,1,L,3]
                elif coords_affinity.dim() == 4:
                    if coords_affinity.shape[1] != 1:
                        coords_affinity = coords_affinity[:, :1, ...]      # 取第一个 ensemble -> [B,1,L,3]
                else:
                    raise RuntimeError(f"Unexpected feats['coords'] shape: {coords_affinity.shape}")

            # ---------- 批处理安全的 mask 构造 ----------
            pad_token_mask = feats["token_pad_mask"].bool()                # [B,L]
            rec_mask = (feats["mol_type"] == 0) & pad_token_mask           # [B,L]
            lig_mask = feats["affinity_token_mask"].bool() & pad_token_mask# [B,L]

            # 交叉/自相互作用的 pair mask（B,L,L）
            cross_pair_mask = (
                (lig_mask[:, :, None] & rec_mask[:, None, :]) |
                (rec_mask[:, :, None] & lig_mask[:, None, :]) |
                (lig_mask[:, :, None] & lig_mask[:, None, :])
            ).float()
            z_affinity = z * cross_pair_mask[..., None]                    # [B,L,L,C]
            #⭐ ---------- 构造坐标：优先用结构采样，否则回退到输入 coords ----------

            s_inputs = self.input_embedder(feats, affinity=True)

            with torch.autocast("cuda", enabled=False):
                if self.affinity_ensemble:
                    dict_out_affinity1 = self.affinity_module1(
                        s_inputs=s_inputs.detach(),
                        z=z_affinity.detach(),
                        x_pred=coords_affinity,
                        feats=feats,
                        multiplicity=1,
                        use_kernels=self.use_kernels,
                    )

                    dict_out_affinity1["affinity_probability_binary"] = (
                        torch.nn.functional.sigmoid(
                            dict_out_affinity1["affinity_logits_binary"]
                        )
                    )
                    dict_out_affinity2 = self.affinity_module2(
                        s_inputs=s_inputs.detach(),
                        z=z_affinity.detach(),
                        x_pred=coords_affinity,
                        feats=feats,
                        multiplicity=1,
                        use_kernels=self.use_kernels,
                    )
                    dict_out_affinity2["affinity_probability_binary"] = (
                        torch.nn.functional.sigmoid(
                            dict_out_affinity2["affinity_logits_binary"]
                        )
                    )
                    # print("affinity_logits_binary1:", dict_out_affinity1["affinity_logits_binary"])
                    # print("affinity_logits_binary2:", dict_out_affinity2["affinity_logits_binary"])
                    dict_out_affinity_ensemble = {
                        "affinity_pred_value": (
                            dict_out_affinity1["affinity_pred_value"]
                            + dict_out_affinity2["affinity_pred_value"]
                        )
                        / 2,
                        "affinity_probability_binary": (
                            dict_out_affinity1["affinity_probability_binary"]
                            + dict_out_affinity2["affinity_probability_binary"]
                        )
                        / 2,
                    }

                    dict_out_affinity1 = {
                        "affinity_pred_value1": dict_out_affinity1[
                            "affinity_pred_value"
                        ],
                        "affinity_probability_binary1": dict_out_affinity1[
                            "affinity_probability_binary"
                        ],
                    }
                    dict_out_affinity2 = {
                        "affinity_pred_value2": dict_out_affinity2[
                            "affinity_pred_value"
                        ],
                        "affinity_probability_binary2": dict_out_affinity2[
                            "affinity_probability_binary"
                        ],
                    }
                    # 训练数据不是小分子要重新拟合
                    # if self.affinity_mw_correction:
                    #     model_coef = 1.03525938
                    #     mw_coef = -0.59992683
                    #     bias = 2.83288489
                    #     mw = feats["affinity_mw"][0] ** 0.3
                    #     dict_out_affinity_ensemble["affinity_pred_value"] = (
                    #         model_coef
                    #         * dict_out_affinity_ensemble["affinity_pred_value"]
                    #         + mw_coef * mw
                    #         + bias
                    #     )

                    dict_out.update(dict_out_affinity_ensemble)
                    dict_out.update(dict_out_affinity1)
                    dict_out.update(dict_out_affinity2)
                else:
                    dict_out_affinity = self.affinity_module(
                        s_inputs=s_inputs.detach(),
                        z=z_affinity.detach(),
                        x_pred=coords_affinity,
                        feats=feats,
                        multiplicity=1,
                        use_kernels=self.use_kernels,
                    )
                    dict_out.update(
                        {
                            "affinity_pred_value": dict_out_affinity[
                                "affinity_pred_value"
                            ],
                            "affinity_probability_binary": torch.nn.functional.sigmoid(
                                dict_out_affinity["affinity_logits_binary"]
                            ),
                        }
                    )

        return dict_out

    def get_true_coordinates(
        self,
        batch: dict[str, Tensor],
        out: dict[str, Tensor],
        diffusion_samples: int,
        symmetry_correction: bool,
        expand_to_diffusion_samples: bool = True,
    ):
        if symmetry_correction:
            msg = "expand_to_diffusion_samples must be true for symmetry correction."
            assert expand_to_diffusion_samples, msg

        return_dict = {}

        assert (
            batch["coords"].shape[0] == 1
        ), f"Validation is not supported for batch sizes={batch['coords'].shape[0]}"

        if symmetry_correction:
            true_coords = []
            true_coords_resolved_mask = []
            for idx in range(batch["token_index"].shape[0]):
                for rep in range(diffusion_samples):
                    i = idx * diffusion_samples + rep
                    best_true_coords, best_true_coords_resolved_mask = (
                        minimum_lddt_symmetry_coords(
                            coords=out["sample_atom_coords"][i : i + 1],
                            feats=batch,
                            index_batch=idx,
                        )
                    )
                    true_coords.append(best_true_coords)
                    true_coords_resolved_mask.append(best_true_coords_resolved_mask)

            true_coords = torch.cat(true_coords, dim=0)
            true_coords_resolved_mask = torch.cat(true_coords_resolved_mask, dim=0)
            true_coords = true_coords.unsqueeze(1)

            true_coords_resolved_mask = true_coords_resolved_mask

            return_dict["true_coords"] = true_coords
            return_dict["true_coords_resolved_mask"] = true_coords_resolved_mask
            return_dict["rmsds"] = 0
            return_dict["best_rmsd_recall"] = 0

        else:
            K, L = batch["coords"].shape[1:3]

            true_coords_resolved_mask = batch["atom_resolved_mask"]
            true_coords = batch["coords"].squeeze(0)
            if expand_to_diffusion_samples:
                true_coords = true_coords.repeat((diffusion_samples, 1, 1)).reshape(
                    diffusion_samples, K, L, 3
                )

                true_coords_resolved_mask = true_coords_resolved_mask.repeat_interleave(
                    diffusion_samples, dim=0
                )  # since all masks are the same across conformers and diffusion samples, can just repeat S times
            else:
                true_coords_resolved_mask = true_coords_resolved_mask.squeeze(0)

            return_dict["true_coords"] = true_coords
            return_dict["true_coords_resolved_mask"] = true_coords_resolved_mask
            return_dict["rmsds"] = 0
            return_dict["best_rmsd_recall"] = 0
            return_dict["best_rmsd_precision"] = 0

        return return_dict

    '只训练亲和力模块'
    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:

        # Sample recycling steps
        if self.no_random_recycling_training:
            recycling_steps = self.training_args.recycling_steps
        else:
            rgn = np.random.default_rng(self.global_step)
            recycling_steps = rgn.integers(
                0, self.training_args.recycling_steps + 1
            ).item()

        if self.training_args.get("sampling_steps_random", None) is not None:
            rgn_samplng_steps = np.random.default_rng(self.global_step)
            sampling_steps = rgn_samplng_steps.choice(
                self.training_args.sampling_steps_random
            )
        else:
            sampling_steps = self.training_args.sampling_steps
         # Compute the forward pass   
        out = self(
            feats=batch,
            recycling_steps=recycling_steps,
            num_sampling_steps=sampling_steps,
            multiplicity_diffusion_train=self.training_args.diffusion_multiplicity,
            diffusion_samples=self.training_args.diffusion_samples,
        )

        # 计算 AffinityModule 模块的损失
        affinity_loss_dict = affinity_loss(out, batch, multiplicity=1)
        loss = affinity_loss_dict["loss"]
        self.log("train/loss_affinity", loss)
        self.training_log()

        return loss

    def training_log(self):
        # 只记录 AffinityModule 的梯度和参数的 norm
        if self.affinity_ensemble:
            # self.log("train/grad_norm_affinity_module1", self.gradient_norm(self.affinity_module1), prog_bar=True)
            # self.log("train/grad_norm_affinity_module2", self.gradient_norm(self.affinity_module2), prog_bar=True)
            self.log("train/param_norm_affinity_module1", self.parameter_norm(self.affinity_module1), prog_bar=True)
            self.log("train/param_norm_affinity_module2", self.parameter_norm(self.affinity_module2), prog_bar=True)

        # 记录学习率
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr, prog_bar=True)

    # 记录梯度
    def on_after_backward(self):
        if self.affinity_ensemble:
            self.log("train/grad_norm_affinity_module1",
                    self.gradient_norm(self.affinity_module1),
                    on_step=True, prog_bar=False)
            self.log("train/grad_norm_affinity_module2",
                    self.gradient_norm(self.affinity_module2),
                    on_step=True, prog_bar=False)
        else:
            self.log("train/grad_norm_affinity_module",
                    self.gradient_norm(self.affinity_module),
                    on_step=True, prog_bar=False)


    def gradient_norm(self, module):
        parameters = [
            p.grad.norm(p=2) ** 2
            for p in module.parameters()
            if p.requires_grad and p.grad is not None
        ]
        if len(parameters) == 0:
            return torch.tensor(
                0.0, device="cuda" if torch.cuda.is_available() else "cpu"
            )
        norm = torch.stack(parameters).sum().sqrt()
        return norm

    def parameter_norm(self, module):
        parameters = [p.norm(p=2) ** 2 for p in module.parameters() if p.requires_grad]
        if len(parameters) == 0:
            return torch.tensor(
                0.0, device="cuda" if torch.cuda.is_available() else "cpu"
            )
        norm = torch.stack(parameters).sum().sqrt()
        return norm

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int):
        # 采样/步数与训练尽量对齐，避免分布飘移
        recycling_steps = self.validation_args.recycling_steps
        sampling_steps  = self.validation_args.sampling_steps

        out = self(
            feats=batch,
            recycling_steps=recycling_steps,
            num_sampling_steps=sampling_steps,
            diffusion_samples=self.validation_args.diffusion_samples,
        )

        # 计算亲和力验证损失（与训练使用同一个函数）
        affinity_loss_dict = affinity_loss(out, batch, multiplicity=1)
        val_loss = affinity_loss_dict["loss"]

        # 记录：只在 epoch 级聚合，进度条显示
        B = batch["token_index"].shape[0]
        self.log("val/loss_affinity", val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=B)

        return val_loss 


    def on_validation_epoch_end(self):
        """Aggregate all metrics for each validator."""
        if self.validate_structure:
            for validator in self.validator_mapper.values():
                # This will aggregate, compute and log all metrics
                validator.on_epoch_end(model=self)
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> dict:
        """Modify predict_step to be similar to validation_step."""
        
        # 采样/步数与训练尽量对齐，避免分布飘移
        recycling_steps = self.validation_args.recycling_steps
        sampling_steps  = self.validation_args.sampling_steps

        out = self(
            feats=batch,
            recycling_steps=recycling_steps,
            num_sampling_steps=sampling_steps,
            diffusion_samples=self.validation_args.diffusion_samples,
        )

        # 用于返回模型的预测结果
        pred_dict = {"exception": False}

        if self.affinity_prediction:
            pred_dict["affinity_pred_value"] = out["affinity_pred_value"]
            pred_dict["affinity_probability_binary"] = out[
                "affinity_probability_binary"
            ]
            if self.affinity_ensemble:
                pred_dict["affinity_pred_value1"] = out["affinity_pred_value1"]
                pred_dict["affinity_probability_binary1"] = out[
                    "affinity_probability_binary1"
                ]
                pred_dict["affinity_pred_value2"] = out["affinity_pred_value2"]
                pred_dict["affinity_probability_binary2"] = out[
                    "affinity_probability_binary2"
                ]

        return pred_dict
    
    # def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> dict:
    #     try:
    #         out = self(
    #             batch,
    #             recycling_steps=self.predict_args["recycling_steps"],
    #             num_sampling_steps=self.predict_args["sampling_steps"],
    #             diffusion_samples=self.predict_args["diffusion_samples"],
    #             max_parallel_samples=self.predict_args["max_parallel_samples"],
    #             run_confidence_sequentially=True,
    #         )
    #         pred_dict = {"exception": False}
    #         if "keys_dict_batch" in self.predict_args:
    #             for key in self.predict_args["keys_dict_batch"]:
    #                 pred_dict[key] = batch[key]

    #         pred_dict["masks"] = batch["atom_pad_mask"]
    #         pred_dict["token_masks"] = batch["token_pad_mask"]

    #         if "keys_dict_out" in self.predict_args:
    #             for key in self.predict_args["keys_dict_out"]:
    #                 pred_dict[key] = out[key]
    #         pred_dict["coords"] = out["sample_atom_coords"]
    #         if self.confidence_prediction:
    #             pred_dict["confidence"] = out.get("ablation_confidence", None)
    #             pred_dict["pde"] = out["pde"]
    #             pred_dict["plddt"] = out["plddt"]
    #             pred_dict["confidence_score"] = (
    #                 4 * out["complex_plddt"]
    #                 + (
    #                     out["iptm"]
    #                     if not torch.allclose(
    #                         out["iptm"], torch.zeros_like(out["iptm"])
    #                     )
    #                     else out["ptm"]
    #                 )
    #             ) / 5

    #             pred_dict["complex_plddt"] = out["complex_plddt"]
    #             pred_dict["complex_iplddt"] = out["complex_iplddt"]
    #             pred_dict["complex_pde"] = out["complex_pde"]
    #             pred_dict["complex_ipde"] = out["complex_ipde"]
    #             if self.alpha_pae > 0:
    #                 pred_dict["pae"] = out["pae"]
    #                 pred_dict["ptm"] = out["ptm"]
    #                 pred_dict["iptm"] = out["iptm"]
    #                 pred_dict["ligand_iptm"] = out["ligand_iptm"]
    #                 pred_dict["protein_iptm"] = out["protein_iptm"]
    #                 pred_dict["pair_chains_iptm"] = out["pair_chains_iptm"]
    #         if self.affinity_prediction:
    #             pred_dict["affinity_pred_value"] = out["affinity_pred_value"]
    #             pred_dict["affinity_probability_binary"] = out[
    #                 "affinity_probability_binary"
    #             ]
    #             if self.affinity_ensemble:
    #                 pred_dict["affinity_pred_value1"] = out["affinity_pred_value1"]
    #                 pred_dict["affinity_probability_binary1"] = out[
    #                     "affinity_probability_binary1"
    #                 ]
    #                 pred_dict["affinity_pred_value2"] = out["affinity_pred_value2"]
    #                 pred_dict["affinity_probability_binary2"] = out[
    #                     "affinity_probability_binary2"
    #                 ]
    #         return pred_dict

    #     except RuntimeError as e:  # catch out of memory exceptions
    #         if "out of memory" in str(e):
    #             print("| WARNING: ran out of memory, skipping batch")
    #             torch.cuda.empty_cache()
    #             gc.collect()
    #             return {"exception": True}
    #         else:
    #             raise e

    "冻结其他模块的权重，只训练亲和力预测部分"

    def configure_optimizers(self) -> torch.optim.Optimizer:
        # 获取所有的参数名
        param_dict = dict(self.named_parameters())

        # 支持三种命名：affinity_module（单头）/ affinity_module1 / affinity_module2（双头）
        allow_prefixes = ("affinity_module", "affinity_module1", "affinity_module2")

        params = [
            p for n, p in param_dict.items()
            if p.requires_grad and n.split(".")[0] in allow_prefixes
        ]
        assert len(params) > 0, "Optimizer 中没有亲和力头参数，请检查 requires_grad 与命名"

        optimizer = torch.optim.AdamW(
            params,  # 只传入 affinity 模块的参数
            lr=self.training_args.base_lr,
            betas=(self.training_args.adam_beta_1, self.training_args.adam_beta_2),
            eps=self.training_args.adam_eps,
            weight_decay=self.training_args.get("weight_decay", 0.0),
        )

        return optimizer

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        # Ignore the lr from the checkpoint
        lr = self.training_args.max_lr
        weight_decay = self.training_args.weight_decay
        if "optimizer_states" in checkpoint:
            for state in checkpoint["optimizer_states"]:
                for group in state["param_groups"]:
                    group["lr"] = lr
                    group["weight_decay"] = weight_decay
        if "lr_schedulers" in checkpoint:
            for scheduler in checkpoint["lr_schedulers"]:
                scheduler["max_lr"] = lr
                scheduler["base_lrs"] = [lr] * len(scheduler["base_lrs"])
                scheduler["_last_lr"] = [lr] * len(scheduler["_last_lr"])

        # Ignore the training diffusion_multiplicity and recycling steps from the checkpoint
        if "hyper_parameters" in checkpoint:
            checkpoint["hyper_parameters"]["training_args"]["max_lr"] = lr
            checkpoint["hyper_parameters"]["training_args"][
                "diffusion_multiplicity"
            ] = self.training_args.diffusion_multiplicity
            checkpoint["hyper_parameters"]["training_args"][
                "recycling_steps"
            ] = self.training_args.recycling_steps
            checkpoint["hyper_parameters"]["training_args"][
                "weight_decay"
            ] = self.training_args.weight_decay

    def configure_callbacks(self) -> list[Callback]:
        """Configure model callbacks.

        Returns
        -------
        List[Callback]
            List of callbacks to be used in the model.

        """
        return [EMA(self.ema_decay)] if self.use_ema else []

    #⭐
    def on_train_start(self):
        self.training_affinity = True
        self.validate_affinity = False

    def on_train_epoch_start(self):
        self.training_affinity = True
        self.validate_affinity = False

    def on_validation_start(self):
        self.training_affinity = False
        self.validate_affinity = True
