from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from models.encoder_registry import (
    build_encoder, encoder_output_dim, encoder_type, load_model_state_for_tasks,
    load_pretrained_encoder,
)
from models.multitask_gnn import CYPDMPNN, batch_graphs, build_head
from models.target_conditioned_gnn import TargetConditionedHeads
from models.transfer_model import resolve_transfer_stages, validate_encoder_checkpoint
from src.constants import FEATURE_SCHEMA_VERSION, MODEL_CODE_VERSION
from src.comparison import model_comparison_table, residual_complementarity
from src.data import DataBundle
from src.training import build_model, masked_loss, parameter_counts, resolve_loss_config
from src.utils import RunContext, atomic_torch_save, load_config, seed_everything


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CYP_TASKS = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")


def synthetic_bundle(purpose: str = "direct_pic50", rows: int = 12) -> DataBundle:
    frame = pd.DataFrame(
        {
            "Molecule_Name": [f"m{index}" for index in range(rows)],
            "canonical_smiles": ["C" * (index % 4 + 1) for index in range(rows)],
            "target": np.linspace(1.0, 3.2, rows),
            "lower": np.linspace(0.9, 3.1, rows),
            "upper": np.linspace(1.1, 3.3, rows),
            "sd": np.repeat(0.1, rows),
        }
    )
    split = rows * 2 // 3
    return DataBundle(
        frame=frame,
        train_mask=pd.Series([True] * split + [False] * (rows - split), index=frame.index),
        validation_mask=pd.Series([False] * split + [True] * (rows - split), index=frame.index),
        task_names=("CYP1A2",), target_columns=("target",), lower_columns=("lower",),
        upper_columns=("upper",), uncertainty_columns=("sd",),
        split_metadata={"purpose": purpose, "outer_fold": 0},
    )


def small_gine_config(experiment: str = "gine-test") -> dict:
    return {
        "experiment": experiment,
        "model": {
            "type": "gine", "architecture": "cyp_specific_heads", "hidden_dim": 8,
            "num_layers": 2, "dropout": 0.1, "residual": True,
            "normalization": "layernorm", "pooling": "mean",
            "ffn_hidden_dim": 8, "ffn_num_layers": 2, "ffn_dropout": 0.1,
        },
        "data": {"target_normalization": "zscore_per_target"},
        "training": {
            "batch_size": 4, "max_epochs": 1, "optimizer": "adamw",
            "learning_rate": 1e-3, "selection_metric": "macro_MAE",
            "metric_direction": "minimize", "precision": "fp32",
        },
        "loss": {"loss_mode": "standard"},
    }


class ConfigCompatibilityTests(unittest.TestCase):
    def test_existing_configs_reproduce_the_original_architecture(self):
        """The unchanged configs must still build the pre-existing D-MPNN exactly."""
        config = load_config(PROJECT_ROOT / "configs/chemprop_multitask.yaml")
        model = build_model(CYP_TASKS, config)
        self.assertIsInstance(model, CYPDMPNN)
        self.assertEqual(model.encoder.hidden_dim, 300)
        self.assertEqual(model.encoder.depth, 3)
        head = model.heads["CYP1A2"]
        self.assertEqual(
            [type(module).__name__ for module in head],
            ["Linear", "ReLU", "Dropout", "Linear"],
        )
        expected_keys = sorted(
            [
                "encoder.input_layer.weight", "encoder.input_layer.bias",
                "encoder.message_layer.weight",
                "encoder.atom_layer.weight", "encoder.atom_layer.bias",
            ]
            + [
                f"heads.{task}.{index}.{parameter}"
                for task in CYP_TASKS for index in (0, 3) for parameter in ("weight", "bias")
            ]
        )
        self.assertEqual(sorted(model.state_dict()), expected_keys)

    def test_existing_transfer_config_loss_resolution_is_unchanged(self):
        """transfer_singleconc.yaml has no stage losses, so both stages use the shared block."""
        config = load_config(PROJECT_ROOT / "configs/transfer_singleconc.yaml")
        self.assertEqual(config["loss"]["loss_mode"], "interval_aware")
        self.assertEqual(resolve_loss_config(config, "pretraining")["loss_mode"], "interval_aware")
        self.assertEqual(resolve_loss_config(config, "transfer")["loss_mode"], "interval_aware")
        direct = load_config(PROJECT_ROOT / "configs/chemprop_multitask.yaml")
        self.assertEqual(resolve_loss_config(direct, "training")["loss_mode"], "standard")

    def test_large_and_xlarge_configs_build_the_declared_capacity(self):
        for name, hidden, depth in (
            ("transfer_singleconc_large", 600, 4),
            ("transfer_singleconc_xlarge", 1024, 5),
        ):
            with self.subTest(config=name):
                config = load_config(PROJECT_ROOT / f"configs/{name}.yaml")
                model = build_model(CYP_TASKS, config)
                self.assertIsInstance(model, CYPDMPNN)
                self.assertEqual(model.encoder.hidden_dim, hidden)
                self.assertEqual(model.encoder.depth, depth)
                self.assertEqual(config["model"]["ffn_num_layers"], 3)
                self.assertEqual(resolve_loss_config(config, "pretraining")["loss_mode"], "standard")
                self.assertEqual(
                    resolve_loss_config(config, "transfer")["loss_mode"], "interval_aware"
                )
                counts = parameter_counts(model)
                self.assertGreater(counts["encoder_parameters"], 0)
                self.assertEqual(
                    counts["total_parameters"],
                    counts["encoder_parameters"] + counts["head_parameters"]
                    + counts["other_parameters"],
                )

    def test_every_new_config_parses_and_builds(self):
        for name in (
            "transfer_singleconc_search", "gine_multitask", "gine_transfer_singleconc",
            "target_conditioned_transfer", "joint_multiassay",
        ):
            with self.subTest(config=name):
                config = load_config(PROJECT_ROOT / f"configs/{name}.yaml")
                tasks = CYP_TASKS
                if config.get("data", {}).get("task_set") == "joint_multiassay":
                    tasks = tuple(f"{cyp}|direct_pic50" for cyp in CYP_TASKS) + tuple(
                        f"{cyp}|single_concentration" for cyp in CYP_TASKS
                    )
                model = build_model(tasks, config)
                output = model(batch_graphs(["CCO", "c1ccccc1"]))
                self.assertEqual(tuple(output.shape), (2, len(tasks)))


class ResidualHeadTests(unittest.TestCase):
    def test_residual_head_preserves_output_shape(self):
        embedding = torch.randn(5, 16)
        for layer_norm in (False, True):
            with self.subTest(layer_norm=layer_norm):
                head = build_head(16, 12, 3, 0.1, "residual_mlp", layer_norm)
                self.assertEqual(tuple(head(embedding).shape), (5, 1))
        model = CYPDMPNN(
            ("CYP1A2", "CYP2D6"), message_hidden_dim=16, ffn_hidden_dim=12,
            ffn_num_layers=3, head_type="residual_mlp", head_layer_norm=True,
        )
        self.assertEqual(tuple(model(batch_graphs(["CCO", "c1ccccc1"])).shape), (2, 2))

    def test_mlp_head_is_unchanged_and_rejects_layer_norm(self):
        plain = build_head(16, 12, 2, 0.1)
        self.assertEqual(
            [type(module).__name__ for module in plain], ["Linear", "ReLU", "Dropout", "Linear"]
        )
        with self.assertRaisesRegex(ValueError, "head_layer_norm"):
            build_head(16, 12, 2, 0.1, "mlp", True)
        with self.assertRaisesRegex(ValueError, "head_type"):
            build_head(16, 12, 2, 0.1, "deep_mlp")

    def test_residual_head_adds_capacity_over_the_plain_head(self):
        plain = parameter_counts(
            CYPDMPNN(("CYP1A2",), message_hidden_dim=16, ffn_hidden_dim=12, ffn_num_layers=3)
        )
        residual = parameter_counts(
            CYPDMPNN(
                ("CYP1A2",), message_hidden_dim=16, ffn_hidden_dim=12, ffn_num_layers=3,
                head_type="residual_mlp",
            )
        )
        self.assertEqual(plain["encoder_parameters"], residual["encoder_parameters"])
        self.assertGreater(residual["head_parameters"], plain["head_parameters"])


class TargetConditionedTests(unittest.TestCase):
    def test_target_conditioned_model_masks_missing_targets(self):
        config = {
            "model": {
                "type": "dmpnn", "architecture": "target_conditioned",
                "message_hidden_dim": 16, "message_passing_depth": 2, "dropout": 0.1,
                "aggregation": "mean", "cyp_embedding_dim": 4,
                "predictor_hidden_dim": 8, "predictor_layers": 2,
            }
        }
        model = build_model(CYP_TASKS, config)
        prediction = model(batch_graphs(["CCO", "c1ccccc1"]))
        self.assertEqual(tuple(prediction.shape), (2, 4))
        target = torch.tensor(
            [[1.0, float("nan"), 2.0, float("nan")], [0.5, float("nan"), 1.5, float("nan")]]
        )
        bounds = torch.full_like(target, float("nan"))
        weights = torch.ones_like(target)
        full = masked_loss(prediction, target, bounds, bounds, weights, "standard")
        observed_only = masked_loss(
            prediction[:, [0, 2]], target[:, [0, 2]], bounds[:, [0, 2]],
            bounds[:, [0, 2]], weights[:, [0, 2]], "standard",
        )
        # Unobserved CYPs contribute nothing, so the masked loss equals the loss over
        # the observed endpoints alone.
        self.assertAlmostEqual(
            float(full.detach()), float(observed_only.detach()), places=6
        )
        full.backward()
        self.assertTrue(torch.isfinite(model.heads.cyp_embedding.weight.grad).all())

    def test_one_shared_predictor_is_used_for_every_cyp(self):
        heads = TargetConditionedHeads(
            CYP_TASKS, input_dim=8, cyp_embedding_dim=4, predictor_hidden_dim=6,
            predictor_layers=2,
        )
        self.assertEqual(tuple(heads(torch.randn(3, 8)).shape), (3, 4))
        self.assertEqual(tuple(heads.cyp_embedding.weight.shape), (4, 4))
        self.assertIsNone(heads.assay_embedding)
        # Different CYPs must produce different predictions from the same molecule.
        embedding = torch.randn(1, 8)
        values = heads(embedding).detach().numpy().ravel()
        self.assertGreater(float(np.std(values)), 0.0)

    def test_joint_multiassay_heads_condition_on_cyp_and_assay(self):
        tasks = tuple(f"{cyp}|direct_pic50" for cyp in CYP_TASKS) + tuple(
            f"{cyp}|single_concentration" for cyp in CYP_TASKS
        )
        heads = TargetConditionedHeads(
            tasks, input_dim=8, cyp_embedding_dim=4, predictor_hidden_dim=6,
            predictor_layers=2, assay_conditioned=True, assay_embedding_dim=3,
        )
        self.assertEqual(tuple(heads(torch.randn(2, 8)).shape), (2, 8))
        self.assertEqual(tuple(heads.assay_embedding.weight.shape), (2, 3))
        # The embedding tables are sized from fixed vocabularies, not the active tasks,
        # so a single-endpoint refinement can reload a full multitask checkpoint.
        self.assertEqual(tuple(heads.cyp_embedding.weight.shape), (4, 4))
        with self.assertRaisesRegex(ValueError, "CYP\\|assay"):
            TargetConditionedHeads(CYP_TASKS, input_dim=8, assay_conditioned=True)


class GINETests(unittest.TestCase):
    def test_gine_encoder_output_dimensions(self):
        encoder = build_encoder({"type": "gine", "hidden_dim": 24, "num_layers": 2})
        self.assertEqual(encoder_type({"type": "gine"}), "gine")
        self.assertEqual(encoder_output_dim({"type": "gine", "hidden_dim": 24}), 24)
        # Includes a single-atom molecule, which has no bonds at all.
        graph = batch_graphs(["CCO", "c1ccccc1", "C"])
        self.assertEqual(tuple(encoder(graph).shape), (3, 24))
        model = build_model(
            CYP_TASKS,
            {
                "model": {
                    "type": "gine", "hidden_dim": 24, "num_layers": 2,
                    "ffn_hidden_dim": 12, "ffn_num_layers": 2,
                }
            },
        )
        self.assertEqual(tuple(model(graph).shape), (3, 4))
        self.assertEqual(parameter_counts(model)["encoder_parameters"], sum(
            parameter.numel() for parameter in model.encoder.parameters()
        ))

    def test_gine_pretraining_to_finetuning_checkpoint_transfer(self):
        from src.training import train_dmpnn

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seed_everything(11)
            pretrain_config = small_gine_config("gine-pretrain")
            pretrain_run = RunContext(root / "pretrain", resume=False)
            train_dmpnn(
                synthetic_bundle("single_concentration_pretraining"), pretrain_config,
                pretrain_run, 0, 11, "cpu", None, stage="pretraining",
            )
            encoder_checkpoint = pretrain_run.checkpoints / "best_encoder.pt"
            self.assertTrue(encoder_checkpoint.exists())
            saved = torch.load(encoder_checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(saved["architecture_type"], "gine")
            self.assertEqual(saved["loss_mode"], "standard")

            finetune_config = small_gine_config("gine-finetune")
            finetune_config["transfer"] = {"loss_mode": "interval_aware"}
            finetune_run = RunContext(root / "finetune", resume=False)
            result = train_dmpnn(
                synthetic_bundle("direct_pic50"), finetune_config, finetune_run, 0, 11,
                "cpu", None, encoder_checkpoint=encoder_checkpoint, transfer_mode="full",
                stage="transfer",
            )
            preload = result["pretrained_encoder"]
            self.assertGreater(preload["loaded_key_count"], 0)
            self.assertEqual(preload["source_architecture_type"], "gine")
            self.assertEqual(result["pretraining_loss_mode"], "standard")
            self.assertEqual(result["finetuning_loss_mode"], "interval_aware")

            # A mismatched GINE encoder width must still be rejected.
            wrong = small_gine_config("gine-wrong")
            wrong["model"]["hidden_dim"] = 16
            with self.assertRaisesRegex(ValueError, "Incompatible encoder checkpoint"):
                load_pretrained_encoder(
                    build_model(("CYP1A2",), wrong), encoder_checkpoint, wrong["model"]
                )

    def test_unknown_encoder_type_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported neural architecture type"):
            build_model(("CYP1A2",), {"model": {"type": "transformer"}})


class StageLossTests(unittest.TestCase):
    def test_stage_specific_loss_selection(self):
        config = {
            "loss": {"loss_mode": "interval_aware", "weight_clip_min": 0.3},
            "pretraining": {"loss_mode": "standard"},
            "transfer": {},
        }
        self.assertEqual(resolve_loss_config(config, "pretraining")["loss_mode"], "standard")
        self.assertEqual(resolve_loss_config(config, "transfer")["loss_mode"], "interval_aware")
        self.assertEqual(resolve_loss_config(config, None)["loss_mode"], "interval_aware")
        # Options that a stage does not override still come from the shared block.
        self.assertEqual(resolve_loss_config(config, "pretraining")["weight_clip_min"], 0.3)

    def test_nested_stage_loss_mapping_and_defaults(self):
        nested = {
            "loss": {"loss_mode": "standard"},
            "transfer": {"loss": {"loss_mode": "clipped_uncertainty_weighted"}},
        }
        self.assertEqual(
            resolve_loss_config(nested, "transfer")["loss_mode"], "clipped_uncertainty_weighted"
        )
        self.assertEqual(resolve_loss_config({}, "transfer")["loss_mode"], "standard")
        with self.assertRaisesRegex(ValueError, "must be a mapping"):
            resolve_loss_config({"transfer": {"loss": "interval_aware"}}, "transfer")


class TransferStageTests(unittest.TestCase):
    def test_checkpoint_self_validation_uses_recorded_encoder_type(self):
        checkpoint = {
            "architecture_type": "gine",
            "model_config": {
                "type": "gine", "hidden_dim": 8, "num_layers": 2, "dropout": 0.1,
                "residual": True, "normalization": "layernorm", "pooling": "mean",
            },
            "task_names": ("CYP1A2",),
            "model_code_version": MODEL_CODE_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "pytorch_version": torch.__version__,
            "chemprop_version": "self-contained-chemprop-style",
        }
        diagnostics = validate_encoder_checkpoint(checkpoint)
        self.assertEqual(diagnostics["metadata_mismatches"], {})

    def test_single_stage_transfer_is_unchanged(self):
        source, records = resolve_transfer_stages({}, None, "direct_pic50")
        self.assertIsNone(source)
        self.assertEqual([record["name"] for record in records], ["direct_pic50"])
        self.assertEqual(records[0]["role"], "current")

    def test_declared_stages_record_each_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "encoder.pt"
            model = CYPDMPNN(("CYP1A2",), message_hidden_dim=8, ffn_hidden_dim=8)
            atomic_torch_save(
                {
                    "architecture_type": "dmpnn",
                    "model_config": {"type": "dmpnn", "message_hidden_dim": 8},
                    "task_names": ("CYP1A2",), "encoder_state_dict": model.encoder.state_dict(),
                    "loss_mode": "standard", "stage": "pretraining",
                    "split_metadata": {"purpose": "single_concentration_pretraining"},
                },
                path,
            )
            config = {
                "transfer": {
                    "stages": [
                        {"name": "molecular_pretraining", "checkpoint": None},
                        {"name": "single_concentration", "checkpoint": str(path)},
                        {"name": "direct_pic50"},
                    ]
                }
            }
            source, records = resolve_transfer_stages(config, None, "direct_pic50")
            self.assertEqual(source, str(path))
            self.assertEqual(
                [record["name"] for record in records],
                ["molecular_pretraining", "single_concentration", "direct_pic50"],
            )
            used = records[1]
            self.assertTrue(used["used_for_initialization"])
            self.assertEqual(used["source_task"], "single_concentration_pretraining")
            self.assertEqual(used["loss_mode"], "standard")
            self.assertEqual(len(used["sha256"]), 64)
            # An explicit --checkpoint overrides the stage immediately before the current one.
            overridden, _records = resolve_transfer_stages(config, path, "direct_pic50")
            self.assertEqual(overridden, str(path))

    def test_malformed_stage_declarations_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one source stage"):
            resolve_transfer_stages(
                {"transfer": {"stages": [{"name": "only"}]}}, None, "direct_pic50"
            )
        with self.assertRaisesRegex(ValueError, "unique"):
            resolve_transfer_stages(
                {"transfer": {"stages": [{"name": "a"}, {"name": "a"}]}}, None, "direct_pic50"
            )


class EndpointRefinementTests(unittest.TestCase):
    def test_partial_load_allows_only_head_shape_mismatches(self):
        source = CYPDMPNN(("CYP2D6",), message_hidden_dim=8, ffn_hidden_dim=16)
        destination = CYPDMPNN(("CYP2D6",), message_hidden_dim=8, ffn_hidden_dim=4)
        diagnostics = load_model_state_for_tasks(
            destination, {"model_state_dict": source.state_dict()}, ("CYP2D6",),
            partial_load=True,
        )
        self.assertTrue(diagnostics["skipped_head_shape_keys"])
        for key, value in source.encoder.state_dict().items():
            self.assertTrue(torch.equal(destination.encoder.state_dict()[key], value))

    def test_refinement_loads_the_encoder_and_only_the_selected_head(self):
        source = CYPDMPNN(CYP_TASKS, message_hidden_dim=8, ffn_hidden_dim=8)
        checkpoint = {
            "model_state_dict": source.state_dict(),
            "task_names": CYP_TASKS,
        }
        destination = CYPDMPNN(("CYP2D6",), message_hidden_dim=8, ffn_hidden_dim=8)
        diagnostics = load_model_state_for_tasks(destination, checkpoint, ("CYP2D6",))
        self.assertEqual(diagnostics["missing_head_keys"], [])
        for key, value in source.encoder.state_dict().items():
            self.assertTrue(torch.equal(destination.encoder.state_dict()[key], value))
        for key, value in source.heads["CYP2D6"].state_dict().items():
            self.assertTrue(torch.equal(destination.heads["CYP2D6"].state_dict()[key], value))
        # The other endpoints' heads are simply not part of the refinement model.
        self.assertEqual(list(destination.heads), ["CYP2D6"])
        self.assertTrue(
            any(key.startswith("heads.CYP1A2") for key in diagnostics["skipped_source_keys"])
        )

    def test_refinement_reloads_a_target_conditioned_multitask_checkpoint(self):
        """The conditioning index buffers must not size the checkpoint by task count."""
        config = {
            "model": {
                "type": "dmpnn", "architecture": "target_conditioned",
                "message_hidden_dim": 16, "message_passing_depth": 2, "dropout": 0.1,
                "aggregation": "mean", "cyp_embedding_dim": 4,
                "predictor_hidden_dim": 8, "predictor_layers": 2,
            }
        }
        source = build_model(CYP_TASKS, config)
        self.assertEqual(
            [key for key in source.state_dict() if "indices" in key], [],
            "derived conditioning buffers must stay out of state_dict",
        )
        destination = build_model(("CYP2D6",), config)
        diagnostics = load_model_state_for_tasks(
            destination, {"model_state_dict": source.state_dict()}, ("CYP2D6",)
        )
        self.assertEqual(diagnostics["missing_head_keys"], [])
        # The shared predictor and the full CYP embedding table both transfer intact.
        for key, value in source.heads.state_dict().items():
            self.assertTrue(torch.equal(destination.heads.state_dict()[key], value))
        self.assertEqual(tuple(destination.heads.cyp_embedding.weight.shape), (4, 4))
        self.assertEqual(tuple(destination(batch_graphs(["CCO"])).shape), (1, 1))

    def test_refinement_reloads_a_joint_multiassay_checkpoint(self):
        config = {
            "model": {
                "type": "dmpnn", "architecture": "joint_multiassay",
                "message_hidden_dim": 16, "message_passing_depth": 2, "dropout": 0.1,
                "aggregation": "mean", "cyp_embedding_dim": 4, "assay_embedding_dim": 3,
                "predictor_hidden_dim": 8, "predictor_layers": 2,
            }
        }
        tasks = tuple(f"{cyp}|direct_pic50" for cyp in CYP_TASKS) + tuple(
            f"{cyp}|single_concentration" for cyp in CYP_TASKS
        )
        source = build_model(tasks, config)
        destination = build_model(("CYP2D6|direct_pic50",), config)
        diagnostics = load_model_state_for_tasks(
            destination, {"model_state_dict": source.state_dict()}, ("CYP2D6|direct_pic50",)
        )
        self.assertEqual(diagnostics["missing_head_keys"], [])
        self.assertEqual(tuple(destination(batch_graphs(["CCO", "CCN"])).shape), (2, 1))

    def test_incompatible_encoder_width_is_rejected(self):
        source = CYPDMPNN(CYP_TASKS, message_hidden_dim=8, ffn_hidden_dim=8)
        destination = CYPDMPNN(("CYP2D6",), message_hidden_dim=16, ffn_hidden_dim=8)
        with self.assertRaisesRegex(ValueError, "incompatible"):
            load_model_state_for_tasks(
                destination, {"model_state_dict": source.state_dict()}, ("CYP2D6",)
            )

    def test_frozen_encoder_excludes_encoder_parameters_from_training(self):
        from src.training import train_dmpnn

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seed_everything(5)
            config = {
                "experiment": "refine-test",
                "model": {
                    "type": "dmpnn", "message_hidden_dim": 8, "message_passing_depth": 2,
                    "dropout": 0.1, "aggregation": "mean", "ffn_hidden_dim": 8,
                    "ffn_num_layers": 2, "ffn_dropout": 0.1,
                },
                "data": {"target_normalization": "zscore_per_target"},
                "training": {
                    "batch_size": 4, "max_epochs": 1, "optimizer": "adamw",
                    "learning_rate": 1e-3, "selection_metric": "macro_MAE",
                    "metric_direction": "minimize", "precision": "fp32",
                },
                "loss": {"loss_mode": "standard"},
            }
            run = RunContext(root / "seed", resume=False)
            train_dmpnn(synthetic_bundle(), config, run, 0, 5, "cpu", None)
            before = torch.load(
                run.checkpoints / "best.pt", map_location="cpu", weights_only=False
            )
            refine_run = RunContext(root / "refine", resume=False)
            result = train_dmpnn(
                synthetic_bundle(), config, refine_run, 0, 5, "cpu", None,
                stage="endpoint_refinement", initial_checkpoint=run.checkpoints / "best.pt",
                freeze_encoder=True, transfer_mode="endpoint_refinement_frozen",
            )
            self.assertFalse(result["encoder_trainable"])
            self.assertIsNotNone(result["endpoint_refinement_source"])
            after = torch.load(
                refine_run.checkpoints / "best.pt", map_location="cpu", weights_only=False
            )
            for key, value in before["encoder_state_dict"].items():
                self.assertTrue(
                    torch.equal(after["encoder_state_dict"][key], value),
                    f"frozen encoder parameter changed: {key}",
                )
            self.assertLess(
                result["trainable_parameters"], result["total_parameters"]
            )


class ExperimentMatrixTests(unittest.TestCase):
    RECORDER = (
        "import json, os, sys, time\n"
        "from pathlib import Path\n"
        "output = Path(sys.argv[sys.argv.index('--output-dir') + 1])\n"
        "output.mkdir(parents=True, exist_ok=True)\n"
        "(output / f'job_{time.time_ns()}_{os.getpid()}.json').write_text(json.dumps(sys.argv))\n"
    )

    def launch(self, directory: Path, config_body: str, extra: list[str]) -> list[list[str]]:
        script = directory / "recorder.py"
        script.write_text(self.RECORDER, encoding="utf-8")
        config = directory / "config.yaml"
        config.write_text(config_body, encoding="utf-8")
        output = directory / "outputs"
        result = subprocess.run(
            [
                sys.executable, "scripts/run_matrix.py", "--script", str(script),
                "--config", str(config), "--folds", "0", "1", "--seeds", "7",
                "--devices", "cpu", "cpu", "--output-dir", str(output), *extra,
            ],
            cwd=PROJECT_ROOT, capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return [json.loads(path.read_text()) for path in sorted(output.glob("job_*.json"))]

    def test_experiment_matrix_expands_only_the_named_configurations(self):
        body = (
            "experiments:\n"
            "  - name: h600_d4\n"
            "    overrides:\n"
            "      model.message_hidden_dim: 600\n"
            "      model.message_passing_depth: 4\n"
            "  - name: h1024_d5\n"
            "    overrides:\n"
            "      model.message_hidden_dim: 1024\n"
            "search_space:\n"
            "  model.message_hidden_dim: [300, 600, 1024]\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            jobs = self.launch(Path(directory), body, ["--experiment-matrix"])
        # Two named configurations x two folds x one seed, and nothing else.
        self.assertEqual(len(jobs), 4)
        names = sorted(job[job.index("--run-name") + 1] for job in jobs)
        self.assertEqual(names, ["h1024_d5", "h1024_d5", "h600_d4", "h600_d4"])
        first = next(job for job in jobs if job[job.index("--run-name") + 1] == "h600_d4")
        overrides = [first[index + 1] for index, token in enumerate(first) if token == "--set"]
        self.assertEqual(
            sorted(overrides), ["model.message_hidden_dim=600", "model.message_passing_depth=4"]
        )
        second = next(job for job in jobs if job[job.index("--run-name") + 1] == "h1024_d5")
        self.assertEqual(
            [second[index + 1] for index, token in enumerate(second) if token == "--set"],
            ["model.message_hidden_dim=1024"],
        )

    def test_experiment_matrix_uses_selected_section_and_checkpoint_alias(self):
        body = (
            "pretraining_experiments:\n"
            "  - name: pre_h600\n"
            "    overrides:\n"
            "      model.message_hidden_dim: 600\n"
            "experiments:\n"
            "  - name: finetune_lr\n"
            "    checkpoint_run_name: pre_h600\n"
            "    overrides:\n"
            "      transfer.encoder_learning_rate: 0.0001\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            jobs = self.launch(
                Path(directory), body,
                [
                    "--experiment-matrix", "--checkpoint-template",
                    "pre/{checkpoint_run_name}/fold{fold}/seed{seed}/best.pt",
                ],
            )
        self.assertEqual(len(jobs), 2)
        for job in jobs:
            checkpoint = job[job.index("--checkpoint") + 1]
            self.assertIn("pre/pre_h600/", checkpoint)

        with tempfile.TemporaryDirectory() as directory:
            jobs = self.launch(
                Path(directory), body,
                ["--experiment-matrix", "--experiment-matrix-section", "pretraining_experiments"],
            )
        self.assertEqual(len(jobs), 2)
        self.assertEqual(
            {job[job.index("--run-name") + 1] for job in jobs}, {"pre_h600"}
        )

    def test_grid_expansion_is_unchanged(self):
        body = (
            "search_grid:\n"
            "  model.message_hidden_dim: [300, 600]\n"
            "  model.dropout: [0.1]\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            jobs = self.launch(Path(directory), body, ["--grid"])
        self.assertEqual(len(jobs), 4)
        self.assertEqual(
            sorted({job[job.index("--run-name") + 1] for job in jobs}), ["grid000", "grid001"]
        )

    def test_plain_launch_requests_no_run_name_or_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            jobs = self.launch(Path(directory), "model:\n  type: dmpnn\n", [])
        self.assertEqual(len(jobs), 2)
        for job in jobs:
            self.assertNotIn("--run-name", job)
            self.assertNotIn("--set", job)

    def test_grid_and_experiment_matrix_are_mutually_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "recorder.py"
            script.write_text(self.RECORDER, encoding="utf-8")
            config = Path(directory) / "config.yaml"
            config.write_text("experiments: []\n", encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable, "scripts/run_matrix.py", "--script", str(script),
                    "--config", str(config), "--folds", "0", "--seeds", "1",
                    "--devices", "cpu", "--output-dir", str(Path(directory) / "outputs"),
                    "--grid", "--experiment-matrix",
                ],
                cwd=PROJECT_ROOT, capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("mutually exclusive", result.stderr)


class ComparisonSafetyTests(unittest.TestCase):
    def test_macro_comparison_rejects_incompatible_endpoint_sets(self):
        frame = pd.DataFrame(
            [
                {"model": "four", "fold": 0, "seed": 1, "macro_ST_RAE": 0.8,
                 "CYP_set": "CYP1A2|CYP2C9|CYP2D6|CYP3A4", "scored_CYPs": 4},
                {"model": "single", "fold": 0, "seed": 1, "macro_ST_RAE": 0.1,
                 "CYP_set": "CYP2D6", "scored_CYPs": 1},
            ]
        )
        comparison, reference = model_comparison_table(frame)
        self.assertEqual(reference, "four")
        single = comparison.loc[comparison["model"].eq("single")].iloc[0]
        self.assertEqual(single["comparison_status"], "incompatible_endpoint_set")
        self.assertTrue(np.isnan(single["delta_vs_reference"]))

    def test_residual_comparison_rejects_disagreeing_truth(self):
        base = pd.DataFrame(
            {"molecule_id": ["m1", "m2", "m3"], "CYP": ["CYP2D6"] * 3,
             "y_true": [1.0, 2.0, 3.0], "y_pred": [1.1, 2.1, 3.1]}
        )
        changed = base.copy()
        changed.loc[0, "y_true"] = 9.0
        with self.assertRaisesRegex(ValueError, "truth values disagree"):
            residual_complementarity({"a": base, "b": changed})


if __name__ == "__main__":
    unittest.main()
