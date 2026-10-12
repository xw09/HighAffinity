import unittest
from types import SimpleNamespace

import torch

from boltz.model.loss.affinity_loss import affinity_loss, focal_loss, huber_loss


def records(regression, classes):
    return [
        SimpleNamespace(
            affinity=SimpleNamespace(true_affinity=reg, true_class=cls)
        )
        for reg, cls in zip(regression, classes, strict=True)
    ]


def model_output(regression, probabilities):
    return {
        "affinity_pred_value": torch.tensor(
            regression, dtype=torch.float32
        ).reshape(-1, 1).requires_grad_(),
        "affinity_probability_binary": torch.tensor(
            probabilities, dtype=torch.float32
        ).reshape(-1, 1).requires_grad_(),
    }


class AffinityLossTests(unittest.TestCase):
    def test_huber_excludes_missing_labels_from_denominator(self):
        prediction = torch.tensor([[0.0], [100.0], [2.0]], requires_grad=True)
        target = torch.tensor([[0.0], [float("nan")], [2.5]])
        loss, _ = huber_loss(prediction, target, delta=0.5)
        # Valid per-record losses are 0 and 0.125; their mean is 0.0625.
        self.assertAlmostEqual(loss.item(), 0.0625)
        loss.backward()
        self.assertEqual(prediction.grad[1].item(), 0.0)

    def test_numeric_zero_is_a_valid_regression_target(self):
        output = model_output([1.0], [0.5])
        result = affinity_loss(
            output,
            {"record": records([0.0], [1])},
            classification_enabled=False,
        )
        self.assertEqual(
            result["loss_breakdown"]["regression_valid_count"].item(), 1
        )
        self.assertAlmostEqual(result["loss"].item(), 0.375)

    def test_all_missing_regression_labels_produce_differentiable_zero(self):
        output = model_output([1.0, 2.0], [0.4, 0.6])
        result = affinity_loss(
            output,
            {"record": records([None, None], [0, 1])},
            classification_enabled=False,
        )
        self.assertEqual(result["loss"].item(), 0.0)
        result["loss"].backward()
        self.assertTrue(
            torch.equal(
                output["affinity_pred_value"].grad,
                torch.zeros_like(output["affinity_pred_value"]),
            )
        )

    def test_supported_joint_objectives_run_on_cpu(self):
        combinations = [("huber", "focal"), ("mse", "focal"), ("huber", "bce")]
        for regression_loss, classification_loss in combinations:
            with self.subTest(
                regression_loss=regression_loss,
                classification_loss=classification_loss,
            ):
                output = model_output([0.1, 0.9], [0.2, 0.8])
                result = affinity_loss(
                    output,
                    {"record": records([0.0, 1.0], [0, 1])},
                    regression_loss=regression_loss,
                    classification_loss=classification_loss,
                )
                self.assertEqual(result["loss"].device.type, "cpu")
                self.assertTrue(torch.isfinite(result["loss"]))
                result["loss"].backward()

    def test_single_task_switches_and_loss_weights(self):
        feats = {"record": records([0.0, 1.0], [0, 1])}
        reg_only = affinity_loss(
            model_output([0.2, 0.8], [0.4, 0.6]),
            feats,
            classification_enabled=False,
            lambda_reg=2.0,
        )
        cls_only = affinity_loss(
            model_output([0.2, 0.8], [0.4, 0.6]),
            feats,
            regression_enabled=False,
            lambda_cls=3.0,
        )
        self.assertAlmostEqual(
            reg_only["loss"].item(),
            2.0
            * reg_only["loss_breakdown"]["affinity_pred_value_loss"].item(),
        )
        self.assertAlmostEqual(
            cls_only["loss"].item(),
            3.0
            * cls_only["loss_breakdown"][
                "affinity_probability_binary_loss"
            ].item(),
        )

    def test_disabling_both_tasks_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "At least one"):
            affinity_loss(
                model_output([0.0], [0.5]),
                {"record": records([0.0], [1])},
                regression_enabled=False,
                classification_enabled=False,
            )

    def test_focal_alpha_uses_binary_class_weighting(self):
        prob = torch.tensor([[0.5], [0.5]])
        target = torch.tensor([[1.0], [0.0]])
        unweighted = focal_loss(prob, target, gamma=1.0)
        balanced = focal_loss(prob, target, gamma=1.0, alpha=0.5)
        self.assertAlmostEqual(balanced.item(), 0.5 * unweighted.item())

    def test_classification_losses_are_autocast_safe(self):
        output = model_output([0.1, 0.9], [0.2, 0.8])
        feats = {"record": records([0.0, 1.0], [0, 1])}
        for classification_loss in ("focal", "bce"):
            with self.subTest(classification_loss=classification_loss):
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    result = affinity_loss(
                        output,
                        feats,
                        regression_enabled=False,
                        classification_loss=classification_loss,
                    )
                self.assertTrue(torch.isfinite(result["loss"]))


if __name__ == "__main__":
    unittest.main()
