"""Unit tests for PrivDisen models, losses, and utilities.

Run with::

    python -m pytest tests/ -v
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BATCH = 8
INPUT_DIM = 784
HIDDEN_DIM = 64
SHARED_DIM = 16
PRIVATE_DIM = 16
NUM_CLASSES = 10
NUM_SENSITIVE = 2


def make_batch():
    """Return a random (x, y, s) batch."""
    x = torch.rand(BATCH, INPUT_DIM)
    y = torch.randint(0, NUM_CLASSES, (BATCH,))
    s = torch.randint(0, NUM_SENSITIVE, (BATCH,))
    return x, y, s


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestSharedEncoder:
    def test_output_shape(self):
        from src.models.encoder import SharedEncoder
        enc = SharedEncoder(INPUT_DIM, HIDDEN_DIM, SHARED_DIM)
        x = torch.rand(BATCH, INPUT_DIM)
        mu, logvar = enc(x)
        assert mu.shape == (BATCH, SHARED_DIM)
        assert logvar.shape == (BATCH, SHARED_DIM)

    def test_gradient_flows(self):
        from src.models.encoder import SharedEncoder
        enc = SharedEncoder(INPUT_DIM, HIDDEN_DIM, SHARED_DIM)
        x = torch.rand(BATCH, INPUT_DIM)
        mu, logvar = enc(x)
        (mu.sum() + logvar.sum()).backward()
        for p in enc.parameters():
            assert p.grad is not None


class TestPrivateEncoder:
    def test_output_shape(self):
        from src.models.encoder import PrivateEncoder
        enc = PrivateEncoder(INPUT_DIM, HIDDEN_DIM, PRIVATE_DIM)
        x = torch.rand(BATCH, INPUT_DIM)
        mu, logvar = enc(x)
        assert mu.shape == (BATCH, PRIVATE_DIM)
        assert logvar.shape == (BATCH, PRIVATE_DIM)


class TestTaskHead:
    def test_output_shape(self):
        from src.models.encoder import TaskHead
        head = TaskHead(SHARED_DIM, NUM_CLASSES)
        z_s = torch.rand(BATCH, SHARED_DIM)
        out = head(z_s)
        assert out.shape == (BATCH, NUM_CLASSES)


class TestDecoder:
    def test_output_shape(self):
        from src.models.decoder import Decoder
        dec = Decoder(SHARED_DIM, PRIVATE_DIM, HIDDEN_DIM, INPUT_DIM)
        z_s = torch.rand(BATCH, SHARED_DIM)
        z_p = torch.rand(BATCH, PRIVATE_DIM)
        x_recon = dec(z_s, z_p)
        assert x_recon.shape == (BATCH, INPUT_DIM)

    def test_output_range(self):
        """Decoder uses Sigmoid, so output must be in [0, 1]."""
        from src.models.decoder import Decoder
        dec = Decoder(SHARED_DIM, PRIVATE_DIM, HIDDEN_DIM, INPUT_DIM)
        z_s = torch.rand(BATCH, SHARED_DIM)
        z_p = torch.rand(BATCH, PRIVATE_DIM)
        x_recon = dec(z_s, z_p)
        assert x_recon.min().item() >= 0.0
        assert x_recon.max().item() <= 1.0


class TestPrivacyDiscriminator:
    def test_output_shape(self):
        from src.models.discriminator import PrivacyDiscriminator
        disc = PrivacyDiscriminator(SHARED_DIM, NUM_SENSITIVE)
        z_s = torch.rand(BATCH, SHARED_DIM)
        logits = disc(z_s)
        assert logits.shape == (BATCH, NUM_SENSITIVE)


class TestReparameterise:
    def test_shape(self):
        from src.models.privdisen import reparameterise
        mu = torch.zeros(BATCH, SHARED_DIM)
        logvar = torch.zeros(BATCH, SHARED_DIM)
        z = reparameterise(mu, logvar)
        assert z.shape == (BATCH, SHARED_DIM)

    def test_mean_close_to_mu(self):
        """With logvar=0, std=1. Over many samples, mean ≈ mu."""
        from src.models.privdisen import reparameterise
        mu = torch.full((10000, 1), 3.0)
        logvar = torch.zeros(10000, 1)
        z = reparameterise(mu, logvar)
        assert abs(z.mean().item() - 3.0) < 0.1


class TestPrivDisen:
    def _make_model(self):
        from src.models.privdisen import PrivDisen
        return PrivDisen(
            input_dim=INPUT_DIM,
            hidden_dim=HIDDEN_DIM,
            shared_dim=SHARED_DIM,
            private_dim=PRIVATE_DIM,
            num_classes=NUM_CLASSES,
            num_sensitive=NUM_SENSITIVE,
        )

    def test_forward_keys(self):
        model = self._make_model()
        x, _, _ = make_batch()
        out = model(x)
        for key in ["x_recon", "y_pred", "s_pred", "z_s", "z_p", "mu_s", "logvar_s", "mu_p", "logvar_p"]:
            assert key in out, f"Missing key: {key}"

    def test_forward_shapes(self):
        model = self._make_model()
        x, _, _ = make_batch()
        out = model(x)
        assert out["x_recon"].shape == (BATCH, INPUT_DIM)
        assert out["y_pred"].shape == (BATCH, NUM_CLASSES)
        assert out["s_pred"].shape == (BATCH, NUM_SENSITIVE)
        assert out["z_s"].shape == (BATCH, SHARED_DIM)
        assert out["z_p"].shape == (BATCH, PRIVATE_DIM)

    def test_encoder_parameters_count(self):
        model = self._make_model()
        enc_params = set(id(p) for p in model.encoder_parameters)
        disc_params = set(id(p) for p in model.discriminator_parameters)
        assert len(enc_params & disc_params) == 0, "Encoder and discriminator params must be disjoint"


# ---------------------------------------------------------------------------
# Loss tests
# ---------------------------------------------------------------------------

class TestLosses:
    def test_reconstruction_loss_scalar(self):
        from src.losses.losses import reconstruction_loss
        x = torch.rand(BATCH, INPUT_DIM)
        x_recon = torch.rand(BATCH, INPUT_DIM)
        loss = reconstruction_loss(x, x_recon)
        assert loss.ndim == 0  # scalar
        assert loss.item() > 0

    def test_reconstruction_loss_perfect(self):
        """When x_recon == x the loss should be near 0 (using near-zero values)."""
        from src.losses.losses import reconstruction_loss
        x = torch.zeros(BATCH, INPUT_DIM)
        loss = reconstruction_loss(x, x.clone())
        assert loss.item() < 0.01

    def test_kl_divergence_non_negative(self):
        from src.losses.losses import kl_divergence
        mu = torch.randn(BATCH, SHARED_DIM)
        logvar = torch.zeros(BATCH, SHARED_DIM)
        loss = kl_divergence(mu, logvar, mu, logvar)
        # KL divergence is non-negative
        assert loss.item() >= 0

    def test_kl_divergence_zero_at_prior(self):
        """KL(N(0,1) || N(0,1)) = 0."""
        from src.losses.losses import kl_divergence
        mu = torch.zeros(BATCH, SHARED_DIM)
        logvar = torch.zeros(BATCH, SHARED_DIM)
        loss = kl_divergence(mu, logvar, mu, logvar)
        assert abs(loss.item()) < 1e-5

    def test_task_loss_scalar(self):
        from src.losses.losses import task_loss
        logits = torch.randn(BATCH, NUM_CLASSES)
        labels = torch.randint(0, NUM_CLASSES, (BATCH,))
        loss = task_loss(logits, labels)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_mutual_information_penalty_range(self):
        from src.losses.losses import mutual_information_penalty
        z_s = torch.randn(BATCH, SHARED_DIM)
        z_p = torch.randn(BATCH, PRIVATE_DIM)
        mi = mutual_information_penalty(z_s, z_p)
        assert 0.0 <= mi.item() <= 1.0

    def test_mi_penalty_zero_for_orthogonal(self):
        """Two orthogonal vectors should have near-zero cosine similarity."""
        from src.losses.losses import mutual_information_penalty
        # Construct perfectly orthogonal batches
        z_s = torch.zeros(BATCH, SHARED_DIM)
        z_s[:, 0] = 1.0  # unit vector along dim 0
        z_p = torch.zeros(BATCH, PRIVATE_DIM)
        z_p[:, 1] = 1.0  # unit vector along dim 1
        mi = mutual_information_penalty(z_s, z_p)
        assert mi.item() < 1e-5

    def test_adversarial_privacy_loss_scalar(self):
        from src.losses.losses import adversarial_privacy_loss
        logits = torch.randn(BATCH, NUM_SENSITIVE)
        labels = torch.randint(0, NUM_SENSITIVE, (BATCH,))
        loss = adversarial_privacy_loss(logits, labels)
        assert loss.ndim == 0

    def test_compute_total_loss_keys(self):
        from src.losses.losses import compute_total_loss
        from src.models.privdisen import PrivDisen
        model = PrivDisen(
            input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM,
            shared_dim=SHARED_DIM, private_dim=PRIVATE_DIM,
            num_classes=NUM_CLASSES, num_sensitive=NUM_SENSITIVE,
        )
        x, y, s = make_batch()
        outputs = model(x)
        losses = compute_total_loss(outputs, x, y, s)
        for k in ["total", "recon", "task", "kl", "mi", "adv"]:
            assert k in losses


# ---------------------------------------------------------------------------
# Utility tests
# ---------------------------------------------------------------------------

class TestUtils:
    def test_accuracy_perfect(self):
        from src.utils.utils import accuracy
        logits = torch.eye(5)  # one-hot, trivial argmax
        labels = torch.arange(5)
        assert accuracy(logits, labels) == 1.0

    def test_accuracy_zero(self):
        from src.utils.utils import accuracy
        # All predictions wrong: argmax is 0, labels are 1
        logits = torch.zeros(5, 2)
        logits[:, 0] = 1.0  # predicts class 0
        labels = torch.ones(5, dtype=torch.long)  # ground truth is class 1
        assert accuracy(logits, labels) == 0.0

    def test_average_meter(self):
        from src.utils.utils import AverageMeter
        meter = AverageMeter("test")
        meter.update(1.0, n=2)
        meter.update(3.0, n=2)
        assert abs(meter.avg - 2.0) < 1e-6

    def test_average_meter_reset(self):
        from src.utils.utils import AverageMeter
        meter = AverageMeter()
        meter.update(5.0)
        meter.reset()
        assert meter.avg == 0.0
        assert meter.count == 0

    def test_set_seed_reproducibility(self):
        from src.utils.utils import set_seed
        set_seed(42)
        a = torch.randn(10)
        set_seed(42)
        b = torch.randn(10)
        assert torch.allclose(a, b)

    def test_save_load_checkpoint(self, tmp_path):
        from src.utils.utils import save_checkpoint, load_checkpoint
        state = {"epoch": 5, "loss": 0.123}
        save_checkpoint(state, tmp_path, filename="ckpt.pt")
        loaded = load_checkpoint(tmp_path / "ckpt.pt")
        assert loaded["epoch"] == 5
        assert abs(loaded["loss"] - 0.123) < 1e-6


# ---------------------------------------------------------------------------
# Dataset wrapper test (no download required)
# ---------------------------------------------------------------------------

class TestDatasetWrapper:
    def test_synthetic_sensitive(self):
        from data.datasets import DatasetWrapper

        class DummyDataset:
            def __len__(self):
                return 10
            def __getitem__(self, idx):
                return torch.rand(INPUT_DIM), idx % NUM_CLASSES

        ds = DatasetWrapper(DummyDataset())
        x, y, s = ds[0]
        assert isinstance(s, int)
        assert s == (0 % 2)

    def test_custom_sensitive_fn(self):
        from data.datasets import DatasetWrapper

        class DummyDataset:
            def __len__(self):
                return 10
            def __getitem__(self, idx):
                return torch.rand(INPUT_DIM), 3

        ds = DatasetWrapper(DummyDataset(), sensitive_fn=lambda y: int(y > 5))
        _, y, s = ds[0]
        assert s == 0  # 3 > 5 is False → 0
