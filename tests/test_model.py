"""Basic tests for the fraud detection model."""


import torch

from app.model import FraudNet


def test_model_output_shape():
    model = FraudNet(input_size=30)
    x = torch.randn(8, 30)

    logits = model(x)

    assert logits.shape == (8,)
    assert torch.isfinite(logits).all()


def test_model_accepts_single_transaction():
    model = FraudNet(input_size=30)
    x = torch.randn(1, 30)

    logits = model(x)

    assert logits.shape == (1,)

