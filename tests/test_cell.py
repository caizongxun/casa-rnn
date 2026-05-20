import torch
import pytest
from casa_rnn import (
    CASARNNCell, CASARNNModel,
    MultiScaleCASARNN, CounterfactualLoss,
    MemoryBank, UncertaintyGatedHead,
)

DEVICE = torch.device("cpu")
B, I, H, T = 4, 16, 32, 20


def test_cell_forward():
    cell = CASARNNCell(I, H)
    x = torch.randn(B, I)
    h_s, h_f = torch.zeros(B, H), torch.zeros(B, H)
    h_s2, h_f2, reg, pe = cell(x, h_s, h_f)
    assert h_s2.shape == (B, H)
    assert h_f2.shape == (B, H)
    assert reg.shape  == (B,)
    assert pe.shape   == (B, 1)


def test_model_forward():
    model = CASARNNModel(I, H, num_layers=2, output_size=1)
    x = torch.randn(B, T, I)
    out, extra = model(x)
    assert out.shape == (B, T, 1)


def test_multiscale_forward():
    model = MultiScaleCASARNN(I, H, output_size=1, use_memory=True)
    x = torch.randn(B, T, I)
    means, stds, extra = model(x)
    assert means.shape  == (B, T, 1)
    assert stds.shape   == (B, T, 1)
    assert (stds > 0).all(), "std must be positive"
    assert extra["scale_weights"].shape == (B, T, 3)


def test_uncertainty_head():
    head = UncertaintyGatedHead(H, 1)
    h = torch.randn(B, H)
    mean, std = head(h)
    assert mean.shape == (B, 1)
    assert std.shape  == (B, 1)
    assert (std > 0).all()


def test_memory_bank():
    mem = MemoryBank(H, num_slots=8)
    h = torch.randn(B, H)
    out = mem.read(h)
    assert out.shape == (B, H)
    regime = torch.ones(B) * 0.9  # high regime -> should write
    mem.write(h, regime)


def test_counterfactual_loss_multiscale():
    model   = MultiScaleCASARNN(I, H, output_size=1, use_memory=True)
    loss_fn = CounterfactualLoss(alpha=0.1, beta=0.01, gamma=0.05, use_nll=True)
    x = torch.randn(B, T, I)
    y = torch.randn(B, T, 1)
    means, stds, extra = model(x)
    loss = loss_fn((means, stds), y, extra)
    loss.backward()
    assert loss.item() > 0
    assert not torch.isnan(loss)


def test_no_nan():
    model = MultiScaleCASARNN(I, H, output_size=1, use_memory=True)
    x = torch.randn(B, T, I)
    means, stds, extra = model(x)
    assert not torch.isnan(means).any()
    assert not torch.isnan(stds).any()
