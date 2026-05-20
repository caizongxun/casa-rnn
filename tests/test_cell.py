import torch
import pytest
from casa_rnn import CASARNNCell, CASARNNModel, CounterfactualLoss


DEVICE = torch.device("cpu")
BATCH, INPUT, HIDDEN, SEQ = 4, 16, 32, 20


def test_cell_forward():
    cell = CASARNNCell(INPUT, HIDDEN).to(DEVICE)
    x = torch.randn(BATCH, INPUT)
    h_slow = torch.zeros(BATCH, HIDDEN)
    h_fast = torch.zeros(BATCH, HIDDEN)
    h_slow_new, h_fast_new, regime, pred_err = cell(x, h_slow, h_fast)
    assert h_slow_new.shape == (BATCH, HIDDEN)
    assert h_fast_new.shape == (BATCH, HIDDEN)
    assert regime.shape == (BATCH,)
    assert pred_err.shape == (BATCH, 1)


def test_model_forward():
    model = CASARNNModel(INPUT, HIDDEN, num_layers=2, output_size=1).to(DEVICE)
    x = torch.randn(BATCH, SEQ, INPUT)
    out, extra = model(x)
    assert out.shape == (BATCH, SEQ, 1)
    assert "regime_probs" in extra
    assert "pred_coding_loss" in extra


def test_counterfactual_loss():
    model = CASARNNModel(INPUT, HIDDEN, num_layers=2, output_size=1).to(DEVICE)
    loss_fn = CounterfactualLoss(alpha=0.1, beta=0.01, task="regression")
    x = torch.randn(BATCH, SEQ, INPUT)
    target = torch.randn(BATCH, SEQ, 1)
    out, extra = model(x)
    loss = loss_fn(out, target, extra)
    loss.backward()
    assert loss.item() > 0


def test_no_nan():
    model = CASARNNModel(INPUT, HIDDEN, num_layers=3, output_size=1, dropout=0.1)
    x = torch.randn(BATCH, SEQ, INPUT)
    out, extra = model(x)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
