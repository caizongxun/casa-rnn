"""Quickstart v2: MultiScaleCASARNN + NLL loss + Contrastive Regime."""
import torch
import torch.optim as optim
import math
from casa_rnn import MultiScaleCASARNN, CounterfactualLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

BATCH, SEQ, FEAT = 32, 60, 8
TOTAL_STEPS = 500
SWITCH_STEPS = [200, 350]   # regime transitions happen here
TRANSITION_LEN = 30         # gradual transition over 30 steps


def regime_weight(step: int) -> float:
    """
    Returns a smooth regime blend weight in [0, 1].
    0 = full regime-0 (low-vol trend)
    1 = full regime-1 (high-vol reversal)
    Transitions are sigmoid-smoothed over TRANSITION_LEN steps.
    """
    w = 0.0
    for i, sw in enumerate(SWITCH_STEPS):
        direction = 1.0 if i % 2 == 0 else -1.0
        progress  = (step - sw) / TRANSITION_LEN
        w += direction / (1.0 + math.exp(-progress * 6))  # sigmoid ramp
    return max(0.0, min(1.0, w))


def make_batch(step: int):
    """
    Gradual regime transition.
    noise_scale and target_sign interpolate smoothly between regimes.
    vol_indicator is explicitly provided so the model can sense the transition.
    """
    rw = regime_weight(step)             # 0.0 (calm) -> 1.0 (volatile)

    noise_scale = 0.5 + 1.5 * rw        # 0.5 -> 2.0
    target_sign = 0.3 - 0.6 * rw        # +0.3 -> -0.3

    x = torch.randn(BATCH, SEQ, FEAT, device=device) * noise_scale
    y = x[:, :, :1].roll(-1, dims=1) * target_sign
    y += 0.1 * torch.randn_like(y)

    # vol_indicator: rolling std of the sequence as a proxy for realised volatility
    # shape (BATCH, SEQ, 1) -- the model uses this to update regime gate
    vol = x.std(dim=-1, keepdim=True)   # (B, SEQ, 1)
    vol = (vol - vol.mean()) / (vol.std() + 1e-6)  # normalise

    return x, y, vol, rw


model = MultiScaleCASARNN(
    input_size=FEAT,
    hidden_size=64,
    output_size=1,
    dropout=0.1,
    use_memory=True,
    memory_slots=32,
    regime_reset_threshold=0.12,
    hidden_decay_rate=0.25,
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
loss_fn   = CounterfactualLoss(
    alpha=0.1, beta=0.01, gamma=0.05,
    use_nll=True,
    nll_clamp=3.0,
)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=100, T_mult=1, eta_min=1e-5
)

best_loss      = float("inf")
prev_regime_w  = 0.0

for step in range(TOTAL_STEPS):
    x, y, vol, rw = make_batch(step)
    optimizer.zero_grad()

    means, stds, extra = model(x, vol_indicator=vol)
    loss = loss_fn((means, stds), y, extra)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step(step)

    loss_val = loss.detach().item()

    # Soft optimizer momentum decay proportional to regime delta
    # Gradual -> gentle decay. Sharp jump -> heavy decay.
    rw_delta = abs(rw - prev_regime_w)
    if rw_delta > 0.02:  # only during active transition
        decay_factor = max(0.1, 1.0 - rw_delta * 2)
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                if 'exp_avg' in state:
                    state['exp_avg'].mul_(decay_factor)
    prev_regime_w = rw

    if loss_val < best_loss:
        best_loss = loss_val

    if step % 50 == 0 or step in SWITCH_STEPS:
        sw  = extra["scale_weights"].mean(0).mean(0).detach()
        reg = extra["regime_probs"].mean().detach().item()
        unc = stds.mean().detach().item()
        lr  = optimizer.param_groups[0]["lr"]
        tag = " <<< TRANSITION" if abs(rw - round(rw)) > 0.05 else ""
        print(
            f"Step {step:3d} [rw={rw:.2f}] "
            f"| Loss: {loss_val:+.4f} "
            f"| Best: {best_loss:+.4f} "
            f"| Regime: {reg:.3f} "
            f"| Unc: {unc:.4f} "
            f"| ScaleW: F={sw[0]:.2f} M={sw[1]:.2f} S={sw[2]:.2f} "
            f"| LR: {lr:.2e}"
            f"{tag}"
        )

print(f"\nFinal output shape : {means.shape}")
print(f"Best loss achieved : {best_loss:+.4f}")
print(f"Final uncertainty  : {stds.mean().detach().item():.4f}")
