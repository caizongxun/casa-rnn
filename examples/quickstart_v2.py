"""Quickstart v2: MultiScaleCASARNN + NLL loss + Contrastive Regime."""
import torch
import torch.optim as optim
from casa_rnn import MultiScaleCASARNN, CounterfactualLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Synthetic data with regime structure ---
# Two regimes: low-vol trend vs high-vol reversal
BATCH, SEQ, FEAT = 32, 60, 8

def make_batch(step: int):
    # Alternate regimes every 200 steps to give regime detector signal
    regime = (step // 200) % 2
    noise_scale = 0.5 if regime == 0 else 2.0
    x = torch.randn(BATCH, SEQ, FEAT, device=device) * noise_scale
    y = x[:, :, :1].roll(-1, dims=1) * (0.3 if regime == 0 else -0.3)
    y += 0.1 * torch.randn_like(y)
    return x, y

# --- Model ---
model = MultiScaleCASARNN(
    input_size=FEAT,
    hidden_size=64,
    output_size=1,
    dropout=0.1,
    use_memory=True,
    memory_slots=32,
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
loss_fn   = CounterfactualLoss(alpha=0.1, beta=0.01, gamma=0.05, use_nll=True)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5, min_lr=1e-5)

# --- Training ---
for step in range(300):
    x, y = make_batch(step)
    optimizer.zero_grad()

    means, stds, extra = model(x)
    loss = loss_fn((means, stds), y, extra)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step(loss)

    if step % 50 == 0:
        sw   = extra["scale_weights"].mean(0).mean(0)   # avg weight per scale
        reg  = extra["regime_probs"].mean().item()
        unc  = stds.mean().item()
        lr   = optimizer.param_groups[0]["lr"]
        print(
            f"Step {step:3d} | Loss: {loss.item():.4f} "
            f"| Regime: {reg:.3f} "
            f"| Uncertainty: {unc:.4f} "
            f"| ScaleW: fast={sw[0]:.2f} mid={sw[1]:.2f} slow={sw[2]:.2f} "
            f"| LR: {lr:.2e}"
        )

print(f"\nFinal output shape: {means.shape}")
print(f"Final std (uncertainty) mean: {stds.mean().item():.4f}")
