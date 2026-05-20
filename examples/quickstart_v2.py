"""Quickstart v2: MultiScaleCASARNN + NLL loss + Contrastive Regime."""
import torch
import torch.optim as optim
from casa_rnn import MultiScaleCASARNN, CounterfactualLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

BATCH, SEQ, FEAT = 32, 60, 8


def make_batch(step: int):
    """Two-regime synthetic data: low-vol trend vs high-vol reversal."""
    regime = (step // 200) % 2
    noise  = 0.5 if regime == 0 else 2.0
    x = torch.randn(BATCH, SEQ, FEAT, device=device) * noise
    y = x[:, :, :1].roll(-1, dims=1) * (0.3 if regime == 0 else -0.3)
    y += 0.1 * torch.randn_like(y)
    return x, y


model = MultiScaleCASARNN(
    input_size=FEAT,
    hidden_size=64,
    output_size=1,
    dropout=0.1,
    use_memory=True,
    memory_slots=32,
    regime_reset_threshold=0.15,
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
loss_fn   = CounterfactualLoss(alpha=0.1, beta=0.01, gamma=0.05, use_nll=True)

# Fix 1: CosineAnnealingWarmRestarts -> warm restart every 100 steps
# so the model gets a chance to re-adapt after regime switches
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=100, T_mult=1, eta_min=1e-5
)

best_loss = float("inf")

for step in range(400):
    x, y = make_batch(step)
    optimizer.zero_grad()

    means, stds, extra = model(x)
    loss = loss_fn((means, stds), y, extra)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    # Fix 2: .detach().item() before passing to scheduler
    scheduler.step(step + loss.detach().item() * 0)

    loss_val = loss.detach().item()
    if loss_val < best_loss:
        best_loss = loss_val

    if step % 50 == 0:
        sw  = extra["scale_weights"].mean(0).mean(0).detach()
        reg = extra["regime_probs"].mean().detach().item()
        unc = stds.mean().detach().item()
        lr  = optimizer.param_groups[0]["lr"]
        print(
            f"Step {step:3d} "
            f"| Loss: {loss_val:+.4f} "
            f"| Best: {best_loss:+.4f} "
            f"| Regime: {reg:.3f} "
            f"| Unc: {unc:.4f} "
            f"| ScaleW: F={sw[0]:.2f} M={sw[1]:.2f} S={sw[2]:.2f} "
            f"| LR: {lr:.2e}"
        )

print(f"\nFinal output shape : {means.shape}")
print(f"Best loss achieved : {best_loss:+.4f}")
print(f"Final uncertainty  : {stds.mean().detach().item():.4f}")
