"""Quickstart: CASARNNModel with FeatureGenome auto feature evolution."""
import torch
import torch.optim as optim
import math
from casa_rnn import CASARNNModel, CounterfactualLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

BATCH, SEQ, RAW_FEAT = 32, 60, 5
TOTAL_STEPS  = 600         # longer: give CrossFeature time to differentiate
SWITCH_STEPS = [200, 400]
TRANSITION   = 30


def regime_weight(step):
    w = 0.0
    for i, sw in enumerate(SWITCH_STEPS):
        d = 1.0 if i % 2 == 0 else -1.0
        w += d / (1.0 + math.exp(-(step - sw) / TRANSITION * 6))
    return max(0.0, min(1.0, w))


def make_batch(step):
    rw    = regime_weight(step)
    noise = 0.5 + 1.5 * rw
    sign  = 0.3 - 0.6 * rw
    x     = torch.randn(BATCH, SEQ, RAW_FEAT, device=device) * noise
    y     = x[:, :, 3:4].roll(-1, dims=1) * sign
    y    += 0.05 * torch.randn_like(y)
    vol   = x.std(dim=-1, keepdim=True)
    vol   = (vol - vol.mean()) / (vol.std() + 1e-6)
    return x, y, vol, rw


model = CASARNNModel(
    raw_size=RAW_FEAT,
    hidden_size=64,
    output_size=1,
    feat_size=16,
    use_genome=True,
    top_k_pairs=6,
    dropout=0.1,
    use_memory=True,
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
loss_fn   = CounterfactualLoss(
    alpha=0.1, beta=0.01, gamma=0.05,
    use_nll=True, nll_clamp=3.0,
    regime_scale=True,   # scale loss down during regime transitions
)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=100, T_mult=1, eta_min=1e-5
)

best_loss = float("inf")
prev_rw   = 0.0

for step in range(TOTAL_STEPS):
    x, y, vol, rw = make_batch(step)
    optimizer.zero_grad()

    means, stds, extra = model(x, vol_indicator=vol)
    loss = loss_fn((means, stds), y, extra)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step(step)

    rw_delta = abs(rw - prev_rw)
    if rw_delta > 0.02:
        decay = max(0.1, 1.0 - rw_delta * 2)
        for group in optimizer.param_groups:
            for p in group['params']:
                s = optimizer.state[p]
                if 'exp_avg' in s:
                    s['exp_avg'].mul_(decay)
    prev_rw = rw

    lv = loss.detach().item()
    if lv < best_loss:
        best_loss = lv

    if step % 50 == 0:
        reg = extra["regime_probs"].mean().item()
        unc = stds.mean().item()
        lr  = optimizer.param_groups[0]["lr"]
        # op entropy: how converged are the op choices?
        ent = model.genome.soft_op.get_op_entropy()
        tag = " <<< TRANSITION" if abs(rw - round(rw)) > 0.05 else ""
        print(
            f"Step {step:3d} [rw={rw:.2f}]"
            f" | Loss: {lv:+.4f}"
            f" | Best: {best_loss:+.4f}"
            f" | Regime: {reg:.3f}"
            f" | Unc: {unc:.4f}"
            f" | OpEnt: {ent:.3f}"
            f" | LR: {lr:.2e}"
            f"{tag}"
        )

# Feature report
print("\n=== Feature Genome Report ===")
report = model.get_feature_report()
print(f"Op Entropy (convergence): {report['op_entropy']:.4f}")
print("  -> Low (<1.5) = ops converged to sharp choices")
print("  -> High (>2.5) = still exploring")

print("\nTop cross-feature interactions learned:")
for i, j, w in report["top_interactions"]:
    print(f"  feat[{i:2d}] x feat[{j:2d}]  weight={w:.4f}")

print("\nSample dominant ops (feat0..2, all raw inputs):")
for k, v in list(report["dominant_ops"].items())[:RAW_FEAT * 3]:
    print(f"  {k:20s} -> {v}")

print(f"\nFinal output shape : {means.shape}")
print(f"Best loss achieved : {best_loss:+.4f}")
print(f"Final uncertainty  : {stds.mean().item():.4f}")
