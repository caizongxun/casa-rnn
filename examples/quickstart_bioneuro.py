"""
Bio-Neuro Quickstart: deep integration of all bio modules.

NeuromodulatorGating + ThalamicAttention + PrefrontalWorkingMemory are now
wired INSIDE the RNN timestep loop (multiscale.py), not just demonstrated.
Every timestep:
  fused_h -> ThalamicAttention -> NeuromodulatorGating -> PFC-WM -> MemoryBank -> head

HippocampalReplayBuffer remains in the training loop (external).
OpEntropy warmup ensures alpha converges in first 150 steps.
"""
import torch
import torch.optim as optim
import math
from casa_rnn import CASARNNModel, CounterfactualLoss
from casa_rnn.neuro_modules import HippocampalReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

BATCH, SEQ, RAW_FEAT = 32, 60, 5
HIDDEN = 64
FEAT   = 16
TOTAL  = 700
SWITCH = [200, 450]
TRANS  = 30

ENTROPY_WARMUP = 150
ENTROPY_W_MAX  = 0.5
ENTROPY_W_MIN  = 0.02


def regime_weight(step):
    w = 0.0
    for i, sw in enumerate(SWITCH):
        d = 1.0 if i % 2 == 0 else -1.0
        w += d / (1.0 + math.exp(-(step - sw) / TRANS * 6))
    return max(0.0, min(1.0, w))


def entropy_weight(step):
    if step >= ENTROPY_WARMUP:
        return ENTROPY_W_MIN
    ratio = step / ENTROPY_WARMUP
    return ENTROPY_W_MIN + (ENTROPY_W_MAX - ENTROPY_W_MIN) * 0.5 * (1 + math.cos(math.pi * ratio))


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
    hidden_size=HIDDEN,
    output_size=1,
    feat_size=FEAT,
    use_genome=True,
    top_k_pairs=6,
    dropout=0.1,
    use_memory=True,
    use_bio=True,      # deep bio integration
).to(device)

replay_buf = HippocampalReplayBuffer(capacity=300, replay_every=25)

alpha_params = [model.genome.soft_op.alpha]
other_params = [p for n, p in model.named_parameters() if 'soft_op.alpha' not in n]
optimizer = optim.AdamW([
    {'params': other_params, 'lr': 3e-4},
    {'params': alpha_params, 'lr': 3e-3, 'weight_decay': 0.0},
], weight_decay=1e-4)

loss_fn   = CounterfactualLoss(alpha=0.1, beta=0.01, gamma=0.05,
                               use_nll=True, nll_clamp=3.0, regime_scale=True)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=100, T_mult=1, eta_min=1e-5
)

best_loss = float("inf")
prev_rw   = 0.0

for step in range(TOTAL):
    x, y, vol, rw = make_batch(step)
    optimizer.zero_grad()

    means, stds, extra = model(x, vol_indicator=vol)

    # RPE for replay and PFC-WM is computed internally now;
    # here we only need it for the replay buffer push
    with torch.no_grad():
        rpe_scalar = (means - y).abs().mean().item()

    task_loss = loss_fn((means, stds), y, extra)
    ent_w     = entropy_weight(step)
    loss      = task_loss + ent_w * model.genome.alpha_entropy_loss()

    # Hippocampal replay
    replay_buf.push(x, y, rpe=rpe_scalar)
    if replay_buf.should_replay(step):
        rx, ry = replay_buf.sample_rpe_biased(BATCH // 2)
        rx, ry = rx.to(device), ry.to(device)
        r_vol  = rx.std(dim=-1, keepdim=True)
        r_vol  = (r_vol - r_vol.mean()) / (r_vol.std() + 1e-6)
        rm, rs, re = model(rx, vol_indicator=r_vol)
        loss   = loss + 0.3 * loss_fn((rm, rs), ry, re)

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

    lv = task_loss.detach().item()
    if lv < best_loss:
        best_loss = lv

    if step % 50 == 0:
        reg  = extra["regime_probs"].mean().item()
        unc  = stds.mean().item()
        ent  = model.genome.soft_op.get_op_entropy()
        neuro = extra.get("neuro", {})
        da   = neuro.get("dopamine", 0)
        ne   = neuro.get("norepinephrine", 0)
        tag  = " <<< TRANSITION" if abs(rw - round(rw)) > 0.05 else ""
        print(
            f"Step {step:3d} [rw={rw:.2f}]"
            f" | Loss: {lv:+.4f}"
            f" | Best: {best_loss:+.4f}"
            f" | Regime: {reg:.3f}"
            f" | Unc: {unc:.4f}"
            f" | OpEnt: {ent:.3f}(w={ent_w:.2f})"
            f" | DA={da:.2f} NE={ne:.2f}"
            f"{tag}"
        )

print("\n=== Feature Genome Report ===")
report = model.get_feature_report()
print(f"Op Entropy (final): {report['op_entropy']:.4f}")

print("\nTop cross-feature interactions:")
for i, j, w in report["top_interactions"]:
    print(f"  feat[{i:2d}] x feat[{j:2d}]  weight={w:.4f}")

print("\nDominant ops (feat 0-2):")
for k, v in list(report["dominant_ops"].items())[:RAW_FEAT * 3]:
    print(f"  {k:20s} -> {v}")

print(f"\nFinal output shape : {means.shape}")
print(f"Best loss achieved : {best_loss:+.4f}")
print(f"Final uncertainty  : {stds.mean().item():.4f}")
