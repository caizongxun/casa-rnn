"""
Full CASA-RNN quickstart — all modules active, SoftModuleRouter governs tradeoffs.

Changelog:
  - Router sparsity weight: 0.05
  - Fix Router T: log_temperature split into separate param group with lr=5e-3
  - Fix CT direction in neuro_modules.py
  - Fix Danger normalization in danger.py
"""
import torch
import torch.optim as optim
import math
from casa_rnn import CASARNNModel, CounterfactualLoss
from casa_rnn.loss import BioConstraintLoss
from casa_rnn.neuro_modules import HippocampalReplayBuffer, STRATEGY_NAMES, MODULE_NAMES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

BATCH, SEQ, RAW_FEAT = 32, 60, 5
HIDDEN = 64
FEAT   = 16
TOTAL  = 800
SWITCH = [200, 500]
TRANS  = 30
WARMUP = 200
ENT_HIGH, ENT_LOW = 0.15, 0.01


def regime_weight(step):
    w = 0.0
    for i, sw in enumerate(SWITCH):
        d = 1.0 if i % 2 == 0 else -1.0
        w += d / (1.0 + math.exp(-(step - sw) / TRANS * 6))
    return max(0.0, min(1.0, w))


def is_transition(step, hw=25):
    return 1.0 if any(abs(step - sw) <= hw for sw in SWITCH) else 0.0


def router_ent_weight(step):
    if step >= WARMUP:
        return ENT_LOW
    ratio = step / WARMUP
    return ENT_LOW + (ENT_HIGH - ENT_LOW) * 0.5 * (1 + math.cos(math.pi * ratio))


def make_batch(step):
    rw    = regime_weight(step)
    noise = 0.5 + 1.5 * rw
    sign  = 0.3 - 0.6 * rw
    x     = torch.randn(BATCH, SEQ, RAW_FEAT, device=device) * noise
    y     = x[:, :, 3:4].roll(-1, dims=1) * sign + 0.05 * torch.randn(BATCH, SEQ, 1, device=device)
    vol   = x.std(dim=-1, keepdim=True)
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
    use_bio=True,
).to(device)

replay_buf  = HippocampalReplayBuffer(capacity=300, replay_every=25)
bio_loss_fn = BioConstraintLoss(ne_weight=0.05, da_weight=0.05)

alpha_params = [model.genome.soft_op.alpha]

# Fix Router T: log_temperature 獨立拆出來，給更高 lr 讓 temperature 更容易下降
log_temp_params = [model.rnn.router.log_temperature]
router_net_params = [
    p for n, p in model.rnn.router.named_parameters()
    if 'log_temperature' not in n
]
other_params = [
    p for n, p in model.named_parameters()
    if 'soft_op.alpha' not in n
    and 'rnn.router' not in n
]
optimizer = optim.AdamW([
    {'params': other_params,      'lr': 3e-4},
    {'params': alpha_params,      'lr': 3e-3,  'weight_decay': 0.0},
    {'params': router_net_params, 'lr': 1e-3,  'weight_decay': 0.0},
    {'params': log_temp_params,   'lr': 5e-3,  'weight_decay': 0.0},  # temperature 快速響應
], weight_decay=1e-4)

loss_fn   = CounterfactualLoss(alpha=0.1, beta=0.01, gamma=0.05, use_nll=True, nll_clamp=3.0, regime_scale=True)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=1, eta_min=1e-5)

best_loss = float("inf")
prev_rw   = 0.0

for step in range(TOTAL):
    x, y, vol, rw = make_batch(step)
    tl  = is_transition(step)
    rew = router_ent_weight(step)
    optimizer.zero_grad()

    means, stds, extra = model(x, vol_indicator=vol, transition_label=tl,
                               t_offset=step * SEQ, router_entropy_weight=rew)

    with torch.no_grad():
        rpe_scalar = (means - y).abs().mean().item()

    task_loss = loss_fn((means, stds), y, extra)
    loss      = task_loss

    loss = loss + 0.005 * model.rnn.cpg.period_loss()

    nt  = extra.get("neuro_tensors", {})
    vov = extra.get("vol_of_vol", None)
    rpe = extra.get("rpe_tensor",  None)
    if nt and vov is not None and rpe is not None:
        loss = loss + bio_loss_fn(ne=nt["norepinephrine"], vol_of_vol=vov,
                                  da=nt["dopamine"],        rpe=rpe)

    sw = extra.get("strategy_w_raw", None)
    if sw is not None:
        loss = loss - 0.03 * model.rnn.strategy_bank.entropy_loss(sw)

    tp = extra.get("trans_prob", None)
    if tp is not None:
        loss = loss + 0.15 * model.rnn.transition_det.loss(tp.mean(dim=1), tl)

    if nt:
        sht = nt.get("serotonin", None)
        if sht is not None:
            loss = loss + model.rnn.homeostasis.homeostatic_loss(
                da=nt["dopamine"], ne=nt["norepinephrine"], sht=sht
            )

    rw_full = extra.get("router_weights", None)
    if rw_full is not None:
        if step < WARMUP:
            loss = loss - rew * model.rnn.router.entropy_loss(rw_full)
        else:
            loss = loss + 0.05 * model.rnn.router.sparsity_loss(rw_full)
            rc_loss = extra.get("regime_consistency_loss", None)
            if rc_loss is not None:
                loss = loss + 0.01 * rc_loss

    ct_loss = extra.get("contrastive_loss", None)
    if ct_loss is not None:
        loss = loss + 0.02 * ct_loss

    loss = loss + 0.002 * model.conformal.sharpness_loss(stds)

    replay_buf.push(x, y, rpe=rpe_scalar)
    if replay_buf.should_replay(step):
        rx, ry = replay_buf.sample_rpe_biased(BATCH // 2)
        rx, ry = rx.to(device), ry.to(device)
        r_vol  = rx.std(dim=-1, keepdim=True)
        rm, rs, re = model(rx, vol_indicator=r_vol, transition_label=0.0,
                           t_offset=0, router_entropy_weight=rew)
        loss = loss + 0.3 * loss_fn((rm, rs), ry, re)

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
        reg   = extra["regime_probs"].mean().item()
        neuro = extra.get("neuro", {})
        da    = neuro.get("dopamine",       0)
        ne    = neuro.get("norepinephrine", 0)
        dom   = extra.get("dominant_strategy", -1)
        sname = STRATEGY_NAMES[dom] if 0 <= dom < len(STRATEGY_NAMES) else "n/a"
        tp_v  = extra.get("trans_prob")
        tp_v  = tp_v.mean().item() if tp_v is not None else 0.0
        ce    = extra.get("cereb_err", 0.0)
        ds    = extra.get("danger_score", 0.0)
        cpg_p = extra.get("cpg_periods", [])
        a_alp = extra.get("astrocyte_alpha", 0.0)
        r_tmp = extra.get("router_temperature", 0.0)
        rwm   = extra.get("router_w_mean", [])
        rc    = extra.get("regime_consistency_loss")
        rc_v  = rc.item() if rc is not None and hasattr(rc, 'item') else 0.0
        ct    = extra.get("contrastive_loss")
        ct_v  = ct.item() if ct is not None and hasattr(ct, 'item') else 0.0
        if rwm:
            top2 = sorted(enumerate(rwm), key=lambda kv: -kv[1])[:2]
            top2_str = "+".join(MODULE_NAMES[i] for i, _ in top2)
        else:
            top2_str = "n/a"
        tag = " <<< TRANSITION" if abs(rw - round(rw)) > 0.05 else ""
        print(
            f"[{step:3d}] Loss={lv:+.4f} Best={best_loss:+.4f}"
            f" | Reg={reg:.3f} DA={da:.2f} NE={ne:.2f}"
            f" | Trans={tp_v:.2f} Cereb={ce:.3f} Danger={ds:.3f}"
            f" | Router: T={r_tmp:.2f} top={top2_str}"
            f" | RC={rc_v:.3f} CT={ct_v:.3f}"
            f" | Astro: a={a_alp:.3f}"
            f" | CPG: {[f'{p:.0f}' for p in cpg_p]}"
            f" | Strat={sname}"
            f"{tag}"
        )

print("\n--- Post-training Conformal Calibration ---")
x_cal, y_cal, vol_cal, _ = make_batch(TOTAL + 1)
q = model.calibrate_conformal(x_cal, vol_cal, y_cal)
print(f"q_hat = {q:.4f}  (prediction intervals are now statistically guaranteed at 90%)")

means_t, stds_t, lo, hi, extra_t = model.predict_with_interval(x_cal, vol_cal)
print(f"Interval width: {(hi - lo).mean().item():.4f}")

print("\n=== Router Module Usage (avg across final batch) ===")
final_rwm = extra.get("router_w_mean", [])
if final_rwm:
    for name, w in sorted(zip(MODULE_NAMES, final_rwm), key=lambda kv: -kv[1]):
        bar = '#' * int(w * 40)
        print(f"  {name:20s} {bar:<40s} {w:.3f}")

print("\n=== Final Model Summary ===")
print(f"Best loss:       {best_loss:+.4f}")
print(f"Final q_hat:     {q:.4f}")
print(f"CPG periods:     {extra.get('cpg_periods', [])}")
print(f"Astrocyte alpha: {extra.get('astrocyte_alpha', 0):.4f}")
print(f"Router temp:     {extra.get('router_temperature', 0):.4f}")
dom_final = extra.get('dominant_strategy', -1)
print(f"Dominant strat:  {STRATEGY_NAMES[dom_final] if dom_final >= 0 else 'n/a'}")
