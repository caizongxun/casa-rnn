"""Quickstart: train CASA-RNN on synthetic financial data."""
import torch
import torch.optim as optim
from casa_rnn import CASARNNModel, CounterfactualLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Synthetic data: predict next return from 10 features ---
BATCH, SEQ, FEAT = 32, 60, 10

def make_batch():
    x = torch.randn(BATCH, SEQ, FEAT, device=device)
    y = x[:, :, :1].roll(-1, dims=1) + 0.05 * torch.randn(BATCH, SEQ, 1, device=device)
    return x, y

# --- Model ---
model = CASARNNModel(
    input_size=FEAT,
    hidden_size=64,
    num_layers=2,
    output_size=1,
    dropout=0.1,
    use_counterfactual_loss=True,
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = CounterfactualLoss(alpha=0.1, beta=0.01, task="regression")
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# --- Training loop ---
for step in range(100):
    x, y = make_batch()
    optimizer.zero_grad()
    out, extra = model(x)
    loss = loss_fn(out, y, extra)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    if step % 20 == 0:
        regime_mean = extra["regime_probs"].mean().item()
        print(f"Step {step:3d} | Loss: {loss.item():.4f} | Regime: {regime_mean:.3f}")

print("Done. Output shape:", out.shape)
