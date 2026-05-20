# CASA-RNN

**Causal Adaptive State Autoregressive RNN** — A breakthrough RNN architecture for financial market sequence modeling.

## Key Features

- **Regime-Aware Memory**: Active market regime detection that resets slow state on structural breaks
- **Complex-Valued Hidden State**: Encodes market periodicity (intraday, weekly cycles) naturally
- **Layered Predictive Coding**: Local prediction-error updates, no global BPTT required
- **Counterfactual Loss**: Trains on causal structure, not just correlation
- **GPU/CPU Auto-Fallback**: Seamless `cuda` → `cpu` switching

## Installation

```bash
pip install casa-rnn
```

Or install from source:

```bash
git clone https://github.com/caizongxun/casa-rnn.git
cd casa-rnn
pip install -e .
```

## Quick Start

```python
import torch
from casa_rnn import CASARNNCell, CASARNNModel

# Auto GPU/CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Single cell
cell = CASARNNCell(input_size=16, hidden_size=64).to(device)
x = torch.randn(8, 16).to(device)          # (batch, input_size)
h_slow = torch.randn(8, 64).to(device)
h_fast = torch.randn(8, 64).to(device)

h_slow_new, h_fast_new, regime_prob = cell(x, h_slow, h_fast)

# Full sequence model
model = CASARNNModel(
    input_size=16,
    hidden_size=64,
    num_layers=3,
    output_size=1,
    use_counterfactual_loss=True,
).to(device)

seq = torch.randn(32, 50, 16).to(device)   # (batch, seq_len, input_size)
out, hidden, regime_probs = model(seq)
print(out.shape)  # (32, 50, 1)
```

## Architecture

```
Input (multi-scale)
  ├── tick  → h_fast
  ├── hourly → h_mid
  └── daily → h_slow
         │
   Regime Detector
         │
   Complex-Valued CASA-RNN Cell
         │
   Layered Predictive Error Update
         │
   Counterfactual Loss
         │
   Output: direction + confidence + uncertainty
```

## License

MIT
