"""CASA-RNN full sequence model with multi-layer support."""
import torch
import torch.nn as nn
from typing import Optional, Tuple
from .cell import CASARNNCell


class CASARNNModel(nn.Module):
    """
    Multi-layer CASA-RNN sequence model.

    Args:
        input_size: Number of input features
        hidden_size: Hidden state dimension
        num_layers: Number of stacked CASA-RNN layers
        output_size: Output dimension
        dropout: Dropout between layers
        use_counterfactual_loss: If True, accumulates predictive coding errors
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.1,
        use_counterfactual_loss: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_counterfactual_loss = use_counterfactual_loss

        # Stack of CASA-RNN cells
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_sz = input_size if i == 0 else hidden_size
            self.cells.append(
                CASARNNCell(
                    input_size=in_sz,
                    hidden_size=hidden_size,
                    dropout=dropout if i < num_layers - 1 else 0.0,
                )
            )

        # Output head: mean + log_variance (uncertainty estimation)
        self.output_head = nn.Linear(hidden_size, output_size * 2)
        self.output_size = output_size

    def forward(
        self,
        x: torch.Tensor,                          # (batch, seq_len, input_size)
        vol_indicator: Optional[torch.Tensor] = None,  # (batch, seq_len, 1)
        init_hidden: Optional[Tuple] = None,
    ):
        batch, seq_len, _ = x.shape
        device = x.device

        # Initialize hidden states
        if init_hidden is None:
            h_slows = [torch.zeros(batch, self.hidden_size, device=device) for _ in range(self.num_layers)]
            h_fasts = [torch.zeros(batch, self.hidden_size, device=device) for _ in range(self.num_layers)]
        else:
            h_slows, h_fasts = init_hidden

        outputs = []
        all_regime_probs = []
        total_pred_error = torch.zeros(1, device=device)

        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch, input_size)
            vol_t = vol_indicator[:, t, :] if vol_indicator is not None else None

            layer_input = x_t
            step_regime = []

            for layer_idx, cell in enumerate(self.cells):
                h_slow_new, h_fast_new, regime_prob, pred_err = cell(
                    layer_input, h_slows[layer_idx], h_fasts[layer_idx], vol_t
                )
                h_slows[layer_idx] = h_slow_new
                h_fasts[layer_idx] = h_fast_new
                layer_input = h_fast_new  # pass fast state to next layer
                step_regime.append(regime_prob)

                if self.use_counterfactual_loss:
                    total_pred_error = total_pred_error + pred_err.mean()

            # Output from top layer fast state
            out_params = self.output_head(h_fasts[-1])  # (batch, output_size*2)
            mean_out, log_var = out_params.chunk(2, dim=-1)
            outputs.append(mean_out.unsqueeze(1))
            all_regime_probs.append(torch.stack(step_regime, dim=1))

        outputs = torch.cat(outputs, dim=1)          # (batch, seq_len, output_size)
        regime_probs = torch.stack(all_regime_probs, dim=1)  # (batch, seq_len, num_layers)

        extra = {
            "pred_coding_loss": total_pred_error / (seq_len * self.num_layers),
            "regime_probs": regime_probs,
            "final_hidden": (h_slows, h_fasts),
        }

        return outputs, extra
