# CASA-RNN

**Causal Adaptive State Autoregressive RNN** — 針對金融市場設計的序列模型。

GPU 優先，自動 fallback 到 CPU，不需修改任何程式碼。

---

## 安裝

```bash
git clone https://github.com/caizongxun/casa-rnn.git
cd casa-rnn
pip install -e .
```

依賴：`torch`（唯一必要套件）

---

## 快速開始

```bash
python examples/quickstart_full.py
```

---

## 核心概念

訓練時你只需要提供：

1. **原始 OHLCV 或任意特徵序列** `x_raw` — 模型內部的 `FeatureGenome` 會自動演化特徵組合
2. **定義 label** `y` — 方向、報酬、分類都可以
3. **波動率指標** `vol_indicator`（可選）— 傳入即可，模型自行決定怎麼用

其餘全部由模型自己做取捨：哪些特徵重要、哪個模組在當前 regime 最有用、預測區間多寬。

---

## API 參考

### `CASARNNModel`

```python
from casa_rnn import CASARNNModel

model = CASARNNModel(
    raw_size      = 5,      # 輸入特徵維度（例如 OHLCV = 5）
    hidden_size   = 64,     # RNN 隱狀態維度，建議 64~256
    output_size   = 1,      # 輸出維度（1 = 單一預測值，多分類則調大）
    feat_size     = 16,     # FeatureGenome 輸出維度，建議 = hidden_size / 4
    use_genome    = True,   # True: 啟用 FeatureGenome 自動特徵演化
    top_k_pairs   = 6,      # FeatureGenome 保留的 top-k 特徵交互對
    dropout       = 0.1,    # Dropout rate
    use_memory    = True,   # True: 啟用 MemoryBank（跨序列記憶）
    memory_slots  = 32,     # MemoryBank 槽位數
    use_bio       = True,   # True: 啟用全部神經仿生模組
)
```

### `model.forward()`

```python
means, stds, extra = model(
    x_raw,                      # Tensor (B, T, raw_size)  — 原始輸入序列
    vol_indicator     = vol,    # Tensor (B, T, 1) 或 None — 波動率指標
    transition_label  = 0.0,    # float 0.0~1.0 — 已知 regime 切換時設為 1.0
    t_offset          = 0,      # int — 當前 batch 的全域時間偏移（用於 CPG 對齊）
    router_entropy_weight = 0.02, # float — Router 探索強度（訓練中由 schedule 控制）
)
# means: (B, T, output_size)   預測均值
# stds:  (B, T, output_size)   預測標準差
# extra: dict                  各模組的中間狀態（見下方）
```

### `extra` 字典常用欄位

| 欄位 | 型別 | 說明 |
|---|---|---|
| `regime_probs` | `(B, T, 2)` | 當前 regime 機率（bull/bear） |
| `router_weights` | `(B, T, 10)` | SoftModuleRouter 對 10 個模組的權重 |
| `router_w_mean` | `list[float]` | 本 batch 平均模組權重（用於印出使用率） |
| `router_temperature` | `float` | Router 當前溫度（越低越收斂） |
| `danger_score` | `float` | 當前 Danger score（Mahalanobis 距離） |
| `neuro` | `dict` | `dopamine`, `norepinephrine`, `serotonin`, `acetylcholine` |
| `cpg_periods` | `list[float]` | CPG 三個振盪器的當前週期 |
| `astrocyte_alpha` | `float` | Astrocyte 慢波調制係數 |
| `dominant_strategy` | `int` | 當前主導學習策略的 index |
| `trans_prob` | `(B, T, 1)` | Regime transition 偵測機率 |
| `cereb_err` | `float` | Cerebellum forward model 預測誤差 |

### Conformal Calibration（訓練後執行一次）

```python
# 用 held-out calibration set 校準
q = model.calibrate_conformal(x_cal, vol_cal, y_cal)
# 之後預測時自動附帶 90% 覆蓋率保證的區間
means, stds, lower, upper, extra = model.predict_with_interval(x, vol)
```

---

## 真實資料接入範例

```python
import torch
from casa_rnn import CASARNNModel

# x: numpy array (N, T, features)，例如 OHLCV
# y: numpy array (N, T, 1)，例如下一根 K 棒報酬率

x_tensor = torch.tensor(x, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)
vol = x_tensor.std(dim=-1, keepdim=True)  # 簡單用輸入 std 當波動率

model = CASARNNModel(raw_size=x.shape[-1])

means, stds, extra = model(x_tensor, vol_indicator=vol)
loss = your_loss_fn(means, stds, y_tensor, extra)
```

label 的定義完全由你決定，模型不假設任何 label 格式。

---

## 訓練時的 Loss 建議結構

```python
from casa_rnn import CounterfactualLoss
from casa_rnn.loss import BioConstraintLoss

loss_fn     = CounterfactualLoss(alpha=0.1, beta=0.01, gamma=0.05,
                                  use_nll=True, nll_clamp=3.0, regime_scale=True)
bio_loss_fn = BioConstraintLoss(ne_weight=0.05, da_weight=0.05)

task_loss = loss_fn((means, stds), y, extra)

# 加上 Router sparsity（讓模組選擇收斂）
rw_full = extra.get("router_weights")
if rw_full is not None:
    task_loss = task_loss + 0.02 * model.rnn.router.sparsity_loss(rw_full)
```

完整訓練範例見 [`examples/quickstart_full.py`](examples/quickstart_full.py)。

---

## 模組說明

| 模組 | 仿生來源 | 功能 |
|---|---|---|
| `FeatureGenome` | 基因演化 | 自動發現特徵交互對，訓練中自行取捨 |
| `CPGEncoder` | 脊髓中樞模式產生器 | 編碼市場週期性節律（日內、週、月） |
| `AstrocyteModulator` | 神經膠質細胞 | 慢波調制隱狀態，抑制高頻噪音 |
| `DangerSignalDetector` | 先天免疫 Danger Signal | Mahalanobis 距離偵測分佈偏移（regime change） |
| `TDAFeatureExtractor` | 拓撲資料分析 | 計算 Betti 數，捕捉價格序列的拓撲結構 |
| `SoftModuleRouter` | — | 每個 timestep 對上方所有模組產生軟性混合權重 |
| `NeuromodulatorGating` | DA/NE/5-HT/ACh | 四種神經調制物質控制學習率與注意力 |
| `ThalamicAttention` | 視丘閘控 | 選擇性放大重要時間步的訊號 |
| `PrefrontalWorkingMemory` | 前額葉皮質 | 跨 timestep 工作記憶，保存 regime 相關上下文 |
| `CerebellarForwardModel` | 小腦前向模型 | 預測下一步隱狀態，誤差訊號回饋給 Router |
| `RegimeTransitionDetector` | — | 偵測市場制度切換，觸發隱狀態 reset |
| `MetaLearningStrategyBank` | — | 動態切換 rehearsal/chunking/contrastive 等學習策略 |
| `ConformalWrapper` | 共形預測理論 | 無分佈假設的預測區間，保證 90% 覆蓋率 |

---

## 關鍵超參數速查

| 參數 | 位置 | 預設 | 說明 |
|---|---|---|---|
| `hidden_size` | `CASARNNModel` | 64 | 越大容量越強，但訓練更慢 |
| `feat_size` | `CASARNNModel` | 16 | FeatureGenome 輸出維度 |
| `top_k_pairs` | `CASARNNModel` | 6 | 特徵交互對數量 |
| `ema_alpha` | `DangerSignalDetector` | 0.05 | 分佈偏移 EMA 更新速度 |
| sparsity weight | `quickstart_full.py` | 0.02 | Router 收斂壓力 |
| `WARMUP` | `quickstart_full.py` | 200 | Router 探索 steps 數 |
| `coverage` | `ConformalWrapper` | 0.90 | 預測區間覆蓋率目標 |
