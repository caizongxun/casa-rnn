# Bio-Neuro Architecture Design

CASA-RNN 的突破性設計，參考 2024-2025 年最新神經科學研究。

---

## 設計哲學

> 傳統 RNN/LSTM 是 1990 年代的工程設計。  
> 大腦是 5 億年演化的最優解。  
> 我們直接抄它。

---

## 四大生物模組

### 1. NeuromodulatorGating（神經調節物質閘控）

**生物來源：**
- Dopamine（多巴胺）：控制「是否要更新記憶」的 RPE 訊號
- Acetylcholine（乙醯膽鹼）：提升訊噪比，讓模型在高不確定性時更「專注」
- Norepinephrine（去甲腎上腺素）：調整全層增益，高波動期自動放大
- Serotonin（血清素）：控制記憶時間視野，穩定市場時看更遠

**論文：**
- Three-Factor Learning in Spiking Neural Networks (arXiv 2504.05341, 2025)
- Computational Models of Neuromodulation (Frontiers, 2026)

**金融應用：**
```
低波動 (VIX<15):
  DA=低  -> 不急著更新，保留舊記憶
  ACh=低 -> 放鬆，看廣一點
  NE=低  -> 低增益
  5HT=高 -> 長期記憶視野

高波動 (VIX>30):
  DA=高  -> RPE大，大量更新
  ACh=高 -> 聚焦，過濾噪音
  NE=高  -> 高增益，捕捉快速變化
  5HT=低 -> 短期焦點
```

---

### 2. ThalamicAttention（丘腦選擇性注意力）

**生物來源：**
丘腦不只是被動中繼站。丘腦網狀核（TRN）可以主動 VETO 整個感覺通道，
讓大腦在需要時完全忽略特定輸入。

**論文：**
- Corticothalamic Synaptic Noise as Selective Attention (Frontiers 2015)
- Neural Circuits That Mediate Selective Attention (PMC 2018)

**金融應用：**
在制度轉換期間，TRN gate 會自動關閉「舊制度相關特徵」的通道，
避免讓過期的趨勢訊號干擾新制度下的判斷。

---

### 3. HippocampalReplayBuffer（海馬迴 RPE 偏向回放）

**生物來源：**
睡眠時海馬迴會「重播」白天的記憶，但不是隨機重播。
最新研究（Nature Comms 2025）確認：RPE 越高（越驚訝）的事件，
重播優先級越高。就像你失眠時一直在回想今天最意外的事。

**論文：**
- Post-learning replay of hippocampal-striatal activity biased by RPE (Nature Comms 2025)
- Brain-Like Replay Naturally Emerges in RL (arXiv 2402.01467, 2025)

**金融應用：**
- 黑天鵝事件、制度切換、跳空缺口 = 高 RPE 事件
- 正常交易日 = 低 RPE 事件
- Replay buffer 會自動讓模型更多次學習異常事件
- 結果：對尾部風險更敏感，不容易被意外擊穿

---

### 4. PrefrontalWorkingMemory（前額葉工作記憶）

**生物來源：**
前額葉皮質維護兩個**正交**子空間（biorxiv 2025）：
- Context subspace：編碼「現在是什麼市場制度」（慢變化）
- Content subspace：編碼「現在要追蹤什麼模式」（快變化）

入閘（Striatum D1）由多巴胺/RPE 控制：只有驚訝事件才能改寫工作記憶。  
出閘（Striatum D2）由任務需求控制：只輸出當前任務需要的東西。

**論文：**
- Adaptive chunking in PFC-Basal Ganglia circuit (eLife 2025)
- Compositional architecture in PFC (biorxiv 2025)

**為何優於 LSTM：**
```
LSTM:    一個 hidden space，context 和 content 混在一起
PFC-WM:  兩個正交 subspace:
         h_context (regime) 不被 content 梯度污染
         h_content (pattern) 可以快速更新
         input gate 只在 RPE 高時開啟（不是每步都更新）
```

---

## OpEntropy 修復

問題診斷：`torch.sign` 梯度=0，`LayerNorm` 讓所有 op 輸出幅度相同。  
修復：`soft_sign = tanh(10x)`，移除 LayerNorm 改用 `gain/bias`，
alpha init scale 從 0.5 提升到 2.0，加入 entropy warmup（前 150 步 entropy loss 佔主導）。

預期 OpEntropy 曲線：
```
Step   0: ~1.8  (init scale=2.0 -> 起點已更低)
Step  50: ~1.5  (warmup 期間 entropy loss 強制收斂)
Step 150: ~1.2  (warmup 結束，ops 已收斂)
Step 300: ~1.0  (穩定)
```

---

## 執行

```bash
# 標準版（含 OpEntropy 修復）
python examples/quickstart_genome.py

# 完整仿生版（全部四個模組）
python examples/quickstart_bioneuro.py
```
