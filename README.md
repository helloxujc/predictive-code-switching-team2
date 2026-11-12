# Predictive Multitask Learning for Streaming Code-Switching

CS6140 Machine Learning

**Team 2**: Hsien-Yu Yang, Jiachen Xu, Xinyi Zhang, Zihan Xiong

**Project Update 1 submission**

The goal is to predict, from the historical prefix `x_1, ..., x_t` only, (i) whether the next token `x_{t+1}` belongs to a different language than `x_t` (binary **switch** label `y_sw`) and (ii) the length bin of the upcoming same-language burst (3-class **duration** label `y_dur` ∈ {0: 1–2 tok, 1: 3–6 tok, 2: 7+ tok}). We evaluate universality across six typologically diverse English–X pairs on the SwitchLingua corpus.

## Repository contents

| File | Purpose |
|---|---|
| `streaming_dataloader.ipynb` | End-to-end Update 1 pipeline: dataset loading, LID, label generation, sub-word alignment, streaming DataLoader, and the naive baseline. |
| `pyproject.toml` / `uv.lock` | Python 3.12 environment (datasets, transformers, torch, lingua-language-detector, wordfreq, pandas, matplotlib). |

## Language pairs

Six English–X pairs chosen to span scripts, families and resource levels:

- **en–ar** (Arabic, Semitic, RTL, script-based LID)
- **en–hi** (Indo-Aryan, SOV, Devanagari — romanized Hinglish rows dropped)
- **en–yue** (Cantonese, Sinitic, low-resource, Han characters)
- **en–ko** (Koreanic, agglutinative, SOV, Hangul)
- **en–es** (Romance, Latin script, frequency-based LID)
- **en–de** (Germanic, Latin script, frequency-based LID)

## Pipeline (Update 1)

```
SwitchLingua text (single- + multi-turn)
  → pair / script filtering + balanced train/val/test split
  → word-level LID (Unicode script OR Zipf-frequency margin for Latin pairs)
  → predictive labels y_sw, y_dur (last word of each turn is masked with -100)
  → sub-word alignment (labels on the last sub-word of each word)
  → Streaming DataLoader yielding left-padded prefix windows [x_{t-N+1}, ..., x_t]
```

**Causal / prefix-only constraint.** Every training example is a left-padded prefix window ending at `t`, and the label is for `x_{t+1}`. When the backbone encoder is plugged in (Update 2), a causal attention mask (lower-triangular) will enforce that position `t` cannot attend to any `t' > t`.

**Multi-turn dialogues.** Each speaker turn is labeled independently and the turns are joined with `</s>`, so prior turns act as prefix context. The last word of every turn is masked (`-100`) because the next token belongs to the other speaker and a change there reflects a speaker switch, not a code-switch.

## Naive baseline

Token-lookup baseline: on the train split, estimate `P(switch | x_t)` for each sub-word id (add-one smoothed toward the global switch rate); predict switch when the probability exceeds a threshold tuned for F1 on val; unseen tokens get the global rate.

Anticipatory F1 of the switch class on the **test** split:

| pair | switch % | F1 always-switch | **F1 token-lookup** |
|---|---:|---:|---:|
| en-ar  | 10.10 | 0.184 | **0.689** |
| en-hi  | 17.02 | 0.291 | **0.520** |
| en-yue |  7.13 | 0.133 | **0.404** |
| en-es  |  8.17 | 0.151 | **0.647** |
| en-ko  | 11.47 | 0.206 | **0.752** |
| en-de  |  8.30 | 0.153 | **0.650** |

Mean F1 = **0.610**, σ_universality (sample, ddof=1) = **0.126**.

Across all pairs a switch is 6–20× more likely right after punctuation than after non-punctuation tokens, so the baseline mostly fires at clause / sentence boundaries. See notebook Section 8 for per-pair precision/recall and top switch-triggering tokens.

## Multitask model (Update 2)

Shared XLM-R (`xlm-roberta-base`) encoder with two linear heads (switch → 2, duration → 3), trained with the
multitask loss `L = λ_sw · CE(switch) + λ_dur · CE(duration, ignore_index=-1)`. The prefix-only (causal)
constraint is enforced by the streaming windows themselves: each example is the prefix `[x_1, …, x_t]` and the
prediction is read at the last position, so `t` never sees `x_{t+1}`. Loss weights were selected by a val-F1
sweep (`switch_pos_weight = 8`, `λ_dur = 0.5`).

Anticipatory switch F1 and duration accuracy on the **test** split, across 13 English–X pairs:

| pair | switch F1 | dur acc |
|---|---:|---:|
| en-ar  | **0.793** | 0.576 |
| en-hi  | **0.658** | 0.570 |
| en-yue | **0.496** | 0.635 |
| en-zh  | **0.486** | 0.742 |
| en-ja  | **0.386** | 0.625 |
| en-ko  | **0.784** | 0.637 |
| en-ru  | **0.618** | 0.494 |
| en-es  | **0.740** | 0.735 |
| en-fr  | **0.783** | 0.642 |
| en-de  | **0.825** | 0.724 |
| en-vi  | **0.765** | 0.640 |
| en-ms  | **0.604** | 0.578 |
| en-fil | **0.478** | 0.519 |

Mean switch F1 = **0.647**, σ_universality (sample, ddof=1) = **0.147**. The model beats the token-lookup
baseline on switch F1; the lowest-F1 pairs (ja / zh / yue / fil) are the CJK / low-resource scripts where
word-level LID and segmentation are hardest.

## mBERT multitask model (Update 2)

Same dual-head architecture, loss and training settings as the XLM-R model above (`switch_pos_weight = 8`,
`λ_dur = 0.5`, `lr = 2e-5`, 1,500 optimizer steps of 8 × 4 windows), with `bert-base-multilingual-cased` as the
shared encoder. The mBERT tokenizer is extended with 241 train-split Han characters it cannot encode (`MBERT_ADDED`,
mostly Cantonese), and the embedding matrix is resized to match (119,788 entries). The code is in notebook Section 6,
"mBERT multitask model"; the model was trained on a Colab T4 GPU.

Anticipatory switch F1 and duration accuracy on the first 300 **test** batches (19,200 positions):

| pair | switch F1 | dur acc |
|---|---:|---:|
| en-ar  | **0.774** | 0.531 |
| en-hi  | **0.663** | 0.238 |
| en-yue | **0.506** | 0.635 |
| en-zh  | **0.566** | 0.716 |
| en-ja  | **0.347** | 0.609 |
| en-ko  | **0.821** | 0.655 |
| en-ru  | **0.561** | 0.428 |
| en-es  | **0.655** | 0.635 |
| en-fr  | **0.734** | 0.581 |
| en-de  | **0.802** | 0.630 |
| en-vi  | **0.777** | 0.570 |
| en-ms  | **0.585** | 0.465 |
| en-fil | **0.468** | 0.281 |

Mean switch F1 = **0.635**, σ_universality (sample, ddof=1) = **0.145**; mean duration accuracy = 0.536, lowest for
en-hi and en-fil. The first 300 batches cover different rows and positions than in the XLM-R evaluation (mBERT has
one token per Han character), so the two tables are not directly comparable yet.

The checkpoint (`checkpoints/mbert_multitask.pt`, 680 MB) and the extended tokenizer (`checkpoints/mbert_tokenizer`)
are not in the repository (`checkpoints/` is git-ignored).

## Running the notebook

The SwitchLingua dataset is gated on Hugging Face — accept its terms on the dataset page and authenticate before launching the notebook.

```bash
# install (uv recommended)
uv sync
# or: pip install -e .

# Hugging Face auth (one-time)
huggingface-cli login

# launch
uv run jupyter lab streaming_dataloader.ipynb
```

Then run the cells top-to-bottom. The notebook is self-contained: it loads SwitchLingua, builds the splits, generates labels, aligns to XLM-R sub-words, materializes the streaming DataLoader, and reports the baseline numbers above.

## Plan for Next

So far we have built the data preprocessing pipeline, generated switch and duration labels, implemented a prefix-only streaming DataLoader, and evaluated a token-lookup baseline. In the coming weeks we plan to:

**Improve data preprocessing and label quality.**
- Improve handling of Romanized Hindi, ambiguous words, and Chinese-character segmentation.
- Expand experiments to at least 12 language pairs.

**Implement and train multitask models (XLM-R, mBERT).**
- Shared multilingual backbone with separate heads for switch and duration prediction.
- Combined loss function with causal constraints to prevent future information leakage.

**Evaluate predictive accuracy and universality.**
- Switch: precision / recall / anticipatory F1; Duration: accuracy and macro-F1.
- Universality: standard deviation of pair-specific F1 across language pairs, with intra- vs. inter-sentential switches analyzed separately.
