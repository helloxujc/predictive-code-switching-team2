# Predictive Multitask Learning for Streaming Code-Switching

CS6140 Machine Learning

**Team 2**: Hsien-Yu Yang, Jiachen Xu, Xinyi Zhang, Zihan Xiong

**Project Update 2 submission**

The goal is to predict, from the historical prefix `x_1, ..., x_t` only, (i) whether the next token `x_{t+1}` belongs to a different language than `x_t` (binary **switch** label `y_sw`) and (ii) the length bin of the upcoming same-language burst (3-class **duration** label `y_dur` ∈ {0: 1–2 tok, 1: 3–6 tok, 2: 7+ tok}). We evaluate universality across 13 typologically diverse English–X pairs on the SwitchLingua corpus.

## Repository contents

| File | Purpose |
|---|---|
| `streaming_dataloader.ipynb` | End-to-end Update 1 pipeline: dataset loading, LID, label generation, sub-word alignment, streaming DataLoader, and the naive baseline. |
| `pyproject.toml` / `uv.lock` | Python 3.12 environment (datasets, transformers, torch, lingua-language-detector, wordfreq, pandas, matplotlib). |

## Language pairs

13 English–X pairs spanning different scripts and language families:

- **en–ar** (Arabic, Semitic, RTL, script-based LID)
- **en–hi** (Indo-Aryan, SOV, Devanagari and romanized Hinglish; romanized rows retained with improved LID)
- **en–yue** (Cantonese, Sinitic, low-resource, Han characters)
- **en–ko** (Koreanic, agglutinative, SOV, Hangul)
- **en–es** (Romance, Latin script, frequency-based LID)
- **en–de** (Germanic, Latin script, frequency-based LID)
- **en–zh** (Mandarin Chinese, Han characters)
- **en–ja** (Japanese, mixed scripts)
- **en–ru** (Russian, Cyrillic)
- **en–fr** (French, Latin script)
- **en–vi** (Vietnamese, Latin script)
- **en–ms** (Malay, Latin script)
- **en–fil** (Filipino, Latin script)

## Pipeline (Update 1)

```
SwitchLingua text (single- + multi-turn)
  → pair / script filtering + balanced train/val/test split
  → word-level LID (Unicode script OR Zipf-frequency margin for Latin pairs)
  → predictive labels y_sw, y_dur (last word of each turn is masked with -100)
  → sub-word alignment (labels on the last sub-word of each word)
  → Streaming DataLoader yielding left-padded prefix windows [x_{t-N+1}, ..., x_t]
```

**Causal / prefix-only constraint.** Every training example is a left-padded prefix window ending at `t`, and the label is for `x_{t+1}`. In Update 2, the encoder receives only the observed prefix, so future tokens are excluded by the input window itself; no causal attention mask is used.

**Multi-turn dialogues.** Each speaker turn is labeled independently and the turns are joined with `</s>`, so prior turns act as prefix context. The last word of every turn is masked (`-100`) because the next token belongs to the other speaker and a change there reflects a speaker switch, not a code-switch.

## Naive baseline

Token-lookup baseline: on the train split, estimate `P(switch | x_t)` for each sub-word id (add-one smoothed toward the global switch rate); predict switch when the probability exceeds a threshold tuned for F1 on val; unseen tokens get the global rate.

Historical Update 1 anticipatory F1 of the switch class on the **test** split (six pairs):

| pair | switch % | F1 always-switch | **F1 token-lookup** |
|---|---:|---:|---:|
| en-ar  | 10.10 | 0.184 | **0.689** |
| en-hi  | 17.02 | 0.291 | **0.520** |
| en-yue |  7.13 | 0.133 | **0.404** |
| en-es  |  8.17 | 0.151 | **0.647** |
| en-ko  | 11.47 | 0.206 | **0.752** |
| en-de  |  8.30 | 0.153 | **0.650** |

Update 1 mean F1 = **0.610**, σ_universality (sample, ddof=1) = **0.126** (six pairs).

**Updated Update 2 baseline across all 13 pairs:** mean F1 = **0.579**, σ_universality (sample, ddof=1) = **0.104**. The historical table above is retained for comparison with Update 1.

Across all pairs a switch is 6–20× more likely right after punctuation than after non-punctuation tokens, so the baseline mostly fires at clause / sentence boundaries. See notebook Section 8 for per-pair precision/recall and top switch-triggering tokens.

## Multitask model (Update 2)

Shared XLM-R (`xlm-roberta-base`) encoder with two linear heads (switch → 2, duration → 3), trained with the
multitask loss `L = λ_sw · CE(switch) + λ_dur · CE(duration, ignore_index=-1)`. The prefix-only (causal)
constraint is enforced by the streaming windows themselves: each example is the prefix `[x_1, …, x_t]` and the
prediction is read at the last position, so `t` never sees `x_{t+1}`. Loss weights were selected by a val-F1
sweep (`switch_pos_weight = 8`, `λ_dur = 0.5`); the final model is trained for 6,000 optimizer steps of 8 × 4
windows on a Colab T4 GPU.

Anticipatory switch F1 (**R-EVAL-1**) and duration accuracy (**R-EVAL-2**) on the full **test** split, across 13
English–X pairs (**R-EVAL-4**):

| pair | switch F1 | dur acc |
|---|---:|---:|
| en-ar  | **0.716** | 0.602 |
| en-hi  | **0.667** | 0.652 |
| en-yue | **0.555** | 0.674 |
| en-zh  | **0.579** | 0.738 |
| en-ja  | **0.612** | 0.787 |
| en-ko  | **0.817** | 0.689 |
| en-ru  | **0.683** | 0.648 |
| en-es  | **0.703** | 0.674 |
| en-fr  | **0.790** | 0.679 |
| en-de  | **0.747** | 0.675 |
| en-vi  | **0.693** | 0.674 |
| en-ms  | **0.633** | 0.628 |
| en-fil | **0.479** | 0.536 |

Mean switch F1 = **0.667**, σ_universality (sample, ddof=1) = **0.095** (**R-EVAL-5**). The model beats the
updated 13-pair token-lookup baseline (0.579) on switch F1 and is slightly more consistent across pairs (σ 0.095 vs. 0.104). The lowest-F1 pairs are en-fil, en-yue, and en-zh; LID and tokenization differences may contribute, but require further analysis.

Macro-average across switch types, all pairs pooled (**R-EVAL-3**):

| switch type | switch F1 | dur acc | dur macro-F1 |
|---|---:|---:|---:|
| inter-sentential | 0.718 | 0.768 | 0.609 |
| intra-sentential | 0.554 | 0.559 | 0.525 |
| extra-sentential / tag | 0.707 | 0.693 | 0.564 |
| **macro-average** | **0.659** | **0.673** | **0.566** |

Intra-sentential switches (a switch inside one sentence) are the hardest for both metrics, as expected.

## mBERT multitask model (Update 2)

Same dual-head architecture, loss and training settings as the XLM-R model above (`switch_pos_weight = 8`,
`λ_dur = 0.5`, `lr = 2e-5`, 6,000 optimizer steps of 8 × 4 windows), with `bert-base-multilingual-cased` as the
shared encoder. The mBERT tokenizer is extended with 241 train-split Han characters it cannot encode (`MBERT_ADDED`,
mostly Cantonese), and the embedding matrix is resized to match (119,788 entries). The code is in notebook Section 6,
"mBERT multitask model"; the model was trained on a Colab T4 GPU.

Anticipatory switch F1 and duration accuracy on the full **test** split:

| pair | switch F1 | dur acc |
|---|---:|---:|
| en-ar  | **0.721** | 0.600 |
| en-hi  | **0.623** | 0.697 |
| en-yue | **0.538** | 0.648 |
| en-zh  | **0.605** | 0.698 |
| en-ja  | **0.621** | 0.757 |
| en-ko  | **0.824** | 0.665 |
| en-ru  | **0.682** | 0.614 |
| en-es  | **0.716** | 0.654 |
| en-fr  | **0.809** | 0.678 |
| en-de  | **0.743** | 0.673 |
| en-vi  | **0.689** | 0.674 |
| en-ms  | **0.605** | 0.594 |
| en-fil | **0.495** | 0.497 |

Mean switch F1 = **0.667**, σ_universality (sample, ddof=1) = **0.098**. Macro-average across switch types
(**R-EVAL-3**), all pairs pooled:

| switch type | switch F1 | dur acc | dur macro-F1 |
|---|---:|---:|---:|
| inter-sentential | 0.713 | 0.760 | 0.541 |
| intra-sentential | 0.545 | 0.553 | 0.474 |
| extra-sentential / tag | 0.708 | 0.675 | 0.527 |
| **macro-average** | **0.655** | **0.662** | **0.514** |

Both models are now scored on the full test split, so the two tables are directly comparable. They tie on mean
switch F1 (0.667); XLM-R has the slightly lower σ_universality (0.095 vs 0.098), i.e. the marginally more
"universal" predictor (**R-EVAL-5**). mBERT emits one token per Han character, so its position counts (`n`,
printed in the notebook) are higher for yue / zh / ja, but the underlying test rows are the same.

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

Then run the cells top-to-bottom. The notebook loads SwitchLingua, builds the splits, generates labels, aligns sub-words, and includes the baseline and Update 2 multitask experiments. Training the models requires substantial GPU time; the mBERT checkpoint and tokenizer are not included in the repository.

## Plan for Next

So far we have expanded the pipeline to 13 language pairs, improved Romanized Hindi labeling, trained XLM-R and mBERT multitask models, and evaluated both on the full test split. Next we plan to:

**Improve training efficiency and convergence analysis.**
- Replace repeated prefix encoding with longer sequence passes using causal attention masks where appropriate, and train for more steps. Each current model has processed only about 5% of one pass through the training pool.
- Log averaged training losses and validation metrics. Current loss curves fluctuate partly because each logged value comes from a single micro-batch of eight windows; zero duration loss can occur when a batch contains no switches.

**Evaluate multitask learning and data partitions.**
- Compare joint multitask training against single-task models.
- Test punctuation kept vs. stripped and single-turn vs. multi-turn settings; explore window lengths and other partitions if time permits.

**Analyze difficult cases.**
- Investigate intra-sentential switches and weaker language pairs (en-fil, en-yue, en-zh).
- Report switch F1, duration accuracy, and cross-language F1 variation, including error examples.
