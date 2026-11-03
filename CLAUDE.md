# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

CS6140 course project: predictive multitask learning for streaming code-switching on SwitchLingua
(`Shelton1013/SwitchLingua_text`, gated on Hugging Face). From the prefix `x_1..x_t` only, predict
(i) `y_sw`: whether `x_{t+1}` is in a different language than `x_t`, and (ii) `y_dur`: the length bin of the
upcoming burst (0 = 1–2, 1 = 3–6, 2 = 7+ tokens). Metrics, duration binning and the loss definition must follow
the spec below; cite requirements by their IDs (e.g. `R-EVAL-3`).

@.claude/SPEC.md

All code lives in `streaming_dataloader.ipynb`. The model and training loop (Phase 2) are not implemented yet.
`README.md` describes the original 6-pair Update 1 and may be behind the notebook.

## Commands

```bash
uv sync                                    # Python 3.12 env in .venv
uv run jupyter lab streaming_dataloader.ipynb
huggingface-cli login                      # needed once; the dataset is gated

# Execute the whole notebook headlessly (≈30–40 min: every row is processed for 2 models × 2 punctuation modes)
.venv/bin/jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
  --ExecutePreprocessor.kernel_name=python3 streaming_dataloader.ipynb --output-dir <dir> --output executed.ipynb

# Fast smoke test (~30 s): run a copy with tiny sample sizes
sed -e 's/ROWS_PER_PAIR_TYPE = 3400/ROWS_PER_PAIR_TYPE = 60/' -e 's/N_EVAL_ROWS = 300/N_EVAL_ROWS = 10/' \
  -e 's/TRAIN_ROWS_PER_PAIR = 5400/TRAIN_ROWS_PER_PAIR = 40/' \
  streaming_dataloader.ipynb > /path/to/smoke.ipynb   # then nbconvert --execute it as above
```

There are no tests or linters. Verify changes with the smoke run first, then the full run. Executing in place
rewrites every output and produces a very large diff.

## Pipeline architecture (notebook sections)

1. **Config** (`SELECTED_PAIRS`: 13 X–English pairs, all but Italian; the dataset's "Philippines" is `fil`; `MODELS`: `xlmr` = `xlm-roberta-base`, `mbert` =
   `bert-base-multilingual-cased`; `PUNCT_MODES` = keep/strip; `CS_TYPES`). The dataset has a single split
   sorted by language pair, so the notebook builds its own balanced, stratified, shuffled train/val/test split at
   row level (Section 4).
2. **Word-level LID** (Section 3). `TOKEN_RE` pre-tokenizes the text, with one token per Han or kana character.
   - **Script-based** for ar/hi/yue/zh/ja/ko/ru.
   - **Hinglish LID** for en-hi Latin-script words (romanized Hindi vs. English): `word_lids` sends them to the LinCE
     hin-eng model (`hinglish_lid`, cached per turn on punctuation-stripped words). Romanized en-hi rows are kept and
     flagged `romanized`. Use `word_lids` (turn level), not `word_lid`, when labeling text.
     LinCE labels are batched (`prefetch_hinglish`, Section 7) and cached on disk in `data/hinglish_lid_*.pkl`
     (git-ignored). Delete that file if the Hinglish model, `TOKEN_RE`, `strip_punct` or `TEXT_NORMALIZE` changes. Any new language needs an entry in `SCRIPT_OF_LANG` (and
     `SCRIPT_RE` for the row filter).
   - **wordfreq Zipf-margin LID** (`make_latin_lid`) for the Latin-script pairs es/fr/de/vi/ms/fil. Section 4
     evaluates it against lingua and the LinCE model.
   - Neutral tokens inherit the previous word's language (`fill_neutral`).
   - `make_labels` produces the word-level `y_sw` / `y_dur`.
3. **Sub-word alignment** (`align_text`, Section 5). It tokenizes the *raw turn text* with the chosen model's
   tokenizer and maps sub-words back to words via `offset_mapping`.
   - Labels sit on the last sub-word of each word. Other sub-words get `-100` / `-1`.
   - When one sub-word merges several CJK characters, only the last merged word keeps its label.
   - Labels and LID are word-level and shared by both tokenizers. Only the positions differ (mBERT has one
     token per Han character).
4. **Streaming** (`process_example` → `StreamingCSDataset` → `collate`, Section 6). Multi-turn rows are labeled
   per turn, joined with the tokenizer's sep token, and the last word of each turn is masked.
   - Windows are left-padded prefixes `[x_{t-N+1}..x_t]` (`WINDOW = 64`), with the prediction at the last
     position.
   - Each window carries `pair_id`, `multi_turn`, `cs_type`, `after_punct`, `row`, `word_id`.
   - `row` + `word_id` identify the same prediction point across tokenizers.
   - Every function takes `model=` and `punct=`. Use `partial(collate, model=m)` so the pad id is right.
5. **Experiment grid** (`REGIMES`, `make_loaders`, `multitask_loss`): `MODELS × PUNCT_MODES × REGIMES` = 8 runs.
   - Window sweep: `WINDOW_EXPERIMENTS` varies `N` over `WINDOWS` (sub-words) for both models on keep + joint (10 runs).
     Pass `window=` to `make_loaders`. The window is cut only at streaming time, so labels and positions are identical
     for every `N`. `positions()` keeps `prefix` (t + 1) for breakdowns by available context.
   - Training setups: `SETUP_EXPERIMENTS` = `multilingual` (one model on all pairs) vs `per_pair` (one model per pair).
     Pass `pairs=` to `make_loaders`. Train then streams only `TRAIN_ROWS` (`TRAIN_ROWS_PER_PAIR` = 5400 rows per pair, at most the smallest pair's train rows,
     stratified by conversation type × `cs_type`, identical in both setups); val/test stream all rows of those pairs.
     `pairs=None` streams the full, unequal train split. Select the multilingual model on macro-averaged per-pair val F1.
   - Conversation type: `CONV_EXPERIMENTS` trains one model per conversation type. Pass `conv=` ("single_turn" /
     "multi-turn") with `pairs=PAIR_NAMES`, so each gets its half of `TRAIN_ROWS`. Evaluate on both types separately
     (`conv=` on the test loader or the `multi_turn` field / column).
   - Dialogue context: `CONTEXT_EXPERIMENTS` (multi-turn rows only). `context="dialogue"` (default) lets windows include
     earlier turns; `context="turn"` cuts them at the current turn's start. Same labels and positions; each window carries
     `turn` (turn index), so report F1 on `turn >= 1` too.
   - `joint` uses loss weights λ = (1, 1).
   - `sequential` trains the switch head first (λ = 1, 0), then the duration head (λ = 0, 1). The duration
     stage streams only `y_sw = 1` positions (`only_switch=True`).
   - Val/test loaders always stream all positions.
6. **Baseline and metrics** (Sections 7–8). `token_lookup` is the baseline: P(switch | token) and the
   per-token most frequent duration bin.
   - `metrics()` computes the spec's metrics: switch F1/P/R, duration accuracy and macro-F1 at true switches.
     `by_group` breaks them down by pair or `cs_type`.
   - σ_universality is the sample std (`ddof=1`) of per-pair scores.
   - Reuse `metrics()` for the trained models.
   - Compare XLM-R vs. mBERT switch F1 on shared (`row`, `word_id`) positions, because yue/zh/ja position sets
     differ.

## Phase 2 training

The plan is the "Phase 2: training along the partitioned dimensions" table at the end of the notebook (Next Steps). Rules:

- Train XLM-R and mBERT for each experiment list (`EXPERIMENTS`, `WINDOW_EXPERIMENTS`, `SETUP_EXPERIMENTS`, `CONV_EXPERIMENTS`,
  `CONTEXT_EXPERIMENTS`) and score them with the Section 8 `metrics()` / `by_group()`, so models and baseline are comparable.
- Evaluate with the same `punct`, `window` and `context` as training; val/test loaders always use `regime="joint"`.
- `sequential`: stage 2 updates the backbone with the duration loss only and can erase switch skill. Record switch F1 after
  stage 1 (or freeze the backbone in stage 2) and add a switch-only model (λ = 1, 0) as a control.
- Window sweep: only points with `prefix > N` differ, and the same `N` covers different numbers of words per pair, so also
  compare at matched words per window. The token-lookup baseline does not depend on `N`.
- Report every model on the baseline's slices: per pair (σ_universality), per `cs_type` (macro-average), `after_punct`,
  `multi_turn`, en-hi `romanized`, `turn >= 1` for the context experiment, and XLM-R vs. mBERT on shared positions.

## Conventions

- `StreamingCSDataset(rows=...)` streams a subset of row indices; `row` stays the index in the full split.
- Memory (a full run holds every split × 2 models × 2 punct modes): `processed` rows store `ids`/`sw`/`dur`/`wid` as small
  numpy arrays (`COMPACT`, no `lid`), and `positions()` uses int8/int32 columns. LID models are loaded on first use with
  `lid_model(name)` and dropped with `LID_MODELS.pop(name, None)` once a section is done; with a full LinCE cache the
  Hinglish model is never loaded. Don't keep a second kernel with the full notebook open while running another one.
- `get_turns` normalizes every turn with `TEXT_NORMALIZE` (curly quotes, dashes, full-width letters/digits -> ASCII; one
  char each, offsets kept) because mBERT encodes those as `[UNK]`.
- mBERT vocabulary is extended with `MBERT_ADDED` (train-split Han characters it cannot encode, mostly Cantonese). Phase 2
  must call `model.resize_token_embeddings(len(TOKENIZERS["mbert"]))` and save that tokenizer with the model.
- `collate` prepends `<s>` / `[CLS]` and returns `position_ids` counted from the first real token. Always pass them to the
  model: BERT counts positions from the left edge including padding, so left-padded windows would shift otherwise.
- Ignore indices: `y_sw` uses `-100`, `y_dur` uses `-1` (the spec requires `ignore_index=-1` for the duration loss).
- `punct="strip"` trims edge punctuation from each word before labeling (`strip_punct_text`). Positions barely
  change except for standalone CJK punctuation, so keep and strip pose the same task without the punctuation cue.
- Markdown cells quote concrete numbers from the outputs. Update them after a run that changes results.
