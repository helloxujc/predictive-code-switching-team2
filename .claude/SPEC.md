# Predictive Multitask Learning for Streaming Code-Switching

Course project details.

**Goal:** Evaluate the universality and predictive accuracy of multilingual transformer architectures in a real-time,
prefix-only context using the SwitchLingua corpus.

## I. Project Objective and Rationale

The primary objective is a comparative study of transformer-based paradigms to determine which architecture most
consistently anticipates code-switch events and their durations across diverse language pairs.

Unlike detection tasks that use full-sentence context, this is a **prediction** task. The model must predict the state
of the next token `x_{t+1}` based solely on the historical prefix `x_1, …, x_t`. This simulates real-world scenarios like
predictive text for bilingual users or real-time dialogue systems.

### Task Definition

| ID | Task | Target Type | Loss Function | Prediction Point |
|---|---|---|---|---|
| R-TASK-1 | Switch Prediction (`y_sw`) | Binary (0/1) | Cross-Entropy | At token `t`, predict if `LID(x_{t+1}) = LID(x_t)` |
| R-TASK-2 | Duration Prediction (`y_dur`) | Multiclass (0/1/2) | Masked Cross-Entropy | At token `t`, predict the length bin of the upcoming burst |

## II. Phase 1: Data Engineering for Streaming

We will utilize the [SwitchLingua_text dataset](https://huggingface.co/datasets/Shelton1013/SwitchLingua_text), which
provides bilingual and multilingual mixed sentences generated through linguistic constraints (Equivalence, Functional
Head) to ensure naturalness.

### A. Dataset Acquisition and Preprocessing

- **R-DATA-1 Source:** Access the dataset via Hugging Face (`Shelton1013/SwitchLingua_text`).
- **R-DATA-2 Language Coverage:** Select diverse pairs from the 12 supported languages (e.g., Arabic–English,
  Hindi–English, Cantonese–English, Spanish–English).
- **R-DATA-3 Sub-word Tokenization:** Use model-specific tokenizers (e.g., BPE for RoBERTa variants). Ensure sub-word
  segments are correctly mapped to their respective Language IDs.

### B. Predictive Label Generation

For every token `x_t` in a sequence, generate labels based on the upcoming token `x_{t+1}`:

- **R-LABEL-1 Switch Label (`y_sw(t)`):** 1 if `x_{t+1}` belongs to a different language than `x_t`; 0 otherwise.
- **R-LABEL-2 Duration Label (`y_dur(t)`):** If `y_sw(t) = 1`, set `y_dur(t)` to the Class ID (0, 1, or 2) corresponding
  to the total length of the new language segment starting at `x_{t+1}`. If `y_sw(t) = 0`, set to −1 (ignorable).

### C. Duration Binning (R-LABEL-3)

| Class ID | Class Name | Burst Length | Rationale |
|---|---|---|---|
| 0 | Small | 1–2 tokens | Short, lexical insertions |
| 1 | Medium | 3–6 tokens | Phrase-level segments (NP/VP) |
| 2 | Large | 7+ tokens | Clausal or full-sentence switches |

## III. Phase 2: Model Architecture and Implementation

Implement a custom dual-head architecture adapted for causal (left-to-right) attention.

### A. Predictive Architecture

- **R-MODEL-1 Encoder Backbone:** Utilize a multilingual transformer (e.g., XLM-R, mBERT).
- **R-MODEL-2 Causal Constraint:** Apply a causal attention mask (look-ahead mask) or use a causal transformer variant to
  ensure token `t` cannot attend to `t+1`.
- **R-MODEL-3 Multitask Heads:**
  - **Switch Head:** Linear layer on top of the hidden state at `t` to predict the switch at `t+1`.
  - **Duration Head:** Parallel linear layer to predict the upcoming segment length.

### B. Multitask Loss Function (R-LOSS-1)

The total loss is a weighted sum:

```
L_total = λ1 · L_switch + λ2 · L_duration
```

Both tasks use Cross-Entropy Loss.

`L_duration` must use `ignore_index=-1`, ensuring the model only learns duration at the exact moment a switch is
anticipated.

## IV. Phase 3: Training and Optimization Roadmap

| Weeks | Phase | Work |
|---|---|---|
| 1–4 | Data Pipeline | Ingest SwitchLingua. Implement "Streaming DataLoaders" yielding context windows `[x_{t−N}, …, x_t]`. |
| 5–7 | Implementation | Build the Causal Encoder and Multitask heads. Ensure gradients flow correctly to the shared backbone. |
| 8–10 | Experimentation | Train and compare different model paradigms. Monitor convergence for both tasks. |
| 11–13 | Evaluation | Analyze cross-lingual performance and variance across different language pairs in SwitchLingua. |
| 14 | Final Synthesis | Finalize the research paper and project demo. |

## V. Evaluation and Universality Metrics

Success is defined by the **Universality Metric**: consistent performance across diverse typological pairs provided in
the SwitchLingua benchmark.

### A. Performance Metrics

- **R-EVAL-1 Anticipatory F1:** F1-score of the SWITCH class at position `t` for the actual event at `t+1`.
- **R-EVAL-2 Duration Accuracy:** Accuracy of `y_dur` specifically at tokens where a switch actually occurs.
- **R-EVAL-3 Macro-Average Accuracy:** Performance across different types of switches (inter-sentential vs.
  intra-sentential).

### B. Variance Analysis

- **R-EVAL-4 Pair-Specific Breakdown:** Report F1-scores for every individual language pair.
- **R-EVAL-5 σ_universality:** Calculate the Standard Deviation of F1-scores across all pairs. The superior "Universal
  Predictor" is the model with the lowest variance.

## VI. Final Deliverables

- **R-DELIV-1 Research Paper:** 6–8 pages in official ACL two-column format.
- **R-DELIV-2 Streaming Simulation:** A Python script/demo that processes text word-by-word and outputs the "Next-Token
  Switch Probability" and "Anticipated Duration."
- **R-DELIV-3 Code Repository:** A documented GitHub repository featuring the causal masking implementation, dataset
  preprocessing for SwitchLingua, and multitask loss logs.
