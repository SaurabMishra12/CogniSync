# CogniSync: Learned-Alpha Hybrid Retrieval with Multi-Signal Adversarial Filtering

This repository contains the empirical artifacts, raw evaluation outputs, configuration logs, and algorithmic pseudocode supporting the CIKM 2026 submission:

**CogniSync: Learned-Alpha Hybrid Retrieval with Multi-Signal Adversarial Filtering for Local-First LLM Agents**

---

## Directory Structure

### `results/`
Raw CSV files corresponding to all quantitative tables reported in the manuscript.

- `MASTER_RAW_EVAL_ALL_QUERIES.csv` — Complete per-query evaluation dataset backing the primary retrieval and statistical analyses.
- `security_comparison_raw.csv` — Per-query adversarial filtering logs.
- `security_baseline_metrics.csv` — Baseline classifier evaluation metrics.
- `security_heuristic_ablation.csv` — Security ablation comparing classifier-only vs. classifier + goal-redirection.
- `table1_main_retrieval.csv` — Main retrieval benchmark results.
- `table2_full_corpus.csv` — Full-corpus evaluation over a 100K-passage MS MARCO index.
- `table3_statistical_significance.csv` — Wilcoxon signed-rank tests and bootstrap confidence intervals.
- `table4_per_dataset.csv` — Dataset-specific performance breakdown.
- `table5_component_ablation.csv` — Learned-alpha vs. cross-encoder component ablation.
- `table6_query_type.csv` — Exact-match vs. semantic query analysis.
- `table7_security_evaluation.csv` — Adversarial filtering metrics (ASR, FPR, MRR drop).
- `table8_jailbreak_stress_test.csv` — Public jailbreak stress-test results.
- `table9_threshold_sensitivity.csv` — Threshold sensitivity analysis for the goal-redirection heuristic.
- `table10_context_ablation.csv` — Synthetic episodic-memory context ablation.

### `logs/`
Configuration snapshots and execution metadata used to ensure deterministic reproducibility.

- `full_corpus_config.json`
- `latency_config.json`
- `security_baseline_config.json`
- `threshold_config.json`

### `plots/`
Generated figures corresponding to visualizations in the manuscript.

- `sensitivity_curve.pdf`
- `main_retrieval_performance.pdf`
- `per_dataset_mrr.pdf`
- `query_type_comparison.pdf`
- `context_ablation.pdf`



### `pseudocode_and_hyperparameters.md`
Algorithmic pseudocode and principal hyperparameter settings for the retrieval and security components.

---

## Reproducibility Protocol

To support anonymous and deterministic reproducibility during double-blind review, this repository provides:

1. **Algorithmic Pseudocode**  
   High-level descriptions of the Multi-Signal Adversarial Filtering layer and the Learned-Alpha Hybrid Retrieval mechanism.

2. **Principal Hyperparameters**  
   Fixed random seeds, model configurations, and threshold values used throughout the experiments.

3. **Pinned Dataset and Model Versions**  
   Public datasets and pretrained models were loaded using fixed revisions to ensure deterministic execution.

4. **Raw Evaluation Outputs**  
   Per-query CSV files backing all tables and figures in the manuscript.

5. **Configuration Logs**  
   Execution metadata and parameter snapshots used during the reported runs.

The full implementation is withheld during peer review to preserve double-blind anonymity, but the provided materials are sufficient to understand the methodological design and independently validate the reported findings.

---

## Core Dependencies

- `torch >= 2.0`
- `sentence-transformers`
- `faiss-cpu`
- `rank_bm25`
- `scikit-learn`
- `datasets`
- `pandas`
- `numpy`

---

## Execution Environment

Experiments were conducted in a controlled  environment using NVIDIA T4 GPU hardware. Random seeds were fixed across all relevant libraries to ensure deterministic execution.

---

## Notes

- All results in the manuscript are directly traceable to the CSV files in the `results/` directory.
- Statistical significance analyses and confidence intervals are fully backed by the provided raw outputs.
- Unevaluated architectural concepts (e.g., persistent episodic memory) are documented for completeness but are clearly identified as design proposals rather than validated components.
