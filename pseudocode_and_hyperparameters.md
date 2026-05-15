# CogniSync Pseudocode

## Algorithm 1: Multi-Signal Adversarial Filtering

```text
Require: chunk c; mean clean embedding μ; logistic classifier F; query embedding q_emb
Ensure: decision ∈ {admit, filter}

1:  Encode c to embedding e_c
2:  s_cos ← cos(e_c / ||e_c||, μ)
3:  has_imperative ← 1 [c matches I]  {I: {ignore, reveal, execute, forget, bypass, output}}
4:  len_ratio ← |c| / |c_clean|
5:  p ← F([s_cos, has_imperative, len_ratio])
6:  if p > 0.5 then
7:      return filter
8:  end if
9:  if has_imperative = 1 and cos(e_c / ||e_c||, q_emb) < 0.3 then
10:     return filter   {goal-redirection heuristic}
11: end if
12: return admit
```

## Algorithm 2: Learned-Alpha Hybrid Retrieval with Gating Mechanism

```text
Require: query q; candidate documents D; Random Forest Regressor R; Cross-Encoder C
Ensure: ranked list of documents D_ranked

1:  // Feature Extraction
2:  dense_scores ← normalize(FAISS.search(q, D))
3:  lexical_scores ← normalize(BM25.search(q, D))
4:  features ← extract_features(q, dense_scores, lexical_scores) 
    // Features: [q_len, std(dense), std(lexical), cv(dense), cv(lexical), has_id]

5:  // Predict Fusion Weight
6:  alpha ← R.predict(features)
7:  alpha ← min(max(alpha, 0.0), 1.0)

8:  // Defensive Gating Heuristic (Protects against out-of-domain lexical degradation)
9:  if alpha > 0.85 or max(dense_scores) > 0.85 or cv(lexical) < 0.1 then
10:     alpha ← 1.0
11: end if

12: // First-Stage Score Fusion
13: for d in D do
14:     fusion_scores[d] ← alpha * dense_scores[d] + (1 - alpha) * lexical_scores[d]
15: end for
16: D_fused ← sort_descending(fusion_scores)

17: // Cross-Encoder Reranking
18: D_topK ← select_top_k(D_fused, K=10)
19: for d in D_topK do
20:     rerank_scores[d] ← C.predict(q, d)
21: end for
22: D_reranked ← sort_descending(rerank_scores)

23: // Reconstruct Final List
24: D_ranked ← append(D_reranked, D_fused[K:])
25: return D_ranked
```

## Hyperparameters & Global Configuration

To ensure zero fabrication and deterministic reproducibility, the following global hyperparameters were fixed across all evaluations:

*   **Random Seed (`RANDOM_SEED`)**: `42` (applied uniformly to `numpy`, `torch`, `random`, `scikit-learn`, and `datasets.shuffle`).
*   **Learned-Alpha Regressor**: `RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)`
*   **Dense Model Dimension**: `384`
*   **Cross-Encoder Max Tokens**: `512`
*   **BM25 Algorithm**: standard `BM25Okapi` with default `k1=1.5`, `b=0.75` parameters.
*   **Candidate Pool Size**: `50` (10 original documents + 40 dynamically injected distractors for ablation).
*   **Logistic Security Classifier**: L2-regularized Logistic Regression, `class_weight='balanced'`, default `C=1.0`.
*   **Gating Thresholds**:
    *   Pure Dense Trigger (Alpha): `0.85`
    *   Pure Dense Trigger (Dense Score): `0.85`
    *   Goal-Redirection Heuristic ($\tau$): `0.30` cosine similarity.
