# RQ1 Methodology — Variance and Regularity in AI vs. Human Syntactic Structure

<!-- Working record only — bullets, not prose. Full LaTeX write-up later. -->
<!-- Equations numbered continuously (N) through the whole doc, each on its own line. -->
<!-- Citations: (Author, Year) inline, matches lit review / natbib. -->

## 1. Notation

- `k_i` — count of one category in document *i*; `n_i` — its sentences (`n_sents`); `r_i = k_i/n_i` — its observed rate
- `p_i` — document *i*'s true rate (its habit): the probability that any given sentence in it is of that category. Unobserved
- `N` — documents in a group; group = one (domain, source) pair, e.g. reuter/claude
- `μ`, `V` — mean and variance of the `p_i` across a group's documents
- `ρ = V / μ(1−μ)` — relative between-document variance: the share of the maximum possible spread at that mean
- `S²` — sample variance of the `r_i` (divisor `N−1`); `r̄` — their mean
- Scheme — `sentence_type` (SIMPLE / COMPLEX / COMPOUND / COMPLEX-COMPOUND / OTHER) or `sentence_structure` (LOOSE / PERIODIC / OTHER), Feng et al. (2012)
- Track 2 only: `H` — Shannon entropy (nats); `D` — parts in a composition

## 2. Research Question & Design

- RQ1: "Does AI prose show reduced variance and greater regularity across interpretable syntactic features, such as loose/periodic sentences and clause complexity, when compared to human prose?"
- RQ0 (`docs/rq0_methodology.md`) establishes how often each source uses each category; RQ1 asks how consistently, given those rates
- **Track 1 — Variance** (complete): how much each category's rate varies *between* documents, per (domain, source, category), human vs each AI (§4)
- **Track 2 — Regularity** (under review): within-document uniformity (§5)
- **Track 3 — per-category diagnostics**: retired (§6)
- Track 1's answer is one bar chart per domain (§10.4), the same layout as RQ0's

## 3. Data, Cleaning & Filtering

- Input: `data/processed/features/doc_features.feather`, built by `src/analysis/features/build_doc_features.py` from sentence-level Feng labels over the Ghostbuster constituency parses
- Two cleaning rules, applied to sentences before aggregation (shared with RQ0):
  - **Sentences with no letters dropped** — 2,459 (lone `"` 1,406 times; also `.`, `*`, list numbers): sentence-splitter artefacts, all labelled OTHER; 6–69% of a group's type-OTHER sentences (reuter GPT 69%, Claude 63%)
  - **Reuter sentences with no author dropped** — 108: orphans the author fix couldn't assign, which formed 43 pseudo-documents of 1–6 sentences
  - Result: 8,991 documents
- How they were found: a scan for documents extreme under a fitted Beta-binomial (p < 10⁻⁴ per document-category, ~7 expected by chance) flagged 25 pairs in 20 documents; inspection traced most to these two artefacts. After cleaning: 17 pairs in 14 documents, mostly genuine writing; six judged non-prose feed the sensitivity pass (§4.6)
- Filter: `n_sents ≥ 5` via the shared `filter_min_sents` (`src/analysis/shared/filters.py`) — 42 documents excluded (0.47%) → **8,949 documents, 987–1,000 per group**; same pool as RQ0
- Track 1: denominator `n_sents` for all 8 categories, both OTHERs included, as in RQ0. No zero-replacement, so the old algo1-OTHER exclusion and `denom ≤ 1` rule don't apply
- Track 2 still uses the algo1 conforming-only composition and `denom ≤ 1` exclusion (§5)

## 4. Track 1 — Between-Document Variance

### 4.1 The quantity

- Each document has a true rate `p_i`; the group's habits vary with variance `V`
- Since `0 ≤ p ≤ 1`, `p² ≤ p`, so the variance has a ceiling, reached only if every document is all-or-nothing (0% or 100%):

  $$V = E[p^2] - \mu^2 \le \mu(1-\mu) \tag{1}$$

- Headline — relative, so groups with different rates (RQ0) compare fairly:

  $$\rho = \frac{V}{\mu(1-\mu)}, \qquad 0 \le \rho \le 1 \tag{2}$$

- Absolute `V` reported alongside (e.g. √V as a typical document-to-document difference in percentage points)
- Second reading: `ρ` is the correlation between two sentences of the same document (both of type *j* with probability `E[p_i²] = V + μ²`, so covariance `V`, variance `μ(1−μ)`). "AI documents vary less from each other" and "AI sentences are less tied to their document's own habits" are one fact
### 4.2 Estimator

- Assumptions: **A1** — given `p_i`, a document's sentences are independent, so `k_i ~ Binomial(n_i, p_i)`; **A2** — documents are independent (handled for reuter in §4.3); `n_i ≥ 2`. Nothing is assumed about the shape of the `p_i`'s distribution
- Observed rate's variance, by the law of total variance — real spread plus sampling noise:

  $$\text{Var}(r_i) = V + \frac{E[p_i(1-p_i)]}{n_i} \tag{3}$$

- What the sample variance estimates (using A2):

  $$E[S^2] = \frac{1}{N}\sum_i \text{Var}(r_i) = V + \frac{1}{N}\sum_i \frac{E[p(1-p)]}{n_i} \tag{4}$$

- Each document's noise, estimated without bias from its own count, for any `p_i` (it's the squared standard error of the document's rate; `n_i − 1` for the same reason as in `S²`):

  $$E\!\left[\frac{r_i(1-r_i)}{n_i-1} \,\Big|\, p_i\right] = \frac{p_i(1-p_i)}{n_i} \tag{5}$$

- The estimator — observed spread minus average estimated noise, then divided by the ceiling:

  $$\hat V = S^2 - \frac{1}{N}\sum_i \frac{r_i(1-r_i)}{n_i-1}, \qquad \hat\rho = \frac{\hat V}{\bar r(1-\bar r)} \tag{6}$$

- `E[V̂] = V` exactly, by Eqs. 4–5. `ρ̂`'s denominator is biased by `Var(r̄) ≈ S²/N` — negligible at `N ≈ 1,000`
- `V̂` and `ρ̂` can be slightly negative when there is no real variation (unbiased, so they must average 0); left unclipped, since clipping would bias them. Only case: reuter GPT type OTHER, `ρ̂ = −0.0008`
- `r̄` is the per-document mean — identical to RQ0's `mean_rate` in all 72 cells (checked)

### 4.3 Uncertainty — resampling

- 2,000 resamples per group (Efron, 1979):
  - essay, wp — documents with replacement
  - reuter — whole authors with replacement (50 per group), all of each chosen author's documents: same-author articles share habits, so documents aren't independent (A2)
- One set of resample indices per group, reused for all 8 categories
- Each group has its own generator, seeded from `SEED` and the group's position in `DOMAINS`/`SOURCES` — reproducible, groups independent, and a group's resamples don't depend on any other group's size (needed for §4.6)
- SE = standard deviation of the 2,000 `ρ̂`; 95% interval = 2.5th–97.5th percentiles (asymmetric near zero)

### 4.4 Test

- Human vs AI, per domain and category — human minus AI, so a positive value supports RQ1:

  $$z = \frac{\hat\rho_h - \hat\rho_a}{\sqrt{SE_h^2 + SE_a^2}} \tag{7}$$

- Two-sided p from the standard normal; 48 tests (8 categories × 3 domains × 2 comparisons) corrected as one BH-FDR family (Benjamini and Hochberg, 1995)
- Verdict: **human more variable** / **AI more variable** (significant, by sign) / **no difference**
- In essay and wp each AI document shares a prompt with a human one; if that correlates the two estimates, Eq. 7's SE is overstated → conservative

### 4.5 Verification

- Consistency check — Eq. 3 rewritten with `E[p(1−p)] = μ(1−μ)(1−ρ)`:

  $$\text{Var}(r_i) = \mu(1-\mu)\left[\rho + \frac{1-\rho}{n_i}\right] \tag{8}$$

| Check | Result |
|---|---|
| True variance recovered under five shapes — one Beta, two-curve mixture, heavy tail, length-dependent rates, 1% category (real wp-human lengths, 400 simulations each) | `V̂` 0.996–1.004 × true; `ρ̂` 0.996–1.002 × true |
| False positives with equal `ρ`, real means and lengths — reuter COMPLEX-COMPOUND human vs Claude (3× mean gap), essay type OTHER human vs GPT, reuter type OTHER human vs GPT (150 simulations each) | 6.0%, 5.3%, 5.3% |
| Author clustering — real reuter author structure, equal `ρ` (150 simulations) | 22.7% resampling documents vs 5.3% resampling authors; `V̂` 1.000 × true |
| Real reuter: SE resampling authors ÷ SE resampling documents | median 1.08 (0.91–1.94); largest all human (SIMPLE 1.94×) |
| Noise share predicted by Eq. 8 (`ρ = 0.03`, `n = 20`) vs observed median over 72 cells | 62% vs 60% |
| Point estimates recomputed independently; FDR re-run; untouched groups across the two passes | match to 10⁻¹⁶; matches; identical |
### 4.6 Sensitivity

- **Six documents judged non-prose on inspection** — kept in the main analysis, because they were found by being extreme on the measured variables and excluding them would bias it; a second pass runs without them:
  - essay/claude 851 — list of interview questions
  - essay/human 841 — sectioned company overview
  - reuter/human 9 (JimGilchrist) — mostly numbered statistics
  - wp/gpt 671 — Reddit-thread format
  - wp/human 627 — terminal-log format
  - wp/human 903 — Google-search format
- Result: **3 of 48 verdicts change, all type OTHER** — where non-prose sentences land:
  - essay human vs Claude: no difference → human more variable
  - reuter human vs GPT: no difference → human more variable
  - wp human vs Claude: human more variable → no difference
- The other 45 verdicts, including every one in the other seven categories, are unchanged
- **Sentence order (A1)** — measured once rather than as a second pass. Noise from contiguous halves vs Eq. 5: median 1.05× (0.86–1.28; largest in OTHER). `ρ̂` with contiguous-halves noise: median 0.93× (10th–90th 0.77–1.13); interleaved halves: 1.00× (0.90–1.12). 6 of 48 human-vs-AI directions differ with contiguous halves, mostly ratios already near 1 (largest: essay and wp LOOSE vs GPT, wp COMPLEX vs Claude); 2 of 48 with interleaved. Whether within-document structure counts as chance is a definition the data can't settle — Eq. 6 uses the standard one

### 4.7 Results

- **14 of 48 significant after FDR: 12 human more variable, 2 AI more variable, 34 no difference**

| Domain | Human more variable | AI more variable | No difference |
|---|---|---|---|
| essay | 0 | 2 (GPT: COMPOUND, type OTHER) | 14 |
| reuter | 2 (Claude: PERIODIC, COMPOUND) | 0 | 14 |
| wp | 10 (5 vs GPT, 5 vs Claude) | 0 | 6 |

| Domain | Comparison | Scheme | Category | ρ human | ρ AI | z | FDR p |
|---|---|---|---|---|---|---|---|
| essay | vs GPT | type | COMPOUND | 0.0207 | 0.0450 | −4.21 | 0.0002 |
| essay | vs GPT | type | OTHER | 0.0987 | 0.1660 | −3.06 | 0.0082 |
| reuter | vs Claude | structure | PERIODIC | 0.0364 | 0.0096 | 3.87 | 0.0005 |
| reuter | vs Claude | type | COMPOUND | 0.0315 | 0.0062 | 3.61 | 0.0012 |
| wp | vs Claude | structure | PERIODIC | 0.0549 | 0.0146 | 6.43 | < 0.0001 |
| wp | vs Claude | structure | OTHER | 0.0732 | 0.0409 | 5.37 | < 0.0001 |
| wp | vs Claude | type | SIMPLE | 0.0464 | 0.0273 | 3.84 | 0.0005 |
| wp | vs Claude | type | COMPLEX-COMPOUND | 0.0511 | 0.0102 | 5.54 | < 0.0001 |
| wp | vs Claude | type | OTHER | 0.0478 | 0.0317 | 2.56 | 0.0364 |
| wp | vs GPT | structure | PERIODIC | 0.0549 | 0.0137 | 6.27 | < 0.0001 |
| wp | vs GPT | structure | OTHER | 0.0732 | 0.0180 | 9.23 | < 0.0001 |
| wp | vs GPT | type | SIMPLE | 0.0464 | 0.0144 | 6.46 | < 0.0001 |
| wp | vs GPT | type | COMPLEX | 0.0371 | 0.0162 | 4.15 | 0.0002 |
| wp | vs GPT | type | COMPLEX-COMPOUND | 0.0511 | 0.0102 | 5.31 | < 0.0001 |

- Reading:
  - **wp — clear yes**: human stories vary 2–5× more between documents; largest gaps COMPLEX-COMPOUND and PERIODIC
  - **reuter — weak**: two categories, against Claude only
  - **essay — no, slightly reversed**: GPT's essays vary more than human ones in COMPOUND and type OTHER
  - **PERIODIC** is the most consistent category: significant in wp against both AIs and in reuter against Claude, always human more variable
  - Type OTHER verdicts depend on a handful of non-prose documents (§4.6); the wp-vs-Claude one is also the weakest significant result

### 4.8 Limitations

- The noise definition: sentence order treated as irrelevant (§4.6)
- Independence: handled for reuter by resampling authors; essay and wp have no author information, so documents are assumed independent
- The six-document judgement is by inspection, which is why it is a sensitivity pass, not an exclusion
- `ρ̂` ratios are unstable near zero (e.g. reuter GPT type OTHER) — the test uses the difference, not the ratio
- Per-category by design — no single composition-level number (the shared-`ρ` model that would give one didn't fit, §7)

## 5. Track 2 — Regularity (under review)

- **Status**: as originally implemented; outputs regenerated on the cleaned data; not yet re-examined with Track 1's level of scrutiny. The known issues below need resolving before its results are reported

### 5.1 Pipeline (current implementation)

- **Step 1**: proportions via `compute_proportions_algo1` (4 conforming categories, `denom ≤ 1` excluded) and `compute_proportions_algo2` (3 categories, `n_sents`); no zero-replacement (`0·ln(0) := 0`)
- **Step 2**: composition-specific `n` for the bias correction:

  $$n = c_S+c_C+c_{Cd}+c_{CC} \ \text{(algo1)}, \qquad n = n\_sents \ \text{(algo2)} \tag{9}$$

- **Step 3**: plug-in Shannon entropy per document:

  $$\hat{H} = -\sum_j p_j \ln(p_j) \tag{10}$$

- **Step 4**: Miller-Madow bias correction:

  $$H_{MM} = \hat{H} + \frac{D-1}{2n} \tag{11}$$

- **Step 5**: Mann-Whitney U on `H_MM`, human vs each AI, per domain and composition; 12 tests, one BH-FDR family; lower AI entropy read as "greater regularity"
### 5.2 Known issues

- **Entropy depends on the mean composition**: a document's entropy is higher the more even its mix, so RQ0's rate differences leak into "regularity" (e.g. wp Claude 52% SIMPLE vs human 40%) — the same kind of confound that broke the original Track 1
- **Residual length dependence** after Miller-Madow: correlation of `H_MM` with `n_sents` differs between sources within a domain by up to 0.17 (measured before cleaning)
- **Mann-Whitney** compares distributions, not means or any single parameter (§7)
- The algo1 conforming-only scope and `denom ≤ 1` exclusion were inherited from the CLR track; their original reason no longer applies
- **Definition**: "regularity" needs a definition distinct from Track 1's `ρ`, which is itself a within-document correlation (§4.1)

## 6. Track 3 — Retired

- Was: per-category Brown-Forsythe (dispersion) and Mann-Whitney (location), two 48-test families
- Location half → RQ0, done properly; Mann-Whitney isn't a test of rates (§7)
- Dispersion half → Track 1, done properly: Brown-Forsythe compared mostly noise and was capped by the mean (§7)
- Removed: `run_rq1_significance.py`, `rq1_diagnostics_significance.csv`

## 7. Approaches Tried and Rejected

| Approach | What it did | Why rejected — evidence |
|---|---|---|
| CLR distance-to-centroid (original Track 1) | Zero-replacement `δ_i = 0.5/n_i`, centred log-ratio transform, Euclidean distance to the group centroid, Mann-Whitney | With equal true variance, real means and lengths: 62% (wp) and 63% (reuter) false positives. Zero-replacement puts a `−(3/4)·ln n_i` term into a replaced category's CLR value (fitted slope −0.785); and CLR spread behaves like a multivariate CV², growing as a mean shrinks (`Var(ln p)` 2.73× and 3.13× larger for GPT than human in reuter COMPOUND and COMPLEX-COMPOUND at identical `ρ`) — even perfect, noise-free data rejected 98–100% |
| Dirichlet-multinomial, one `ρ` per composition | One relative variance per composition, likelihood-ratio test | Model didn't fit: per-category `ρ` spread 1.3–6.1× within a group; observed ÷ predicted variance 0.69–1.49; the single number hid categories pulling in opposite directions (essay COMPOUND) |
| Beta-binomial per category, maximum likelihood | One smooth curve for the true rates; likelihood-ratio test | Test well calibrated (3.5–6.5%), but the curve shape biases `ρ` in whichever direction the true shape dictates (two-curve truth: 106% of true variance); on real wp human, 12–43% below Eq. 6 |
| Brown-Forsythe, variance ratio, CV (Track 3) | Raw spread of document rates | Raw variance is mostly sampling noise (59–92% in reuter) and capped by `μ(1−μ)`; variance ratio and CV disagreed on direction in 23 of 33 significant tests |
| Mann-Whitney as a test of rates (Track 3 location) | Ranks of document rates | Same true mean, different spread: 100% false positives (essay type OTHER, human vs GPT); pooled chi-square 44%; Beta-binomial LR 3% |

## 8. Multiple Comparisons

| Family | Tests | Question |
|---|---|---|
| Track 1 — Variance | 48 | between-document variance, per category |
| Track 2 — Regularity (current implementation) | 12 | within-document uniformity |

- Benjamini-Hochberg (`fdr_bh`) via the shared `apply_fdr`, called once per family — never per domain or per scheme
- Families corrected separately: they answer different questions
- BH over Bonferroni: controls the expected false-discovery proportion; Bonferroni (α/48) is too conservative for a descriptive multi-category design

## 9. Implementation Mapping

### 9.1 Script → output

| Script | Outputs |
|---|---|
| `src/analysis/features/build_doc_features.py` | `data/processed/features/doc_features.feather` (cleaning, §3) |
| `src/analysis/rq1/run_rq1_variance.py` | `rq1_variance.csv`, `rq1_variance_tests.csv`, `rq1_variance_tests_sensitivity.csv` |
| `src/analysis/rq1/plot_rq1_variance.py` | `rq1_variance_{essay,reuter,wp}.{png,pdf}` |
| `src/analysis/rq1/run_rq1_regularity.py` (Track 2, under review) | `data/processed/rq1/doc_entropy.feather`, `rq1_regularity_significance.csv` |
| `src/analysis/shared/filters.py`, `significance_utils.py` | — (shared with RQ0) |

### 9.2 Function → step (`run_rq1_variance.py`)

| Function | Does |
|---|---|
| `variance_components` | Eq. 6 — `r̄`, `V̂`, `ρ̂` from counts and lengths |
| `_resample_indices` | §4.3 — 2,000 resamples of row positions, by document or by author |
| `_estimate` | §4.2–4.3 — point estimates, SE and interval for all 72 cells; per-group generator |
| `_test` | Eq. 7 and FDR — 48 rows |
| `_verdict` | §4.4 — verdict labels |
| `_exclude_assessed` | §4.6 — removes the six documents, failing loudly unless each matches exactly one |
| `main` | main pass, then the sensitivity pass; writes all three CSVs and prints a summary |

- Track 2: `compositions.compute_proportions_algo1/2`, `entropy.plugin_entropy`, `entropy.miller_madow`, `shared.significance_utils.run_test_family`
## 10. Output Specifications

### 10.1 `results/rq1/rq1_variance.csv` — 72 rows

| Column | Type | Description |
|---|---|---|
| `domain`, `source`, `scheme`, `category` | str | the cell |
| `n_docs` | int | documents in the group |
| `n_clusters` | int | units resampled — authors in reuter (50), documents elsewhere |
| `mean_rate` | float | `r̄` (equals RQ0's `mean_rate`) |
| `V` | float | Eq. 6, absolute between-document variance |
| `rho` | float | Eq. 6, headline |
| `rho_se` | float | SD of the 2,000 resampled `ρ̂` |
| `rho_ci_low`, `rho_ci_high` | float | 2.5th and 97.5th percentiles |

### 10.2 `results/rq1/rq1_variance_tests.csv` — 48 rows

| Column | Type | Description |
|---|---|---|
| `domain`, `comparison`, `scheme`, `category` | str | `comparison` is `human_vs_gpt` / `human_vs_claude` |
| `rho_human`, `rho_ai` | float | the two `ρ̂` |
| `rho_diff` | float | human minus AI |
| `rho_ratio` | float | human ÷ AI, descriptive only; NaN if `rho_ai ≤ 0` |
| `se_diff` | float | Eq. 7's denominator |
| `z`, `p_value`, `p_value_fdr` | float | Eq. 7, two-sided, BH within the 48 |
| `significant` | bool | `p_value_fdr < 0.05` |
| `verdict` | str | human more variable / AI more variable / no difference |

### 10.3 `results/rq1/rq1_variance_tests_sensitivity.csv` — 48 rows

- `domain`, `comparison`, `scheme`, `category`; `rho_diff`, `p_value_fdr`, `verdict` for each pass (suffixes `_main`, `_excl`); `verdict_changed` (bool)

### 10.4 Figures — `results/rq1/rq1_variance_{essay,reuter,wp}.{png,pdf}`

- **Track 1's answer** — RQ0's layout: per domain, two panels (sentence type, sentence structure), grouped bars human / GPT / Claude, `ρ̂` as % of maximum, asymmetric 95% resampling intervals, zero line

### 10.5 Track 2 outputs (current implementation)

- `data/processed/rq1/doc_entropy.feather` — `doc_id`, `domain`, `source`, `author`, `composition`, `entropy_plugin`, `entropy_mm`
- `results/rq1/rq1_regularity_significance.csv` — 12 rows: `composition`, `domain`, `comparison`, `n_human`, `n_ai`, `mean_entropy_human`, `mean_entropy_ai`, `median_entropy_human`, `median_entropy_ai`, `entropy_ratio`, `U_stat`, `p_value`, `p_value_fdr`, `significant`

## 11. Open Items

- Track 2: definition, the known issues in §5.2, and a calibration check like Track 1's (§4.5) before results are reported
- Reuters author clustering is handled in Track 1 but not yet in Track 2, nor in RQ0's descriptive error bars
