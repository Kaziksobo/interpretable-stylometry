# RQ1 Methodology — Variance and Regularity in AI vs. Human Syntactic Structure

<!-- Working record only — bullets, not prose. Full LaTeX write-up later. -->
<!-- Equations numbered continuously (N) through the whole doc, each on its own line. -->
<!-- Citations: (Author, Year) inline, matches lit review / natbib. -->

## 1. Notation & Definitions

- `p_i` — proportion of category *i* within a composition (sums to 1 across the composition's parts)
- `D` — number of parts in a composition (4 for algo1's Track 1/2 scope, 3 for algo2, 8 for Track 3's full battery)
- `Z` — number of zero-valued parts in a document's composition vector
- `n_i` — document *i*'s own denominator for a given composition (sum of the 4 conforming counts for algo1, `n_sents` for algo2)
- `δ_i` — document-specific zero-replacement constant
- `g(x)` — geometric mean of a composition vector
- `CLR(x)` — centred log-ratio transform
- `H` — Shannon entropy (nats)
- `n_sents` — sentences per document
- `μ, σ, σ²` — mean, std, variance of a per-document rate across a group
- `CV` — coefficient of variation, `σ/μ`
- Category counts (raw `doc_features.feather` columns, algo1): `c_S`=`sent_simple`, `c_C`=`sent_complex`, `c_Cd`=`sent_compound`, `c_CC`=`sent_complex_compound`, `c_O1`=`sent_other`
- Category counts (algo2): `c_L`=`struct_loose`, `c_P`=`struct_periodic`, `c_O2`=`struct_other`

## 2. Research Question & Design Overview

- RQ1: "Does AI prose show reduced variance and greater regularity across interpretable syntactic features, such as loose/periodic sentences and clause complexity, when compared to human prose?"
- Two tracks, each testing one clause of RQ1 as its own statistical object:
  - **Track 1 — Variance**: between-document dispersion — do AI documents resemble *each other* more?
  - **Track 2 — Regularity**: within-document uniformity — does *one* AI document spread across types more/less evenly?
- **Track 3 — per-category diagnostics**: explanatory, not a third RQ1 claim — explains *which* categories drive Tracks 1/2
- Categories within one composition aren't independent (sum to 1) — this is why Track 1/2 use a genuine multivariate treatment rather than testing each part separately and FDR-correcting as if unrelated

## 3. Data & Preprocessing

- Source: `doc_features.feather` — Feng et al. (2012) Algorithm 1 (sentence type) + Algorithm 2 (structure), over Ghostbuster corpus constituency parses
- Two compositions:
  - algo1: {SIMPLE, COMPLEX, COMPOUND, COMPLEX-COMPOUND, OTHER}
  - algo2: {LOOSE, PERIODIC, OTHER}
- Filter: exclude `n_sents < 5` — degenerate compositions below this (e.g. `n_sents=1` is 100% one category, tells you nothing about tendency)
  - Cost: 80 documents (0.89%) — essay/human 6, reuter/claude 16, reuter/gpt 19, reuter/human 15, wp/claude 8, wp/gpt 3, wp/human 13
  - **Applies to all three tracks** — one shared `filter_min_sents` (`src/analysis/shared/filters.py`, shared with RQ0), called by all three driver scripts, so every track tests the same document pool
- algo1 conforming-count filter (Tracks 1 and 2 only; Track 3 is unaffected): exclude documents with `denom ≤ 1`, where `denom = c_S+c_C+c_Cd+c_CC`
  - `denom = 0`: proportions undefined — 4 documents left after the `n_sents` filter
  - `denom = 1`: one conforming sentence forces `Z=3`, and with `δ = 0.5/1` the replacement scaling factor is `1 − 3·0.5 = −0.5`, i.e. negative. Left in, that one document's `NaN` passed through `clr_transform` and silently voided an entire 2,008-document Mann-Whitney cell (reuter, human_vs_claude, sentence_type) — 1 document
  - algo2 is structurally immune: its denominator is `n_sents ≥ 5` and `Z ≤ 2`, so `Z/(2·n_sents) ≤ 0.2`
  - Total cost: 5 documents (0.06% of the 8,954 left after the `n_sents` filter). Track 2 inherits this exclusion by reusing `compute_proportions_algo1`, so both tracks test the identical documents
  - `multiplicative_replacement` also asserts a positive scaling factor, so any future violation fails loudly
- `OTHER` asymmetry — settled design decision, evidence-based, kept here for transparency:
  - algo1 `sent_other` zero-rate: 11–84.5% across domain/source (e.g. essay/claude 84.5%) — near-structural-absence in some cells, multiplicative replacement would be dominated by δ
  - algo2 `struct_other` zero-rate: 0.9–7.1% everywhere (essay: claude 1.3%/gpt 1.1%/human 3.5%; reuter: claude 2.0%/gpt 4.5%/human 7.1%; wp: claude 0.9%/gpt 2.5%/human 2.2%) — genuinely rare-but-real everywhere, no structural-absence pattern
  - → algo1 `OTHER` excluded from Track 1/2, tested only in Track 3; algo2 `OTHER` kept in the full 3-part composition for Track 1/2

## 4. Track 1 — Variance

### 4.1 Pipeline

- **Step 1**: start from the filtered document pool (§3)

- **Step 2**: compute proportions per composition — algo1 (Eq. 1a), algo2 (Eq. 1b); algo1 also excludes the 5 documents with `denom ≤ 1` (§3)

  $$p_i = \frac{c_i}{c_S+c_C+c_{Cd}+c_{CC}}, \quad i \in \{S,C,Cd,CC\} \tag{1a}$$

  $$p_i = \frac{c_i}{n\_sents}, \quad i \in \{L,P,O2\} \tag{1b}$$

- **Step 3**: compute each document's own zero-replacement constant — count-zero treatment, not a corpus-wide constant (Martín-Fernández et al., 2003)

  $$δ_i = \frac{0.5}{n_i} \tag{2}$$

  — smaller `n_i` (shorter document) gives a *larger* `δ_i`: a zero in a short document is weaker evidence of "never occurs" and should be treated that way. `n_i` is composition-specific (Eq. 1a/1b's own denominator), so `δ_i` differs between algo1 and algo2 automatically.

- **Step 4**: zero-replacement, per document, per composition

  $$r_i = \begin{cases} δ_i & p_i = 0 \\ p_i(1-Zδ_i) & p_i > 0 \end{cases} \tag{3}$$

- **Step 5**: CLR-transform each replaced vector

  $$CLR(x)_i = \ln\!\left(\frac{x_i}{g(x)}\right), \qquad g(x) = \left(\prod_{j=1}^{D} x_j\right)^{1/D} \tag{4}$$

- **Step 6**: group by `(domain, source)`, compute centroid

  $$c = \frac{1}{n}\sum_i CLR(x_i) \tag{5}$$

- **Step 7**: compute per-document distance to its **own** group's centroid — both forms, plain is the one reported

  $$d_i = \lVert CLR(x_i) - c \rVert \tag{6}$$

  $$d_i^2 = \lVert CLR(x_i) - c \rVert^2 \tag{7}$$

- **Step 8**: run Mann-Whitney U on `{d_i}` (Eq. 6, plain), human vs. each AI, per domain, per composition — H₀: equal location

- **Step 9**: collect the 12 p-values (2 compositions × 3 domains × 2 comparisons), apply BH-FDR as its own family

- **Step 10**: report `mean(d_human) / mean(d_AI)` (plain, Eq. 6) as the headline effect size; `mean(d²_human)/mean(d²_AI)` (squared, Eq. 7) reported alongside as the exact generalised-variance check (Eq. 10), not the primary number

### 4.2 Why Not Raw Euclidean Distance on Proportions

- Boundedness: for `X` on `[0,1]`, `x² ≤ x` pointwise, so `E[X²] ≤ E[X]`, giving

  $$\text{Var}(X) = E[X^2] - \mu^2 \le \mu - \mu^2 = \mu(1-\mu) \tag{8}$$

  — variance ceiling depends on where the mean sits, nothing to do with real consistency

- Spurious negative correlation: sum-to-1 constraint forces parts to move oppositely regardless of any real relationship (Pearson, 1897)
- Scale-relativity: a 0.01→0.02 move (rare category) is a doubling; 0.50→0.51 (common category) is a 2% nudge — raw Euclidean weights both identically
- Fix: CLR moves the simplex into unconstrained real space; `∑ᵢ CLR(x)ᵢ = 0`; Euclidean distance in CLR space = Aitchison distance (Aitchison, 1986)
- CLR over ALR: CLR keeps 1:1 correspondence with original categories (interpretable); ALR picks one category as a fixed denominator, loses it as its own dimension, asymmetric

- Closure check — confirms (3) preserves a valid composition regardless of which `δ_i` is used:

  $$\sum_i r_i = Zδ_i + (1-Zδ_i)\sum_{\text{nonzero } i} p_i = Zδ_i + (1-Zδ_i)\cdot 1 = 1 \tag{9}$$

- Trace identity — proves *why* distance-to-centroid formalises "variance" at all; this is the theoretical grounding for the whole approach, not the reported statistic (Eq. 6/step 10 is):

  $$\frac{1}{n}\sum_i \lVert CLR(x_i)-c \rVert^2 = \sum_k \text{Var}_k = \text{tr}(\Sigma) \tag{10}$$

### 4.3 Dataset-Specific Design Decisions

- algo1 scoped to the 4 conforming categories only (step 2), algo2 kept full 3-part — justified in §3
- Zero-replacement (step 4) needed for algo1 (rare-but-real COMPLEX-COMPOUND, up to 52.7% zero in essay/claude) and algo2 (all cells well under 10% zero — replacement barely perturbs anything there, included for consistency)
- Replacement guard: `multiplicative_replacement` asserts `1 − Z·δ_i > 0` for every document. The only algo1 case that can violate it (`denom = 1`) is excluded upstream (§3); algo2 cannot violate it
- Headline statistic is plain distance (Eq. 6), not squared (Eq. 7) — matches Anderson's (2006) PERMDISP convention, robust to outlier documents (real risk here — the Reuters motif-mining track already surfaced boilerplate contamination in this corpus)

### 4.4 Worked Example
- *(pending — fill once the pipeline runs on real data)*

## 5. Track 2 — Regularity

### 5.1 Pipeline

- **Step 1**: same filtered pool, same proportions as Track 1 step 2 (Eq. 1a/1b) — no zero-replacement needed here (entropy handles `p=0` natively, `0·ln(0) := 0`)

- **Step 2**: composition-specific `n` for the bias correction (same `n_i` as Eq. 2)

  $$n = c_S+c_C+c_{Cd}+c_{CC} \ \text{(algo1, i.e. } n\_sents - c_{O1}\text{)}, \qquad n = n\_sents \ \text{(algo2)} \tag{11}$$

- **Step 3**: plug-in Shannon entropy per document

  $$\hat{H} = -\sum_i p_i \ln(p_i) \tag{12}$$

- **Step 4**: Miller-Madow bias correction

  $$H_{MM} = \hat{H} + \frac{D-1}{2n} \tag{13}$$

- **Step 5**: run Mann-Whitney U on `{H_MM}`, human vs. each AI, per domain, per composition — H₀: equal location

- **Step 6**: collect the 12 p-values (2 compositions × 3 domains × 2 comparisons), apply BH-FDR as its own family

- **Step 7**: report mean `H_MM` per group alongside each test — lower AI entropy supports "greater regularity"

### 5.2 Why Not Plug-In Entropy

- `Ĥ` is negatively biased for small samples — rare categories under-observed in few draws, entropy mechanically undershoots; the bias term is exactly what Eq. 13 adds back
- Empirical motivation for correcting rather than ignoring — document length differs by source:
  - WP: human median 38 vs. claude/gpt 32 sentences
  - Essay: human mean 31.6 vs. claude 22.7 (shortest)
  - Reuter: comparable (17.7–21.4) — no length confound here
- → uncorrected entropy comparison in WP/essay would partly reflect length bias, not real regularity difference; Reuters unaffected either way

### 5.3 Dataset-Specific Design Decisions

- `n` in Eq. 13 is composition-specific (step 2), not raw `n_sents` — algo1's `n` is smaller by `c_O1`, so its bias correction term is larger than algo2's for the same document
- Same algo1 (conforming-only, 4-part) / algo2 (full, 3-part) scoping as Track 1, for consistency between the two tracks' numbers
- Track 2 has no replacement step, so a `denom = 1` document is mathematically fine for it (entropy `0`). It's excluded anyway, deliberately: `compute_proportions_algo1` is shared, so both tracks test the identical document pool (§3)

### 5.4 Worked Example
- *(pending — fill once the pipeline runs on real data)*

## 6. Track 3 — Per-Category Diagnostics

- Framing: explains what drives Tracks 1/2 — not a third independent RQ1 claim
- Unchanged from the original design, all 8 categories retained (including both `OTHER`s)
- **Denominator note**: rate here (Eq. 14) uses `n_sents` for *every* category, including algo1's — different from Track 1/2's algo1 denominator (sum of 4 conforming counts only, Eq. 1a). Don't compare a Track 3 algo1 rate directly against a Track 1 algo1 proportion — they're on different scales by construction.

### 6.1 Pipeline

- **Step 1**: filtered pool (§3), via the shared `filter_min_sents`

- **Step 2**: per-document rate, all 8 categories

  $$rate_i = \frac{c_i}{n\_sents} \tag{14}$$

- **Step 3**: per `(domain, category, comparison)` — Brown-Forsythe (`levene(..., center="median")`) on `{rate}_human` vs. `{rate}_AI` → `p_disp`

- **Step 4**: Mann-Whitney U on the same two groups → `p_loc`

- **Step 5**: effect sizes

  $$var\_ratio = \frac{\sigma^2_{human}}{\sigma^2_{AI}} \tag{15}$$

  $$CV = \frac{\sigma}{\mu}, \qquad CV\_ratio = \frac{CV_{human}}{CV_{AI}} \tag{16}$$

- **Step 6**: collect 48 `p_disp` + 48 `p_loc`; BH-FDR correction, **two separate families**

- **Step 7**: flag rows where `p_disp_fdr<0.05` but `p_loc_fdr≥0.05` as "pure regularisation"

- **Step 8**: flag rows where `var_ratio` (Eq. 15) and `CV_ratio` (Eq. 16) disagree in direction (one `>1`, one `<1`) as **"dispersion direction unstable"** — report the location result (step 4) as the reliable finding for that category, don't assert a dispersion winner

### 6.2 Why Add CV

- Raw variance ratio conflates "different spread" with "different mean → mechanically different room to vary" (same boundedness issue as Eq. 8, univariate case)
- `CV` (Eq. 16) is scale-free — divides out the mean-dependence
- Worked examples (real data):
  - **[REUTER] SIMPLE, human_vs_gpt**: human μ=0.276 σ=0.135 (n=992), gpt μ=0.399 σ=0.153 (n=999) → `var_ratio=0.77×` vs. `CV_ratio=1.27×` — direction flips, flagged "dispersion direction unstable" (step 8), location result is the reliable one for this row
  - **[WP] OTHER, human_vs_gpt** (`p_disp_fdr≈1.4e-68`, `p_loc_fdr≈2.5e-190`): human μ=0.141 σ=0.100 (n=987), gpt μ=0.026 σ=0.051 (n=997) → `var_ratio=3.83×` vs. `CV_ratio=0.37×` — also flags "dispersion direction unstable"; GPT's mean sits near the `μ→0` edge where CV is noise-sensitive. **Resolution**: report location instead — human mean 14.1%/median 12.8% vs. GPT mean 2.6%/median 0% is unambiguous and is the actual finding for this cell. This instability is confined to Track 3 — algo1's Track 1/2 composition excludes `OTHER` entirely, so it never reaches the RQ1 headline tests.

### 6.3 Dataset-Specific Design Decisions

- Rank-based tests (steps 3–4) need no positivity — no zero-replacement required even for algo1 `OTHER` here, contrast with Track 1
- WP conformity finding: human median `sent_other` rate 12.8% vs. GPT median **0%** — over half of GPT's WP documents contain no non-conforming sentence at all — directly connects to Heuser's (2025a) cultural-collapse framing, worth foregrounding in write-up
- Full-battery result, re-run with the filter and CV in place: 33/48 dispersion significant, 42/48 location significant after FDR, 3 pure regularisation — the same counts as the earlier unfiltered run, so the `n_sents` filter changed no significance call. 28/48 cells are flagged `dispersion_direction_unstable` (every domain/comparison has at least 2 stable rows), so for more than half the battery the location result is the reportable finding (step 8)

### 6.4 Worked Examples
- See §6.2 — REUTER SIMPLE and WP OTHER

## 7. Multiple Comparisons Strategy

- Four independent FDR families, corrected separately (different substantive questions):

| Family | Tests | Question |
|---|---|---|
| Track 1 — Variance | 12 | between-document dispersion |
| Track 2 — Regularity | 12 | within-document uniformity |
| Track 3 — dispersion | 48 | per-category spread |
| Track 3 — location | 48 | per-category median |

- **120 tests total**, Benjamini-Hochberg (`fdr_bh`) throughout
- BH over Bonferroni: at uncorrected α=0.05, expect ~6 false positives across 120 tests by chance alone; Bonferroni (α/120 ≈ 0.0004) too conservative for an exploratory multi-feature design — BH controls expected false-discovery proportion among calls made "significant," standard choice here

## 8. Implementation Mapping

### 8.1 Script → Output Mapping

| Track | Script | Output |
|---|---|---|
| Shared preprocessing | `src/analysis/rq1/compositions.py` | — (utility module, no direct output) |
| Shared preprocessing | `src/analysis/rq1/entropy.py` | — (utility module, no direct output) |
| Shared (RQ0 and RQ1) | `src/analysis/shared/significance_utils.py` | — (utility module, no direct output) |
| Shared (RQ0 and RQ1) | `src/analysis/shared/filters.py` | — (utility module: `filter_min_sents`) |
| Track 1 (Variance) | `src/analysis/rq1/run_rq1_variance.py` | `data/processed/rq1/doc_distances.feather`, `results/rq1/rq1_variance_significance.csv`, `results/rq1/centroids.csv` |
| Track 2 (Regularity) | `src/analysis/rq1/run_rq1_regularity.py` | `data/processed/rq1/doc_entropy.feather`, `results/rq1/rq1_regularity_significance.csv` |
| Track 3 (Diagnostics) | `src/analysis/rq1/run_rq1_significance.py` (+filter, +CV, +stability flag; uses `run_test_family` and `apply_fdr`) | `results/rq1/rq1_diagnostics_significance.csv` (renamed from `rq1_significance.csv`; the old `.txt` summary is no longer produced) |
| Synthesis | `src/analysis/rq1/build_rq1_answer.py` | `results/rq1/rq1_answer.txt` |

Motif-mining track (`src/analysis/motifs/`) is a separate, unrelated analysis — not part of RQ1, not included here.

### 8.2 Function → Track/Step Mapping

The point of this table: before writing a function, check which track(s) actually call it and at which step — several of these are shared, not Track-1-exclusive despite living in a file `compositions.py` mostly discussed under Track 1.

**`compositions.py`**

| Function                                                  | Used by                | Step(s)                               | Notes                                                                                                                                                                                          |
| --------------------------------------------------------- | ---------------------- | ------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `filter_min_sents`                                        | Tracks **1, 2, and 3** | §4.1 step 1, §5.1 step 1, §6.1 step 1 | Single source of truth for the `n_sents<5` filter — all three driver scripts call it. Moved to `src/analysis/shared/filters.py` so RQ0 uses the same filter; no longer in `compositions.py`                                                                          |
| `compute_proportions_algo1` / `compute_proportions_algo2` | Tracks 1, 2            | §4.1 step 2, §5.1 step 1              | **Not** Track 3 — Track 3 has its own denominator (Eq. 14, full `n_sents` including `OTHER`); don't reuse this function there, the two are deliberately different scales (§6 denominator note) |
| `compute_deltas`                                          | Track 1 only           | §4.1 step 3                           | Track 2 has no zero-replacement step at all                                                                                                                                                    |
| `multiplicative_replacement`                              | Track 1 only           | §4.1 step 4                           |                                                                                                                                                                                                |
| `clr_transform`                                           | Track 1 only           | §4.1 step 5                           |                                                                                                                                                                                                |
| `inverse_clr`                                             | Track 1 only           | §4.1 step 6 (centroid back-transform) | Feeds `centroids.csv` (§9.6) directly                                                                                                                                                          |
| `compute_centroid`                                        | Track 1 only           | §4.1 step 6                           |                                                                                                                                                                                                |
| `compute_distances`                                       | Track 1 only           | §4.1 step 7                           | Returns both Eq. 6 (plain) and Eq. 7 (squared)                                                                                                                                                 |

**`entropy.py`** — Track 2 only, nothing cross-track:

| Function | Used by | Step(s) |
|---|---|---|
| `plugin_entropy` | Track 2 | §5.1 step 3 |
| `miller_madow` | Track 2 | §5.1 step 4 |

**`src/analysis/shared/significance_utils.py`** — the most shared file, all three RQ1 tracks and RQ0:

| Function | Used by | Step(s) | Notes |
|---|---|---|---|
| `run_test_family` | Tracks 1, 2, 3 | §4.1 step 8, §5.1 step 5, §6.1 steps 3–4 | Track 3 calls it **twice** per cell (Levene, then Mann-Whitney); Tracks 1/2 call it once |
| `apply_fdr` | Tracks 1, 2, 3 | §4.1 step 9, §5.1 step 6, §6.1 step 6 | Track 3 calls it **twice** (dispersion family, location family — kept separate per §7). Tracks 1/2 call it once, in `main()` on the concatenated 12-row frame — not per composition, which would correct over 6 tests instead of the 12-test family §7 specifies |

**Driver-script-only logic** — orchestration, or simple enough not to warrant a shared function; lives directly in the relevant `run_rq1_*.py`, not in a utility file:

| Step | Track | Lives in |
|---|---|---|
| Step 10 — report effect size, save outputs | 1 | `run_rq1_variance.py` |
| Step 7 — report, save outputs | 2 | `run_rq1_regularity.py` |
| Steps 5, 7, 8 — `var_ratio`, `CV`, `pure_regularisation`, `dispersion_direction_unstable` | 3 | `run_rq1_significance.py` — small enough to stay inline; could move to `significance_utils.py` as generic effect-size helpers later, not required now |

## 9. Output Specifications

**The answer to RQ1 lives in exactly two files**: `rq1_variance_significance.csv` (Track 1) and `rq1_regularity_significance.csv` (Track 2). Everything else — both `.feather` intermediates, `centroids.csv`, `rq1_diagnostics_significance.csv` — produces, explains, or supports those two; none of them are the answer on their own.

**How to read any result row** — needs `significant` AND the ratio's direction together, never one alone:

| `significant` | ratio | verdict |
|---|---|---|
| True | > 1 | ✅ effect real, supports RQ1 (reduced variance / greater regularity) |
| True | < 1 | ❌ effect real, opposite direction — AI *more* variable / *less* regular |
| False | either | no detectable effect — ratio value is noise, ignore it |

### 9.1 `data/processed/rq1/doc_distances.feather` — Track 1 per-document intermediate

| Column | Type | Description |
|---|---|---|
| `doc_id` | str | document identifier |
| `domain` | str | essay / reuter / wp |
| `source` | str | human / gpt / claude |
| `author` | str/null | Reuters author, null elsewhere |
| `composition` | str | `sentence_type` (algo1) or `sentence_structure` (algo2) |
| `distance` | float | Eq. 6, plain distance to own group's centroid — headline |
| `distance_sq` | float | Eq. 7, squared distance — supplementary |

### 9.2 `data/processed/rq1/doc_entropy.feather` — Track 2 per-document intermediate

| Column | Type | Description |
|---|---|---|
| `doc_id` | str | |
| `domain` | str | |
| `source` | str | |
| `author` | str/null | |
| `composition` | str | `sentence_type` / `sentence_structure` |
| `entropy_plugin` | float | Eq. 12, uncorrected |
| `entropy_mm` | float | Eq. 13, Miller-Madow corrected — used downstream |

### 9.3 `results/rq1/rq1_variance_significance.csv` — 12 rows (answer file)

| Column | Type | Description |
|---|---|---|
| `composition` | str | `sentence_type` / `sentence_structure` |
| `domain` | str | essay / reuter / wp |
| `comparison` | str | human_vs_gpt / human_vs_claude |
| `n_human`, `n_ai` | int | documents per group, post-filtering |
| `mean_dist_human`, `mean_dist_ai` | float | mean of Eq. 6 (plain) |
| `median_dist_human`, `median_dist_ai` | float | |
| `dist_ratio` | float | `mean_dist_human/mean_dist_ai` — **headline effect size** |
| `mean_distsq_human`, `mean_distsq_ai`, `distsq_ratio` | float | Eq. 7 versions, supplementary |
| `U_stat` | float | Mann-Whitney U statistic |
| `p_value` | float | raw |
| `p_value_fdr` | float | BH-corrected within this 12-test family |
| `significant` | bool | `p_value_fdr < 0.05` |

### 9.4 `results/rq1/rq1_regularity_significance.csv` — 12 rows (answer file)

| Column | Type | Description |
|---|---|---|
| `composition`, `domain`, `comparison` | str | as above |
| `n_human`, `n_ai` | int | |
| `mean_entropy_human`, `mean_entropy_ai` | float | mean of `H_MM` (Eq. 13) |
| `median_entropy_human`, `median_entropy_ai` | float | |
| `entropy_ratio` | float | `mean_entropy_human/mean_entropy_ai` — **headline effect size**; `>1` supports "AI less entropic, more regular" |
| `U_stat`, `p_value`, `p_value_fdr`, `significant` | — | as §9.3 |

### 9.5 `results/rq1/rq1_diagnostics_significance.csv` — 48 rows (supporting, not answer)

| Column | Type | Description |
|---|---|---|
| `feature` | str | e.g. `"Sentence type: OTHER"` — matches existing report string format |
| `domain`, `comparison` | str | |
| `n_human`, `n_ai` | int | |
| `mean_human`, `std_human`, `mean_ai`, `std_ai` | float | rate (Eq. 14) stats |
| `var_ratio` | float | Eq. 15 |
| `cv_human`, `cv_ai`, `cv_ratio` | float | Eq. 16 |
| `dispersion_direction_unstable` | bool | §6.1 step 8 — `var_ratio`/`cv_ratio` disagree in sign |
| `disp_statistic`, `p_disp`, `p_disp_fdr`, `disp_significant` | — | Brown-Forsythe W statistic, raw p-value, BH-corrected within the 48-test dispersion family |
| `loc_statistic`, `p_loc`, `p_loc_fdr`, `loc_significant` | — | Mann-Whitney U statistic, raw p-value, BH-corrected within the 48-test location family |
| `pure_regularisation` | bool | `disp_significant & !loc_significant` |

### 9.6 `results/rq1/centroids.csv` — companion to Track 1

| Column | Type | Description |
|---|---|---|
| `domain`, `source` | str | |
| `composition` | str | `sentence_type` / `sentence_structure` |
| `category` | str | e.g. `COMPLEX-COMPOUND`, `LOOSE` |
| `geometric_mean_proportion` | float | that group's centroid (Eq. 5), back-transformed from CLR to the simplex — the "typical document" composition for that group |

### 9.7 `results/rq1/rq1_answer.txt` — synthesis, not a data table

- Generated by `build_rq1_answer.py`, reading §9.3/9.4/9.5 — not raw pipeline output; written as UTF-8 plain text
- One block per `(domain, comparison)` — 6 total:
  - **Variance**: both compositions' verdict from §9.3 (significant? + `dist_ratio` direction), read via the §9 table
  - **Regularity**: both compositions' verdict from §9.4, same table
  - **Supporting evidence**: from §9.5, filtered to that `(domain, comparison)`; top 1 row by `p_loc_fdr` (location) and, **excluding** `dispersion_direction_unstable` rows, top 1 row by `p_disp_fdr` (dispersion) — reported as the strongest per-category evidence for that cell. 28 of the 48 rows are flagged unstable, but every `(domain, comparison)` cell still has at least 2 stable rows, but a pick must itself be significant (`loc_significant` / `disp_significant`), and that does bind: reuter / human_vs_gpt has no stable, significant dispersion row, so that line reads "none significant" rather than naming a non-significant row. FDR values tie in this data, so ties are broken on the raw p-value (`p_loc` / `p_disp`)
- Header block (summary, then a units legend):
  - Track 1 and Track 2: number significant of 12 (§7), split by the §9 reading rule into supports / opposite direction / not significant
  - Track 3: number significant on dispersion and on location (of 48, the two separate families), and how many rows are flagged `dispersion_direction_unstable`
  - Dispersion direction at three levels (answers "of the significant Brown-Forsythe cases, what percentage was the human more variant"): share of significant dispersion tests with `var_ratio > 1`, share with `cv_ratio > 1`, and share among stable rows only, each with its n. Currently 25/33 (76%) by `var_ratio`, 12/33 (36%) by `cv_ratio`, 7/10 (70%) among stable rows — the gap between the first two is the mean-variance confound in aggregate, which is why all three are reported
  - Units legend: ratios are human ÷ AI; distances in Aitchison (log-ratio) units; entropy in nats (Miller-Madow corrected); Track 3 rates as percentages of a document's sentences
- Per-genre and overall levels: the 6 per-cell blocks are the per-genre level, the header is the overall level
- Every verdict line prints its ratio and both group values next to it. No "negligible effect" label — that would need a magnitude threshold the spec doesn't define, so the ratio is left for the reader to judge (a significant ratio of 1.006 is normal at ~1,000 documents per group)
- Compositions are **not** collapsed into one number per cell — algo1 and algo2 can legitimately disagree (e.g. variance reduced in sentence-type but not sentence-structure), and that's a reportable finding, not something to average away

## 10. Resolved Design Decisions & Remaining Limitations

**Resolved:**
- Headline dispersion statistic is **plain** distance-to-centroid (Eq. 6), not squared — matches PERMDISP convention, robust to outlier documents; squared distance (Eq. 7) kept only as the theoretical proof (Eq. 10) that this formalises "variance"
- Zero-replacement uses a **per-document** constant `δ_i = 0.5/n_i` (Eq. 2), not a single corpus-wide value — corrects an earlier flawed proposal (a `min()` against a global minimum, which turned out to be a no-op for nearly the whole corpus)
- `var_ratio`/`CV_ratio` sign disagreements are handled by an explicit **"dispersion direction unstable"** flag (§6.1 step 8), not by picking one number as authoritative — applied to WP `OTHER`, resolves by falling back to the (unambiguous) location result
- Output files fully specified (§9) — 2 answer files, 2 feather intermediates, 2 supporting CSVs, 1 synthesis text file
- algo1 exclusion rule is `denom ≤ 1`, not `denom = 0` — a single conforming sentence drives the replacement scaling factor negative (§3); only found by running the full pipeline on real data
- FDR is applied once per family, matching §7: in `main()` on the concatenated 12-row frame for Tracks 1 and 2 (an earlier draft corrected per composition, over 6 tests), and inside `_process_features` for Track 3's two 48-test families

**Settled, not open** (kept here for transparency, not indecision):
- algo1/algo2 `OTHER` asymmetry (§3) — evidence-based, different zero-rate behaviour in each composition

**Remaining limitation:**
- `δ_i = 0.5/n_i` is a standard count-zero heuristic, not a full posterior-based estimate (e.g. Bayesian-multiplicative replacement via a Dirichlet posterior) — reasonable given project scope, worth naming explicitly as a threat-to-validity rather than leaving implicit; a full Bayesian treatment is a possible future robustness check, not required for this pass
