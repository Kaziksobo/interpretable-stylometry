# RQ0 Methodology — Do Humans, GPT and Claude Use Each Sentence Type and Structure at Different Rates?

<!-- Working record only — bullets, not prose. Full LaTeX write-up later. -->
<!-- Equations numbered continuously (N) through the whole doc, each on its own line. -->
<!-- Citations: (Author, Year) inline, matches lit review / natbib. -->

## 1. Notation

- `k_i` — count of one category in document *i*
- `n_i` — sentences in document *i* (`n_sents`)
- `r_i` — document *i*'s rate for the category
- `N` — documents in a group; group = one (domain, source) pair, e.g. reuter/claude
- Scheme — `sentence_type` (SIMPLE / COMPLEX / COMPOUND / COMPLEX-COMPOUND / OTHER) or `sentence_structure` (LOOSE / PERIODIC / OTHER), Feng et al. (2012)

## 2. Research Question & Design

- RQ0: do humans, GPT and Claude use each sentence type and sentence structure at different rates?
- Descriptive, and a prerequisite for RQ1: before asking whether AI prose *varies* less (RQ1), establish how often each source uses each category
- Per domain (essay, reuter, wp), per category (8), per source (human, GPT, Claude)
- Output: one bar chart per domain — mean rate per category and source, with 95% error bars

## 3. Data

- Input: `data/processed/rq1/doc_features.feather` — per-document Feng category counts and `n_sents`
- Filter: `n_sents ≥ 5` via the shared `filter_min_sents` (`src/analysis/shared/filters.py`) — 8,954 documents, 987–1,002 per group; same starting pool as RQ1
- Denominator is `n_sents` for every category, both `OTHER`s included
- Each scheme's counts sum to `n_sents` in every document (asserted in `build_doc_features.py`), so each document's rates within a scheme sum to exactly 1

## 4. Method

- **Step 1** — each document's rate; this is the length adjustment: every document becomes a share of its own sentences

  $$r_i = \frac{k_i}{n_i} \tag{1}$$

- **Step 2** — the group's mean rate

  $$\bar r = \frac{1}{N}\sum_{i=1}^{N} r_i \tag{2}$$

- **Step 3** — standard error and 95% interval, with `s` the standard deviation of the `r_i` (`ddof=1`)

  $$\text{SE} = \frac{s}{\sqrt{N}}, \qquad \bar r \pm 1.96\,\text{SE} \tag{3}$$

- The document is the unit throughout: one document = one writing task, matching the corpus design (~1,000 comparable tasks per group)
- Because each document's rates sum to 1 within a scheme (§3), the mean rates do too — each chart panel sums to exactly 100%

## 5. Why This Is Enough

### 5.1 Noise cancels in a mean

- Write each observed rate as the document's true rate plus sampling noise:

  $$r_i = p_i + \varepsilon_i, \qquad E[\varepsilon_i] = 0 \tag{4}$$

- A short document's rate is noisier, not biased — so averaging ~1,000 documents, the noise washes out and Eq. 2 estimates the mean true rate whatever the lengths
- The noise only widens the error bar, and Eq. 3 already includes it: `s` is measured from the observed rates

### 5.2 Noise adds in a variance — why RQ1 is harder

- Variance squares deviations, so the noise term can't cancel:

  $$\text{Var}(r) = \underbrace{\text{Var}(p)}_{\text{real spread}} + \underbrace{E\!\left[\varepsilon^2\right]}_{\text{noise, always } > 0} \tag{5}$$

- Observed spread always overstates real spread, more so for shorter documents — which is why RQ1 needs a model to separate the two terms, and RQ0 doesn't

### 5.3 Checked against the alternatives

- **Full Beta-binomial model** (weights documents by `n_i/(1+(n_i−1)ρ)`): Eq. 2 is within 0.95 pp of it at worst, 0.22 pp in a typical cell; error bars (Eq. 3) are ±0.21 to ±1.00 pp
  - 2 of 24 bar groups swap order between the two — only possible between bars closer than the methods ever differ (< 1 pp), i.e. within the error bars either way
  - The model only tightens error bars slightly; it doesn't change the answer — reserved for RQ1
- **Pooling** (sum `k_i` over sum `n_i`) weights by sentence, so a few very long documents dominate — unevenly by source:

| Domain | Human: median / max sentences / longest 5% of docs' share of sentences | GPT | Claude |
|---|---|---|---|
| essay | 22 / 365 / **20%** | 28 / 80 / 9% | 22 / 45 / 8% |
| reuter | 20 / 53 / 9% | 22 / 46 / 8% | 18 / 31 / 7% |
| wp | 38 / 235 / **15%** | 32 / 143 / 9% | 32 / 100 / 9% |

  - Pooled human bars would describe how humans write very long essays; AI bars would describe typical documents
  - Pooled vs Eq. 2: max gap 2.51 pp, median 0.42 pp; human/GPT/Claude order flips in 3 of 24 bar groups (essay struct OTHER, reuter struct OTHER, wp SIMPLE)

## 6. Pipeline

### 6.1 `src/analysis/rq0/run_rq0.py`

- **Step 1**: load `doc_features.feather`, `filter_min_sents` → 8,954 documents
- **Step 2**: on a copy, convert each category count to a rate (Eq. 1) — overwrite the count columns so nothing downstream can read counts by mistake
- **Step 3**: per (domain, source, category): `n_docs`, mean (Eq. 2), SE and interval (Eq. 3) — 72 rows
- **Step 4**: assert 72 rows; each (domain, source, scheme)'s mean rates sum to 1; every `ci_low ≥ 0`
- **Step 5**: write `results/rq0/rq0_rates.csv`

### 6.2 `src/analysis/rq0/plot_rq0.py`

- **Step 1**: read `rq0_rates.csv` — no recomputation, so charts can be restyled freely
- **Step 2**: one figure per domain, two panels (`sentence_type`, 5 categories; `sentence_structure`, 3; width ratios 5:3)
- **Step 3**: grouped bars human / GPT / Claude per category, in a fixed category order; one colour per source, the same in every figure; y-axis in % of sentences
- **Step 4**: error bars = 1.96 × SE
- **Step 5**: save PNG (viewing) and PDF (vector, for the LaTeX write-up)

## 7. Output Specifications

### 7.1 `results/rq0/rq0_rates.csv` — 72 rows

| Column | Type | Description |
|---|---|---|
| `domain`, `source` | str | the group |
| `scheme` | str | `sentence_type` / `sentence_structure` |
| `category` | str | e.g. `SIMPLE`, `LOOSE`, `OTHER` |
| `n_docs` | int | documents in the group, post-filter |
| `mean_rate` | float | Eq. 2, share of sentences (0–1) |
| `se` | float | Eq. 3 |
| `ci_low`, `ci_high` | float | `mean_rate ∓ 1.96·se` |

### 7.2 Figures — `results/rq0/rq0_rates_{essay,reuter,wp}.{png,pdf}`

- The answer to RQ0 — one figure per domain, as §6.2

## 8. Limitations & Notes

- **Reuters author clustering**: human reuter is 50 authors × ~20 documents; same-author documents are correlated, so reuter error bars are somewhat too narrow. Affects RQ1 equally
- **No formal test**: RQ0 is descriptive; the error bars show which gaps are clear. If a test is wanted later, Welch's t-test on the document rates matches this method — it compares means directly
- **For the RQ1 rebuild**: Track 3's location test (Mann-Whitney) is not a test of rates. Simulated with the same true rate in both groups, each keeping its real variance and lengths, it flagged a difference in 100% of runs for essay OTHER, human vs GPT (Beta-binomial LR: 3%) — GPT's documents are more spread out, so more sit at 0% at the same mean, and Mann-Whitney detects that shape difference. Its `pure_regularisation` flag and the "Strongest location difference" lines in `rq1_answer.txt` rest on it