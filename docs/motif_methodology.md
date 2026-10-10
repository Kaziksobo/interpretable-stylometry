# Motif Mining Methodology — Which Phrase-Structure Patterns Do Humans, GPT and Claude Use at Different Rates?

<!-- Working record only — bullets, not prose. Full LaTeX write-up later. -->
<!-- Equations numbered continuously (N) through the whole doc, each on its own line. -->
<!-- Citations: (Author, Year) inline, matches lit review / natbib. -->

## 1. Notation

- **Pattern** `p` — an induced subtree of a constituency parse: a node and its descendants down to depth `d`, with the nodes at the cut shown as bare labels, e.g. `(VP (VBD) (SBAR))`. The root is level 1, so depth 2 is a node and its children
- **Slot** — a leaf label in a pattern. A slot with a phrase label (`NP`, `SBAR`) stands for any structure below it. The slots partition the tokens under the pattern's root
- Source `s` — human, GPT or Claude. Domain — essay, reuter, wp. Every comparison is within one domain
- `c_{s,p}` — occurrences of `p` in source `s`'s sentences; `T_s` — tokens in source `s`'s text (parse-tree leaves, so punctuation counts); `docs_{s,p}` — documents of source `s` containing `p`
- `parent(p)` — `p` cut off one level higher: `(VP (VBD) (SBAR (S)))` has parent `(VP (VBD) (SBAR))`. Every pattern deeper than 2 has exactly one parent, so patterns form a forest

## 2. Research Question & Design

- Descriptive and bottom-up: which phrase-structure patterns do GPT and Claude use more or less than humans? RQ0 asks the same about two hand-defined schemes (Feng et al., 2012); here the units are discovered from the parses rather than chosen in advance
- Each AI source is compared with human only. GPT-vs-Claude contrasts are not made (they were in the first, text-report version of this track and were dropped); the explorer instead has a "both agree" view that keeps patterns where the two AIs differ from human in the same direction
- Per domain, per pattern; the unit of exposure is the token (§5)
- Output: a self-contained interactive explorer (`results/motifs/motif_explorer.html`), the statistics table behind it, and the per-document count table that the repertoire-diversity and concentration analyses will use

## 3. Data, Cleaning & Filtering

- Input: `data/processed/parses/constituency_parses.feather` — one row per sentence, parsed with spaCy + benepar (`src/parsing/constituency_parse.py`)
- Same document pool as RQ0 and RQ1, by construction (shared `src/analysis/shared/filters.py`):
  - `drop_junk_sentences` — sentences with no letters (2,575), Reuters sentences with no author (232) and bare heading lines (452: a sentence that is only a label ending in a colon, such as `Introduction:` in GPT essays or `SAM:` in WritingPrompts); the rules overlap, so 3,043 of 253,049 sentences go → 250,006
  - `filter_min_sents` (≥ 5 sentences) on `doc_features.feather` — 42 documents (8,991 → 8,949; `doc_features` has 8,991 of the nominal 9,000) and 120 sentences → **249,886 sentences, 8,949 documents, 5,191,038 tokens**
  - `restrict_to_docs` keeps only the sentences of those documents and raises if a pool document has no sentences, so the pools cannot drift apart
- Pool by cell (documents / sentences / tokens):

| Domain | Human | GPT | Claude |
|---|---|---|---|
| essay | 987 / 31,256 / 737,953 | 998 / 28,128 / 649,409 | 1,000 / 22,566 / 517,274 |
| reuter | 992 / 20,334 / 574,876 | 996 / 21,512 / 575,667 | 1,000 / 17,836 / 447,755 |
| wp | 987 / 43,929 / 646,280 | 997 / 31,081 / 580,993 | 992 / 33,244 / 460,831 |

### 3.1 Whitespace tokens leave empty nodes in the parses

- The parser tags newline tokens like any other token. When the bracketed parse string is read back the whitespace vanishes, leaving a preterminal with no word, e.g. `(PRP \n)`. They sit at the start of sentences that follow a paragraph break
- 14.2% of pool sentences contain one — very unevenly: 57.1% of human Reuters sentences, against 12.7–12.9% for GPT and Claude Reuters, and 4–17% elsewhere. Left in, they put a human-vs-AI difference in the patterns that is really about paragraph formatting (and slots with no word in them)
- **RQ0 and RQ1 are unaffected**: removing these nodes changes no Feng label for any of the 250,338 sentences of the pool at the time (before the heading rule) (the algorithms only test for S, SBAR and VP). Caveat: nodes were removed from existing trees, not re-parsed without the whitespace tokens
- Fix: `mining.remove_empty_nodes` drops the empty nodes and any ancestor left with no words. A node that lost a child and is left with a single child of its own label is contracted into it, so `(NP (NP (PRP \n)) (NP (PRP He)))` becomes `(NP (PRP He))`
  - Only nodes that lost a child are contracted: benepar itself outputs `(NP (NP (PRP it)))` (1.7% of ghost-free sentences), which must stay
  - Tokens are unchanged (`leaves()` is identical before and after on 30,000 sampled sentences)
- Effect (measured before the heading rule was added; final totals are in §7): patterns 937,396 → 919,510; occurrences 8,017,419 → 7,989,459; the top-50 lists by z keep 42–50 of 50 patterns. The change is concentrated in a few patterns, e.g. Reuters `(S (``) (``) (S) (,) ('') (NP) (VP) (.))` fell from 1.87 to 0.02 per 1,000 tokens for human, while the clean `(S (``) (S) (,) ('') (NP) (VP) (.))` rose from 0.97 to 2.84 and its GPT z from −8.7 to −25.8: the artefact had split one construction across two variants and hidden a strong AI under-use

## 4. Method

### 4.1 Patterns

- Every node of every cleaned parse is the root of up to three patterns, at depths 2, 3 and 4 (`MAX_DEPTH`, `MIN_DEPTH` in `build_motif_counts.py`)
- A pattern needs at least two countable leaves. A leaf label is countable if it matches `[A-Z$][A-Z0-9$-]*`, so punctuation tags (`,` `.` `:`) and bracket tags (`-LRB-`) do not count towards the minimum
- A pattern that is identical at two depths of one node (the deeper nodes were already leaves) is counted once for that node. A pattern is counted once per node it occurs at, so it counts twice if it occurs at two nodes of a sentence

### 4.2 Counting

- Occurrences are summed per document first (`motif_counts.feather`), then per source and domain. The document table is kept so that document frequencies and, later, resampling by document or author are possible
- `mining.count_patterns` builds each node's depth-d string bottom-up from its children's depth-(d−1) strings instead of constructing trees; it is equivalent to the original extractor (§8)

### 4.3 Rates

- Rate per 1,000 tokens

  $$r_{s,p} = 1000\,\frac{c_{s,p}}{T_s} \tag{1}$$

- Tokens, not sentences (§5). Counts are occurrences, so a rate is not the share of sentences containing the pattern
- A pattern enters the statistics table if it occurs in ≥ 10 documents of its domain (summed over the three sources): 14,858 / 14,407 / 13,807 patterns for essay / reuter / wp. The explorer lists those in ≥ 20 documents: 8,042 / 7,796 / 7,503

### 4.4 Comparing an AI source with human

- Log rate ratio with additive smoothing `a = 0.5` on each count (so a pattern a source never uses has a finite value):

  $$\delta_{p} = \ln\frac{c_{\text{AI},p}+a}{T_{\text{AI}}} - \ln\frac{c_{\text{human},p}+a}{T_{\text{human}}} \tag{2}$$

- Reported as `log2_ratio = δ / ln 2` (+1 is twice as often as human, −1 half as often)
- z-score, in the style of Monroe et al.'s "Fightin' Words"

  $$z_p = \frac{\delta_p}{\sqrt{\dfrac{1}{c_{\text{AI},p}+a} + \dfrac{1}{c_{\text{human},p}+a}}} \tag{3}$$

- z ranks patterns by weight of evidence, where the raw ratio would put the rarest patterns first. It is **a ranking statistic, not a test**: it treats occurrences as independent, so it overstates the evidence when a few documents account for most of a pattern's occurrences. The document counts shown next to it are the check
- "Both agree": GPT and Claude must have the same sign of z; the score is `sign · min(|z_GPT|, |z_Claude|)`, and likewise for log2 ratios

### 4.5 Parents and echo variants

- Patterns overlap: one occurrence yields a pattern at each depth, so a phenomenon appears as a family of rows. The family is a tree by `parent(p)` (18,919 of the 23,341 explorer rows have their parent in the table)
- A child is an **echo** of its parent for a comparison if the log2 ratios have the same sign and differ by less than θ (default 0.3, adjustable in the page). An echo says nothing its parent does not; for "both agree" it must be an echo for both AIs
- Echoes are folded under their *leader*: the first non-echo ancestor reached by following parents. A row is folded only if its leader passes the current filters, so a search can never hide its own matches
- Variants that are *not* echoes stay visible: they show where the parent's difference comes from. Worked example, Reuters, GPT:

| Pattern | Human | GPT | log2 ratio |
|---|---|---|---|
| `(VP (VBD) (SBAR))` | 6.17 | 1.57 | −1.98 |
| `(VP (VBD) (SBAR (S)))` — no "that" | 4.92 | 0.22 | −4.45 |
| `(VP (VBD) (SBAR (IN) (S)))` — with "that" | 0.98 | 1.17 | +0.25 |

  - The parent's gap comes almost entirely from one child: GPT almost never writes "said ∅ …", but uses "said that …" as often as humans do
- At θ = 0.3, 27% (GPT) and 32% (Claude) of rows with a parent in the table are echoes, but only 3–9 rows of each top-50 list, so folding tidies the lists rather than reshaping them

### 4.6 Examples and slots

- For every pattern in the statistics table and every source that uses it, up to 20 example occurrences are sampled (`build_motif_examples.py`): 124,007 (domain, source, pattern) groups, 1,237,228 examples; 23% of groups have the full 20 (most are rare)
- Sampling is deterministic and independent of processing order: each (sentence, pattern) pair gets a priority from a seeded hash (BLAKE2b, seed 0) and a group keeps its 20 lowest. Rank 0 is the lowest, so the first *n* examples are a uniform random sample of size *n*
- A sentence offers at most one occurrence of a pattern (the leftmost, innermost), so examples are distinct sentences
- Each example stores the token span of the pattern's root and the boundaries between its leaf slots. A constituent covers a contiguous run of tokens and its slots partition it, so the page can mark each slot without matching any text. Sentences are stored once, as space-separated parse tokens
- Sentence tokens are the parser's, so contractions and punctuation are separate tokens; the page re-joins them for display only

## 5. Why Tokens Rather Than Sentences

- A source with longer sentences scores higher on nearly every construction when exposure is sentences, so the rankings partly measure sentence length. Mean words per sentence:

| Domain | Human | GPT | Claude |
|---|---|---|---|
| essay | 20.4 | 19.8 | 20.0 |
| reuter | 24.3 | 23.4 | 21.7 |
| wp | 12.1 | 16.0 | 11.7 |

- Example: WritingPrompts `(PP (IN) (NP))`, GPT — 1.75× human per sentence (z = +86.7), 1.38× per token (z = +49.6). GPT's sentences there are about 32% longer, which accounts for most of the inflation
- Tokens remove the first-order effect. They do not make constructions scale exactly linearly with length, so rates are not perfectly length-free

## 6. Pipeline

Run from the project root; each step reads the previous step's output.

### 6.1 `src/analysis/motifs/build_motif_counts.py`

- **Step 1**: load parses, apply the shared cleaning and pool restriction (§3)
- **Step 2**: group sentences by document (sorted by key, so ids are reproducible); check the document table against `doc_features` (same documents, same sentence counts)
- **Step 3**: per sentence, parse, `remove_empty_nodes` (§3.1), `count_patterns`; count tokens (`leaves()`); parallel over documents
- **Step 4**: raise if any parse fails (a dropped sentence would bias its document's rates); write the three tables to `data/processed/motifs/` (git-ignored; rebuilt in under a minute)

### 6.2 `src/analysis/motifs/compute_motif_stats.py`

- Sum counts to (domain, source, pattern), apply the ≥ 10 document floor, compute Eq. 1–3 for both AIs, add depth and leaf count; write `results/motifs/motif_stats.csv` and `motif_pool.csv`; print each domain's top and bottom five patterns as a check

### 6.3 `src/analysis/motifs/build_motif_examples.py`

- Re-derives the pool's sentences in `(doc_idx, sent_idx)` order, cleans each tree as in 6.1, samples examples (§4.6) and writes `motif_sentences.feather` and `motif_examples.feather`. It asserts that its token totals equal `motif_docs.n_tokens` and that every group the statistics table says occurs has examples

### 6.4 `src/analysis/motifs/build_explorer.py`

- Fills `explorer_template.html` with the data and writes `results/motifs/motif_explorer.html` (about 11 MB). Edit the template, never the output
- Embedded: every pattern in ≥ 20 documents of its domain (as one array per column, plus each row's parent index), and examples for the patterns that can come out near the top of a view — in each domain, the `--n-cut` (250) highest and lowest by z and by log2 ratio for GPT, for Claude, and for both agreeing — with `--per-source` (5) examples per source; about 1,570 patterns per domain and 49,452 sentences. Embedding examples for every pattern would make the file tens of megabytes

```
python src/analysis/motifs/build_motif_counts.py
python src/analysis/motifs/compute_motif_stats.py
python src/analysis/motifs/build_motif_examples.py
python src/analysis/motifs/build_explorer.py        # --n-cut, --per-source, --min-docs
```

### 6.5 The explorer

- **Controls**: domain tabs; compare with GPT, Claude or both agreeing; AI uses the pattern more, less or either; sort by z, log2 ratio, human rate or AI rate; minimum documents; depth; search (plain words match labels, text containing a bracket matches that fragment); fold echo variants and the threshold θ; only patterns with examples
- **Rows**: pattern, rates per 1,000 tokens, document counts, and for each AI a diverging bar of the log2 ratio (clipped at ±3) with its value and z
- **Open row**: the pattern's tree (dashed = structure below is left out, red = clause), per-source counts, its parent, its variants one level deeper, and examples with each slot marked and labelled
- All filtering, sorting and folding runs in the browser over the embedded data; nothing is fetched

### 6.6 Earlier text report

- `run_feature_analysis.py` and `analyze_corpus.py` produced `results/motifs/stylometric_report.txt` (log-ratio ranking by raw ratio, GPT-vs-Claude sections, first-met examples). The runner now uses the shared document pool, but the explorer supersedes the report

## 7. Output Specifications

### 7.1 `data/processed/motifs/` (git-ignored)

| File | Rows | Columns |
|---|---|---|
| `motif_docs.feather` | 8,949 | `doc_idx`, `domain`, `source`, `doc_id`, `author`, `n_sents`, `n_tokens` |
| `motif_patterns.feather` | 919,325 | `pattern_id`, `pattern` |
| `motif_counts.feather` | 4,807,773 | `doc_idx`, `pattern_id`, `count` (one row per document and pattern used) |
| `motif_sentences.feather` | 249,886 | `sent_id`, `doc_idx`, `tokens` |
| `motif_examples.feather` | 1,237,228 | `domain`, `source`, `pattern_id`, `rank`, `sent_id`, `start`, `end`, `bounds` |

- `bounds` — comma-separated token indices where one slot ends and the next begins; slots are `[start, b0), [b0, b1), …, [b_last, end)`, in the order the leaves appear in the pattern string

### 7.2 `results/motifs/motif_stats.csv` — 43,072 rows

| Column | Description |
|---|---|
| `domain`, `pattern_id`, `pattern` | the pattern (ids index `motif_patterns`) |
| `depth`, `n_leaves` | levels in the pattern; leaf slots |
| `count_`, `docs_`, `rate_` + `human` / `gpt` / `claude` | occurrences, documents containing it, rate per 1,000 tokens (Eq. 1) |
| `log2_ratio_`, `z_` + `gpt` / `claude` | against human (Eq. 2, 3) |

- `results/motifs/motif_pool.csv` — documents, sentences and tokens per (domain, source)

### 7.3 `results/motifs/motif_explorer.html`

- The interactive result of this track (§6.5)

## 8. Verification

- `count_patterns` equals the original extractor (`extract_patterns_with_examples`) on 3,000 random sentences: 0 mismatches
- `find_patterns_with_spans` finds exactly `count_patterns`' occurrences (0 mismatches over 95,614 occurrences); an independent re-derivation of the slot spans from tree positions agrees for 9,940 occurrences, and for 400 randomly drawn rows of the saved examples table (sentence, source, span and slot boundaries): 0 failures
- After cleaning, slot boundaries are well-formed for all 949,097 occurrences in 30,000 sampled sentences. Before cleaning, 2,134 of 95,614 occurrences had a zero-width slot — which is how the whitespace-token problem (§3.1) was found
- The document table matches `doc_features` (documents and sentence counts); token totals per document match between the counts and examples builds
- Rebuilding the examples file gives a byte-identical file; rebuilding the count tables gives identical totals
- Checked in Chrome: no console errors; domain tabs, comparison toggle, fold, bracket search and example rendering work. Not checked: dark mode, narrow screens, other browsers

## 9. Limitations & Notes

- **z is not a test** (§4.4). It is anti-conservative when occurrences cluster in few documents, and Reuters human text is 50 authors × ~20 documents, so author clustering is not accounted for. Use the document counts, and treat z as an ordering
- **Overlapping patterns are not independent.** A phenomenon shows up in several rows; the echo folding (§4.5) is a presentation aid with an arbitrary threshold, not a statistical grouping
- **Not re-parsed after removing whitespace tokens** (§3.1). benepar saw those tokens in its input, so the structure around them may differ slightly from a clean parse. The Feng labels are unchanged; a full re-parse of whitespace-normalised text would settle it
- **Tokens are the parser's.** Punctuation counts as tokens, which affects rates for sources that use more of it
- **Document floor.** Patterns used in fewer than 10 (statistics) or 20 (explorer) documents of a domain are not listed, so the rare tail of the vocabulary is out of view; the per-document count table keeps all of it for the repertoire analyses
- **Examples are a random draw**, not a curated selection; patterns outside the embedded cut have none in the page (they are still in `motif_examples.feather`)
- **Smoothing** (`a = 0.5`) matters only for very small counts; a pattern one source never uses gets a large but finite ratio
- **How the AI texts were produced** (Ghostbuster, Verma et al., 2024: templates in Tables 7–8 of the paper, per-document prompts in `data/raw/ghostbuster-data/{essay,wp}/prompts`). Differences in citations, quotations and parentheticals reflect the tasks as much as the writers:
  - Essays: ChatGPT wrote a prompt from each human essay and a second call wrote the essay to it, with no mention of sources. 8.9% of human essay sentences contain an author-year or numbered citation, against 0.06% (GPT) and 0.18% (Claude); 16.5% contain a parenthesis, against 2.9% and 2.1%
  - News: written from a ChatGPT-made headline alone, with no source material, which plausibly accounts for the missing quotations and reported speech
  - WritingPrompts: the real Reddit prompt; the human texts are the last 100 posts of each of the top 50 posters, so they cluster by author
  - Models: GPT is gpt-3.5-turbo; the Claude version is not stated (2023). Length was matched to the nearest 100 words for ChatGPT, but Claude's documents are shorter (median words human / ChatGPT / Claude: essays 529 / 559 / 442, news 498 / 510 / 384, stories 455 / 512 / 384). That lowers Claude's document counts, since a pattern is less likely to appear in a shorter document, but not its rates per token
  - These patterns are left in, not filtered out. A column flagging patterns that sit mostly in sentences with a citation, parenthesis or quotation is a planned addition
