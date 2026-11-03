# Project: Calculate Basketball Scoring Probabilities

## Project summary

This project computes probabilities from **Steph Curry's shot log for the 2022-2023 NBA season** (1,434 attempts × 15 columns) and reports them in the four slide-deck sections required by `docs/3. Presentation Components.md`: overall shooting statistics, binomial probabilities, forward-looking conditional probabilities, and retrospective (past) conditional probabilities. A single reproducible Python script — `curry_analysis.py` — downloads the CSV from the Udacity CDN with `requests`, loads it with pandas, and executes every requested calculation in order: overall frequencies (P(Make) **0.4902**, P(Miss) **0.5098**, P(3-pointer) **0.5481**, P(2-pointer) **0.4519**); binomial probabilities computed twice, by manual formula and by `scipy.stats.binom.pmf`, agreeing to ~1e-17 (exactly 3 of the next 4 made → **0.2402**, exactly 4 of the next 5 are threes → **0.2039**), followed by the four required assumptions and an explicit discussion of whether they hold for shot data; four conditional probabilities each computed two independent ways — direct counting and Bayes' Theorem — matching to ~1e-16 (P(Made|3) **0.4186**, P(Lead|3) **0.4936**, P(Made|2) **0.5772**, P(Lead|2) **0.5154**); and the retrospective answers P(3|Made) **0.4680** and P(2|Made) **0.5320** derived through Bayes' Theorem with every intermediate value printed, summing to **1.000000** as required by the total-probability identity. All probabilities are rounded to four decimal places at output time only and written — together with percentages — to `curry_probability_summary.txt`, whose content is byte-for-byte identical to the console output.

## How to run

1. Install the dependencies (Python 3.14 verified; no other version constraints):

   ```bash
   pip install pandas scipy requests
   ```

2. Run the analysis from the project root:

   ```bash
   python curry_analysis.py
   ```

   The script is idempotent: if `stephen_curry_shots_2023.csv` already exists and is non-empty it is reused instead of re-downloaded, and the summary file is overwritten with identical results. A network connection is needed only for the first run.

3. Outputs land in the current working directory:
   - `curry_probability_summary.txt` — the complete run: all counts, formulas, Bayes steps, and the four slide sections
   - `stephen_curry_shots_2023.csv` — the dataset, stored byte-for-byte as downloaded (reused on re-runs)

## Data sources

- **Stephen Curry shots, 2022-2023** (Udacity CDN): <https://video.udacity-data.com/topher/2024/November/673ba86a_stephen_curry_shots_2023/stephen_curry_shots_2023.csv>
- Original dataset on Kaggle: <https://www.kaggle.com/datasets/dhavalrupapara/nba-2023-player-shot-dataset>
- Key columns: `result` (TRUE = made), `shot_type` (2 or 3), `lead` (TRUE = Curry's team leading when the shot was taken)

## Key results (copy into the slide deck)

| Slide | Result |
| :--- | :--- |
| **Overall Shooting Statistics** | 1,434 shots: P(Make) 0.4902 (703) · P(Miss) 0.5098 (731) · P(3-pointer) 0.5481 (786) · P(2-pointer) 0.4519 (648). |
| **Probabilities (binomial)** | P(3 of next 4 made) = 0.2402 · P(4 of next 5 are threes) = 0.2039. Manual `C(n,k)·pᵏ(1−p)ⁿ⁻ᵏ` matches `binom.pmf` to ~1e-17. Assumptions: fixed n, binary outcomes, independence, constant p — the last two hold only approximately. |
| **Conditional Probabilities — Future** | P(Made\|3) 0.4186 (329/786) · P(Lead\|3) 0.4936 (388/786) · P(Made\|2) 0.5772 (374/648) · P(Lead\|2) 0.5154 (334/648). Each reproduced by Bayes' Theorem to ~1e-16. |
| **Conditional Probabilities — Past** | P(3\|Made) 0.4680 · P(2\|Made) 0.5320 · sum 1.000000. Both derived via Bayes with every intermediate printed, then sanity-checked. |

## Rubric conformance

Reviewed against `docs/5. Rubric.md`, with every reported figure independently recomputed in a second implementation that shares no code or libraries with the script (pure Python standard library: `csv` + `math.factorial`, no pandas and no scipy) — **47 checks, 0 failures**:

| Rubric criterion | Status |
| :--- | :--- |
| Binomial distribution used correctly | ✅ Manual formula and `binom.pmf` agree for both questions (0.2402, 0.2039) |
| Conditional probability calculations correct | ✅ All four computed by direct counting *and* by Bayes' Theorem; both routes agree to ~1e-16 |
| Bayes' rule calculations correct | ✅ Retrospective P(3\|Made), P(2\|Made) derived through Bayes with all intermediates shown; sum = 1.000000 |
| Assumptions of the binomial distribution provided | ✅ Four numbered assumptions plus a written discussion of which hold for shot data (independence and constant p only approximately) |
| *Standout:* show your work | ✅ Every step prints its counts, formula, and intermediate values — nothing is asserted without derivation |
| *Standout:* are the assumptions violated? | ✅ §4C argues each assumption against the data (hot-hand effects, fatigue, shot-quality variation) |

### Review findings (resolved before writing this README)

An audit of the implementation against the specification found **no numerical errors** (47/47 independent checks passed) and three non-numeric inconsistencies, all fixed and re-verified:

1. **Step 1 confirmation incomplete** — the brief asks for the row count and column names right after the download; they were only printed in Step 2. The download step now loads the file with pandas and prints both (in the fresh-download and reuse branches).
2. **Latent null-handling bug** — `_to_bool()` coerces NaN to `False`, and it ran *before* the printed missingness check, so a null in `result`/`lead` would have been silently converted and the check would still report zero. The null check now runs first and raises `ValueError` (fault-injection test: a null injected into `lead` aborts with `Missing values in key columns: {'lead': 1}`). The dataset itself has no missing values.
3. **Console output ≠ saved file** — the trailing confirmation line was printed outside the capture stream, so the console had 286 lines against the file's 284. The line is now written while the file is open; console and `curry_probability_summary.txt` are byte-identical (289 lines).

## Files

- `curry_analysis.py` — the complete analysis script (Steps 1–7 of the project brief), 589 lines
- `stephen_curry_shots_2023.csv` — the downloaded dataset (1,434 × 15, 109,488 bytes)
- `curry_probability_summary.txt` — every count, formula, and probability under the four slide headers (byte-identical to the console output)
- `docs/` — Udacity project documentation, rubric, and candidate notes
- `LICENSE.md` — project license

## Submission

The graded component is the slide deck (Google Slides or MS PowerPoint template — see `docs/3. Presentation Components.md` and `docs/1. Project Overview.md`). The four slide sections in `curry_probability_summary.txt` map one-to-one to the required slides and can be copied directly into the deck. Submission methods:

1. **Project Link**: submit the viewable link if you used the Google Slides template (make sure link access is set so anyone with the link can view).
2. **Single File**: submit the completed deck saved as a PDF if you used the MS PowerPoint template.
