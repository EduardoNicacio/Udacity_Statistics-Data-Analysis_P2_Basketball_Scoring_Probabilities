"""
Steph Curry Shot Probability Analysis - 2022-2023 NBA Season
=============================================================

Udacity Statistics for Data Analysis - Project 2: Basketball Scoring Probabilities

This self-contained script:
    1. Downloads the shot dataset (CSV) from the Udacity CDN.
    2. Loads and cleans the data with pandas, normalizing the key columns.
    3. Computes overall shooting statistics.
    4. Applies the binomial distribution (manual formula + scipy) and discusses assumptions.
    5. Computes forward-looking conditional probabilities, verified with Bayes' Theorem.
    6. Computes retrospective conditional probabilities using Bayes' Theorem.
    7. Prints a slide-by-slide summary and saves the whole run to a text file.

Every probability below is derived from the loaded data. Nothing is hardcoded.

Usage:
    python curry_analysis.py
"""

import math
import os
import sys

import pandas as pd
import requests
from scipy.stats import binom

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

DATASET_URL = (
    "https://video.udacity-data.com/topher/2024/November/"
    "673ba86a_stephen_curry_shots_2023/stephen_curry_shots_2023.csv"
)
CSV_PATH = "stephen_curry_shots_2023.csv"
SUMMARY_PATH = "curry_probability_summary.txt"

SECTION_RULE = "=" * 78
SUBSECTION_RULE = "-" * 78


# ---------------------------------------------------------------------------
# OUTPUT HELPERS
# ---------------------------------------------------------------------------


class Tee:
    """Write everything written to stdout to one or more streams at once."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def fmt(value):
    """Format a probability as a 4-decimal decimal plus a percentage."""
    return f"{value:.4f}  ({value * 100:.2f}%)"


def banner(title, rule=SECTION_RULE):
    print()
    print(rule)
    print(title)
    print(rule)


# ---------------------------------------------------------------------------
# STEP 1 - DOWNLOAD THE DATASET
# ---------------------------------------------------------------------------


def download_dataset(url=DATASET_URL, path=CSV_PATH):
    """Download the CSV to ``path`` (skipped when the file already exists)."""
    banner("STEP 1: DOWNLOAD THE DATASET")

    if os.path.exists(path) and os.path.getsize(path) > 0:
        print(f"[ok]   Dataset already present: {os.path.abspath(path)}")
        print(f"       Size: {os.path.getsize(path):,} bytes")
        print("[..]   Reusing the stored CSV (no re-download).")
        probe = pd.read_csv(path)
        print(f"[ok]   Row count after loading with pandas: {len(probe):,}")
        print(f"[ok]   Column names: {list(probe.columns)}")
        return path

    print(f"[..]   Downloading from:\n       {url}")
    try:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        content = response.content
    except Exception as exc:  # fall back to urllib if requests is unavailable/blocked
        print(f"[warn] requests failed ({exc}); retrying with urllib")
        import urllib.request

        with urllib.request.urlopen(url, timeout=120) as handle:
            content = handle.read()

    with open(path, "wb") as handle:
        handle.write(content)

    print(f"[ok]   Saved to: {os.path.abspath(path)}")
    print(f"       Size: {len(content):,} bytes")

    # Confirm the saved file is a loadable CSV and report the row count and
    # column names straight after the download, as the brief requires.
    probe = pd.read_csv(path)
    print(f"[ok]   Row count after loading with pandas: {len(probe):,}")
    print(f"[ok]   Column names: {list(probe.columns)}")
    return path


# ---------------------------------------------------------------------------
# STEP 2 - DATA INSPECTION AND CLEANING
# ---------------------------------------------------------------------------


def _to_bool(series):
    """Normalize a column holding bools / 'TRUE' / 'FALSE' strings to bool."""
    if series.dtype == bool:
        return series.astype(bool)

    def convert(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and not pd.isna(value):
            return bool(value)
        if pd.isna(value):
            return False
        return str(value).strip().lower() in {"true", "t", "1", "yes", "y"}

    return series.map(convert).astype(bool)


def load_and_clean(path=CSV_PATH):
    """Load the CSV and normalize ``result``, ``lead`` and ``shot_type``."""
    banner("STEP 2: DATA INSPECTION AND CLEANING")

    df = pd.read_csv(path)

    print(f"[ok]   Loaded: {path}")
    print(f"[ok]   Rows: {df.shape[0]:,}    Columns: {df.shape[1]}")
    print(f"[ok]   Column names: {list(df.columns)}")

    print(f"\n-- Shape --\n{df.shape}")
    print(f"\n-- dtypes (before cleaning) --\n{df.dtypes.to_string()}")
    print(f"\n-- First 5 rows (before cleaning) --\n{df.head(5).to_string()}")

    # --- Missingness check must run BEFORE normalization --------------------
    # _to_bool() maps NaN to False, so a null in result/lead would otherwise be
    # silently turned into a value instead of being reported (and the later
    # isna() print would always show zeros). Fail loudly instead.
    key_nulls = df[["result", "shot_type", "lead"]].isna().sum()
    if key_nulls.any():
        raise ValueError(f"Missing values in key columns: "
                         f"{key_nulls[key_nulls > 0].to_dict()}")

    # --- Normalize the key columns -----------------------------------------
    df["result"] = _to_bool(df["result"])
    df["lead"] = _to_bool(df["lead"])
    df["shot_type"] = df["shot_type"].astype(int)  # 2 = two-pointer, 3 = three-pointer

    print("\n-- Normalization applied --")
    print("       result    -> bool   (TRUE = made, FALSE = missed)")
    print("       lead      -> bool   (TRUE = Curry's team leading)")
    print("       shot_type -> int    (2 = two-pointer, 3 = three-pointer)")

    print(f"\n-- dtypes (after cleaning) --\n{df.dtypes.to_string()}")

    # --- Integrity checks ---------------------------------------------------
    print("\n-- value_counts: result --")
    print(df["result"].value_counts().to_string())
    print("\n-- value_counts: shot_type --")
    print(df["shot_type"].value_counts().sort_index().to_string())
    print("\n-- value_counts: lead --")
    print(df["lead"].value_counts().to_string())

    print("\n-- Missing values in key columns --")
    print(df[["result", "shot_type", "lead"]].isna().sum().to_string())

    unexpected_types = set(df["shot_type"].unique()) - {2, 3}
    if unexpected_types:
        raise ValueError(f"Unexpected shot_type values found: {unexpected_types}")
    print("\n[ok]   Key columns are clean: only 2-pointers and 3-pointers present,")
    print("       result/lead are boolean, and there are no missing values.")

    return df


# ---------------------------------------------------------------------------
# STEP 3 - OVERALL SHOOTING STATISTICS
# ---------------------------------------------------------------------------


def overall_statistics(df):
    """Compute P(Make), P(Miss), P(3-pointer) and P(2-pointer)."""
    banner("STEP 3: OVERALL SHOOTING STATISTICS")

    total_shots = len(df)
    made = int((df["result"] == True).sum())          # noqa: E712
    missed = int((df["result"] == False).sum())       # noqa: E712
    threes = int((df["shot_type"] == 3).sum())
    twos = int((df["shot_type"] == 2).sum())

    p_make = made / total_shots
    p_miss = missed / total_shots
    p_three = threes / total_shots
    p_two = twos / total_shots

    print(f"Total shots: {total_shots:,}")
    print()
    print(f"  P(Make)      = {made:>5,} / {total_shots:,} = {fmt(p_make)}")
    print(f"  P(Miss)      = {missed:>5,} / {total_shots:,} = {fmt(p_miss)}")
    print(f"  P(3-pointer) = {threes:>5,} / {total_shots:,} = {fmt(p_three)}")
    print(f"  P(2-pointer) = {twos:>5,} / {total_shots:,} = {fmt(p_two)}")
    print()
    print(f"  Check: P(Make) + P(Miss)   = {p_make + p_miss:.4f}")
    print(f"  Check: P(3)    + P(2)      = {p_three + p_two:.4f}")

    return {
        "total_shots": total_shots,
        "made": made,
        "missed": missed,
        "threes": threes,
        "twos": twos,
        "p_make": p_make,
        "p_miss": p_miss,
        "p_three": p_three,
        "p_two": p_two,
    }


# ---------------------------------------------------------------------------
# STEP 4 - BINOMIAL DISTRIBUTION
# ---------------------------------------------------------------------------


def binomial_probability(n, k, p):
    """Return (manual, library) for exactly ``k`` successes in ``n`` trials."""
    manual = math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
    library = float(binom.pmf(k, n, p))
    return manual, library


def binomial_section(p_make, p_three):
    """Steps 4A, 4B and 4C: binomial probabilities and their assumptions."""
    banner("STEP 4: BINOMIAL DISTRIBUTION")

    # --- 4A: exactly 3 of the next 4 shots are made -------------------------
    print()
    print("4A) P(exactly 3 made out of the next 4 shots)")
    print(SUBSECTION_RULE)
    n, k, p = 4, 3, p_make
    manual, library = binomial_probability(n, k, p)
    print(f"  n = {n}, k = {k}, p = P(Make) = {p:.4f}")
    print(f"  Manual : C({n},{k}) * p^{k} * (1-p)^{n - k}")
    print(f"         = {math.comb(n, k)} * {p:.4f}^{k} * {1 - p:.4f}^{n - k}")
    print(f"         = {manual:.6f}")
    print(f"  Library: scipy.stats.binom.pmf({k}, {n}, p) = {library:.6f}")
    print(f"  Result : {fmt(library)}")
    print(f"  Match  : {'YES' if abs(manual - library) < 1e-12 else 'NO'}"
          f"  (difference = {abs(manual - library):.2e})")

    # --- 4B: exactly 4 of the next 5 shots are three-pointers ---------------
    print()
    print("4B) P(exactly 4 of the next 5 shots are three-pointers)")
    print(SUBSECTION_RULE)
    n_b, k_b, p_b = 5, 4, p_three
    manual_b, library_b = binomial_probability(n_b, k_b, p_b)
    print(f"  n = {n_b}, k = {k_b}, p = P(3-pointer) = {p_b:.4f}")
    print(f"  Manual : C({n_b},{k_b}) * p^{k_b} * (1-p)^{n_b - k_b}")
    print(f"         = {math.comb(n_b, k_b)} * {p_b:.4f}^{k_b} * {1 - p_b:.4f}^{n_b - k_b}")
    print(f"         = {manual_b:.6f}")
    print(f"  Library: scipy.stats.binom.pmf({k_b}, {n_b}, p) = {library_b:.6f}")
    print(f"  Result : {fmt(library_b)}")
    print(f"  Match  : {'YES' if abs(manual_b - library_b) < 1e-12 else 'NO'}"
          f"  (difference = {abs(manual_b - library_b):.2e})")

    # --- 4C: assumptions ----------------------------------------------------
    print()
    print("4C) ASSUMPTIONS OF THE BINOMIAL DISTRIBUTION")
    print(SUBSECTION_RULE)
    print("""  1. A fixed number of trials (n).
     Every shot is one trial, and n is decided in advance (4 shots, 5 shots).

  2. Each trial has exactly two possible outcomes (success / failure).
     A shot is either made or missed, so this one is satisfied - "make" is
     success for 4A, "three-pointer" is success for 4B.

  3. The trials are independent.
     The outcome of one shot must not affect the others. This is the weakest
     assumption in basketball: shooters ride hot/cold streaks, defenders adjust,
     and fatigue accumulates during a game, so consecutive shots are only
     approximately independent.

  4. The probability of success p is constant for every trial.
     p is estimated from the whole season, but the true probability varies with
     distance, shot difficulty, defender pressure, quarter, and game situation
     (e.g. garbage time vs. clutch time). Treating p as one fixed number is
     therefore a simplification.""")

    print("""
  Do the assumptions hold? Partially.
    - Assumptions 1 and 2 hold exactly: n is chosen by us and a shot is binary.
    - Assumption 3 (independence) is approximately true at best. Shot outcomes
      are close to independent across a large sample, but "hot hand" effects,
      defensive schemes and fatigue can create serial dependence.
    - Assumption 4 (constant p) is clearly violated in spirit: shot quality
      differs a lot (layup vs. deep three, open vs. contested). The season-wide
      p is an average over very different situations.
  Practical consequence: the binomial probabilities above are useful ballpark
  estimates under an "all shots are alike and independent" model, and should be
  read as approximations rather than exact forecasts.""")

    return {
        "4a_n": n, "4a_k": k, "4a_p": p, "4a": library,
        "4b_n": n_b, "4b_k": k_b, "4b_p": p_b, "4b": library_b,
    }


# ---------------------------------------------------------------------------
# STEP 5 - CONDITIONAL PROBABILITIES (FUTURE)
# ---------------------------------------------------------------------------


def future_conditionals(df, overall):
    """Compute P(Made|3), P(Lead|3), P(Made|2), P(Lead|2) and verify with Bayes."""
    banner("STEP 5: CONDITIONAL PROBABILITIES - FUTURE (FORWARD-LOOKING)")

    made_and_three = int(((df["shot_type"] == 3) & (df["result"] == True)).sum())   # noqa: E712
    lead_and_three = int(((df["shot_type"] == 3) & (df["lead"] == True)).sum())     # noqa: E712
    made_and_two = int(((df["shot_type"] == 2) & (df["result"] == True)).sum())     # noqa: E712
    lead_and_two = int(((df["shot_type"] == 2) & (df["lead"] == True)).sum())       # noqa: E712

    threes = overall["threes"]
    twos = overall["twos"]
    total = overall["total_shots"]
    p_make = overall["p_make"]
    p_three = overall["p_three"]
    p_two = overall["p_two"]

    p_lead = int((df["lead"] == True).sum()) / total                                   # noqa: E712
    p_three_given_lead = lead_and_three / int((df["lead"] == True).sum())              # noqa: E712
    p_two_given_lead = lead_and_two / int((df["lead"] == True).sum())                  # noqa: E712
    p_three_given_made = made_and_three / overall["made"]
    p_two_given_made = made_and_two / overall["made"]

    # Direct comparisons
    p_made_given_three = made_and_three / threes
    p_lead_given_three = lead_and_three / threes
    p_made_given_two = made_and_two / twos
    p_lead_given_two = lead_and_two / twos

    # Bayes' Theorem reconstructions
    bayes_made_given_three = p_three_given_made * p_make / p_three
    bayes_lead_given_three = p_three_given_lead * p_lead / p_three
    bayes_made_given_two = p_two_given_made * p_make / p_two
    bayes_lead_given_two = p_two_given_lead * p_lead / p_two

    # --- Joint and marginal counts -----------------------------------------
    print("\n-- Joint and marginal counts (directly from the data) --")
    print(f"  Made 3-pointers            count(3 AND Made)  = {made_and_three:>4,}")
    print(f"  3-pointers with a lead     count(3 AND Lead)  = {lead_and_three:>4,}")
    print(f"  Made 2-pointers            count(2 AND Made)  = {made_and_two:>4,}")
    print(f"  2-pointers with a lead     count(2 AND Lead)  = {lead_and_two:>4,}")
    print()
    print(f"  count(3-pointers)  = {threes:>4,}   count(2-pointers)  = {twos:>4,}")
    print(f"  count(Made)        = {overall['made']:>4,}   count(Lead)        = "
          f"{int((df['lead'] == True).sum()):>4,}")  # noqa: E712
    print(f"  count(All shots)   = {total:>4,}")

    # --- Derived conditional probabilities + Bayes verification -------------
    checks = [
        {
            "label": "P(Made | 3)",
            "direct": p_made_given_three,
            "bayes": bayes_made_given_three,
            "terms": (p_three_given_made, p_make, p_three),
            "display": "P(Made|3) = P(3|Made) * P(Made) / P(3)",
        },
        {
            "label": "P(Lead | 3)",
            "direct": p_lead_given_three,
            "bayes": bayes_lead_given_three,
            "terms": (p_three_given_lead, p_lead, p_three),
            "display": "P(Lead|3) = P(3|Lead) * P(Lead) / P(3)",
        },
        {
            "label": "P(Made | 2)",
            "direct": p_made_given_two,
            "bayes": bayes_made_given_two,
            "terms": (p_two_given_made, p_make, p_two),
            "display": "P(Made|2) = P(2|Made) * P(Made) / P(2)",
        },
        {
            "label": "P(Lead | 2)",
            "direct": p_lead_given_two,
            "bayes": bayes_lead_given_two,
            "terms": (p_two_given_lead, p_lead, p_two),
            "display": "P(Lead|2) = P(2|Lead) * P(Lead) / P(2)",
        },
    ]

    print("\n-- Conditional probabilities: direct count vs. Bayes' Theorem --")
    print(SUBSECTION_RULE)
    for check in checks:
        numerator, prior_a, prior_b = check["terms"]
        print(f"\n  {check['display']}")
        print(f"    P(B|A) * P(A) / P(B)"
              f" = {numerator:.6f} * {prior_a:.6f} / {prior_b:.6f}")
        print(f"    Bayes' Theorem  : {check['bayes']:.6f}")
        print(f"    Direct counting : {check['direct']:.6f}")
        print(f"    Match           : "
              f"{'YES' if abs(check['bayes'] - check['direct']) < 1e-12 else 'NO'}"
              f"  (difference = {abs(check['bayes'] - check['direct']):.2e})")
        print(f"    Result          : {fmt(check['direct'])}")

    return {
        "made_and_three": made_and_three,
        "lead_and_three": lead_and_three,
        "made_and_two": made_and_two,
        "lead_and_two": lead_and_two,
        "p_lead": p_lead,
        "p_three_given_lead": p_three_given_lead,
        "p_two_given_lead": p_two_given_lead,
        "p_three_given_made": p_three_given_made,
        "p_two_given_made": p_two_given_made,
        "p_made_given_three": p_made_given_three,
        "p_lead_given_three": p_lead_given_three,
        "p_made_given_two": p_made_given_two,
        "p_lead_given_two": p_lead_given_two,
    }


# ---------------------------------------------------------------------------
# STEP 6 - CONDITIONAL PROBABILITIES (PAST) VIA BAYES' THEOREM
# ---------------------------------------------------------------------------


def past_conditionals(overall, future):
    """Compute P(3|Made) and P(2|Made) with every intermediate value shown."""
    banner("STEP 6: CONDITIONAL PROBABILITIES - PAST (RETROSPECTIVE) VIA BAYES")

    p_make = overall["p_make"]
    p_three = overall["p_three"]
    p_two = overall["p_two"]
    p_made_given_three = future["p_made_given_three"]
    p_made_given_two = future["p_made_given_two"]

    print("\n-- Notation mapping --")
    print(f"  P(Made)     = {p_make:.6f}   (Step 3)")
    print(f"  P(3)        = {p_three:.6f}   (Step 3)")
    print(f"  P(2)        = {p_two:.6f}   (Step 3)")
    print(f"  P(Made | 3) = {p_made_given_three:.6f}   (Step 5)")
    print(f"  P(Made | 2) = {p_made_given_two:.6f}   (Step 5)")

    # --- P(3 | Made) --------------------------------------------------------
    print("\n-- P(3 | Made): given that a shot went in, was it a three? --")
    print(SUBSECTION_RULE)
    print("  P(3|Made) = P(Made|3) * P(3) / P(Made)")
    print(f"  Step 1: P(Made|3) = {p_made_given_three:.6f}")
    print(f"  Step 2: P(3)      = {p_three:.6f}")
    print(f"  Step 3: P(Made|3) * P(3) = "
          f"{p_made_given_three * p_three:.6f}   <- P(Made AND 3)")
    print(f"  Step 4: P(Made)   = {p_make:.6f}")
    p_three_given_made = p_made_given_three * p_three / p_make
    print(f"  Step 5: P(3|Made) = {p_made_given_three * p_three:.6f} / {p_make:.6f}"
          f" = {p_three_given_made:.6f}")
    print(f"  Result: {fmt(p_three_given_made)}")

    # --- P(2 | Made) --------------------------------------------------------
    print("\n-- P(2 | Made): given that a shot went in, was it a two? --")
    print(SUBSECTION_RULE)
    print("  P(2|Made) = P(Made|2) * P(2) / P(Made)")
    print(f"  Step 1: P(Made|2) = {p_made_given_two:.6f}")
    print(f"  Step 2: P(2)      = {p_two:.6f}")
    print(f"  Step 3: P(Made|2) * P(2) = "
          f"{p_made_given_two * p_two:.6f}   <- P(Made AND 2)")
    print(f"  Step 4: P(Made)   = {p_make:.6f}")
    p_two_given_made = p_made_given_two * p_two / p_make
    print(f"  Step 5: P(2|Made) = {p_made_given_two * p_two:.6f} / {p_make:.6f}"
          f" = {p_two_given_made:.6f}")
    print(f"  Result: {fmt(p_two_given_made)}")

    # --- Sanity check -------------------------------------------------------
    total = p_three_given_made + p_two_given_made
    print("\n-- Sanity check --")
    print(f"  P(3|Made) + P(2|Made) = {p_three_given_made:.6f} + "
          f"{p_two_given_made:.6f} = {total:.6f}")
    print(f"  Approximately 1.0: {'YES' if abs(total - 1.0) < 1e-9 else 'NO'}")

    return {
        "p_three_given_made": p_three_given_made,
        "p_two_given_made": p_two_given_made,
        "sanity_total": total,
    }


# ---------------------------------------------------------------------------
# STEP 7 - SUMMARY OUTPUT
# ---------------------------------------------------------------------------


def print_summary(overall, binomial, future, past):
    """Print the slide-by-slide summary of every computed probability."""
    banner("STEP 7: SUMMARY - SLIDE-BY-SLIDE PROBABILITIES")

    print(f"\n=== SLIDE 1: Overall Shooting Statistics ===")
    print(f"  Total shots analysed      : {overall['total_shots']:,}")
    print(f"  P(Make)                   : {fmt(overall['p_make'])}")
    print(f"  P(Miss)                   : {fmt(overall['p_miss'])}")
    print(f"  P(3-pointer)              : {fmt(overall['p_three'])}")
    print(f"  P(2-pointer)              : {fmt(overall['p_two'])}")

    print("\n=== SLIDE 2: Binomial Probabilities ===")
    label_a = f"P(exactly {binomial['4a_k']} made of next {binomial['4a_n']} shots)"
    label_b = f"P(exactly {binomial['4b_k']} of next {binomial['4b_n']} are 3-pointers)"
    width = max(len(label_a), len(label_b))
    print(f"  {label_a.ljust(width)} : {fmt(binomial['4a'])}")
    print(f"  {label_b.ljust(width)} : {fmt(binomial['4b'])}")
    print("  Assumptions: fixed n, binary outcomes, independent trials, constant p.")
    print("  Independence and constant p hold only approximately for shot data.")

    print("\n=== SLIDE 3: Conditional Probabilities - Future ===")
    print(f"  P(Made | 3-pointer)       : {fmt(future['p_made_given_three'])}")
    print(f"  P(Lead | 3-pointer)       : {fmt(future['p_lead_given_three'])}")
    print(f"  P(Made | 2-pointer)       : {fmt(future['p_made_given_two'])}")
    print(f"  P(Lead | 2-pointer)       : {fmt(future['p_lead_given_two'])}")
    print("\n  Supporting quantities:")
    print(f"    P(Made)                 : {fmt(overall['p_make'])}")
    print(f"    P(3)                    : {fmt(overall['p_three'])}")
    print(f"    P(2)                    : {fmt(overall['p_two'])}")
    print(f"    P(Lead)                 : {fmt(future['p_lead'])}")
    print(f"    P(3 | Lead)             : {fmt(future['p_three_given_lead'])}")
    print(f"    P(2 | Lead)             : {fmt(future['p_two_given_lead'])}")
    print(f"    P(Made | 3) x P(3)      : {fmt(future['p_made_given_three'] * overall['p_three'])}")
    print(f"    P(Made | 2) x P(2)      : {fmt(future['p_made_given_two'] * overall['p_two'])}")

    print("\n=== SLIDE 4: Conditional Probabilities - Past ===")
    print(f"  P(3-pointer | Made)       : {fmt(past['p_three_given_made'])}")
    print(f"  P(2-pointer | Made)       : {fmt(past['p_two_given_made'])}")
    print(f"  Sanity check (sum = 1.0)  : {past['sanity_total']:.6f}")

    print()
    print(SECTION_RULE)
    print("END OF REPORT")
    print(SECTION_RULE)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


def main():
    # Echo everything to the console and to the summary file simultaneously.
    original_stdout = sys.stdout
    with open(SUMMARY_PATH, "w", encoding="utf-8") as summary_file:
        sys.stdout = Tee(original_stdout, summary_file)
        try:
            banner("STEPH CURRY SHOT PROBABILITY ANALYSIS - 2022-2023 NBA SEASON")
            print("Source dataset: stephen_curry_shots_2023.csv")
            print("All probabilities are computed directly from the loaded data.")

            path = download_dataset()
            df = load_and_clean(path)
            overall = overall_statistics(df)
            binomial = binomial_section(overall["p_make"], overall["p_three"])
            future = future_conditionals(df, overall)
            past = past_conditionals(overall, future)
            print_summary(overall, binomial, future, past)
            # Printed while the summary file is still open, so the console and
            # curry_probability_summary.txt stay byte-for-byte identical.
            print(f"\n[ok]   Full output saved to: {os.path.abspath(SUMMARY_PATH)}")
        finally:
            sys.stdout = original_stdout


if __name__ == "__main__":
    main()
