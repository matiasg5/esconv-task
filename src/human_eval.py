"""
src/human_eval.py

Human side of the GenAI evaluation, and the assignment's required kappa/QWK
agreement check. Two steps of one sequence -- the first step's only output is
the second step's only input:

  parse : reads the hand-filled blind quiz (full32_scoring_quiz.txt) plus the
          blind key, writes full32_human_scores.csv.
  kappa : joins those human scores with the Gemini judge's scores
          (full32_judge_scores_gemini.json), computes Cohen's kappa + QWK for
          quality and strategy_adherence, ranks the prompt levels on a
          human+AI blended score, and reports the correct/wrong (matched vs.
          mismatched strategy signal) breakdown for both raters.

The quiz presented each item's 4 responses shuffled into options A-D with no
level labels, so hand-scoring was blind both to which prompt level produced a
response and to the AI judge's scores. full32_scoring_blind_key.json holds the
per-item letter -> level mapping and is applied only at parse time, after
scoring was done. Quiz parsing is block-based, not line-based -- see
report_notes.md for the real bug that fixed.

Kappa/QWK are implemented from scratch (no sklearn) for auditability --
verified against known textbook values, see report_notes.md.

Usage:
    python human_eval.py parse
    python human_eval.py kappa
    python human_eval.py kappa --human_weight 0.5
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

# --- quiz parsing ---------------------------------------------------------

ITEM_HEADER_RE = re.compile(r"--- Item idx=(\d+)\s+strategy=")
OPTION_START_RE = re.compile(r"^  ([A-D])\. ", re.MULTILINE)
QUALITY_RE = re.compile(r"Quality \(1-5\):\s*(\d+)")
ADHERENCE_RE = re.compile(r"Strategy adherence \(1-5\):\s*(\d+)")

# --- kappa / QWK ----------------------------------------------------------

RATING_MIN, RATING_MAX = 1, 5
HUMAN_WEIGHT_DEFAULT = 0.65


def parse_quiz(quiz_text):
    """Returns {idx_str: {letter: (quality, strategy_adherence)}}. Block-based
    (not line-based) -- see module docstring for the real bug this fixed.

    Called by: cmd_parse().
    """
    chunks = ITEM_HEADER_RE.split(quiz_text)
    results = {}
    for i in range(1, len(chunks), 2):
        idx = chunks[i]
        chunk = chunks[i + 1]
        starts = [(m.group(1), m.start()) for m in OPTION_START_RE.finditer(chunk)]
        options = {}
        for j, (letter, start) in enumerate(starts):
            end = starts[j + 1][1] if j + 1 < len(starts) else len(chunk)
            block = chunk[start:end]
            qm = QUALITY_RE.search(block)
            am = ADHERENCE_RE.search(block)
            if qm and am:
                options[letter] = (int(qm.group(1)), int(am.group(1)))
        results[idx] = options
    return results


def cmd_parse(genai_dir):
    """parse subcommand: reads the hand-filled blind quiz, un-blinds the letters back
    to prompt levels using the saved key, and writes full32_human_scores.csv.

    Called by: main().
    """
    quiz_path = genai_dir / "full32_scoring_quiz.txt"
    key_path = genai_dir / "full32_scoring_blind_key.json"

    if not quiz_path.exists():
        raise FileNotFoundError(f"{quiz_path} not found -- this is the hand-filled quiz, not a generated file")
    if not key_path.exists():
        raise FileNotFoundError(f"{key_path} not found -- needed to un-blind letters back to prompt levels")

    quiz_text = quiz_path.read_text(encoding="utf-8")
    with open(key_path) as f:
        blind_key = json.load(f)  # {idx: {letter: level}}

    parsed = parse_quiz(quiz_text)

    rows = []
    n_missing = 0
    n_total = 0
    for idx, letter_to_level in blind_key.items():
        item_scores = parsed.get(idx, {})
        for letter, level in letter_to_level.items():
            n_total += 1
            scores = item_scores.get(letter)
            if scores is None:
                n_missing += 1
                rows.append({"idx": idx, "level": level, "human_quality": "", "human_strategy_adherence": ""})
                continue
            quality, adherence = scores
            rows.append({
                "idx": idx, "level": level,
                "human_quality": quality, "human_strategy_adherence": adherence,
            })

    out_path = genai_dir / "full32_human_scores.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["idx", "level", "human_quality", "human_strategy_adherence"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {out_path}")
    print(f"Parsed {n_total - n_missing}/{n_total} scored responses.")
    if n_missing:
        print(f"WARNING: {n_missing} response(s) still have blank/unparseable scores -- "
              f"check full32_scoring_quiz.txt for any '____' left unfilled or a malformed "
              f"number, then re-run: python human_eval.py parse")
    print("\nNext: python human_eval.py kappa")


# --- kappa / QWK ----------------------------------------------------------

def load_human_scores(path):
    """Loads the hand scores keyed by (idx, level). Unscored rows are skipped, not
    counted as zero. Returns (scores, total row count).

    Called by: cmd_kappa().
    """
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_key = {}
    for row in rows:
        key = (row["idx"], row["level"])
        q = row["human_quality"].strip()
        a = row["human_strategy_adherence"].strip()
        if q == "" or a == "":
            continue  # not yet scored -- excluded, not treated as 0
        by_key[key] = {"quality": int(q), "strategy_adherence": int(a)}
    return by_key, len(rows)


def load_ai_scores(path):
    """Loads the judge's JSON scores into a dict keyed by (idx, level).

    Called by: cmd_kappa().
    """
    with open(path) as f:
        entries = json.load(f)
    by_key = {}
    for e in entries:
        by_key[(e["idx"], e["level"])] = e
    return by_key


def cohens_kappa(pairs, rating_min=RATING_MIN, rating_max=RATING_MAX):
    """Unweighted Cohen's kappa. pairs: list of (human_rating, ai_rating) ints.

    Called by: cmd_kappa().
    """
    n = len(pairs)
    if n == 0:
        return None
    labels = list(range(rating_min, rating_max + 1))
    idx_of = {v: i for i, v in enumerate(labels)}
    k = len(labels)
    O = [[0] * k for _ in range(k)]
    for h, a in pairs:
        O[idx_of[h]][idx_of[a]] += 1
    row_sums = [sum(O[i]) for i in range(k)]
    col_sums = [sum(O[i][j] for i in range(k)) for j in range(k)]
    po = sum(O[i][i] for i in range(k)) / n
    pe = sum(row_sums[i] * col_sums[i] for i in range(k)) / (n * n)
    if pe == 1:
        return 1.0
    return (po - pe) / (1 - pe)


def quadratic_weighted_kappa(pairs, rating_min=RATING_MIN, rating_max=RATING_MAX):
    """QWK -- same confusion-matrix machinery as Cohen's kappa, but weighted
    by squared rating distance, appropriate for an ordinal 1-5 scale.

    Called by: cmd_kappa().
    """
    n = len(pairs)
    if n == 0:
        return None
    labels = list(range(rating_min, rating_max + 1))
    idx_of = {v: i for i, v in enumerate(labels)}
    k = len(labels)
    O = [[0] * k for _ in range(k)]
    for h, a in pairs:
        O[idx_of[h]][idx_of[a]] += 1
    row_sums = [sum(O[i]) for i in range(k)]
    col_sums = [sum(O[i][j] for i in range(k)) for j in range(k)]
    E = [[row_sums[i] * col_sums[j] / n for j in range(k)] for i in range(k)]
    span = (rating_max - rating_min) if rating_max != rating_min else 1
    W = [[((labels[i] - labels[j]) ** 2) / (span ** 2) for j in range(k)] for i in range(k)]
    num = sum(W[i][j] * O[i][j] for i in range(k) for j in range(k))
    den = sum(W[i][j] * E[i][j] for i in range(k) for j in range(k))
    if den == 0:
        return 1.0
    return 1 - num / den


def normalize_1_5(x):
    """Maps a 1-5 rating onto 0-1, for the weighted blend of human and AI scores.

    Called by: cmd_kappa().
    """
    return (x - RATING_MIN) / (RATING_MAX - RATING_MIN)


def cmd_kappa(genai_dir, human_weight):
    """kappa subcommand: computes Cohen's kappa and QWK between the hand scores and
    the Gemini judge, on both quality and strategy adherence.

    Called by: main().
    """
    human_path = genai_dir / "full32_human_scores.csv"
    ai_path = genai_dir / "full32_judge_scores_gemini.json"
    if not human_path.exists():
        raise FileNotFoundError(f"{human_path} not found -- run: python human_eval.py parse")
    if not ai_path.exists():
        raise FileNotFoundError(f"{ai_path} not found -- run: python genai/judge step (judge.py --provider gemini)")

    human_scores, n_template_rows = load_human_scores(human_path)
    ai_scores = load_ai_scores(ai_path)

    n_missing_human = n_template_rows - len(human_scores)
    if n_missing_human > 0:
        print(f"NOTE: {n_missing_human}/{n_template_rows} rows have no human score yet -- excluded from all "
              f"analysis below (not treated as 0).")

    # --- Build matched pairs (only where BOTH a human score and a valid AI score exist) ---
    quality_pairs, adherence_pairs = [], []
    per_level_human, per_level_ai, per_level_combined = defaultdict(list), defaultdict(list), defaultdict(list)
    per_level_human_quality, per_level_human_adherence = defaultdict(list), defaultdict(list)
    per_level_ai_quality, per_level_ai_adherence = defaultdict(list), defaultdict(list)
    per_level_human_correct, per_level_human_wrong = defaultdict(list), defaultdict(list)
    per_level_ai_correct, per_level_ai_wrong = defaultdict(list), defaultdict(list)
    n_ai_missing = 0

    for key, h in human_scores.items():
        ai = ai_scores.get(key)
        if ai is None:
            continue
        if ai["quality"] is None or ai["strategy_adherence"] is None:
            n_ai_missing += 1
            continue

        quality_pairs.append((h["quality"], ai["quality"]))
        adherence_pairs.append((h["strategy_adherence"], ai["strategy_adherence"]))

        idx, level = key
        h_mean = (h["quality"] + h["strategy_adherence"]) / 2
        a_mean = (ai["quality"] + ai["strategy_adherence"]) / 2
        h_norm = normalize_1_5(h_mean)
        a_norm = normalize_1_5(a_mean)
        combined = human_weight * h_norm + (1 - human_weight) * a_norm

        per_level_human[level].append(h_mean)
        per_level_ai[level].append(a_mean)
        per_level_combined[level].append(combined)
        per_level_human_quality[level].append(h["quality"])
        per_level_human_adherence[level].append(h["strategy_adherence"])
        per_level_ai_quality[level].append(ai["quality"])
        per_level_ai_adherence[level].append(ai["strategy_adherence"])

        correct = ai.get("correct")
        if correct == "True":
            per_level_human_correct[level].append(h_mean)
            per_level_ai_correct[level].append(a_mean)
        elif correct == "False":
            per_level_human_wrong[level].append(h_mean)
            per_level_ai_wrong[level].append(a_mean)

    if n_ai_missing:
        print(f"NOTE: {n_ai_missing} row(s) have a human score but no valid AI score (parse error) -- "
              f"excluded from paired analysis.")

    n = len(quality_pairs)
    print(f"=== Kappa/QWK agreement (human vs. Gemini judge, n={n} paired responses) ===")
    if n == 0:
        print("No matched pairs yet -- nothing to compute.")
        return

    q_kappa = cohens_kappa(quality_pairs)
    q_qwk = quadratic_weighted_kappa(quality_pairs)
    a_kappa = cohens_kappa(adherence_pairs)
    a_qwk = quadratic_weighted_kappa(adherence_pairs)
    print(f"  quality:            Cohen's kappa={q_kappa:.3f}   QWK={q_qwk:.3f}")
    print(f"  strategy_adherence: Cohen's kappa={a_kappa:.3f}   QWK={a_qwk:.3f}")
    print("  (QWK is the more appropriate measure here -- ordinal 1-5 scale, penalizes "
          "far-apart disagreements more than adjacent ones.)")

    print(f"\n=== Per-level mean scores (human_weight={human_weight}) ===")
    ranked = sorted(per_level_combined.items(), key=lambda kv: -sum(kv[1]) / len(kv[1]))
    for level, vals in ranked:
        h_mean = sum(per_level_human[level]) / len(per_level_human[level])
        a_mean = sum(per_level_ai[level]) / len(per_level_ai[level])
        print(f"  level {level}: human_mean={h_mean:.3f}  ai_mean={a_mean:.3f}  "
              f"combined_mean={sum(vals) / len(vals):.3f}  (n={len(vals)})")
    if len(ranked) >= 2:
        print(f"\nTop 2 candidates for the assignment's xlsx step: level {ranked[0][0]}, level {ranked[1][0]}")

    print(f"\n=== Per-level breakdown, quality vs. strategy_adherence separately ===")
    print(f"{'level':>5} {'h_quality':>9} {'h_adherence':>11}   {'ai_quality':>10} {'ai_adherence':>12}")
    for level, _ in ranked:
        hq = per_level_human_quality[level]
        ha = per_level_human_adherence[level]
        aq = per_level_ai_quality[level]
        aa = per_level_ai_adherence[level]
        print(f"{level:>5} {sum(hq) / len(hq):>9.3f} {sum(ha) / len(ha):>11.3f}   "
              f"{sum(aq) / len(aq):>10.3f} {sum(aa) / len(aa):>12.3f}")

    print(f"\n=== Correct (matched signal) vs. wrong (mismatched signal), human AND AI ===")
    print(f"{'level':>5} {'human_correct':>13} {'human_wrong':>11} {'h_gap':>6}   "
          f"{'ai_correct':>10} {'ai_wrong':>8} {'ai_gap':>6}")
    for level in sorted(per_level_combined.keys(), key=int):
        hc = per_level_human_correct.get(level, [])
        hw = per_level_human_wrong.get(level, [])
        ac = per_level_ai_correct.get(level, [])
        aw = per_level_ai_wrong.get(level, [])
        hc_m = sum(hc) / len(hc) if hc else float("nan")
        hw_m = sum(hw) / len(hw) if hw else float("nan")
        ac_m = sum(ac) / len(ac) if ac else float("nan")
        aw_m = sum(aw) / len(aw) if aw else float("nan")
        h_gap = hc_m - hw_m if hc and hw else float("nan")
        a_gap = ac_m - aw_m if ac and aw else float("nan")
        print(f"{level:>5} {hc_m:>13.3f} {hw_m:>11.3f} {h_gap:>6.3f}   "
              f"{ac_m:>10.3f} {aw_m:>8.3f} {a_gap:>6.3f}")
    print("\n(gap > 0 means quality/adherence dropped when the requested strategy didn't "
          "match the conversation content. Comparing human_gap to ai_gap shows whether "
          "the AI judge and the human agree on WHETHER generation degrades under a wrong "
          "signal, not just on individual response scores.)")


def main():
    """CLI entry point: dispatches to parse / kappa.

    Called by: the __main__ guard at the bottom of the file.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("parse", help="Parse the hand-filled blind quiz into full32_human_scores.csv.")
    p_kappa = sub.add_parser("kappa", help="Cohen's kappa + QWK, human vs. Gemini judge.")
    p_kappa.add_argument("--human_weight", type=float, default=HUMAN_WEIGHT_DEFAULT)
    parser.add_argument("--dir", default=None, help="outputs/genai directory (default: repo_root/outputs/genai)")
    args = parser.parse_args()

    genai_dir = Path(args.dir) if args.dir else Path(__file__).resolve().parent.parent / "outputs" / "genai"
    genai_dir.mkdir(parents=True, exist_ok=True)

    if args.cmd == "parse":
        cmd_parse(genai_dir)
    else:
        cmd_kappa(genai_dir, args.human_weight)


if __name__ == "__main__":
    main()
