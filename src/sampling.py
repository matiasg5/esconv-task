"""
src/sampling.py

Builds every input CSV the GenAI runs consume, and assembles the 200-item
run's generations. Three steps of one sequence:

  hand32     : the 32-item hand-annotation sample for the required kappa/QWK
               check. Stratified 2-per-cell across 8 strategies x
               {correct, wrong} RoBERTa prediction (16 cells x 2 = 32), which
               guarantees every strategy appears on both sides of the
               correct/wrong split -- unlike plain proportional sampling (see
               report_notes.md for why). -> hand_annotation_sample.csv
  final200   : the 200-item production sample: those same 32 items plus 168
               new ones allocated proportionally by true_label frequency
               (largest-remainder rounding, so per-strategy counts sum to
               exactly 168). -> final_200_sample.csv + final_200_new168.csv
               (the second is just the 168 new rows, the input to generation --
               the 32 already have level-2/4 predicted-condition responses
               from the earlier pilot and are not regenerated).
  combine200 : stitches those reused 32-item pilot rows together with the
               freshly generated 168-item rows into one 400-row file for
               judging. -> final_200_prompt_comparison.csv

Everything is seeded (SEED=42) and deterministic: re-running any step
reproduces its file byte-for-byte.

Source of truth for both samples: outputs/8class/roberta/test_predictions.csv.

Usage:
    python sampling.py hand32
    python sampling.py final200
    python sampling.py combine200
"""

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

SEED = 42  # same seed used throughout the project for reproducibility

# --- hand32 ---------------------------------------------------------------
N_PER_CELL = 2  # 8 strategies * 2 correctness * 2 per cell = 32 total

# --- final200 -------------------------------------------------------------
N_NEW = 168

# --- combine200 -----------------------------------------------------------
LEVELS = [2, 4]
COMBINED_FIELDNAMES = [
    "idx", "level", "strategy", "context", "response", "word_count",
    "input_tokens", "output_tokens", "latency_seconds",
    "meta_commentary_stripped", "leading_header_stripped",
    "condition", "true_label", "pred_label", "correct",
]


def load_predictions(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(out_path, rows, fieldnames):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})


# --- hand32 ---------------------------------------------------------------

def select_hand32(rows, n_per_cell=N_PER_CELL, seed=SEED):
    by_cell = defaultdict(list)
    for row in rows:
        cell = (row["true_label"], row["correct"] == "True")
        by_cell[cell].append(row)

    rng = random.Random(seed)
    selected = []
    shortfalls = []
    for cell, group in sorted(by_cell.items()):
        strategy, is_correct = cell
        rng.shuffle(group)
        take = group[:n_per_cell]
        if len(take) < n_per_cell:
            shortfalls.append((strategy, is_correct, len(take)))
        selected.extend(take)
    return selected, shortfalls


def cmd_hand32(repo_root):
    rows = load_predictions(repo_root / "outputs" / "8class" / "roberta" / "test_predictions.csv")
    selected, shortfalls = select_hand32(rows)

    if shortfalls:
        print("WARNING: fewer than N_PER_CELL examples available for these cells:")
        for strategy, is_correct, n in shortfalls:
            print(f"  {strategy} ({'correct' if is_correct else 'wrong'}): only {n} available")
    else:
        print("All 16 cells fully filled -- no shortfalls.")

    target = N_PER_CELL * 8 * 2
    print(f"\nSelected {len(selected)} items (target={target})")
    by_strategy_correct = defaultdict(int)
    for row in selected:
        by_strategy_correct[(row["true_label"], row["correct"])] += 1
    for (strategy, correct), n in sorted(by_strategy_correct.items()):
        print(f"  {strategy:32s} {'correct' if correct == 'True' else 'wrong':8s} n={n}")

    out_dir = repo_root / "outputs" / "genai"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hand_annotation_sample.csv"
    write_rows(out_path, selected, ["idx", "context", "true_label", "pred_label", "correct", "confidence"])
    print(f"\nSaved: {out_path}")


# --- final200 -------------------------------------------------------------

def largest_remainder_allocation(counts_by_strategy, total_pool_size, n_new):
    """Proportional allocation of n_new across strategies, by each strategy's
    share of the pool, using largest-remainder rounding so the per-strategy
    counts sum to exactly n_new."""
    raw = {s: n_new * count / total_pool_size for s, count in counts_by_strategy.items()}
    floor_alloc = {s: int(v) for s, v in raw.items()}
    remainder = n_new - sum(floor_alloc.values())
    fractional = sorted(raw.items(), key=lambda kv: -(kv[1] - int(kv[1])))
    for i in range(remainder):
        s = fractional[i % len(fractional)][0]
        floor_alloc[s] += 1
    return floor_alloc


def select_new(rows, exclude_idx, n_new=N_NEW, seed=SEED):
    pool = [r for r in rows if r["idx"] not in exclude_idx]
    by_strategy = defaultdict(list)
    for r in pool:
        by_strategy[r["true_label"]].append(r)

    counts_by_strategy = {s: len(v) for s, v in by_strategy.items()}
    allocation = largest_remainder_allocation(counts_by_strategy, len(pool), n_new)

    rng = random.Random(seed)
    selected = []
    shortfalls = []
    for strategy, group in by_strategy.items():
        n_take = allocation.get(strategy, 0)
        rng.shuffle(group)
        take = group[:n_take]
        if len(take) < n_take:
            shortfalls.append((strategy, n_take, len(take)))
        selected.extend(take)
    return selected, allocation, shortfalls


def cmd_final200(repo_root):
    genai_dir = repo_root / "outputs" / "genai"
    rows = load_predictions(repo_root / "outputs" / "8class" / "roberta" / "test_predictions.csv")

    with open(genai_dir / "hand_annotation_sample.csv", newline="", encoding="utf-8") as f:
        existing_idx = {row["idx"] for row in csv.DictReader(f)}
    assert len(existing_idx) == 32, f"expected 32 existing items, found {len(existing_idx)}"

    new_items, allocation, shortfalls = select_new(rows, existing_idx)

    if shortfalls:
        print("WARNING: fewer than allocated examples available for these strategies:")
        for strategy, wanted, got in shortfalls:
            print(f"  {strategy}: wanted {wanted}, got {got}")

    print(f"Proportional allocation for the {N_NEW} new items:")
    for strategy, n in sorted(allocation.items(), key=lambda kv: -kv[1]):
        print(f"  {strategy:32s} n={n}")
    print(f"Total new: {sum(allocation.values())} (target={N_NEW})")

    existing_rows = [r for r in rows if r["idx"] in existing_idx]
    assert len(existing_rows) == 32

    all_rows = [{**r, "hand_scored": True} for r in existing_rows]
    all_rows += [{**r, "hand_scored": False} for r in new_items]

    assert len(all_rows) == 200, f"expected 200 total, got {len(all_rows)}"
    assert len({r["idx"] for r in all_rows}) == 200, "duplicate idx found -- overlap bug"

    n_mismatched = sum(1 for r in all_rows if r["correct"] == "False")
    print(f"\nOf the 200 items, {n_mismatched} ({n_mismatched/200:.0%}) are predicted != gold "
          f"(the items that carry real predicted-vs-gold signal).")

    fieldnames = ["idx", "context", "true_label", "pred_label", "correct", "confidence", "hand_scored"]
    full_path = genai_dir / "final_200_sample.csv"
    new_path = genai_dir / "final_200_new168.csv"
    write_rows(full_path, all_rows, fieldnames)
    write_rows(new_path, [r for r in all_rows if not r["hand_scored"]], fieldnames)

    print(f"\nSaved: {full_path}  (all 200)")
    print(f"Saved: {new_path}  ({N_NEW} new items -- the generation input)")
    print(f"\nDesign: levels 2 and 4, predicted condition only -- 200 items x 2 levels = "
          f"400 generations. The 32 hand-scored items already have their level-2/4 "
          f"predicted-condition responses from the earlier pilot, so only the 168 new "
          f"items need fresh API calls.")
    print(f"\nNext:")
    print(f"  python generate_responses.py --csv final_200_new168.csv --levels 2,4 --out final_200_new168_prompt_comparison.csv")
    print(f"  python sampling.py combine200")
    print(f"  python judge.py --provider gemini --csv final_200_prompt_comparison.csv --out final_200_judge_scores_gemini.json")


# --- combine200 -----------------------------------------------------------

def cmd_combine200(repo_root):
    import pandas as pd

    genai_dir = repo_root / "outputs" / "genai"
    pilot_csv = genai_dir / "full32_prompt_comparison.csv"
    new_csv = genai_dir / "final_200_new168_prompt_comparison.csv"
    out_csv = genai_dir / "final_200_prompt_comparison.csv"

    if not new_csv.exists():
        raise SystemExit(
            f"{new_csv} not found -- generate the 168 new items first:\n"
            f"  python generate_responses.py --csv final_200_new168.csv "
            f"--levels 2,4 --out final_200_new168_prompt_comparison.csv"
        )

    pilot = pd.read_csv(pilot_csv)
    pilot = pilot[pilot["level"].isin(LEVELS)].copy()
    # The pilot predates the condition/pred_label/leading_header_stripped columns;
    # backfill them -- "strategy" in the pilot is always the RoBERTa prediction
    # (confirmed: strategy == true_label exactly where correct == True).
    pilot["condition"] = "predicted"
    pilot["pred_label"] = pilot["strategy"]
    if "leading_header_stripped" not in pilot.columns:
        pilot["leading_header_stripped"] = False
    pilot = pilot[COMBINED_FIELDNAMES]

    new = pd.read_csv(new_csv)
    new = new[new["level"].isin(LEVELS)].copy()
    new = new[new["condition"] == "predicted"]
    new = new[COMBINED_FIELDNAMES]

    assert len(pilot) == 32 * len(LEVELS), f"expected {32*len(LEVELS)} pilot rows, got {len(pilot)}"
    assert len(new) == N_NEW * len(LEVELS), f"expected {N_NEW*len(LEVELS)} new rows, got {len(new)}"
    assert set(pilot["idx"]).isdisjoint(set(new["idx"])), "overlap between pilot and new idx -- bug"

    combined = pd.concat([pilot, new], ignore_index=True).sort_values(["idx", "level"]).reset_index(drop=True)
    assert len(combined) == 400, f"expected 400 total rows, got {len(combined)}"
    assert combined["idx"].nunique() == 200, f"expected 200 unique idx, got {combined['idx'].nunique()}"

    combined.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    print(f"{len(combined)} rows total: {len(pilot)} reused from the 32-item pilot "
          f"+ {len(new)} freshly generated for the {N_NEW} new items.")
    print(f"Next: python judge.py --provider gemini --csv final_200_prompt_comparison.csv "
          f"--out final_200_judge_scores_gemini.json")


def main():
    parser = argparse.ArgumentParser(description="Build the GenAI runs' input CSVs.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("hand32", help="32-item hand-annotation sample (kappa/QWK check).")
    sub.add_parser("final200", help="200-item production sample + the 168-item generation input.")
    sub.add_parser("combine200", help="Stitch reused pilot rows + new rows into the 400-row judging input.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    if args.cmd == "hand32":
        cmd_hand32(repo_root)
    elif args.cmd == "final200":
        cmd_final200(repo_root)
    else:
        cmd_combine200(repo_root)


if __name__ == "__main__":
    main()
