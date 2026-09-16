"""
src/results.py

Turns a production run's judge scores into the report's numbers and the
assignment's required deliverables. Two steps, both reading the same run:

  breakdown : per-level quality vs. strategy_adherence, reported separately
              (judge.py prints only the blended mean, which can hide a level
              that wins on one dimension and loses on the other). This is the
              source of Section 4.4's per-level table.
  xlsx      : builds the 2 required xlsx files, one per winning prompt level,
              written to outputs/ (not outputs/genai/) as the deliverables.
              Spec columns plus a 'condition' column and a 'word_count'
              column, both added deliberately -- see report_notes.md.

conversation_id/turn_index/problem_type aren't in the sample CSVs; they're
recovered by re-running the same deterministic pipeline evaluate.py used to
assign idx (data.py's example building + apply_conversation_level_split(
seed=42) reproduces the same test_examples list, idx = position in it).
Generation rows and judge scores are joined POSITIONALLY, not by key -- see
report_notes.md for why idx+level+strategy alone isn't a safe key here.

Usage:
    python results.py breakdown                 # final_200 (default)
    python results.py breakdown --run final_100
    python results.py xlsx
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
# openpyxl is imported inside write_xlsx() -- only the xlsx subcommand needs it,
# so breakdown runs without it installed.

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import (
    load_and_clean,
    build_8class_examples,
    apply_conversation_level_split,
)


# ========================================================================
# --- from breakdown_quality_adherence.py --------------------------
# ========================================================================

def load_scores(json_path):
    """Reads the judge's JSON scores.

    Called by: main_breakdown().
    """
    with open(json_path) as f:
        return json.load(f)


def per_level_breakdown(scores):
    """Returns per_level_quality, per_level_adherence: {level: [scores]}.
    PARSE_ERROR rows excluded, not treated as 0.

    Called by: main_breakdown().
    """
    per_level_quality = defaultdict(list)
    per_level_adherence = defaultdict(list)
    n_excluded = 0
    for s in scores:
        if s.get("quality") is None or s.get("strategy_adherence") is None:
            n_excluded += 1
            continue
        per_level_quality[s["level"]].append(s["quality"])
        per_level_adherence[s["level"]].append(s["strategy_adherence"])
    return per_level_quality, per_level_adherence, n_excluded


def per_level_by_condition(gen_rows, scores):
    """Same breakdown, split by condition. Joined positionally (row i <->
    score i), not by an (idx, level) key -- that key isn't unique when a run
    carries both conditions, since both share idx+level for a turn (see
    report_notes.md for the real bug this caused/fixed).

    Called by: main_breakdown().
    """
    per = defaultdict(lambda: defaultdict(lambda: {"quality": [], "adherence": []}))
    for gen, s in zip(gen_rows, scores):
        if s.get("quality") is None or s.get("strategy_adherence") is None:
            continue
        per[s["level"]][gen.get("condition", "predicted")]["quality"].append(s["quality"])
        per[s["level"]][gen.get("condition", "predicted")]["adherence"].append(s["strategy_adherence"])
    return per


def main_breakdown():
    """breakdown subcommand: prints mean quality and strategy adherence per prompt
    level, split by condition when the run has more than one.

    Called by: main(), via the "breakdown" subcommand.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="final_200",
                        help="Run prefix under outputs/genai/ (default: final_200).")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    genai_dir = repo_root / "outputs" / "genai"
    scores_path = genai_dir / f"{args.run}_judge_scores_gemini.json"
    gen_csv_path = genai_dir / f"{args.run}_prompt_comparison.csv"

    if not scores_path.exists():
        raise FileNotFoundError(f"{scores_path} not found -- check --run")

    scores = load_scores(scores_path)
    per_level_quality, per_level_adherence, n_excluded = per_level_breakdown(scores)
    if n_excluded:
        print(f"NOTE: {n_excluded} response(s) excluded (parse error / missing score).")

    print(f"=== {args.run}: per-level breakdown, quality vs. strategy_adherence separately "
          f"(n={len(scores)} total, AI judge only) ===")
    print(f"{'level':>5} {'quality':>9} {'adherence':>11} {'n':>5}")
    ranked = sorted(
        per_level_quality.items(),
        key=lambda kv: -((sum(kv[1]) / len(kv[1])) + (sum(per_level_adherence[kv[0]]) / len(per_level_adherence[kv[0]]))) / 2,
    )
    for level, qvals in ranked:
        avals = per_level_adherence[level]
        print(f"{level:>5} {sum(qvals) / len(qvals):>9.3f} {sum(avals) / len(avals):>11.3f} {len(qvals):>5}")

    # --- Further split by condition, when the run carries more than one.
    # A single-condition run (predicted only) just reports that one column.
    if gen_csv_path.exists():
        with open(gen_csv_path, newline="", encoding="utf-8") as f:
            gen_rows = list(csv.DictReader(f))
        if len(gen_rows) == len(scores):
            n_mismatch = sum(
                1 for gen, s in zip(gen_rows, scores)
                if str(gen["idx"]) != str(s["idx"]) or str(gen["level"]) != str(s["level"])
            )
            if n_mismatch:
                print(f"\nNOTE: {n_mismatch} row(s) failed the positional idx+level join -- "
                      f"skipping the by-condition breakdown (files may be out of sync).")
            else:
                per = per_level_by_condition(gen_rows, scores)
                conditions = sorted({c for lvl in per.values() for c in lvl})
                if len(conditions) > 1:
                    print(f"\n=== Same breakdown, further split by condition ===")
                else:
                    print(f"\n=== Single-condition run ({conditions[0]}) ===")
                print(f"{'level':>5} {'condition':>10} {'quality':>9} {'adherence':>11} {'n':>5}")
                for level in sorted(per.keys(), key=int):
                    for condition in conditions:
                        d = per[level].get(condition)
                        if not d or not d["quality"]:
                            continue
                        qm = sum(d["quality"]) / len(d["quality"])
                        am = sum(d["adherence"]) / len(d["adherence"])
                        print(f"{level:>5} {condition:>10} {qm:>9.3f} {am:>11.3f} {len(d['quality']):>5}")
        else:
            print(f"\nNOTE: {gen_csv_path.name} has {len(gen_rows)} rows but scores has {len(scores)} -- "
                  f"skipping the by-condition breakdown (can't positionally align).")
    else:
        print(f"\nNOTE: {gen_csv_path.name} not found -- skipping the by-condition breakdown "
              f"(level-only breakdown above still valid).")


# ========================================================================
# --- from assemble_final_xlsx.py ----------------------------------
# ========================================================================

SPLIT_SEED = 42  # must match evaluate_checkpoints.py's SEED, which produced idx
LEVELS = ["2", "4"]

COLUMNS = [
    "conversation_id", "turn_index", "problem_type", "dialogue_context",
    "gold_strategy", "predicted_strategy", "condition", "generated_response",
    "word_count", "judged_quality", "judged_strategy_adherence",
]


def build_test_examples_by_idx():
    """Reproduces the exact test_examples list evaluate_checkpoints.py used to
    assign idx = range(len(test_examples)) -- see module docstring.

    Called by: main_xlsx().
    """
    convs = load_and_clean()
    examples = build_8class_examples(convs)
    _, _, test_examples = apply_conversation_level_split(convs, examples, seed=SPLIT_SEED)
    return test_examples


def load_generation_rows(csv_path):
    """Reads the generation CSV into a list of dicts, in file order -- the order the
    positional join depends on.

    Called by: main_xlsx().
    """
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_judge_scores(json_path):
    """Reads the judge scores in file order, to be joined positionally.

    Called by: main_xlsx().
    """
    with open(json_path) as f:
        return json.load(f)


def assemble_rows(gen_rows, judge_scores, test_examples):
    """Joins generations to judge scores by position (row i to score i) and looks up
    each item's conversation metadata by idx. Positional because (idx, level,
    strategy) isn't unique when both conditions are present.

    Called by: main_xlsx().
    """
    assert len(gen_rows) == len(judge_scores), (
        f"generation rows ({len(gen_rows)}) and judge scores ({len(judge_scores)}) "
        f"must be the same length for the positional join to be valid"
    )
    n_key_mismatches = 0
    records = []
    for gen, score in zip(gen_rows, judge_scores):
        if gen["idx"] != score["idx"] or gen["level"] != score["level"] or gen["strategy"] != score["strategy"]:
            n_key_mismatches += 1
        ex = test_examples[int(gen["idx"])]
        records.append({
            "level": gen["level"],
            "conversation_id": ex["conversation_id"],
            "turn_index": ex["turn_index"],
            "problem_type": ex["problem_type"],
            "dialogue_context": gen["context"],
            "gold_strategy": gen["true_label"],
            "predicted_strategy": gen["pred_label"],
            "condition": gen["condition"],
            "generated_response": gen["response"],
            "word_count": int(gen["word_count"]),
            "judged_quality": score.get("quality"),
            "judged_strategy_adherence": score.get("strategy_adherence"),
        })
    if n_key_mismatches:
        raise AssertionError(
            f"{n_key_mismatches} row(s) failed the idx+level+strategy positional-join "
            f"sanity check -- generation and judge files are not in matching order, "
            f"do not proceed until this is fixed."
        )
    return records


def write_xlsx(records, out_path):
    """Writes one deliverable spreadsheet and applies uniform Arial formatting with a
    bold header row.

    Called by: main_xlsx().
    """
    df = pd.DataFrame(records, columns=COLUMNS)
    df.to_excel(out_path, index=False, sheet_name="responses")

    from openpyxl import load_workbook
    from openpyxl.styles import Font

    wb = load_workbook(out_path)
    ws = wb["responses"]
    for row in ws.iter_rows():
        for cell in row:
            cell.font = Font(name="Arial", size=10, bold=(cell.row == 1))
    for col_cells in ws.columns:
        max_len = max(len(str(c.value)) if c.value is not None else 0 for c in col_cells)
        ws.column_dimensions[col_cells[0].column_letter].width = min(max(max_len + 2, 10), 60)
    ws.freeze_panes = "A2"
    wb.save(out_path)


def main_xlsx():
    """xlsx subcommand: builds the two required deliverables, one per prompt level,
    written to outputs/ itself rather than the genai working directory.

    Called by: main(), via the "xlsx" subcommand.
    """
    repo_root = Path(__file__).resolve().parent.parent
    genai_dir = repo_root / "outputs" / "genai"
    # The two required deliverables are written to outputs/ itself, not the
    # genai/ working directory, so they're the first thing visible there.
    deliverable_dir = repo_root / "outputs"

    gen_rows = load_generation_rows(genai_dir / "final_200_prompt_comparison.csv")
    judge_scores = load_judge_scores(genai_dir / "final_200_judge_scores_gemini.json")
    test_examples = build_test_examples_by_idx()

    all_records = assemble_rows(gen_rows, judge_scores, test_examples)
    print(f"Assembled {len(all_records)} rows, 0 idx+level+strategy mismatches (positional join verified).")

    over_cap = [r for r in all_records if r["word_count"] > 40]
    print(f"NOTE: {len(over_cap)}/{len(all_records)} responses exceed the 40-word cap "
          f"(left as generated, see word_count column) -- flag in report.")

    for level in LEVELS:
        # write_xlsx selects only COLUMNS and drops "level" -- don't del it here (shared dict objects)
        level_records = [r for r in all_records if r["level"] == level]
        assert len(level_records) == 200, f"expected 200 rows for level {level}, got {len(level_records)}"
        out_path = deliverable_dir / f"genai_final_level{level}.xlsx"
        write_xlsx(level_records, out_path)
        print(f"Saved: {out_path}  (n={len(level_records)}: 200 items, predicted-condition only)")


# ========================================================================
# --- dispatcher --------------------------------------------------------
# ========================================================================

COMMANDS = {
    "breakdown": (main_breakdown, "Per-level quality vs. adherence for a run."),
    "xlsx": (main_xlsx, "Build the 2 required xlsx deliverables."),
}


def main():
    """Subcommand dispatcher: prints usage on an unknown command, otherwise strips it
    off sys.argv so each step keeps its own argparse.

    Called by: the __main__ guard at the bottom of the file.
    """
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print("usage: python %s <command> [options]\n" % Path(__file__).name)
        print("commands:")
        for name, (_, help_) in COMMANDS.items():
            print(f"  {name:14s} {help_}")
        print("\nRun a command with --help for its own options.")
        sys.exit(2)
    cmd = sys.argv.pop(1)   # strip the subcommand; each step keeps its own argparse
    COMMANDS[cmd][0]()


if __name__ == "__main__":
    main()
