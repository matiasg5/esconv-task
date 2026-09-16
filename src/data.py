"""
src/data.py

The data foundation for every experiment in this project: loading and
cleaning the raw ESConv dataset, flattening conversations into
(context, gold_strategy) examples, and the split protocol applied to them.
Preprocessing and splitting live together because they are never used apart
-- every consumer that needs a split also needs the examples it splits.

Preprocessing: see notebooks/00_explore_data.ipynb for the exploratory
analysis behind the context-window and label-mapping choices.

Splits: two grouping units, for the required split-protocol comparison --
conversation_level_split (chosen, no leakage) vs. random_utterance_split
(the leakage-prone comparison point, not for model selection). Ratios
80/10/10. Justification: notebooks/01_split_protocol.ipynb.

Also carries the whole-dataset invariant checks for this module (all 1,300
conversations, not a sample) -- run before every retrain that touches it:
(1) no cross-conversation context mixing, (2) turn_index correctness, (3) no
future leakage, (4) context matches the documented rule (reconstructed
independently from raw data), (5) empty context only at true conversation-
openers, (6) speaker-tagged flatten_context correctness, (7) no
conversation_id overlap across train/val/test, (8) 3-class label integrity.

Usage:
    python data.py           # summary: example counts + split label distribution
    python data.py --test    # invariant checks; exits non-zero if any fail
"""

import argparse
import itertools
import json
import random
import sys
from collections import Counter
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "ESConv.json"

TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
TEST_FRAC = 0.1

# --- Known data-quality fixes ---------------------------------------------

PROBLEM_TYPE_MERGES = {
    # 'conflict with parents' and 'Issues with Parents' overlap in content
    # (e.g. both contain a "grounded by parents" situation) with no
    # categorical distinction found on manual read.
    "conflict with parents": "Issues with Parents",
}

# --- Label vocabularies -----------------------------------------------------

STRATEGY_LABELS = sorted([
    "Question",
    "Others",
    "Providing Suggestions",
    "Affirmation and Reassurance",
    "Self-disclosure",
    "Reflection of feelings",
    "Information",
    "Restatement or Paraphrasing",
])
STRATEGY_TO_ID = {label: i for i, label in enumerate(STRATEGY_LABELS)}

COARSE_LABELS = ["Action", "Comforting", "Exploration"]
COARSE_TO_ID = {label: i for i, label in enumerate(COARSE_LABELS)}

# 3-class mapping. "Others" excluded (heterogeneous catch-all -- see
# 00_explore_data.ipynb). Ambiguous strategies resolved via mean normalized
# turn position, see notebook.
STRATEGY_TO_COARSE = {
    "Question": "Exploration",
    "Restatement or Paraphrasing": "Exploration",
    "Reflection of feelings": "Exploration",
    "Self-disclosure": "Comforting",
    "Affirmation and Reassurance": "Comforting",
    "Providing Suggestions": "Action",
    "Information": "Action",
}


# --- Loading & cleaning ------------------------------------------------------

def load_raw_conversations(path=DATA_PATH):
    """Reads ESConv.json off disk, unmodified. Separate from load_and_clean so the
    raw file is still reachable for comparison.

    Called by: load_and_clean().
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def normalize_problem_type(problem_type):
    """Applies PROBLEM_TYPE_MERGES; returns the value unchanged if not merged.

    Called by: load_and_clean().
    """
    return PROBLEM_TYPE_MERGES.get(problem_type, problem_type)


def load_and_clean(path=DATA_PATH):
    """Load raw conversations and apply known data-quality fixes.

    Called by: run_invariants(), summary(), evaluate.py build_test_examples_by_idx(), evaluate.py main_test(), results.py build_test_examples_by_idx(), train_roberta.py main().
    """
    conversations = load_raw_conversations(path)
    for conv in conversations:
        conv["problem_type"] = normalize_problem_type(conv["problem_type"])
    return conversations


# --- Turn structure & context construction -----------------------------------

def get_runs(dialog):
    """Segment a dialog into (speaker, turns) runs of consecutive same-speaker turns.

    Called by: build_examples(), run_invariants().
    """
    return [(speaker, list(group)) for speaker, group in itertools.groupby(dialog, key=lambda t: t["speaker"])]


def get_context_for_supporter_turn(runs, run_idx, pos_in_run):
    """Context = the preceding seeker run + this supporter's own earlier
    turn(s) in the current burst. Returns [] only for a conversation's very
    first turn being a supporter turn (dropped by build_examples).

    Called by: build_examples().
    """
    context = []
    if run_idx > 0:
        _, prev_group = runs[run_idx - 1]
        context.extend(prev_group)
    _, current_group = runs[run_idx]
    context.extend(current_group[:pos_in_run])
    return context


def flatten_context(context_turns, include_speaker_tags=True):
    """Render a list of turn dicts as a single text string for model input.

    Called by: find_duplicate_examples(), run_invariants(), evaluate.py __init__(), evaluate.py main_test(), train_roberta.py __init__().
    """
    if not include_speaker_tags:
        return " ".join(t["content"].strip() for t in context_turns)
    return " ".join(f"[{t['speaker']}] {t['content'].strip()}" for t in context_turns)


# --- Example construction -----------------------------------------------------

def build_examples(conversations):
    """One example per supporter turn with a defined strategy label:
    conversation_id, turn_index, context, gold_strategy, problem_type,
    emotion_type. Drops conversation-opening supporter turns (empty
    context, 3.7% of turns -- see 00_explore_data.ipynb).

    turn_index looked up via an identity-keyed index (id(turn)), not
    conv["dialog"].index(turn) -- see report_notes.md "AI was wrong" #4 for
    the bug that fixed (duplicate turns broke .index()).

    Called by: build_3class_examples(), build_8class_examples().
    """
    examples = []
    for conv_id, conv in enumerate(conversations):
        dialog = conv["dialog"]
        true_index_by_id = {id(t): i for i, t in enumerate(dialog)}
        runs = get_runs(dialog)
        for run_idx, (speaker, group) in enumerate(runs):
            if speaker != "supporter":
                continue
            for pos_in_run, turn in enumerate(group):
                strategy = turn["annotation"].get("strategy")
                if strategy is None:
                    continue
                context = get_context_for_supporter_turn(runs, run_idx, pos_in_run)
                if not context:
                    continue  # conversation-opening supporter turn, dropped
                examples.append({
                    "conversation_id": conv_id,
                    "turn_index": true_index_by_id[id(turn)],
                    "context": context,
                    "gold_strategy": strategy,
                    "gold_strategy_id": STRATEGY_TO_ID[strategy],
                    "problem_type": conv["problem_type"],
                    "emotion_type": conv["emotion_type"],
                })
    return examples


def build_8class_examples(conversations):
    """Alias for build_examples -- exists so both tasks have parallel entry points.

    Called by: run_invariants(), summary(), evaluate.py build_test_examples_by_idx(), results.py build_test_examples_by_idx().
    """
    return build_examples(conversations)


def build_3class_examples(conversations):
    """Same as build_examples, but 'Others' turns are excluded (see STRATEGY_TO_COARSE).

    Called by: run_invariants(), summary().
    """
    examples = [ex for ex in build_examples(conversations) if ex["gold_strategy"] in STRATEGY_TO_COARSE]
    for ex in examples:
        coarse = STRATEGY_TO_COARSE[ex["gold_strategy"]]
        ex["gold_coarse"] = coarse
        ex["gold_coarse_id"] = COARSE_TO_ID[coarse]
    return examples


# --- Duplicate detection -----------------------------------------------------

def find_duplicate_examples(examples):
    """Detects exact duplicate (context text, label) pairs across different
    conversations (e.g. generic replies like "Ok, take care"). Detection
    only, not automatic removal -- see report_notes.md for the inspection
    and the decision to retain them (low-information-content, not leakage).
    Returns {(context_text, gold_strategy): [examples]} for keys with >1 example.

    Called by: summary().
    """
    seen = {}
    for ex in examples:
        key = (flatten_context(ex["context"], include_speaker_tags=False), ex["gold_strategy"])
        seen.setdefault(key, []).append(ex)
    return {key: exs for key, exs in seen.items() if len(exs) > 1}


# --- Split protocol -----------------------------------------------------------

def conversation_level_split(conversations, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC, seed=42):
    """Splits whole conversations into train/val/test. Returns three sets of
    conversation_id (same convention as build_examples').

    Called by: apply_conversation_level_split().
    """
    n = len(conversations)
    indices = list(range(n))
    random.Random(seed).shuffle(indices)

    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_ids = set(indices[:n_train])
    val_ids = set(indices[n_train:n_train + n_val])
    test_ids = set(indices[n_train + n_val:])
    return train_ids, val_ids, test_ids


def filter_examples_by_conversation_ids(examples, conversation_ids):
    """Keeps only examples whose conversation is in the given id set. This is the
    line that makes split leakage structurally impossible.

    Called by: apply_conversation_level_split().
    """
    return [ex for ex in examples if ex["conversation_id"] in conversation_ids]


def apply_conversation_level_split(conversations, examples, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC, seed=42):
    """Convenience wrapper: split conversations, then filter a pre-built examples list by the result.

    Called by: run_invariants(), summary(), evaluate.py build_test_examples_by_idx(), evaluate.py main_test(), results.py build_test_examples_by_idx(), train_roberta.py main().
    """
    train_ids, val_ids, test_ids = conversation_level_split(conversations, train_frac, val_frac, seed)
    return (
        filter_examples_by_conversation_ids(examples, train_ids),
        filter_examples_by_conversation_ids(examples, val_ids),
        filter_examples_by_conversation_ids(examples, test_ids),
    )


def label_distribution(examples, label_key="gold_strategy"):
    """Normalized label distribution (%) for a list of examples.

    Called by: compare_split_label_distributions().
    """
    counts = Counter(ex[label_key] for ex in examples)
    total = sum(counts.values())
    return {label: 100 * count / total for label, count in counts.items()}


def compare_split_label_distributions(overall_examples, train, val, test, label_key="gold_strategy"):
    """Not stratified by design -- measures whether the per-split label
    distribution stays close to overall anyway. Returns {label: {overall,
    train, val, test} percentages}.

    Called by: summary().
    """
    overall_dist = label_distribution(overall_examples, label_key)
    train_dist = label_distribution(train, label_key)
    val_dist = label_distribution(val, label_key)
    test_dist = label_distribution(test, label_key)
    return {
        label: {
            "overall": overall_dist.get(label, 0.0),
            "train": train_dist.get(label, 0.0),
            "val": val_dist.get(label, 0.0),
            "test": test_dist.get(label, 0.0),
        }
        for label in overall_dist
    }


def random_utterance_split(examples, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC, seed=42):
    """Splits individual examples independently, ignoring conversation
    membership -- the leakage-prone comparison point, not for model selection.

    Called by: summary().
    """
    n = len(examples)
    indices = list(range(n))
    random.Random(seed).shuffle(indices)

    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train = [examples[i] for i in indices[:n_train]]
    val = [examples[i] for i in indices[n_train:n_train + n_val]]
    test = [examples[i] for i in indices[n_train + n_val:]]
    return train, val, test


# ========================================================================
# --- invariant checks (from test_pipeline_invariants.py) ----------------
# ========================================================================

FAILURES = []


def check(name, condition, detail=""):
    """Prints one PASS/FAIL line and records failures in FAILURES. Used instead of
    assert so every check runs, not just up to the first failure.

    Called by: run_invariants().
    """
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append((name, detail))


def run_invariants():
    """Runs the eight whole-dataset invariant checks (see the module docstring for
    what each verifies) and exits non-zero if any fail.

    Called by: main().
    """
    convs = load_and_clean()
    n_convs = len(convs)
    print(f"Loaded {n_convs} conversations.\n")

    ex8 = build_8class_examples(convs)
    ex3 = build_3class_examples(convs)
    print(f"8-class examples: {len(ex8)}   3-class examples: {len(ex3)}\n")

    # Per-conversation id-sets of turns actually belonging to that conversation.
    conv_turn_ids = [set(id(t) for t in conv["dialog"]) for conv in convs]
    conv_true_index = [{id(t): i for i, t in enumerate(conv["dialog"])} for conv in convs]

    # --- Checks 1-3, 5: run per example, across the FULL 8-class set ----------
    cross_conv_violations = 0
    turn_index_mismatches = 0
    future_leakage_violations = 0
    n_checked = 0

    for ex in ex8:
        cid = ex["conversation_id"]
        n_checked += 1

        # Check 1: every context turn belongs (by identity) to THIS conversation.
        for ct in ex["context"]:
            if id(ct) not in conv_turn_ids[cid]:
                cross_conv_violations += 1
                break

        # Check 2: turn_index must point at a valid supporter turn whose strategy matches gold_strategy.
        dialog = convs[cid]["dialog"]
        ti = ex["turn_index"]
        if not (0 <= ti < len(dialog)):
            turn_index_mismatches += 1
        else:
            true_turn = dialog[ti]
            if true_turn["speaker"] != "supporter" or true_turn["annotation"].get("strategy") != ex["gold_strategy"]:
                turn_index_mismatches += 1

        # Check 3: no context turn's true position >= turn_index (no future leakage).
        for ct in ex["context"]:
            ct_true_idx = conv_true_index[cid].get(id(ct))
            if ct_true_idx is None or ct_true_idx >= ti:
                future_leakage_violations += 1
                break

    check("1. No cross-conversation context mixing (8-class, all examples)",
          cross_conv_violations == 0, f"{cross_conv_violations}/{n_checked} examples had a foreign-conversation context turn")
    check("2. turn_index correctness (8-class, all examples)",
          turn_index_mismatches == 0, f"{turn_index_mismatches}/{n_checked} examples had a turn_index not pointing at their own labeled turn")
    check("3. No future leakage (8-class, all examples)",
          future_leakage_violations == 0, f"{future_leakage_violations}/{n_checked} examples had a context turn at/after the labeled turn")

    # --- Check 4: context reconstruction from first principles ---------------
    context_mismatches = 0
    n_reconstructed = 0
    empty_context_conversation_openers = 0
    empty_context_not_openers = 0

    example_by_key = {(e["conversation_id"], e["turn_index"]): e for e in ex8}

    for conv_id, conv in enumerate(convs):
        dialog = conv["dialog"]
        runs = get_runs(dialog)
        for run_idx, (speaker, group) in enumerate(runs):
            if speaker != "supporter":
                continue
            for pos_in_run, turn in enumerate(group):
                strategy = turn["annotation"].get("strategy")
                if strategy is None:
                    continue
                n_reconstructed += 1

                # ground-truth expected context, built independently from raw runs
                expected = []
                if run_idx > 0:
                    prev_speaker, prev_group = runs[run_idx - 1]
                    assert prev_speaker == "seeker", "structural guarantee violated: run before a supporter run must be seeker"
                    expected.extend(prev_group)
                expected.extend(group[:pos_in_run])

                is_conversation_opener = (run_idx == 0 and pos_in_run == 0)
                if not expected:
                    if is_conversation_opener:
                        empty_context_conversation_openers += 1
                    else:
                        empty_context_not_openers += 1
                    continue  # matches build_examples' drop rule

                # Find the matching built example: conv_id + true turn_index.
                true_idx = conv_true_index[conv_id][id(turn)]
                match = example_by_key.get((conv_id, true_idx))
                if match is None:
                    context_mismatches += 1
                    continue
                actual_ids = [id(t) for t in match["context"]]
                expected_ids = [id(t) for t in expected]
                if actual_ids != expected_ids:
                    context_mismatches += 1

    check("4. Context construction matches documented rule exactly (all conversations, reconstructed independently)",
          context_mismatches == 0, f"{context_mismatches}/{n_reconstructed} labeled supporter turns had a context mismatch")
    check("5. Empty context occurs ONLY at true conversation-opening supporter turns",
          empty_context_not_openers == 0,
          f"{empty_context_not_openers} non-opening supporter turns had empty context (should be impossible); "
          f"{empty_context_conversation_openers} true openers correctly had empty context")

    # --- Check 6: speaker-tagged flatten_context -------------------------------
    tag_violations = 0
    n_tag_checked = 0
    for ex in ex8:
        n_tag_checked += 1
        tagged = flatten_context(ex["context"], include_speaker_tags=True)
        expected_tagged = " ".join(f"[{t['speaker']}] {t['content'].strip()}" for t in ex["context"])
        if tagged != expected_tagged:
            tag_violations += 1
    check("6. Speaker-tagged flatten_context correct (8-class, all examples)",
          tag_violations == 0, f"{tag_violations}/{n_tag_checked} examples had a tagging mismatch")

    # --- Check 7: split leakage --------------------------------------------------
    for name, examples in [("8-class", ex8), ("3-class", ex3)]:
        train, val, test = apply_conversation_level_split(convs, examples, seed=42)
        train_ids = set(e["conversation_id"] for e in train)
        val_ids = set(e["conversation_id"] for e in val)
        test_ids = set(e["conversation_id"] for e in test)
        overlap = (train_ids & val_ids) | (train_ids & test_ids) | (val_ids & test_ids)
        check(f"7. No conversation_id overlap across train/val/test ({name})",
              len(overlap) == 0, f"{len(overlap)} conversation_ids appear in more than one split")

    # --- Check 8: 3-class label integrity ---------------------------------------
    others_present = sum(1 for e in ex3 if e["gold_strategy"] == "Others")
    coarse_mismatches = sum(1 for e in ex3 if e["gold_coarse"] != STRATEGY_TO_COARSE[e["gold_strategy"]])
    check("8a. No 'Others' examples in 3-class set", others_present == 0, f"{others_present} 'Others' examples found")
    check("8b. gold_coarse matches STRATEGY_TO_COARSE for every 3-class example",
          coarse_mismatches == 0, f"{coarse_mismatches} mismatches")

    print()
    if FAILURES:
        print(f"=== {len(FAILURES)} CHECK(S) FAILED ===")
        for name, detail in FAILURES:
            print(f"  - {name}: {detail}")
        sys.exit(1)
    else:
        print(f"=== ALL CHECKS PASSED (across {n_convs} conversations, {len(ex8)} 8-class / {len(ex3)} 3-class examples) ===")


# ========================================================================
# --- entry points -------------------------------------------------------
# ========================================================================

def summary():
    """Example counts, duplicate check, and the split label distribution.

    Called by: main().
    """
    convs = load_and_clean()
    ex8 = build_8class_examples(convs)
    ex3 = build_3class_examples(convs)
    print(f"Conversations: {len(convs)}")
    print(f"8-class examples: {len(ex8)}")
    print(f"3-class examples: {len(ex3)} ({len(ex8) - len(ex3)} 'Others' turns excluded)")

    dupes = find_duplicate_examples(ex8)
    n_dupe_examples = sum(len(exs) for exs in dupes.values())
    print(f"Duplicate (context, label) keys: {len(dupes)}, covering {n_dupe_examples} examples")

    conv_train, conv_val, conv_test = apply_conversation_level_split(convs, ex8)
    utt_train, utt_val, utt_test = random_utterance_split(ex8)
    print("\nConversation-level split:", len(conv_train), len(conv_val), len(conv_test))
    print("Random-utterance split:  ", len(utt_train), len(utt_val), len(utt_test))

    print("\nLabel distribution check (conversation-level split), overall vs. train/val/test %:")
    dist = compare_split_label_distributions(ex8, conv_train, conv_val, conv_test)
    for label, pcts in sorted(dist.items(), key=lambda kv: -kv[1]["overall"]):
        print(f"  {label:32s} overall={pcts['overall']:5.1f}  train={pcts['train']:5.1f}  "
              f"val={pcts['val']:5.1f}  test={pcts['test']:5.1f}")


def main():
    """CLI entry point: --test runs the invariant checks, no flag runs the summary.

    Called by: the __main__ guard at the bottom of the file.
    """
    parser = argparse.ArgumentParser(description="ESConv data pipeline: examples + splits.")
    parser.add_argument("--test", action="store_true",
                        help="Run the whole-dataset invariant checks instead of the summary. "
                             "Exits non-zero if any check fails.")
    args = parser.parse_args()
    if args.test:
        run_invariants()
    else:
        summary()


if __name__ == "__main__":
    main()
