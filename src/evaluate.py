"""
src/evaluate.py

Everything done with a trained checkpoint. All three steps read the same
artifacts -- the checkpoint and the test_predictions.csv the first step
writes -- so they live together:

  test        : the single official test-set touch for a checkpoint. Run once
                per checkpoint you want reported; don't use its output to pick
                between checkpoints (that's done on val). Inference only, CPU
                is fine. Writes test_predictions.csv, classification_report.txt
                and confusion_matrix.csv per checkpoint. idx in
                test_predictions.csv is range(len(test_examples)) -- results.py
                and the error-sample step below rely on reproducing that exact
                list to look up conversation_id/turn_index/problem_type by idx.
  interpret   : Integrated Gradients (primary, via captum) + Occlusion
                (leave-one-token-out cross-check, hand-rolled -- captum's
                Occlusion class targets continuous/image inputs). Not using raw
                attention as evidence; see report_notes.md for the citations.
  error-sample: pulls the hand-read error-analysis items required by the
                assignment -- items the model got wrong, for the user to read
                and diagnose by hand. Only PULLS candidates; it does not write
                the "your thoughts" column, that's the user's own read.

Usage:
    python evaluate.py test --task 8class --checkpoint ../outputs/8class/roberta/lr1e-5_wd0.01/best_model --name roberta --speaker_tags
    python evaluate.py interpret --task 8class --checkpoint ../outputs/8class/roberta/lr1e-5_wd0.01/best_model
    python evaluate.py error-sample [--n 10]
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, classification_report, confusion_matrix

# torch / transformers / captum are imported inside the two commands that need
# them (test, interpret), so `error-sample` -- which only reads CSVs -- runs
# without the deep-learning stack installed.

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import (
    load_and_clean,
    build_8class_examples,
    build_3class_examples,
    flatten_context,
    apply_conversation_level_split,
    STRATEGY_LABELS,
    COARSE_LABELS,
)


# ========================================================================
# --- from evaluate_checkpoints.py ---------------------------------
# ========================================================================

MAX_SEQ_LEN = 96   # must match training -- see train_roberta.py
BATCH_SIZE = 32    # inference only, no gradient memory -- larger batch than training is fine
SEED = 42          # must match training's split seed so "test" here is the same held-out set

TASKS = {
    "8class": {
        "build_examples": build_8class_examples,
        "labels": STRATEGY_LABELS,
        "label_key": "gold_strategy_id",
    },
    "3class": {
        "build_examples": build_3class_examples,
        "labels": COARSE_LABELS,
        "label_key": "gold_coarse_id",
    },
}


def _make_inference_dataset_cls():
    """Defined lazily: subclassing torch's Dataset requires torch at
    class-creation time, and error-sample must run without it. torch itself is
    imported here too, not in main_test -- a function-level import binds the
    name locally, so __getitem__ below can only see it via this scope.

    Called by: main_test().
    """
    import torch
    from torch.utils.data import Dataset

    class InferenceDataset(Dataset):
        """Feeds examples to the DataLoader for inference: renders each context
        to text once, tokenises one item at a time. include_speaker_tags is a
        parameter here (unlike training) because the DistilBERT checkpoints were
        trained untagged.

        Called by: main_test(), via _make_inference_dataset_cls().
        """

        def __init__(self, examples, label_key, tokenizer, max_length=MAX_SEQ_LEN, include_speaker_tags=False):
            """Renders and stores the context texts and labels; no tokenising yet.

            Called by: the DataLoader, once per InferenceDataset.
            """
            # must match how the checkpoint was trained -- pass --speaker_tags for RoBERTa's final retrain
            self.contexts = [flatten_context(ex["context"], include_speaker_tags=include_speaker_tags) for ex in examples]
            self.labels = [ex[label_key] for ex in examples]
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            """Number of examples -- what the DataLoader iterates over.

            Called by: the DataLoader, to size the epoch.
            """
            return len(self.contexts)

        def __getitem__(self, idx):
            """Tokenises one example to fixed-length (96,) tensors: input_ids,
            attention_mask and a scalar label.

            Called by: the DataLoader, once per item per pass.
            """
            enc = self.tokenizer(
                self.contexts[idx],
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "label": torch.tensor(self.labels[idx], dtype=torch.long),
            }

    return InferenceDataset


def main_test():
    """The official test-set evaluation for one checkpoint: rebuilds the same
    held-out split training used, runs a forward-only pass, and writes
    test_predictions.csv, classification_report.txt and confusion_matrix.csv.

    Called by: main(), via the "test" subcommand.
    """
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    InferenceDataset = _make_inference_dataset_cls()

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["8class", "3class"], required=True)
    parser.add_argument("--checkpoint", required=True, help="Path to a best_model/ directory (save_pretrained output).")
    parser.add_argument("--name", required=True, help="Label for this checkpoint, e.g. 'roberta' or 'distilbert'. Used in output paths.")
    parser.add_argument("--speaker_tags", action="store_true",
                         help="Use include_speaker_tags=True when flattening context. Must match how the "
                              "checkpoint was trained -- only RoBERTa's final retrain (lr1e-5_wd0.01, post "
                              "speaker-tag fix) needs this; every other checkpoint was trained without tags.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    task_cfg = TASKS[args.task]
    labels = task_cfg["labels"]
    label_key = task_cfg["label_key"]

    convs = load_and_clean()
    examples = task_cfg["build_examples"](convs)
    _, _, test_examples = apply_conversation_level_split(convs, examples, seed=SEED)
    print(f"Test set: {len(test_examples)} examples (task={args.task})")
    print(f"*** This touches the held-out test set for '{args.name}' on '{args.task}'. "
          f"This should be the ONLY time this checkpoint is evaluated on test. ***")
    print(f"include_speaker_tags={args.speaker_tags} -- must match training, see --speaker_tags help.")

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint).to(device)
    model.eval()

    ds = InferenceDataset(test_examples, label_key, tokenizer, include_speaker_tags=args.speaker_tags)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    all_preds, all_labels, all_confidences = [], [], []
    inference_start = time.time()  # wall-clock, forward-pass only -- for the report's cost/time table
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch = batch["label"]

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.softmax(logits, dim=-1)
            confs, preds = probs.max(dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels_batch.tolist())
            all_confidences.extend(confs.cpu().tolist())

    inference_seconds = time.time() - inference_start
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    print(f"\n=== {args.task} / {args.name} -- TEST macro-F1 = {macro_f1:.4f} ===")
    print(f"Inference wall-clock: {inference_seconds:.1f}s for {len(test_examples)} examples "
          f"({1000*inference_seconds/len(test_examples):.1f} ms/example, batch={BATCH_SIZE}, device={device}) "
          f"-- for the report's cost/time table.\n")
    report_str = classification_report(all_labels, all_preds, target_names=labels, zero_division=0)
    print(report_str)

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(labels))))

    output_dir = Path(__file__).resolve().parent.parent / "outputs" / args.task / args.name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-example predictions -- kept for the later hand-read error analysis,
    # so wrong examples can be pulled straight from this file rather than re-run.
    context_texts = [flatten_context(ex["context"], include_speaker_tags=args.speaker_tags) for ex in test_examples]
    df = pd.DataFrame({
        "idx": range(len(test_examples)),
        "context": context_texts,
        "true_label": [labels[i] for i in all_labels],
        "true_label_id": all_labels,
        "pred_label": [labels[i] for i in all_preds],
        "pred_label_id": all_preds,
        "correct": [t == p for t, p in zip(all_labels, all_preds)],
        "confidence": all_confidences,
    })
    df.to_csv(output_dir / "test_predictions.csv", index=False)

    with open(output_dir / "classification_report.txt", "w") as f:
        f.write(f"task={args.task}  checkpoint={args.checkpoint}  name={args.name}\n")
        f.write(f"test macro-F1 = {macro_f1:.4f}\n\n")
        f.write(report_str)

    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(output_dir / "confusion_matrix.csv")

    print(f"\nSaved: {output_dir / 'test_predictions.csv'}")
    print(f"Saved: {output_dir / 'classification_report.txt'}")
    print(f"Saved: {output_dir / 'confusion_matrix.csv'}")


# ========================================================================
# --- from interpretability_roberta.py -----------------------------
# ========================================================================

N_WRONG_EXAMPLES = 5    # per task -- same confident-wrong selection the error analysis will use
N_CORRECT_EXAMPLES = 3  # per task -- contrast set, confident AND correct
N_IG_STEPS = 50         # integrated gradients path resolution -- captum's own default



def _select_confident(task, correct, n):
    """Reads test_predictions.csv and returns the n most-confident rows that were
    either right or wrong, per `correct`. Shared by the two wrappers below.

    Called by: select_confident_correct(), select_confident_wrong().
    """
    path = Path(__file__).resolve().parent.parent / "outputs" / task / "roberta" / "test_predictions.csv"
    df = pd.read_csv(path)
    subset = df[df["correct"] == correct].sort_values("confidence", ascending=False)
    return subset.head(n).reset_index(drop=True)


def select_confident_wrong(task, n=N_WRONG_EXAMPLES):
    """The examples to attribute: confident mistakes, where the model committed.

    Called by: main_interpret().
    """
    return _select_confident(task, correct=False, n=n)


def select_confident_correct(task, n=N_CORRECT_EXAMPLES):
    """Contrast set: confident and correct, so error-specific patterns stand out.

    Called by: main_interpret().
    """
    return _select_confident(task, correct=True, n=n)


def build_baseline_input(input_ids, tokenizer):
    """All-pad baseline; special tokens (CLS/SEP/etc.) kept as-is.

    Called by: process_examples().
    """
    special_ids = set(tokenizer.all_special_ids)
    ref_ids = input_ids.clone()
    for i in range(input_ids.shape[1]):
        if input_ids[0, i].item() not in special_ids:
            ref_ids[0, i] = tokenizer.pad_token_id
    return ref_ids


def process_examples(examples, model, tokenizer, lig, labels, device):
    """Runs IG + occlusion for one group of examples. Returns (records,
    occlusion_summaries, viz_data) -- records feed captum's HTML renderer,
    viz_data is the plain-JSON version the notebook renders directly.

    Called by: main_interpret().
    """
    import torch
    from captum.attr import visualization as viz

    records = []
    occlusion_summaries = []
    viz_data = []

    for _, row in examples.iterrows():
        text = row["context"]
        true_label = row["true_label"]
        pred_label = row["pred_label"]
        pred_id = labels.index(pred_label)
        confidence = row["confidence"]

        enc = tokenizer(text, truncation=True, max_length=MAX_SEQ_LEN, padding="max_length", return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        ref_input_ids = build_baseline_input(input_ids, tokenizer).to(device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.softmax(logits, dim=-1)[0]

        # --- Integrated Gradients ---
        attributions, delta = lig.attribute(
            inputs=input_ids,
            baselines=ref_input_ids,
            additional_forward_args=(attention_mask,),
            target=pred_id,
            n_steps=N_IG_STEPS,
            return_convergence_delta=True,
        )
        attr_sum = attributions.sum(dim=-1).squeeze(0)
        attr_norm = attr_sum / (torch.norm(attr_sum) + 1e-12)

        tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu())
        seq_len = int(attention_mask[0].sum().item())  # real tokens only, drop padding

        # positional args (not keyword) -- captum's kwarg name for this has changed across versions
        record = viz.VisualizationDataRecord(
            attr_norm[:seq_len].cpu().detach().numpy(),  # word_attributions
            float(probs[pred_id]),                        # pred_prob
            pred_label,                                   # pred_class
            true_label,                                   # true_class
            pred_label,                                   # attr_class
            float(attr_norm[:seq_len].sum()),              # attr_score
            tokens[:seq_len],                              # raw_input_tokens / raw_input_ids
            float(delta),                                  # convergence_score
        )
        records.append(record)

        # --- Occlusion (leave-one-token-out), cross-check ---
        base_prob = float(probs[pred_id])
        drops = []
        for i in range(seq_len):
            if input_ids[0, i].item() in tokenizer.all_special_ids:
                continue
            occluded = input_ids.clone()
            occluded[0, i] = tokenizer.pad_token_id
            with torch.no_grad():
                occ_logits = model(input_ids=occluded, attention_mask=attention_mask).logits
                occ_prob = float(torch.softmax(occ_logits, dim=-1)[0, pred_id])
            drops.append((tokens[i], base_prob - occ_prob))
        drops.sort(key=lambda x: -x[1])  # largest drop first = most responsible for the prediction
        occlusion_summaries.append({
            "idx": row["idx"], "true_label": true_label, "pred_label": pred_label,
            "confidence": confidence, "top_occlusion_tokens": drops[:8],
        })

        viz_data.append({
            "idx": int(row["idx"]),
            "true_label": true_label,
            "pred_label": pred_label,
            "confidence": float(confidence),
            "pred_prob": float(probs[pred_id]),
            "convergence_delta": float(delta),
            "tokens": tokens[:seq_len],
            "ig_scores": [float(x) for x in attr_norm[:seq_len].cpu().detach().numpy()],
            "occlusion_top": [[tok, float(drop)] for tok, drop in drops[:8]],
        })

    return records, occlusion_summaries, viz_data


def occlusion_section_html(title, occlusion_summaries):
    """Renders the occlusion results as an HTML block -- per example, which tokens
    cost the most predicted probability when removed.

    Called by: main_interpret().
    """
    html = f"<h2>{title}</h2>"
    for s in occlusion_summaries:
        html += (
            f"<h4>idx={s['idx']} true={s['true_label']} pred={s['pred_label']} "
            f"(confidence={s['confidence']:.3f})</h4><ul>"
        )
        for tok, drop in s["top_occlusion_tokens"]:
            html += f"<li>{tok!r}: removing it drops P(pred class) by {drop:+.4f}</li>"
        html += "</ul>"
    return html


def main_interpret():
    """Token attribution for a checkpoint: Integrated Gradients (captum) plus a
    hand-rolled occlusion cross-check, over confident-wrong and confident-correct
    examples. Writes interpretability.html and interpretability_data.json.

    Called by: main(), via the "interpret" subcommand.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from captum.attr import LayerIntegratedGradients
    from captum.attr import visualization as viz   # used by the HTML writer below

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["8class", "3class"], required=True)
    parser.add_argument("--checkpoint", required=True, help="Path to a best_model/ directory (save_pretrained output).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_total = N_WRONG_EXAMPLES + N_CORRECT_EXAMPLES
    print(f"Device: {device} (CPU is fine here -- {n_total} examples, no training)")

    labels = TASKS[args.task]["labels"]

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint).to(device)
    model.eval()

    wrong_examples = select_confident_wrong(args.task)
    print(f"Selected {len(wrong_examples)} confident-but-WRONG examples for task={args.task}:")
    print(wrong_examples[["idx", "true_label", "pred_label", "confidence"]].to_string(index=False))

    correct_examples = select_confident_correct(args.task)
    print(f"\nSelected {len(correct_examples)} confident-and-CORRECT examples for task={args.task}:")
    print(correct_examples[["idx", "true_label", "pred_label", "confidence"]].to_string(index=False))

    def forward_fn(input_ids, attention_mask):
        """Logits-only wrapper -- captum needs a plain tensor, not a HF output object.

        Called by: captum, inside LayerIntegratedGradients.attribute().
        """
        return model(input_ids=input_ids, attention_mask=attention_mask).logits

    lig = LayerIntegratedGradients(forward_fn, model.roberta.embeddings)

    wrong_records, wrong_occlusion, wrong_viz = process_examples(wrong_examples, model, tokenizer, lig, labels, device)
    correct_records, correct_occlusion, correct_viz = process_examples(correct_examples, model, tokenizer, lig, labels, device)

    output_dir = Path(__file__).resolve().parent.parent / "outputs" / args.task / "roberta"
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = output_dir / "interpretability_data.json"
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump({"task": args.task, "wrong": wrong_viz, "correct": correct_viz}, f, indent=1)
    print(f"\nSaved: {data_path}  (load this in the notebook -- no torch/captum needed to render it)")

    out_path = output_dir / "interpretability.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("<h1>Integrated Gradients -- confident but WRONG predictions</h1>")
        f.write(viz.visualize_text(wrong_records).data)
        f.write(occlusion_section_html(
            "Occlusion (leave-one-token-out) cross-check -- confident but WRONG",
            wrong_occlusion,
        ))
        f.write("<hr><h1>Integrated Gradients -- confident and CORRECT predictions (contrast set)</h1>")
        f.write(viz.visualize_text(correct_records).data)
        f.write(occlusion_section_html(
            "Occlusion (leave-one-token-out) cross-check -- confident and CORRECT",
            correct_occlusion,
        ))
    print(f"Saved: {out_path}  (bonus -- same numbers, pre-rendered, open directly in a browser)")


# ========================================================================
# --- from pull_error_analysis_sample.py ---------------------------
# ========================================================================

SPLIT_SEED = 42  # must match evaluate_checkpoints.py's SEED, which produced idx


def build_test_examples_by_idx():
    """Rebuilds the exact test example list so a CSV idx can be resolved back to
    conversation_id / turn_index / problem_type. idx is positional, so this must
    reproduce main_test's list exactly.

    Called by: main_errorsample().
    """
    convs = load_and_clean()
    examples = build_8class_examples(convs)
    _, _, test_examples = apply_conversation_level_split(convs, examples, seed=SPLIT_SEED)
    return test_examples


def load_predictions(csv_path):
    """Reads test_predictions.csv into a list of dicts.

    Called by: main_errorsample().
    """
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def select_by_confidence(wrong_sorted, n):
    """First-pass selection: the n most-confident wrong predictions. Kept
    because the report documents it -- and documents why it was replaced:
    it over-represents one systematic failure mode (see selection_bias).

    Called by: main_errorsample(), when --select confidence is passed.
    """
    return wrong_sorted[:n]


def select_diverse(wrong_sorted, n):
    """The selection actually used: highest-confidence wrong prediction per
    distinct (true_label, pred_label) pair, then the top n of those. Gives a
    genuinely diverse set of confusions instead of repeating the same
    closing-turn -> Others failure seven times.

    Called by: main_errorsample(), the default --select diverse.
    """
    best = {}
    for r in wrong_sorted:
        key = (r["true_label"], r["pred_label"])
        if key not in best:
            best[key] = r
    return sorted(best.values(), key=lambda r: -float(r["confidence"]))[:n]


def selection_bias(wrong_sorted, pred_label="Others"):
    """Quantifies the bias in pure top-N-by-confidence selection: the share of
    wrong predictions that predicted `pred_label`, within each confidence
    percentile vs. the overall base rate. Source of Appendix F's table.

    Called by: main_errorsample().
    """
    total = len(wrong_sorted)
    out = []
    for pct in (1, 5, 10, 25):
        k = max(1, int(total * pct / 100))   # floor, matching the reported percentiles
        head = wrong_sorted[:k]
        share = sum(1 for r in head if r["pred_label"] == pred_label) / len(head)
        out.append((pct, k, share))
    base = sum(1 for r in wrong_sorted if r["pred_label"] == pred_label) / total
    return out, base, total


def main_errorsample():
    """Pulls the wrong test items for the hand-read error analysis and writes them
    to error_analysis_sample.txt. Only selects candidates -- the diagnosis lines
    are left blank for the user to fill in.

    Called by: main(), via the "error-sample" subcommand.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument(
        "--select", choices=["diverse", "confidence"], default="diverse",
        help="diverse (default) = highest-confidence wrong item per distinct "
             "(true, predicted) pair, the set the report uses. confidence = the "
             "superseded first pass, top-n by confidence only.",
    )
    parser.add_argument(
        "--csv", default=None,
        help="Path to test_predictions.csv (default: outputs/8class/roberta/test_predictions.csv)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    csv_path = Path(args.csv) if args.csv else repo_root / "outputs" / "8class" / "roberta" / "test_predictions.csv"
    rows = load_predictions(csv_path)
    test_examples = build_test_examples_by_idx()

    wrong = [r for r in rows if r["correct"] == "False"]
    wrong_sorted = sorted(wrong, key=lambda r: -float(r["confidence"]))
    top_n = (select_diverse if args.select == "diverse" else select_by_confidence)(wrong_sorted, args.n)

    bias_rows, base_rate, total_wrong = selection_bias(wrong_sorted)
    print(f"{len(wrong)}/{len(rows)} test predictions are wrong.")
    print("Selection-bias check -- share of wrong predictions where the model said "
          "'Others', by confidence percentile:")
    for pct, k, share in bias_rows:
        print(f"  top {pct:>2}% (n={k:>4}): {share:.1%}")
    print(f"  all {total_wrong} wrong:  {base_rate:.1%}  (base rate)")
    print(f"\nSelection: {args.select}. {len(top_n)} items:\n")

    out_lines = []
    for i, r in enumerate(top_n, start=1):
        idx = int(r["idx"])
        ex = test_examples[idx]
        block = (
            f"=== Item {i}: idx={idx} conversation_id={ex['conversation_id']} "
            f"turn_index={ex['turn_index']} problem_type={ex['problem_type']} ===\n"
            f"  confidence: {float(r['confidence']):.4f}\n"
            f"  true (gold) strategy: {r['true_label']}\n"
            f"  predicted strategy:   {r['pred_label']}\n"
            f"  context:\n    {r['context']}\n"
            f"  your read (fill in by hand -- error vs. ground truth, and why you think it happened):\n"
            f"    ____\n"
        )
        print(block)
        out_lines.append(block)

    out_dir = repo_root / "outputs" / "genai"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = repo_root / "outputs" / "8class" / "roberta" / "error_analysis_sample.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"{len(wrong)}/{len(rows)} test predictions are wrong. "
                f"{len(top_n)} items, selection={args.select}.\n\n")
        f.writelines(out_lines)

    print(f"Saved: {out_path}")
    print("\nNext: read each item's context, fill in your own diagnosis for the '____' lines, "
          "and that becomes the report's error-analysis section.")


# ========================================================================
# --- dispatcher --------------------------------------------------------
# ========================================================================

COMMANDS = {
    "test": (main_test, "Official test-set evaluation for one checkpoint."),
    "interpret": (main_interpret, "Integrated Gradients + Occlusion attributions."),
    "error-sample": (main_errorsample, "Pull the hand-read error-analysis items."),
}


def main():
    """Subcommand dispatcher: prints usage if the command is missing or unknown,
    otherwise strips it off sys.argv so each step keeps its own argparse.

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
