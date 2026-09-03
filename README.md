# ESConv: Emotional-Support Strategy Classification and GenAI Response Generation

Two subtasks on the ESConv corpus (Liu et al., 2021):

1. **Classification** — predicting the supporter's next support strategy from the
   preceding dialogue, as both an 8-class and a coarser 3-class problem
   (Exploration / Comforting / Action).
2. **Generative AI** — using an LLM to draft the supporter's response text
   conditioned on a target strategy, scored by an LLM judge and by human
   annotation, with the required kappa/QWK agreement check.

The full write-up is [`ESConv_Report_Matias_Guernik.pdf`](ESConv_Report_Matias_Guernik.pdf), included in this repository.

## Setup

```bash
python -m venv .venv && .venv/Scripts/activate     # Windows
pip install -r requirements.txt
```

Then place `ESConv.json` in `data/raw/` (not redistributed here).

Two environment variables are needed for the GenAI subtask only:
`ANTHROPIC_API_KEY` (generation) and `GEMINI_API_KEY` or `GOOGLE_API_KEY`
(judging).

## Layout

`src/` is seven modules, each covering one stage. Every module with more than
one job takes a subcommand; run any of them with no arguments for usage.

| Module | Purpose |
|---|---|
| `data.py` | Loads and cleans ESConv, builds the same-speaker-burst context window, produces 8-class / 3-class examples, and applies the split protocol (conversation-level, plus the random-utterance comparison point). `--test` runs eight whole-dataset invariant checks. |
| `train_roberta.py` | Fine-tunes DistilBERT / RoBERTa. `SEED = 42`, early stopping on validation macro-F1. |
| `evaluate.py` | `test` — the single official test-set touch per checkpoint. `interpret` — Integrated Gradients + a hand-rolled Occlusion cross-check. `error-sample` — pulls the hand-read error-analysis items. |
| `sampling.py` | `hand32` — the 32-item stratified sample for the kappa/QWK check. `final200` — the 200-item production sample. `combine200` — assembles the run's 400 generations for judging. |
| `genai.py` | `generate` — response generation across 4 prompt levels (Claude). `judge` — LLM-judge scoring (Gemini for every reported score; Claude for the same-model pilot only). |
| `human_eval.py` | `parse` — turns the hand-filled blind quiz into scores. `kappa` — Cohen's kappa + QWK, human vs. AI judge, implemented from scratch. |
| `results.py` | `breakdown` — per-level quality vs. strategy adherence. `xlsx` — builds the two required spreadsheet deliverables, written to `outputs/`. |

`notebooks/` holds the exploratory analysis (`00`), the split-protocol
comparison and its significance tests (`01`), and classifier evaluation
including the baselines and interpretability rendering (`02`).

## Pipeline order

```bash
cd src
python data.py --test                        # verify preprocessing invariants

python train_roberta.py --task 8class        # -> outputs/8class/roberta/.../best_model
python evaluate.py test --task 8class --checkpoint ../outputs/8class/roberta/lr1e-5_wd0.01/best_model --name roberta --speaker_tags
python evaluate.py interpret --task 8class --checkpoint ../outputs/8class/roberta/lr1e-5_wd0.01/best_model
python evaluate.py error-sample

python sampling.py hand32                    # 32-item kappa/QWK sample
python genai.py generate --csv hand_annotation_sample.csv --levels 1,2,3,4 --out full32_prompt_comparison.csv
python genai.py judge --provider gemini --csv full32_prompt_comparison.csv --out full32_judge_scores_gemini.json
python human_eval.py parse                   # after hand-scoring the blind quiz
python human_eval.py kappa

python sampling.py final200                  # 200-item production sample
python genai.py generate --csv final_200_new168.csv --levels 2,4 --out final_200_new168_prompt_comparison.csv
python sampling.py combine200
python genai.py judge --provider gemini --csv final_200_prompt_comparison.csv --out final_200_judge_scores_gemini.json
python results.py breakdown
python results.py xlsx                       # -> outputs/genai_final_level{2,4}.xlsx
```

## What is not in this repository

**Trained model weights** (`outputs/**/best_model/`). They are 268–499 MB each,
past GitHub's 100 MB per-file limit. Everything derived from them *is*
committed — `metrics.jsonl` for each sweep run, per-class classification
reports, confusion matrices, test predictions, and the interpretability
attributions — so every number in the report remains checkable without them.
To regenerate: `python train_roberta.py --task 8class` at `SEED = 42`
(~4 minutes per task on an RTX 4090).

**The raw dataset** (`data/raw/ESConv.json`), which is downloaded rather than
redistributed.

## Reproducibility

Every sampling and split step is seeded at 42 and deterministic: re-running
`data.py`, `sampling.py` or `human_eval.py parse` reproduces its outputs
byte-for-byte. Transformer results come from a single seed; the 20-seed TF-IDF
baseline (std 0.0136 macro-F1 on 3-class) is the reference for how much of a
gap is seed noise.
