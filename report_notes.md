# Report working notes

Running scratchpad of evidenced findings, exact numbers, and citations gathered
while building this project. NOT report prose — write the actual report in
your own words from this. Purpose is to make sure nothing gets lost across a
long working session. Update as new results come in.

---

## Split protocol section (notebooks/01_split_protocol.ipynb)

Task: 3-class (Exploration/Comforting/Action), TF-IDF(max_features=20000,
ngram_range=(1,2)) + LogisticRegression(class_weight="balanced", random_state=42).
Chosen as a cheap baseline for this comparison only -- not the final classifier.

- Single seed=42: gap = +0.0124 (random-utterance F1=0.4341 vs conversation-level=0.4217).
  Bootstrap 95% CI [-0.0223, +0.0477], one-sided p=0.2425 -- not significant.
- 20 repeated seeds (paired, sign test + Wilcoxon): mean gap = +0.0008
  (conv mean=0.4293 std=0.0136, utt mean=0.4301 std=0.0117), 9/11
  positive/negative split, sign p=0.7483, Wilcoxon p=0.4204 -- null.
  (SUPERSEDED numbers, pre token_pattern fix -- see "TF-IDF tokenizer fix"
  section below: gap=+0.0221, 20-seed gap=+0.0019, exact 10/10 split,
  sign p=0.588, Wilcoxon p=0.337. Conclusion -- null, no significant
  leakage -- unchanged.)
- Conclusion: no significant leakage gap detected with this baseline. Does not
  overturn the conversation-level split choice (model-independent argument:
  90.6% of conversations have same-speaker-in-a-row turns, supporter context
  windows overlap heavily across turns -- see src/splits.py docstring).
- Citation used to correct an earlier draft's hedge ("baseline too weak to
  detect leakage"): Soltaniani & Ghafari, "From Data Leak to Secret Misses"
  (arXiv:2601.22946) -- Random Forest lost 26.97% MCC when duplicates were
  removed from a leaked secret-detection dataset, vs only 7.22% for a
  transformer (GraphCodeBERT). Low-capacity models can be MORE, not less,
  leakage-sensitive -- so "the baseline lacked capacity to find the gap" isn't
  the right explanation; more likely this dataset/task has little exploitable
  literal overlap between splits.

## TF-IDF+LogReg baseline, reusable numbers (3-class, same config as above)

- Single seed=42, conversation-level split: macro-F1 = 0.4217
- Mean over 20 seeds, conversation-level split: macro-F1 = 0.4293 (std 0.0136)

- Single seed=42, 8-class, official test split: macro-F1 = 0.2189
- Mean over 20 seeds, 8-class: macro-F1 = 0.2264 (std 0.0117)

(SUPERSEDED, pre token_pattern fix: 3-class 0.4138 / 0.4289 (0.0144),
8-class 0.2213 / 0.2267 (0.0114) -- see "TF-IDF tokenizer fix" section
below. No ranking against the tree/RoBERTa baselines changed.)

## Position-only decision tree baseline (notebooks/02_classifier_evaluation.ipynb)

Features (2, no text at all): `normalized_position` (turn_index /
conversation length) and `supporter_turn_ordinal` (how-many-th supporter
turn in the conversation). Purpose: measure how much of strategy choice is
explainable by conversational position alone -- a floor, not intended to
compete for best score.

**Methodology pivot (worth keeping for the report):** started with CART's
own pruning method, cost-complexity pruning via `ccp_alpha` selected on
argmax(val macro-F1) -- the originally planned approach. Found it doesn't
control tree size on this data: val macro-F1 kept marginally improving all
the way to a near-unpruned 3,390-leaf tree on 8-class (~4 examples/leaf,
pure memorization -- val=0.2004, test=0.1900, not a usable interpretable
baseline). Diagnosed why: with only 2 features (one continuous), finer
splits keep finding a little more val signal almost indefinitely, so
argmax-over-the-pruning-path never converges on a small tree. Fix: grid
search over `max_leaf_nodes` (explicit size cap) x `min_samples_leaf`
(statistical floor -- no leaf trusted on too few examples) instead, both
first-class sklearn hyperparameters, selected on val macro-F1. Cost of
capping size, since this baseline was never competing for best score
anyway: ~0.011 macro-F1 on 8-class test (0.1900 uncapped -> 0.1795 capped),
0.0 on 3-class (capped and uncapped tie at val=0.4515) -- negligible against
a model that trails the text baselines on 8-class regardless of size.

| Task | Selected hyperparameters | Depth / leaves | Val macro-F1 | Test macro-F1 |
|---|---|---|---|---|
| 8-class | max_leaf_nodes=20, min_samples_leaf=100 | 8 / 20 | 0.1771 | 0.1795 |
| 3-class | max_leaf_nodes=7, min_samples_leaf=20 | 3 / 7 | 0.4515 | 0.4418 |

(Numbers reflect the turn_index bugfix -- see "AI was wrong" #4. Effect was
negligible: 3-class test is bit-for-bit unchanged, 8-class test moved
0.1791->0.1795.)

Feature importances: `normalized_position` dominates both tasks (0.975 on
8-class, 0.918 on 3-class); `supporter_turn_ordinal` adds little once
position is available (the two are correlated), though it carries more
weight on 3-class (8.2%) than 8-class (2.5%).

**8-class:** trails the text-based approaches (0.1795 vs. TF-IDF 0.2189,
RoBERTa 0.2398). At only 20 leaves it never predicts 4 of the 8 classes at
all (Information, Reflection of feelings, Restatement or Paraphrasing,
Self-disclosure all get precision=recall=0) -- an honest limitation: these
four strategies aren't position-determined the way
Question/Others/Providing Suggestions/Affirmation and Reassurance partially
are, so with a small leaf budget they get folded entirely into whichever
majority-class prediction dominates their position range.

**3-class -- still a notable, report-worthy result, but the specific claim
below changed after the RoBERTa speaker-tag retrain (see that section):**
against the ORIGINAL (pre-tag-fix) RoBERTa checkpoint, the position-only
tree (0.4418 test) edged it out (0.4395 test). After RoBERTa was retrained
with the speaker-tag fix, RoBERTa reclaimed the lead (0.4463 test > 0.4418).
The tree still beats the TF-IDF baseline (0.4217 single-seed / 0.4293
20-seed mean) using zero text, and still supports the same underlying point:
the 8->3 coarse mapping was itself built around measured mean normalized
turn position (see `preprocessing.py` docstring: Exploration/Comforting/
Action were resolved into position-ordered bands), so conversational stage
alone getting within ~0.004 of a fine-tuned transformer reading the actual
text is real, meaningful evidence the mapping decision was sound -- it's
just no longer literally "beats RoBERTa." Worth stating both numbers plainly
in the report (pre-fix tree>RoBERTa, post-fix RoBERTa>tree) rather than only
reporting whichever one is currently true -- the reversal itself is evidence
the speaker-tag fix had a real, measurable effect, not a rounding change.

**Full ranking, all baselines/models, official test split (post RoBERTa
speaker-tag retrain -- see that section for the source numbers):**
- 8-class: RoBERTa 0.2398 > TF-IDF 0.2189 > position tree 0.1795 (DistilBERT
  comparison point: 0.2174, pending re-eval on tagged input)
- 3-class: RoBERTa 0.4463 > position tree 0.4418 > TF-IDF 20-seed mean
  0.4293 > DistilBERT comparison point 0.4269 (pending re-eval on tagged input)

## DistilBERT sweep results

Setup: full fine-tuning, distilbert-base-uncased, max_seq_len=96, batch=16,
class-weighted CrossEntropyLoss (sklearn "balanced" formula), AdamW,
weight_decay=0.01, linear warmup(10%)+decay schedule calibrated to
MAX_EPOCHS=12 (NOTE: known imprecision -- early stopping usually triggers
well before 12 epochs, so the LR never fully decays to the schedule's
intended floor by the time training stops; flagged but not fixed before this
sweep ran, for consistency across all 3 configs). Early stopping:
patience=2 epochs, min_delta=0.001 macro-F1. Conversation-level split, seed=42.
Trained on RunPod, RTX 4090.

### 8-class (train=14190ish, val/test conversation-level split)
| lr | best val macro-F1 | epochs run | stopped at |
|---|---|---|---|
| 3e-5 | **0.2312** (winner) | 5 | epoch 3 peak, stopped epoch 5 |
| 2e-5 | 0.2238 | 5 | epoch 3 peak, stopped epoch 5 |
| 5e-5 | 0.2194 | 6 | epoch 4 peak, stopped epoch 6 |

Winning checkpoint: outputs/8class/lr3e-5/best_model/

### 3-class (train=11672, val=1458, test=1417)
| lr | best val macro-F1 | epochs run | stopped at |
|---|---|---|---|
| 2e-5 | **0.4348** (winner) | 5 | epoch 3 peak, stopped epoch 5 |
| 5e-5 | 0.4311 | 3 | epoch 1 peak, stopped epoch 3 |
| 3e-5 | 0.4261 | 3 | epoch 1 peak, stopped epoch 3 |

Winning checkpoint: outputs/3class/lr2e-5/best_model/

Pattern on BOTH tasks, all 3 LRs: val macro-F1 peaks within 1-4 epochs, then
declines while val_loss climbs and train_loss keeps falling (classic
overfitting). Narrow spread across LRs (~0.012-0.019) -> LR is not the
binding constraint; capacity/regularization relative to dataset size
(~14k/~11.7k train examples, 66M trainable params) is the likely bottleneck.

3-class DistilBERT (0.4348) is only marginally above the TF-IDF+LogReg
baseline's 20-seed mean (0.4293, post tokenizer fix -- see "TF-IDF tokenizer
fix") -- essentially within noise range. Worth
stating honestly in the report: full fine-tuning may not meaningfully
outperform the much simpler/cheaper baseline on this task.

### External calibration -- published ESConv 8-class strategy-prediction results

Source: EmoDynamiX (Yang et al., NAACL 2025, arXiv:2408.08782), full 8-class
label set, official ESConv test split, 5-utterance context window:

| Model | Macro-F1 |
|---|---|
| ChatGPT (zero/few-shot) | 18.14 |
| RoBERTa (fine-tuned) | 25.04 |
| BART (fine-tuned) | 25.66 |
| LLaMA3-8B | 25.91 |
| TransESC (prior SOTA, specialized architecture) | 26.28 |
| EmoDynamiX (proposed) | 27.70 |

Our final model, RoBERTa (post speaker-tag-fix, test macro-F1=0.2398): 23.98
-- within ~1 point of the published RoBERTa baseline (25.04) on the same
backbone, official test split. Our DistilBERT: 23.12 (0.2312). Within ~2-3
points of RoBERTa/BART despite
being a smaller model with a much simpler training recipe (no commonsense
injection, no emotion-dynamics modeling like TransESC/EmoDynamiX use to gain
their last few points). Use this comparison directly in the report -- reframes
"our number looks low" into a calibrated, defensible result: 8-class ESConv
strategy prediction is genuinely hard even for specialized research systems.

## RoBERTa-base comparison (8-class) -- backbone swap, same task

Setup: identical recipe to DistilBERT (max_seq_len=96, batch=16, class-weighted
CrossEntropyLoss, AdamW, linear warmup(10%)+decay schedule, early stopping
patience=2/min_delta=0.001, conversation-level split, seed=42). Only the
backbone (roberta-base, ~125M params vs DistilBERT's ~66M) and, across runs
below, learning rate / weight_decay change. Trained on RunPod, RTX 4090.

| lr | weight_decay | best val macro-F1 | peak epoch | stopped at |
|---|---|---|---|---|
| 1e-5 | 0.01 | **0.2402 (new 8-class winner)** | 5 | epoch 5 peak, stopped epoch 7 |
| 1e-5 | 0.05 | 0.2379 | 4 | epoch 4 peak, stopped epoch 4 |
| 2e-5 | 0.01 | 0.2318 | 2 | epoch 2 peak, stopped epoch 4 |
| 3e-5 (DistilBERT-winning LR, reused unmodified) | 0.01 | 0.2110 | 1 | epoch 1 peak, stopped epoch 3 |

Checkpoint: outputs/8class/roberta/lr1e-5_wd0.01/best_model/

**Story, in order:** the first RoBERTa run reused DistilBERT's winning LR
(3e-5) unmodified -- a real confound, not a fair backbone comparison. It
underperformed DistilBERT (0.2110 vs 0.2312), peaking at epoch 1 and
degrading immediately after. Diagnosis at the time: RoBERTa has ~2x
DistilBERT's trainable params, and the already-established bottleneck for
this task (failed-experiment #1) is capacity vs. ~14k-example dataset size --
a bigger model at the same LR should overfit *faster*, matching what was
observed. Follow-up mini-sweep (lr in {2e-5, 1e-5}, plus wd=0.05 at 1e-5 to
test DistilBERT's best regularization value on the new backbone) confirms
this directly: as LR drops, the peak epoch pushes later (1 -> 2 -> 5) and the
best score rises monotonically (0.2110 -> 0.2318 -> 0.2402). At lr=1e-5,
RoBERTa finally overtakes DistilBERT. Adding weight_decay=0.05 on top of the
lr=1e-5 run did NOT help further (0.2379 < 0.2402) -- worth reporting
honestly rather than cherry-picking: the extra regularization was unneeded
once LR itself was correctly lowered for the larger backbone; combining both
fixes doesn't stack additively here.

**Official test-set result (first touch, pre speaker-tag-fix, see
evaluate_checkpoints.py): RoBERTa-base, 8-class, macro-F1 = 0.2413** (val
was 0.2402 -- consistent, no val-test gap of concern; also now 24.13 on the
EmoDynamiX external calibration table, essentially matching the published
RoBERTa baseline of 25.04). **SUPERSEDED** by the speaker-tag retrain (see
"RoBERTa speaker-tag retrain (final numbers)" section below) -- kept here
as the historical record of the pre-fix result, not the reported number.

**Decision: RoBERTa-base (lr=1e-5, wd=0.01, macro-F1=0.2402) is now the
final 8-class model**, superseding DistilBERT (0.2312). Still within ~0.6
points of the published RoBERTa baseline (25.04, EmoDynamiX paper, official
test split) -- our val-set number lands almost exactly where the literature
would predict, which is a strong external sanity check on the whole pipeline.
(Note: at the time this decision was written, 3-class RoBERTa had not been
run yet -- that changed almost immediately after, see the "RoBERTa-base
comparison (3-class)" section directly below. Left this sentence's original
scope-decision reasoning intact rather than deleting it, since it's still
accurate as a record of the thinking at that point in the process; just
flagging it's since been superseded by the two RoBERTa 3-class runs and the
later speaker-tag retrain.)

**Report framing note:** the naive first RoBERTa run (reusing DistilBERT's
LR, underperforming) is itself a legitimate 3rd failed-experiment candidate
if the section wants more depth -- "hyperparameters don't transfer across
backbones even in the same family/recipe" is a genuine, evidenced lesson,
distinct from failed-experiments #1 (LR sweep within one backbone) and #2
(regularization/schedule confound). Optional; #1 and #2 already satisfy the
>=2 requirement.

## RoBERTa-base comparison (3-class) -- single follow-up run

One run only (not a sweep): lr=1e-5, wd=0.01 -- the recipe that won on
8-class, applied directly to 3-class rather than re-tuning from scratch
(deliberate scope decision, see prior turn). Same setup otherwise
(max_seq_len=96, batch=16, class-weighted loss, linear schedule, early
stopping patience=2/min_delta=0.001, conversation-level split, seed=42).

| lr | weight_decay | best val macro-F1 | peak epoch | stopped at |
|---|---|---|---|---|
| 1e-5 | 0.01 | **0.4409** | 3 | epoch 3 peak, stopped epoch 5 |

Checkpoint: outputs/3class/roberta/lr1e-5_wd0.01/best_model/

Compare: DistilBERT (lr=2e-5) = 0.4348. TF-IDF+LogReg baseline, 20-seed mean
= 0.4293 (std 0.0136, post tokenizer fix -- see "TF-IDF tokenizer fix").
RoBERTa edges out DistilBERT by +0.0061 -- a real
result from a single run, but SMALLER than the seed-to-seed noise already
measured on this task (std=0.0144 across 20 TF-IDF seeds) and not itself
tested across multiple seeds. Honest framing for the report: RoBERTa is the
best point estimate, but the DistilBERT-vs-RoBERTa gap on 3-class is not
distinguishable from noise with the runs available -- unlike the 8-class
case, where the RoBERTa-vs-DistilBERT gap (0.2402 vs 0.2312, +0.0090) is
comparable in size but backed by the LR-sensitivity story (monotonic
improvement 0.2110->0.2318->0.2402 as LR dropped), giving it more evidential
weight than a single 3-class data point can carry alone.

**Official test-set result (first touch, pre speaker-tag-fix, see
evaluate_checkpoints.py): RoBERTa-base, 3-class, macro-F1 = 0.4395** (val
was 0.4409 -- consistent, no val-test gap of concern). **SUPERSEDED** by the
speaker-tag retrain (see "RoBERTa speaker-tag retrain (final numbers)"
section below) -- kept here as the historical record of the pre-fix result,
not the reported number.

**Decision: RoBERTa-base (lr=1e-5, wd=0.01, macro-F1=0.4409) becomes the
3-class model**, on point-estimate grounds, superseding DistilBERT (0.4348)
-- but report BOTH numbers with the noise caveat above, rather than
presenting RoBERTa as a clear winner. This continues the pattern already
flagged for 3-class: full fine-tuning (whichever backbone) only marginally
outperforms the much simpler/cheaper TF-IDF+LogReg baseline on this task,
and that's a legitimate, defensible finding to state plainly rather than
downplay.

## RoBERTa speaker-tag retrain (final numbers)

Retrained both RoBERTa checkpoints (`train_roberta.py`, same lr=1e-5/wd=0.01
recipe, only `include_speaker_tags` changed False->True -- see "AI was
wrong" #3) after the migration to a new RunPod pod (original pod's GPUs
became unavailable; auto-migrated to a new one, RTX 4090). Ran
`src/test_pipeline_invariants.py` clean (all 1,300 conversations) as a gate
immediately before retraining.

Reproducibility note, worth keeping for the report: re-running the UNTAGGED
DistilBERT config on this new pod (before the tag change) reproduced
0.2240/0.4410 val vs. the originally recorded 0.2312/0.4348 -- a ~0.006-0.007
drift despite identical code/data/seed, attributable to GPU/driver/cuDNN
differences between pods rather than the code. A same-pod, same-config
REPEAT of the tagged 8-class DistilBERT run then reproduced bit-for-bit
identical results (0.2130 both times, identical per-epoch losses) --
confirming this specific pod's training IS deterministic given a fixed
seed, so the earlier drift really is a cross-machine effect, not per-run
noise. Evidence this matters for interpreting the RoBERTa numbers below:
they were NOT re-run on this pod before the tag fix, so there's no
same-pod baseline to isolate "tag effect" from "pod effect" for RoBERTa
specifically -- the val-score improvement is consistent with the earlier
TF-IDF tag ablation (+0.0029/+0.0125 macro-F1), which used no pod at all,
so the improvement direction is corroborated by an independent measurement;
the exact magnitude on this pod carries some of that same cross-machine
uncertainty.

| Task | Val (pre-fix) | Val (post-fix) | Test (pre-fix) | Test (post-fix) |
|---|---|---|---|---|
| 8-class | 0.2402 | 0.2493 | 0.2413 | 0.2398 |
| 3-class | 0.4409 | 0.4496 | 0.4395 | **0.4463** |

Checkpoints: `outputs/8class/roberta/lr1e-5_wd0.01/best_model/`,
`outputs/3class/roberta/lr1e-5_wd0.01/best_model/` (both overwritten in
place by the retrain).

**8-class:** val improved (+0.0091) but test moved slightly the other way
(-0.0015, essentially flat/noise) -- val is what selection is based on, and
a single test-set draw is noisy; not a concerning val-test gap, just a
reminder the tag fix's test-set effect on 8-class is not clearly
distinguishable from zero with one run.

**3-class:** both val (+0.0087) and test (+0.0068) improved, consistently
in the same direction -- the strongest single piece of evidence the
speaker-tag fix has a real, positive effect on RoBERTa. This also flips the
position-tree-vs-RoBERTa 3-class ranking (see "Position-only decision tree
baseline" section above) -- RoBERTa is back in front, 0.4463 vs 0.4418.

**This second official test-set touch for both RoBERTa checkpoints is
justified by the same standard applied throughout this project:** the
checkpoint materially changed due to a genuine, documented preprocessing
bug fix (missing speaker separation, see "AI was wrong" #3), not because
the first result was unsatisfactory. Both old and new numbers are recorded
above rather than only the new one, so the change is auditable.

## Per-class results & label-skew reporting (assignment requirement)

Assignment text (verbatim): "Report per-class results, not only aggregates
- the label distribution is heavily skewed and an aggregate number alone
will not be accepted." Added a dedicated section to
`notebooks/02_classifier_evaluation.ipynb` (RoBERTa, both tasks): a
true-vs-predicted-vs-uniform distribution comparison plot, and a per-class
precision/recall/F1/support table+chart. Real numbers:

8-class per-class F1 (sorted by support): Question 0.476 (n=334), Others
0.420 (n=307), Providing Suggestions 0.242 (n=283), Affirmation and
Reassurance 0.183 (n=267), Self-disclosure 0.156 (n=169), Reflection of
feelings 0.171 (n=142), Restatement or Paraphrasing 0.160 (n=126),
Information 0.109 (n=96).

3-class per-class F1: Exploration 0.531 (n=602, precision=0.634/recall=
0.457 -- conservative), Comforting 0.413 (n=436), Action 0.395 (n=379).

**Why aggregate alone is misleading (direct evidence):** 8-class accuracy
is 27%, weighted-F1 0.28 -- both inflated by Question+Others (36% of test
set) performing reasonably. Macro-F1 (0.2398) is dragged down by the
bottom four classes, all F1<0.20: Information, Restatement or Paraphrasing,
Self-disclosure, Reflection of feelings. Support size alone doesn't explain
this (Restatement/Self-disclosure/Reflection all have MORE support than
Information but are equally weak) -- more likely these three sit in a
semantically overlapping band (paraphrasing/reflecting/personal-anecdote
are all similar supportive-listening moves), genuinely harder to separate
from each other than from more distinctively-worded classes.

**Distribution finding:** true vs. predicted percentages checked directly
(not just eyeballed from the plot). 3-class cleanly shows RoBERTa's
predictions compressed toward uniform relative to the true skew (Action
26.7%->30.5% predicted, Comforting 30.8%->38.9%, Exploration 42.5%->30.6%)
-- consistent with the `class_weight="balanced"` loss weighting used in
training working as intended. 8-class mostly matches this pattern but has
one genuine exception: Restatement or Paraphrasing (true=7.3%,
predicted=3.6%) is a small class that got MORE suppressed, not pulled
toward uniform -- ties to it having the single lowest recall of any 8-class
label (0.119), i.e. the model essentially avoids predicting it rather than
over-calling it. Reported as a real exception, not smoothed into "the
pattern holds everywhere."

## TF-IDF tokenizer fix (apostrophe-splitting)

Found while reading the TF-IDF top-coefficient words for interpretability:
sklearn's `TfidfVectorizer` default `token_pattern` is `r"(?u)\b\w\w+\b"` --
`\w` doesn't include the apostrophe, so it's treated as a token boundary.
"we're" split into two separate tokens "we"/"re" (both kept, length>=2);
"that's" split into "that"/"s", and "s" was silently dropped (length<2,
filtered by the `\w\w+` requirement). Verified directly (not just inferred)
via `TfidfVectorizer().build_analyzer()("we're not sure that's the case")`
-> `['we', 're', 'not', 'sure', 'that', 'the', 'case', ...]`. Not a bug in
this project's code -- a default sklearn behavior -- but it made the
interpretability word lists misleading ("re" and "s" showing up as
independently "important" words when they're contraction fragments).

Contrast: RoBERTa's tokenizer (byte-level BPE, same scheme as GPT-2) never
drops characters -- every byte maps to some token by construction, so it
doesn't have this failure mode. It still splits contractions into subword
pieces (e.g. "we're" -> "we" + "'re"), same as it does for many whole
words, but nothing is silently lost the way "that's" -> "s" was.

Fix: `token_pattern=r"(?u)\b\w[\w']*\w\b"` -- keeps contractions whole,
still drops single-character tokens (matches the default's filtering
behavior otherwise). Scope decision: applied everywhere `TfidfVectorizer`
is used, not just the interpretability refit -- 5 call sites total
(`01_split_protocol.ipynb` cells 4/12, `02_classifier_evaluation.ipynb`
cells 13/17/33) -- since 2 of those are the actual reported TF-IDF+LogReg
baseline score, not just cosmetic interpretability output. Re-ran both
notebooks in full (`jupyter nbconvert --execute`), re-validated
(`nbformat.validate`). All old vs. new numbers logged inline above/below
where each was originally reported, old numbers marked SUPERSEDED rather
than removed. No ranking among baselines/models changed on either task;
the split-protocol conclusion (null result, no significant leakage) is
unchanged. Interpretability word lists are qualitatively unchanged, just
cleaner -- e.g. "we're" now shows up whole in Affirmation and Reassurance /
Comforting instead of as disconnected "we"/"re" fragments.

## TF-IDF interpretability (top coefficients)

Assignment requirement: "interpretability / explainability of the proposed
model's behavior." Structure: 3 methods matched to 3 model types --
decision tree gets a tree plot (already in the notebook), TF-IDF+LogReg
gets its own coefficients (this section, cheap since the coefficients ARE
the explanation for a linear model), RoBERTa gets Integrated Gradients +
Occlusion via captum (separate, needs a local run -- see "Still open").
Deliberately not using RoBERTa's raw attention weights as primary evidence
-- Jain & Wallace 2019 ("Attention is not Explanation", NAACL) vs Wiegreffe
& Pinter 2019 ("...is not not Explanation", EMNLP) is an active, unresolved
debate; gradient-based attribution (IG) has cleaner theoretical grounding
(completeness axiom) and sidesteps it.

Refit the same TF-IDF+LogReg pipeline (config/seed unchanged) and pulled
`clf.coef_` top-10 per class, both tasks. Two real findings:

1. **3-class top words track whichever 8-class sub-strategy has more
   training support, not an even blend of its components.** (Support
   numbers here are TRAIN-set counts, since that's what the coefficients
   are actually fit on -- train=14190 total, 8-class.) Action's top words
   (check, with that, usually, suggestions...) echo Providing Suggestions
   (train n=2359) far more than Information (train n=1008) -- "with
   that"/"usually" appear verbatim in both lists. Exploration's top words
   (hi, hello, how are) match Question (train n=2659) almost exactly, not
   Restatement or Paraphrasing (train n=866) or Reflection of feelings
   (train n=1172). Comforting blends both its components more evenly
   (helpful/we're from Affirmation and Reassurance, to this/have you from
   Self-disclosure) -- its two source classes are closer in train support
   (2242 vs 1366, ratio ~1.6x) than Action's (2359 vs 1008, ~2.3x) or
   Exploration's (2659 vs 866/1172, ~2.3-3.1x), consistent with "closer
   support -> more even blend."
2. **Many top words are conversational position markers (hi/hello/hey for
   Question, bye/thanks/goodbye for Others), not content words.** These are
   literally the greeting/closing phrases already catalogued in
   `00_explore_data.ipynb`'s opening/closing-phrase analysis. Directly ties
   to the position-tree baseline: TF-IDF is independently re-discovering
   the same positional structure through vocabulary, not a coincidence --
   worth stating in the report as convergent evidence across three
   different methods (position tree, TF-IDF coefficients, and the original
   greeting/closing lexical analysis) for the same underlying structure.

## Failed experiments section -- draft entries

**#1 (ready): Learning-rate tuning for DistilBERT fine-tuning.**
Swept lr in {5e-5, 3e-5, 2e-5} (BERT paper Appendix A.3 grid) for both tasks.
Result: narrow band on both (8-class 0.2194-0.2312, 3-class 0.4261-0.4348,
spread ~0.01-0.02) -- LR was not the binding constraint. Diagnosis: all
configs, both tasks, show the same fast-overfitting signature (val F1 peaks
epoch 1-4, val loss climbs while train loss keeps falling) regardless of LR --
points to model capacity vs. dataset size as the actual bottleneck.

**#2 (ready): weight_decay + layer-freezing grid to counter the overfitting from #1.**
Ran weight_decay in {0.01, 0.05, 0.1} x frozen_layers in {0, 2} at the 8-class
winning LR (3e-5), fixed schedule changed to constant-with-warmup (see below
for why). 6 configs, RTX 4090, RunPod.

| config | weight_decay | frozen_layers | best val macro-F1 |
|---|---|---|---|
| wd0.05_frz0 | 0.05 | 0 | 0.2267 |
| wd0.01_frz2 | 0.01 | 2 | 0.2257 |
| wd0.1_frz0 | 0.1 | 0 | 0.2233 |
| wd0.05_frz2 | 0.05 | 2 | 0.2219 |
| wd0.1_frz2 | 0.1 | 2 | 0.2191 |
| wd0.01_frz0 | 0.01 | 0 | 0.2190 |

Result: **none of the 6 configs beat the original winner (0.2312, lr=3e-5,
wd=0.01, frozen_layers=0, LINEAR schedule)**. Best here is 0.2267.

Diagnosis -- and this is the useful part, not just "it didn't work": this grid
changed two things at once relative to the original sweep -- the regularization
axes (weight_decay, frozen_layers) AND the LR schedule (switched from linear-
decay-to-12-epochs to constant-with-warmup, to fix a separate miscalibration
issue where early stopping usually fired well before epoch 12 so the linear
schedule never reached its intended low tail). Isolating the schedule's own
effect: `wd0.01_frz0` in this grid is the exact same weight_decay/freezing as
the original winner, only the schedule differs -- 0.2190 (constant) vs. 0.2312
(linear), a ~0.012 drop attributable to the schedule change ALONE. Raising
weight_decay to 0.05 recovered part of that (0.2190 -> 0.2267, +0.0077) but not
all of it. So: linear decay was doing real implicit-regularization work (LR
shrinking over training naturally curbs late-training overfitting) that
constant-with-warmup doesn't provide, and this confound -- not necessarily a
failure of weight_decay/freezing themselves -- is the most defensible
explanation for why nothing in this grid beat baseline. A cleaner follow-up
(not done, out of scope for this pass) would vary regularization and schedule
one at a time rather than together.

**Decision: kept the ORIGINAL checkpoint as the actual 8-class DistilBERT
model** (outputs/8class/lr3e-5/best_model/, macro-F1=0.2312) -- nothing in the
regularization grid beat it. Did NOT run the equivalent 3-class regularization
grid (deliberate scope decision, given #2 already produced a clear, reportable
result on 8-class and the marginal expected value of repeating it on 3-class
was low). 3-class DistilBERT model stays outputs/3class/lr2e-5/best_model/
(macro-F1=0.4348) from the original sweep, unchanged.

## GenAI generation artifact: word-count meta-commentary leak (mini-sample run 1)

Real first run of generate_responses.py (3 items x 4 levels, claude-haiku-4-5)
surfaced a genuine generation-quality artifact: idx=231, level 4 response
ended with literal trailing text "(14 words)" -- e.g. "...whatever you're
considering? (14 words)" -- despite every level's prompt explicitly saying
"Reply with ONLY the response text, nothing else." The model echoed its own
word count as visible text instead of just counting silently before
answering. Not treated as a one-off cosmetic issue -- fixed at the source:
added `strip_meta_commentary()` to generate_responses.py (regex matching a
trailing `(N word(s))` annotation, case-insensitive), applied to every
generated response before it's saved or used for word_count. Verified
against the actual real leaked string (strips cleanly to the intended
reply) AND against all 11 other real responses from this same run (zero
false positives -- nothing else gets modified). A new `meta_commentary_stripped`
boolean column is written per row in mini_sample_prompt_comparison.csv so
this stays auditable rather than silently invisible, and the run summary
line now reports how many responses needed stripping.

Also worth noting from this same run (a real finding, not a bug -- see
docstring's own stated hypothesis): idx=1317 (context is a closing
exchange, gold strategy = Providing Suggestions), level 3 correctly offers
a concrete suggestion ("setting a specific time tomorrow to check in with
yourself") while level 4 -- despite having the MOST scaffolding (guidance +
2 few-shot examples) -- gives a pure farewell with no suggestion at all.
Direct evidence the level-over-level design is genuinely not rigged toward
level 4 winning; more scaffolding did not guarantee better strategy
adherence on this item.

## GenAI judge artifact: markdown code-fence JSON parsing failure (mini-sample run 2)

Real second run (generate_responses.py + judge_mini_sample.py together) fixed
the word-count leak above, but surfaced a new real failure: 9 of the 12
judge calls (judge_mini_sample.py, same claude-haiku-4-5 model) came back
with the JSON reply wrapped in markdown code fences (` ```json\n{...}\n``` `)
despite the prompt explicitly saying "Reply with ONLY a JSON object, no
other text." A 75% failure rate even with the instruction present -- a
documented LLM behavior (models default to fencing code-shaped output),
not something an instruction alone reliably prevents. All 9 failures showed
`quality=None, strategy_adherence=None` with the raw fenced text preserved
in a `PARSE_ERROR: ...` string, so nothing was silently lost, but the run's
scores were unusable as-is.

Fixed at two levels, not just patched around: (1) strengthened the judge
prompt's instruction to explicitly forbid code fences ("no markdown code
fences (no ```), no commentary before or after"); (2) added defensive
parsing regardless -- `extract_json_object()` regex-strips a leading/trailing
` ``` ` or ` ```json ` fence before `json.loads()`, falling back to the raw
text when no fence is present, so a genuinely unfenced reply still parses
normally. Verified against the exact real failed string from this run
(idx=231 level=1) plus 3 more representative fenced variants (no language
tag, no trailing newline before the closing fence, trailing whitespace
after it) -- all recover the intended quality/strategy_adherence values --
and against 3 already-successful unfenced responses from the same run to
confirm no regression. Re-run of judge_mini_sample.py needed to regenerate
a clean mini_sample_judge_scores.json (current one on disk has 9/12 null
scores from before this fix).

Confirmed live on the real re-run: 12/12 judge calls parsed cleanly, zero
PARSE_ERROR entries, zero word-count-annotation strips on the regenerated
generations either. Both fixes hold under real conditions, not just the
offline regression tests.

## Finding: meta-acknowledgment openers correlate with less scaffolding (levels 1-2)

Real pattern noticed reading the mini-sample's 12 generations: level 1/2
responses often open with a meta-acknowledgment before answering --
"I appreciate the question, ...", "I'm here to listen. Would you like...",
"You're very welcome! Before you go, ..." -- while level 3/4 (which add the
few-shot examples) go straight into the conversational move itself, e.g.
level 4 idx=757: "What's been the hardest part of your day so far?" with no
preamble. Likely explanation: without a concrete example of what a real
chat turn looks like, the model falls back to its default assistant
register (acknowledge, then answer); the few-shot examples in levels 3-4
show it a directer conversational pattern instead.

Decision: NOT fixed at the generation-prompt level. Patching this directly
(e.g. "don't open with an acknowledgment") would suppress the very
difference the level comparison is designed to surface -- if thin-guidance
levels need an explicit anti-preamble rule to sound natural while few-shot
levels don't need it, that gap is itself evidence for which design works
better, not noise to remove. Also NOT added to the judge prompt in either
direction (neither an instruction to penalize nor one to ignore openers):
the existing "quality" criterion ("overall naturalness, appropriateness,
and helpfulness as a supportive chat reply") already blends naturalness in
as one read among three, with no rule singling out this specific pattern --
adding language either way would be an explicit thumb on the scale that
wasn't there before. Left as a logged finding, same treatment as the
idx=1317 farewell case: real evidence for how few-shot examples affect
generation, kept in the record rather than papered over.

Follow-up: the "quality" criterion's tone was previously undefined ("overall
naturalness" with no elaboration), which is itself a validity gap independent
of the opener question -- a vague criterion is harder for a judge to apply
consistently. Sharpened (judge_mini_sample.py build_judge_prompt()) with one
added clause: tone should read like one person messaging another in an
ongoing conversation, "in whatever phrasing achieves that -- this is not a
preference for shorter or longer replies, or for or against opening with an
acknowledgment." This defines the criterion without taking a side on the
opener pattern -- deliberately kept neutral so it doesn't reintroduce the
explicit-penalty question that was just ruled out above. No new scored field
added (stays quality + strategy_adherence, matching the assignment's required
xlsx columns); this only changes wording within the existing "quality" field.
Requires re-running judge_mini_sample.py (not generate_responses.py -- the
generation prompt is untouched) to get scores under the new definition;
mini_sample_judge_scores.json on disk still reflects the old, undefined
wording until that re-run happens.

Second follow-up (reopened, then resolved narrower than first proposed):
after further discussion, decided to add one tone word to the GENERATION
prompt after all, but scoped to preserve the level comparison rather than
the "add an anti-preamble rule" idea that was ruled out earlier. Levels 2,
3, and 4 already shared an identical persona phrase ("You are a warm,
professional supporter..."); that phrase is changed uniformly across all
three (same wording, same position) to "warm, professional, and
conversational". Level 1 is deliberately persona-free by design (bare
"You are a chat assistant.", isolating whether the model knows the
strategy unaided) and was explicitly left untouched -- adding a persona
phrase there for the first time would change what level 1 tests, not just
how it's worded. Because the new word is identical across 2/3/4 rather than
escalating, the level-to-level scaffolding comparison stays valid; this is
a deliberate, milder move than the "don't open with an acknowledgment"
rule considered and rejected earlier, and doesn't erase the opener-pattern
finding logged above -- levels 2/3/4 already differ from level 1 by having
a persona at all, so this doesn't introduce a new confound between them.
Verified: generate_responses.py compiles; level 1's system prompt string
confirmed unchanged; levels 2/3/4 confirmed to share the exact same new
phrase. Requires re-running generate_responses.py (not just
judge_mini_sample.py this time) to regenerate responses under the new
wording before the next judge/ranking pass.

## Mini-sample final result (after tone wording + CoT + fence fix, real run)

Real final run, under the current prompts (levels 2-4 "warm, professional,
and conversational"; judge with CoT + sharpened tone-definition + fence
fix): 12/12 generations clean (no leaks), 12/12 judge calls parsed cleanly
(no PARSE_ERROR). Human ranking (real, hand-done, blinded) + AI implied
ranking combined 65/35 (combine_mini_sample_scores.py):

  level 3: human_mean=0.443  ai_mean=0.890  combined_mean=0.600 (n=3)
  level 4: human_mean=0.443  ai_mean=0.667  combined_mean=0.522 (n=3)
  level 2: human_mean=0.780  ai_mean=0.000  combined_mean=0.506 (n=3)
  level 1: human_mean=0.333  ai_mean=0.443  combined_mean=0.372 (n=3)

Worth noting from the breakdown: the human ranking alone actually favors
level 2 (human_mean=0.780, well above level 3's 0.443), but the AI judge
strongly favors level 3 (ai_mean=0.890) and level 3 wins once the two are
combined 65/35. This is a real, visible example of the low Spearman
agreement above translating into an actual disagreement about which level
is best -- not just an abstract correlation number. combine_mini_sample_scores.py
now prints this human/AI breakdown per level by default (previously only
the combined mean was shown), so this kind of disagreement is visible
without re-deriving it by hand from the per-response table.

Winner: level 3

Human-vs-AI rank agreement: Spearman rho=0.067 averaged over the 3 items --
close to zero, i.e. at this n the human's full ranking and the AI judge's
implied ranking barely track each other item-by-item, even though the
AGGREGATE winner (level 3) still comes out ahead under the combined score.
This is a real, reportable finding, not swept under the rug: it's a
concrete instantiation of the self-preference/pointwise-scoring risk
flagged in the literature review (same-model judge, pointwise not
pairwise) -- worth citing directly when justifying why Gemini (a separate
model) is used as the judge for the larger 32-item run rather than reusing
Claude. n=3 items is far too small to trust the rho value itself (it's
reported as an early signal, per the script's own printed caveat), but the
qualitative direction -- low agreement -- is consistent with the concern
that motivated bringing in an independent judge in the first place.

Note: for idx=757, two prompt levels produced byte-identical response text
this run ("What's been the hardest part of your day so far?") -- not a
bug, just means the human ranking between those two options was decided
with no content difference to go on for that one pairwise slot.

## "AI was wrong" section -- instances so far

1. Misread a user-pasted example conversation as containing a casing
   inconsistency ("Reflection of Feelings" vs "Reflection of feelings") --
   on systematic recount (8 distinct raw label strings total), the first
   example never actually contained that string at all. Caught by re-checking
   against the actual data rather than trusting the first read.
2. Misread the ESConv paper's stage-diagram figure: initially described only
   2 strategies (Reflection of Feelings, Self-disclosure) as spanning
   multiple stages. User corrected precisely: Reflection of Feelings spans
   stages 1+2, Self-disclosure spans ALL 3 stages, Affirmation and
   Reassurance spans stages 2+3 -- 3 strategies with two different kinds of
   overlap, not 2. Corrected the 8->3 mapping justification paragraph
   accordingly after the correction.
3. Built the entire flattened-context pipeline (`flatten_context` calls in
   every training/eval script, all `include_speaker_tags=False`) with no
   separator or speaker marker at all between turns -- presented as correct,
   never flagged as a gap. A `[seeker]`/`[supporter]` tagging option existed
   in `flatten_context` (`include_speaker_tags=True`) but nothing used it.
   Caught by the user manually inspecting a raw X/y example (via a
   from-scratch code chunk, not something I proactively offered) and asking
   "isn't there supposed to be a separation between seeker and supporter?"
   -- a direct catch of an unflagged design gap, not something I identified
   myself. Verified the gap was real and measurable via a cheap TF-IDF
   ablation (tagged vs. untagged): +0.0029 macro-F1 on 8-class, +0.0125 on
   3-class from adding tags -- small but consistently positive, confirming
   this wasn't a false alarm. Fixed for RoBERTa's final checkpoints only
   (`train_roberta.py`'s `StrategyDataset`, now `include_speaker_tags=True`
   unconditionally); DistilBERT and the TF-IDF baseline were deliberately
   NOT retrained/rerun initially (later revisited -- see "Still open"), since
   they're kept only as comparison points, not the final model -- see
   "RoBERTa speaker-tag retrain (final numbers)" section below for the
   retrained numbers: 8-class test 0.2413->0.2398, 3-class test
   0.4395->0.4463.
4. Asked AI to self-audit `preprocessing.py`/`splits.py`/the notebooks for
   more mistakes of this kind. Found two, both verified against real data
   before being called bugs (not asserted from reading the code):
   - `build_examples` computed `turn_index` via `conv["dialog"].index(turn)`
     -- returns the position of the FIRST turn dict `==` to `turn`, wrong
     whenever a conversation has two turns with identical (speaker, content,
     annotation). Measured: 97/1,300 conversations (7.5%) contain such a
     duplicate; 100/18,376 labeled supporter turns (0.54%) got a wrong
     `turn_index`. This directly corrupts `normalized_position`, the feature
     that drives 92-98% of the decision-tree baseline's predictions (see
     "Position-only decision tree baseline" above) -- i.e. the same feature
     behind the "beats RoBERTa on 3-class" result. Fixed by looking up each
     turn's true position from an identity-keyed index (`id(turn)`) built
     once per conversation, instead of value-equality search. Re-ran both
     tasks after the fix: 3-class test score is bit-for-bit unchanged
     (0.4418 -- only 2/1,417 test examples were affected), 8-class test
     moved 0.1791->0.1795. Headline claims survive the fix.
   - `00_explore_data.ipynb` cell 14 referenced `first_strategy_counts`,
     `first_supporter_strategy`, `first_others_pct`, `last_others_pct` --
     none defined anywhere in the saved notebook (the cell that computed
     them had been deleted at some point without re-running/removing the
     cell that depended on it). Not reproducible: a clean Restart & Run All
     would `NameError`. The cached output (9.9% opening + 22.9% closing =
     32.8%) matched the "~33%" figure cited in `preprocessing.py`'s
     docstring justifying dropping "Others" from the 3-class mapping, so the
     underlying number was real -- fixed by restoring the missing
     computation cell and confirming the notebook now executes cleanly
     top-to-bottom, reproducing the same 25.5%/58.8%/9.9%/22.9%.
   Also explicitly checked (my request, after finding #3): could the context
   builder ever attach a turn from a DIFFERENT conversation to an example
   (e.g. the first supporter turn of one conversation picking up the last
   seeker turn of the previous one)? Wrote `src/test_pipeline_invariants.py`
   -- reconstructs the expected context for every one of 18,376 labeled
   supporter turns across all 1,300 conversations directly from raw
   `conv["dialog"]`, independent of `get_context_for_supporter_turn`'s own
   code, and asserts every context turn belongs (by Python object identity)
   to its own conversation. Ran clean: 0/17,693 cross-conversation
   violations. Sanity-checked the test itself isn't vacuous by reverting the
   turn_index fix in a throwaway copy and re-running -- it correctly failed
   (100 future-leakage violations, 193 context mismatches), confirming the
   suite actually catches bugs rather than passing by construction.
5. Report-drafting agent claimed RoBERTa interpretability (Integrated
   Gradients + Occlusion) was "designed and coded but never actually
   executed" and wrote the report section as an honest gap. Wrong: the real
   `interpretability_data.json` files (both tasks) existed on the user's
   machine the whole time -- the container's local mirror was just stale
   (never staged). Caught when the user pasted a screenshot of real rendered
   output from notebook 02. Fixed by staging the real files and correcting
   the report section with the actual data. Reinforces the standing
   local-sandbox-vs-real-machine distinction: an empty/missing file in the
   container is evidence the container's copy is stale, not evidence the
   user's real file doesn't exist.
6. Report-drafting agent wrote that "Hello" leads its next-highest occlusion
   token in idx=1624 by "roughly 1.6x" -- fabricated, not computed from the
   real `interpretability_data.json`. Caught by asking a fresh audit pass to
   re-verify every specific number against source data rather than trust
   prior drafting. Real ratio vs. the literal next-ranked token is ~1.13x
   (barely different from idx=757's own ~1.09x, i.e. not a real contrast);
   the honest fix uses the first genuine content-word competitor ("job",
   0.022, since the literal 2nd/3rd tokens "supp"/"orter" are wordpiece
   fragments of the "[supporter]" tag, not content) for a real ~2.2x margin.
   Fixed in the report. General lesson: a plausible-sounding specific number
   in AI-drafted prose still needs re-verification against the source file,
   even when the surrounding claim is directionally correct.

## Context-window design -- burst window vs. full history

`get_context_for_supporter_turn` uses a same-speaker-burst window (preceding
seeker run + this supporter's own earlier turns in the current burst), not
full conversation history back to turn 1. Real justification, now in the
report (Section 3.1): bounded input length, and 90.6% of conversations have
same-speaker-in-a-row turns (line ~27 above) so a burst captures the locally
relevant exchange without unbounded growth. External validation: the
original ESConv paper (Liu et al. 2021) also bounds context rather than
using full history -- "we cut each dialog into conversation pieces with 5
utterances, which contain one supporter's response and the preceding 4
utterances" (different windowing strategy, same bounded-context principle).
Named explicitly as an untested alternative for further work: full-history
context vs. the current burst window -- no comparative claim made, since it
was never tested.

## Generation methodology alignment with the original paper

Liu et al. 2021's own generation pipeline is two-stage: predict/designate a
strategy token first, then generate the response conditioned on it --
P(y~|x) = P([st]|x) * prod_i P(y_i | x, [st], y_<i) -- and they evaluate
exactly a predicted-vs-gold distinction as separate variants ("Joint"
predicts at inference, "Oracle" conditions on gold strategy, "Vanilla"
ignores strategy, "Random" uses a random one). This directly matches this
project's GenAI subtask design (`generate_responses.py --conditions
predicted,gold`) -- real, citable validation that the generation setup
mirrors established methodology from the paper the dataset comes from. Now
stated in the report's GenAI section (4.1).

## Still open / not yet done

- DONE: Position-only decision tree baseline -- both tasks. See "Position-only
  decision tree baseline" section above (max_leaf_nodes/min_samples_leaf grid,
  not ccp_alpha -- see methodology pivot note there).
- DONE: TF-IDF+LogReg baseline, 8-class -- see numbers above.
- DONE (superseded, see below): Single official test-set evaluation of the
  winning checkpoints, pre speaker-tag-fix: BOTH tasks are RoBERTa-base --
  8-class test=0.2413 (val was 0.2402), 3-class test=0.4395 (val was 0.4409).
  DistilBERT checkpoints evaluated on test too, kept as the reported "first
  attempt" comparison point (8-class test=0.2174, 3-class test=0.4269).
- DONE: RoBERTa retrained with the speaker-tag fix and re-evaluated on test
  (second, justified touch -- see "RoBERTa speaker-tag retrain (final
  numbers)" section). Final numbers: 8-class test=0.2398 (val=0.2493),
  3-class test=0.4463 (val=0.4496). This is now the reported result for both
  tasks, superseding the pre-fix numbers above.
- RESOLVED: explored retraining DistilBERT on tagged input too (on the new
  pod, post-migration) -- 8-class val=0.2130 (reproduced bit-for-bit on an
  exact repeat, confirming this specific pod trains deterministically and
  the result isn't noise), 3-class val=0.4397. Neither beat the untagged
  originals (0.2312/0.4348), so speaker tags appear to hurt DistilBERT
  specifically -- plausibly a capacity/sequence-length interaction (smaller
  model, same MAX_SEQ_LEN=96 budget, tags eat more of a tighter budget).
  Official test-set re-eval for these two was never run. DECISION: keep the
  ORIGINAL pre-migration, untagged DistilBERT numbers as the reported
  comparison point throughout (8-class val=0.2312/test=0.2174, 3-class
  val=0.4348/test=0.4269) -- fully tested, already the basis for the
  LR-sweep/failed-experiments/external-calibration sections above. The
  post-migration exploration (both tagged and untagged reruns) was useful
  for the reproducibility-noise finding (see "RoBERTa speaker-tag retrain
  (final numbers)" section) but isn't itself the reported number.
  CODE CLEANUP: `train_distilbert.py` and `train_distilbert_regularized.py`
  removed from the repo (end-of-project cleanup, reducing code surface area
  to what's still relevant to the final pipeline) -- the numbers/analysis
  above are unaffected and remain the full record; only the training code
  that already did its job is gone. Verified the local `outputs/8class/` and
  `outputs/3class/` checkpoint/eval folders (distilbert, lr3e-5, lr2e-5,
  regularized) are untouched originals from before the pod migration --
  recomputed macro-F1 directly from the saved `test_predictions.csv` files
  (0.2174 8-class, 0.4269 3-class) and confirmed they match the reported
  numbers exactly, so nothing local needed deleting. The post-migration
  DistilBERT retrain (tagged and untagged) only ever existed on the RunPod
  pod, never copied locally -- no local cleanup needed for it either.
- DONE: `turn_index` bug + notebook reproducibility bug found and fixed (see
  "AI was wrong" #4); pipeline correctness verified at full-dataset scale via
  `src/test_pipeline_invariants.py` (10 checks, all 1,300 conversations /
  17,693 examples, incl. no cross-conversation context leakage) -- all pass.
  Run this before the RoBERTa speaker-tag retrain and again after, as a
  no-regression gate.
- Interpretability: Integrated Gradients (primary) + Occlusion (cross-check),
  via captum, for RoBERTa on BOTH tasks now (same backbone/tokenizer for
  both, simpler than the earlier plan of two different backbones). Tree
  visualization for the decision tree baseline. Considered and deliberately
  not using raw attention as primary evidence -- cite Jain & Wallace 2019
  ("Attention is not Explanation", NAACL) vs Wiegreffe & Pinter 2019
  ("...is not not Explanation", EMNLP) for why.
- ~10-item hand-read error analysis: PULLED, not yet hand-read/written up
  (see "Error analysis -- scope, selection, and pulled sample" below).
- GenAI subtask: response generation under predicted vs gold strategy, xlsx
  output, LLM-judge, >=30 human-annotated items with kappa/QWK agreement,
  multiple prompting approaches, 2 best output files. IN PROGRESS.

## GenAI subtask -- plan and decisions

Re-read the assignment PDF directly (not from memory) before starting --
exact spec: for every test-set supporter turn, generate the response under
predicted AND gold strategy; output xlsx with conversation id, turn index,
problem type, dialogue context, gold strategy, predicted strategy, generated
response (<=40 words), judged quality score, judged strategy adherence;
LLM-judge scores quality+adherence, >=30 items hand-annotated for
kappa/QWK agreement (low agreement is a legitimate, reportable finding);
try multiple prompting approaches, submit the 2 best as separate output
files; plus a cost/latency table (tokens, cost, 100x-scale discussion).

Decisions (confirmed with user before building anything, since these are
real cost/scope calls):
- Generator + judge: Claude API, same model for both roles (not a
  different provider) -- self-preference-bias risk explicitly noted, to be
  stated in the report.
- Model: claude-haiku-4-5 ($1/$5 per MTok in/out as of Aug 2026 pricing --
  cheap/fast tier, reasonable given short (<=40 word) generation and
  judging tasks; also gives a clean answer to the assignment's own
  "100x scale" question -- already on the cheapest efficient tier).
- Scope: NOT the full 1,724-item 8-class test set (thousands of paid calls
  x2 conditions x multiple prompting approaches). Subsampled instead:
  100 items, stratified proportionally by the 8 strategy labels (a plain
  random sample of 100 would give ~5-6 examples of the rarest classes --
  stratifying keeps the same skew-awareness already applied throughout the
  classification part).
- Prompting: 4 candidate prompt designs at increasing levels of detail
  (minimal / +strategy definition+persona / +do-dont guidance / +one real
  TRAIN-set few-shot example per strategy), tested first on a cheap 3-item
  mini-sample (gold strategy only, 12 total calls) before committing to a
  larger pilot -- same pilot-then-scale pattern as the DL LR sweeps. Winners
  from this mini-test get run on a larger pilot, then the 2 best overall
  get run on the full 100-item sample as the final 2 required output files.

`src/generate_responses.py` (NEW): implements the 4 prompt levels + the
3-item mini-sample. STRATEGY_DEFINITIONS (one-line description per
strategy) and STRATEGY_GUIDANCE (do/don't per strategy) are original
one-liners, standard ESConv-taxonomy descriptions. FEWSHOT_EXAMPLES: one
real example per strategy, pulled directly from the TRAIN split (seed=42,
same split used throughout) -- never val/test, so no leakage into what's
being evaluated. Verified before delivery (sys.modules-stubbed anthropic,
same technique used for interpretability_roberta.py): every strategy used
in the mini-sample has a definition/guidance/few-shot entry, build_prompt()
produces well-formed prompts at all 4 levels for all 3 items, checked by
direct string inspection -- not yet run against the real API (needs the
user's key, run on their machine per the project's standing pattern for
anything cost/execution-bearing).

**Prompt-design revision round (post user-shared external proposal):**
user pasted an outside "improvement" suggestion for the 4 levels (XML tags,
starker naive baseline, absolute banned-phrase list). Evaluated critically
rather than adopted wholesale: agreed XML-tagged structure (docs.claude.com
best practice) and an explicit banned-phrase list are legitimate additions;
pushed back on the proposal's "widen the gap between levels"/"make the
delta unmistakable" framing -- designing the prompts toward a predetermined
outcome conflicts with the project's honest-reporting standard (see level
1-4 docstring: "deliberately NOT designed to guarantee a visible
level-over-level improvement"). Adopted:
- Level 1: stripped further to a starker naive baseline -- system prompt is
  now literally `"You are a chat assistant."` (was a supporter persona),
  no strategy definition, no XML.
- Levels 3-4: restructured with XML tags (`<conversation>`,
  `<strategy_directive>`, `<successful_example>`/`<context>`/
  `<supporter_reply>`), and `BANNED_PHRASES = ["I understand", "I'm sorry
  to hear that", "I hear you"]` is now inserted directly into the prompt
  text at levels 3+ (previously only documented, not enforced in-prompt).
  Level-3 structure+content confound (already flagged in the docstring)
  is unchanged and still explicitly disclosed, not resolved by this round.
- FEWSHOT_EXAMPLES improved for the two strategies flagged as too
  superficial: Question changed from a bare "Hello"/greeting exchange to a
  substantive one (financial-hardship context -> genuine clarifying
  question); Others changed from a one-word "bye" exchange to a closing
  remark that still does real conversational work (acknowledgment +
  reflection tied to context), both still pulled from TRAIN only (seed=42).
- Re-verified via the sys.modules-stub technique (stub `anthropic`): all
  MINI_SAMPLE strategies still have definition/guidance/few-shot entries,
  `build_prompt` produces well-formed XML at levels 3-4 for every strategy
  used, banned phrases confirmed present in the level 3-4 prompt text,
  full 8-strategy FEWSHOT_EXAMPLES coverage confirmed, printed all 4 levels
  for one strategy (Question) and eyeballed the structure. Still not run
  against the real API.

**Level 4 upgraded to 2-shot (user's own methodological call, not adopted
from the external proposal):** user asked whether one few-shot example is
enough, and whether there's an instruction against copying it too closely.
Answer given: one example risks "example anchoring" (the model reproducing
the demonstration's specific phrasing rather than generalizing the
strategy) -- a known risk for short-output few-shot prompts, not just
speculation. Presented 3 options (keep 1-shot + add instruction / upgrade
level 4 to 2-shot + instruction / add a new level 5 for 2-shot, keeping
1-4 untouched) via AskUserQuestion rather than deciding unilaterally, since
it changes what level 4 measures. User chose: upgrade level 4 to 2-shot.
- `FEWSHOT_EXAMPLES[strategy]` is now a list of 2 (context, response) pairs
  per strategy (was 1), pulled from TRAIN (seed=42), deliberately chosen to
  differ in length/register/topic from each other (e.g. Question: financial-
  hardship context vs. job-anxiety context; Others: a closing remark vs. a
  plain greeting) so the pair shows the strategy's range rather than one
  fixed template.
- `build_prompt` level 4 now emits two `<successful_example>` XML blocks,
  plus an explicit instruction: "do not copy either one's wording or
  sentence structure; express the strategy in your own words."
- Disclosed confound (documented in the script's docstring, same honesty
  standard as level 3's structure+content bundle): level 4 vs level 3 now
  isolates "any example vs none" AND "diversity of multiple examples" at
  once -- not cleanly separable with the current 4-level design.
- Incidental finding while sourcing examples: several TRAIN-set "Others"-
  labeled responses read like genuine Questions or Reflections (e.g. "Is it
  possible to reframe how you look at your clients' dire financial
  situations?" labeled Others) -- consistent with "Others" being a noisier
  catch-all label in the source annotation, not cherry-picked evidence of a
  new bug; not pursued further, just noted for context if it comes up.
- Re-verified via the sys.modules-stub technique: all 8 strategies have
  exactly 2 distinct examples, both appear verbatim in the level-4 prompt,
  the anti-mimicry instruction and BANNED_PHRASES are present in the level-4
  prompt text, levels 1-3 unaffected, XML tags well-formed (2 matched
  `<successful_example>` pairs). Still not run against the real API.

**Mini-sample judging: blinded human pick + AI judge, combined with a
weighted score.** User's question ("who judges the 4 levels?") surfaced
that nothing scored the 12 generations yet. Presented 3 options (human-only
/ AI-only pilot / both) via AskUserQuestion; user chose both, then specified
the exact mechanism themselves: a blinded multiple-choice quiz (order
shuffled per item so it's never level-1-first) where they pick one favorite
response per item, combined with the AI judge's score at a 65% human / 35%
AI weighting.

Built as two new scripts:
- `src/judge_mini_sample.py`: (1) groups the 12-row mini-sample CSV by idx,
  shuffles each item's 4 responses into options A-D (seeded per-idx so the
  order varies across items, not fixed to level order), writes a blinded
  quiz (`mini_sample_quiz.txt`, no level labels) plus the hidden A-D->level
  key (`mini_sample_blind_key.json`); (2) calls the same generator model as
  an LLM-judge, scoring each of the 12 responses independently on quality
  (1-5) and strategy_adherence (1-5) -- these are the exact two fields the
  assignment's judge output requires, so this doubles as an early pilot of
  that judge prompt before the 100-item run, not just a mini-sample-only
  artifact. Noted explicitly in the docstring: blinding the option ORDER
  protects the human pick from anchoring on prompt-level identity, but does
  NOT address the underlying self-preference-bias risk for the AI judge
  (same model judging its own generations) -- that risk stays as already
  flagged, unresolved by blinding.
- `src/combine_mini_sample_scores.py`: reads the blind key + judge scores +
  a human-picks file (you create after reading the quiz, `{"231": "B", ...}`
  format), computes per-response human_score (1.0 if it was your pick, else
  0.0) and ai_score (mean(quality, adherence) normalized 1-5 -> 0-1), then
  `combined = human_weight * human_score + (1 - human_weight) * ai_score`,
  averaged per level across the 3 items to rank all 4. `--human_weight`
  defaults to 0.65 (your number) but is a CLI override, not hardcoded --
  re-running at different weights and seeing whether the winner changes is
  itself a cheap, worthwhile robustness check to report.

Verified offline (no API needed for this part): confirmed the 4 levels are
always fully represented across each item's A-D options, confirmed no
individual option line in the quiz text reveals its source level (real
blinding, not just "the word doesn't appear anywhere"), confirmed
build_judge_prompt is well-formed. For the combiner, ran a synthetic
end-to-end test with human picks and AI scores deliberately set to *prefer
opposite levels* (human picks level 1 every time, AI judge scores level 4
highest every time): at human_weight=0.65 the combiner correctly picks
level 1 as winner; at human_weight=0.1 the winner flips to level 4 --
confirms the weighting parameter actually does what it claims, not just
that the script runs without error.

Not yet run against the real API (needs `judge_mini_sample.py --model
claude-haiku-4-5` on your machine, after `generate_responses.py` has
produced the mini-sample CSV). Next step after that: read
`mini_sample_quiz.txt`, write `mini_sample_human_picks.json`, run
`combine_mini_sample_scores.py`.

## GenAI subtask -- hand-annotation sample design (32 items, 8x2 stratified)

User's question ("how would examples distribute?") led to a real redesign of
the required >=30-item hand-annotation set. Proposed and adopted: stratify
by BOTH gold strategy (8) AND whether RoBERTa's 8-class prediction was
correct (2) -- 16 cells, 2 items each = 32 total. Chosen over plain
proportional stratification because several strategies have very low
RoBERTa recall (e.g. Restatement or Paraphrasing: only 15 correctly-predicted
examples in the ENTIRE 1,724-item test set -- pulled directly from
outputs/8class/roberta/test_predictions.csv on the real machine, not a
possibly-stale sandbox mirror). A plain proportional 32-100 item sample
would likely get 0-2 correctly-predicted examples for that class -- too few
to say anything about generation quality when the classifier got it right.
Real per-class correct/wrong counts (test set):

| Strategy | Correct | Wrong |
|---|---|---|
| Question | 154 | 180 |
| Others | 120 | 187 |
| Providing Suggestions | 64 | 219 |
| Affirmation and Reassurance | 40 | 227 |
| Self-disclosure | 33 | 136 |
| Reflection of feelings | 31 | 111 |
| Restatement or Paraphrasing | 15 | 111 |
| Information | 16 | 80 |

Considered n_per_cell=16 (255 items) and n_per_cell=32 (478 items) first --
cost was trivial either way (~$0.81 / ~$1.55 at claude-haiku-4-5 pricing,
computed from REAL token counts measured off actual build_prompt() output,
not guessed), but both far exceeded the originally-discussed ~100-item
scope and both required capping some cells (Restatement always capped at
its 15-item ceiling). User clarified the intended scale was much smaller:
16 or 32 TOTAL items across all 16 cells, sized specifically to be the
required >=30-item hand-annotation subset, not the full generation sample.
Chose 32 total (n_per_cell=2) -- meets the >=30 minimum on its own; 16
would not have.

**Important scope clarification, flagged before building:** this design
picks WHICH 32 conversations get hand-annotated -- it is NOT the same thing
as the mini-sample's blinded 1-4 RANKING (judge_mini_sample.py /
combine_mini_sample_scores.py), which is a separate, already-built step
used only to pick the winning prompt level(s). Ranking isn't the right task
for the kappa/QWK requirement -- Cohen's kappa / QWK need matched,
independently-given ABSOLUTE scores (quality 1-5, strategy_adherence 1-5,
same fields the AI judge produces) on the same items, not a relative
preference order. The actual hand-scoring step (and the "2 best prompting
approaches" xlsx output it depends on) is deliberately deferred until the
mini-sample winner(s) are chosen -- this script only locks in which 32
items will be used once that happens.

`src/select_hand_annotation_sample.py` (NEW): stratified sampler, seed=42,
reads outputs/8class/roberta/test_predictions.csv directly (idx, context,
true_label, pred_label, correct), writes outputs/genai/
hand_annotation_sample.csv (32 rows). Verified live against the real
predictions file on the user's machine (not just the sandbox mirror): all
16 cells filled exactly at n=2, zero shortfalls, output file confirmed
written. conversation_id/turn_index/problem_type (needed for the final
xlsx per the assignment spec) deliberately NOT joined yet -- explicitly
deferred to the full generation/xlsx-assembly step.

**Mini-sample judging upgraded from single-pick to full ranking.** User
asked for ranking (1-4, best to worst) instead of "pick one favorite" for
both sides -- more informative than a binary pick, and lets human/AI
agreement be measured directly (Spearman rho) instead of just win/loss.
- `judge_mini_sample.py`'s quiz now asks you to rank all 4 blinded options
  best-to-worst per item, not just name a favorite.
- `combine_mini_sample_scores.py` rewritten: your ranking gives each
  response a human_score = (4-rank)/3 (rank1->1.0, rank4->0.0); the AI's
  own implied ranking is derived by sorting each item's 4 responses by
  (quality+strategy_adherence) -- no extra API call needed, reuses the
  scores judge_mini_sample.py already produces -- giving an ai_score on
  the same 0-1 scale. Combined the same way as before (human_weight=0.65
  default, CLI-overridable). Also reports mean Spearman rank correlation
  between your ranking and the AI's per item, as an early, cheap signal of
  human/AI agreement -- explicitly NOT a substitute for the required
  >=30-item kappa/QWK check (that needs absolute scores on both sides, not
  rankings -- see the hand-annotation-sample section above), just a first
  look before spending the real annotation effort.
- Verified offline: quiz file confirmed to ask for a ranking, not a pick;
  full synthetic test with human and AI rankings set to be EXACT OPPOSITES
  on all 3 items -- Spearman correctly computes rho=-1.000 (perfect
  disagreement, not a near-zero or buggy value), the weighted combiner
  still correctly picks level 1 at human_weight=0.65 and flips to level 4
  at human_weight=0.1, confirming both the ranking math and the weighting
  behavior are correct under an adversarial, fully-worked-out case, not
  just a happy-path run.

**Judge prompt strengthened with chain-of-thought, backed by literature
check (not just intuition).** User asked whether the judge's criteria
(quality, strategy_adherence) were themselves research-backed. Checked
directly rather than assuming:
- Zheng et al. 2023 ("Judging LLM-as-a-Judge with MT-Bench and Chatbot
  Arena", NeurIPS, arXiv:2306.05685) -- MT-Bench's own single-answer
  grading is actually a SINGLE holistic 1-10 score ("please rate the
  response on a scale of 1 to 10... consider factors such as the
  helpfulness, relevance, accuracy, depth, creativity, and level of
  detail"), not split into named sub-dimensions. So our two-dimension,
  1-5 design is NOT drawn from this paper -- it's dictated by the
  assignment spec's own required xlsx fields ("judged quality score,
  judged strategy adherence"), not a literature-recommended format. Same
  paper names the three biases already relevant to this project's design:
  self-enhancement bias ("the effect that LLM judges may favor the
  answers generated by themselves" -- directly describes the current
  same-model Claude-judges-Claude setup, motivating the Gemini-judge
  idea), position bias (sidestepped by construction here, since the judge
  scores each response alone, never side-by-side), and verbosity bias
  (mitigated by the shared MAX_WORDS=40 cap across all 4 levels). Same
  paper recommends reference-guided grading and chain-of-thought as
  reliability improvements.
- Self-Preference Bias in LLM-as-a-Judge (arXiv:2410.21819, 2024) --
  GPT-4 showed the strongest self-preference bias among 8 tested models,
  but the mechanism may be general preference for low-perplexity/familiar
  text ("LLMs assign significantly higher evaluations to texts with lower
  perplexity than human evaluators, regardless of whether the texts were
  self-generated"), not literal self-recognition. Honest caveat logged:
  switching the judge to Gemini removes the literal same-model
  precondition for self-enhancement bias, but does NOT guarantee full
  immunity from this related, more general preference.
- G-Eval (Liu et al., EMNLP 2023, arXiv:2303.16634) -- checked whether
  detailed per-score-band rubrics (what a 5 vs 3 vs 1 means) are standard
  practice. They are NOT, even in this paper: most criteria get a name +
  one-line definition only (e.g. "Coherence (1-5) -- the collective
  quality of all sentences..."), structurally the same pattern already
  used in build_judge_prompt(). Per-band anchors appear only once, for
  one specific criterion (dialogue engagingness) -- a real but selectively-
  used technique, not a dominant one. Conclusion: the existing
  quality/strategy_adherence criteria phrasing did NOT need
  strengthening. What IS consistently backed across all three papers:
  chain-of-thought before scoring.

Applied: `build_judge_prompt()` in judge_mini_sample.py rewritten so the
required JSON's "reasoning" field comes FIRST (2-3 sentences), before
"quality"/"strategy_adherence" -- since the model fills JSON fields in
written order, this makes it reason through both dimensions before
committing to a number, not score-then-justify (which the old
last-positioned one-sentence "rationale" field allowed). Explicit
instruction added: "do not decide the numbers first and justify them
afterward." max_tokens raised 150->250 so the longer reasoning field
doesn't get truncated (which would break json.loads()). Verified: field
order confirmed literally (reasoning's position in the JSON template
precedes quality's), max_tokens change confirmed, a simulated realistic
judge reply parses correctly end to end. No parsing code changes needed --
still one clean JSON object either way.

Not yet decided: exact stratified-sampling implementation for the larger
~100-item fully-automated run (separate from the 32-item hand-annotation
set above), judge-prompt design for that full run (the mini-sample judge
prompt, already built, is a starting point, not necessarily final), xlsx
assembly script, and the actual hand-scoring step (quality/adherence, 1-5
each) for the 32-item hand-annotation set -- waits on the "2 best prompting
approaches" decision from the mini-sample.
- Cost/latency table for the generative component. Confirmed required by the
  assignment (verbatim: "tokens used, cost, and what you would change to run
  it at 100x scale"). Scope decision: built from the WINNING prompt
  approach(es)' actual production run, not the exploratory mini-sample --
  the assignment's own framing ("the generative component," "100x scale")
  is about the real deliverable pipeline. The mini-sample's own cost stays
  in the methodology narrative (~$0.03-0.05 estimated for 32/48/96 calls at
  the various scales considered -- see "hand-annotation sample design"
  section above), not the formal table.
  **Instrumentation fix, done proactively before any real run happens:**
  neither generate_responses.py's generate() nor judge_mini_sample.py's
  judge-scoring loop measured per-call wall-clock latency -- the assignment
  needs latency, not just tokens, and this project already learned once
  (train_roberta.py, DL wall-clock time) that retrofitting timing after the
  run already happened doesn't work, since the actual runs used are gone by
  the time you notice the gap. Fixed now: generate() returns
  (text, input_tokens, output_tokens, latency_seconds); judge_mini_sample.py
  times each judge call the same way and now also records input_tokens/
  output_tokens per response (previously only totals were printed, not kept
  per-row). Both scripts' output files (mini_sample_prompt_comparison.csv,
  mini_sample_judge_scores.json) now carry latency_seconds per row, and
  print a total-wall-clock/avg-per-call summary line. Verified via a fake
  Anthropic client stub (returns canned tokens after a real
  time.sleep(0.01)) confirming generate() returns a real, non-zero measured
  duration, not a placeholder. This means the FULL production run (once the
  winning approach(es) are chosen and this same generate()/judge call
  pattern is reused at scale) will have real per-call latency data
  automatically, with no separate instrumentation pass needed.
  (Note: this is separate from the RunPod GPU cost for the classifiers,
  which was trivial -- ~$0.75/hr RTX 4090, total classifier training time on
  the order of tens of minutes across both sweeps.)

## Training/inference wall-clock time (needed for the report)

Simple baselines -- measured directly, fresh timing run, CPU (sklearn, no
GPU used or needed):

| Model | Task | Wall-clock |
|---|---|---|
| TF-IDF + LogReg | 8-class | 5.51s (fit_transform + fit, train n=14190) |
| TF-IDF + LogReg | 3-class | 1.65s (fit_transform + fit, train n=11672) |
| Decision tree (winning config) | 8-class | 0.022s |
| Decision tree (winning config) | 3-class | 0.138s |

RoBERTa-base (`lr1e-5_wd0.01`, the final reported config) -- the original
training runs predate the `time`-based instrumentation added to
`train_roberta.py`, so those runs have no real timing. Re-ran both configs
on a fresh RunPod pod (RTX 4090) with the instrumented script to get real
numbers; both reproduced val macro-F1 **bit-for-bit identical**, epoch by
epoch, to the original runs (8-class: 0.2076/0.2262/0.2048/0.2375/0.2493/
0.2323/0.2284; 3-class: 0.3937/0.4291/0.4425/0.4392/0.4496/0.4225/0.4439),
so the timing below is a fully trustworthy stand-in, not just an estimate:

| Model | Task | Per-epoch | Epochs (early-stopped) | Total wall-clock |
|---|---|---|---|---|
| RoBERTa-base | 8-class | ~36.7s | 7 | 261s (4.3 min) |
| RoBERTa-base | 3-class | ~30.2s | 7 | 215s (3.6 min) |

DistilBERT -- still not captured (same instrumentation gap, not re-run
since RoBERTa already gives the DL-vs-classical-baseline comparison point
the report needs); qualitative note only: ~tens of minutes total across
both sweeps, RTX 4090, ~$0.75/hr.
- STRATEGY_LABELS/COARSE_LABELS ordering in src/preprocessing.py: user
  reordered by mean normalized position (confirmed via later screenshot
  showing position comments alongside each label) -- earlier flagged
  discrepancy appears resolved.

## GenAI subtask -- Phase 2: 32-item x 4-level prompt competition

Purpose: pick the winning prompt level(s) (the "2 best prompt candidates"
the assignment's xlsx step needs) on the real 32-item hand-annotation
sample, using an INDEPENDENT judge (Gemini, not Claude) to avoid the
same-model self-preference-bias risk that the mini-sample directly
surfaced (low Spearman agreement between human and Claude-judge rankings).
Decided (real user calls, not assumed):
  - Strategy target: pred_label (RoBERTa's PREDICTED strategy), not
    true_label. Deliberate design, not a shortcut: for the 16
    correctly-predicted items pred_label==true_label so this is just the
    gold strategy; for the 16 incorrectly-predicted items, this generates
    against a strategy that does NOT match what the conversation content
    actually calls for. The mismatch is the point -- testing whether
    generation quality/strategy_adherence degrades under a wrong signal,
    which is the realistic failure mode of a pipeline that depends on
    RoBERTa's own predictions.
  - Judge: Gemini (gemini-2.5-flash), independent of the Claude generator.
  - Scoring: AI absolute scores only (quality + strategy_adherence, 1-5,
    same rubric as the mini-sample) -- no manual human ranking of 128
    responses at this scale. The hand-annotation effort is saved for the
    real >=30-item kappa/QWK check on the eventual winning level(s), per
    select_hand_annotation_sample.py's original design.

Implementation:
  - generate_responses.py generalized (`--csv`, `--out` args added) rather
    than duplicated into a new file -- `load_competition_items()` reads
    hand_annotation_sample.csv and uses pred_label as the strategy passed
    to build_prompt(), carrying true_label/correct through to the output
    CSV (full32_prompt_comparison.csv) for the later matched-vs-mismatched
    analysis. Mini-sample behavior (MINI_SAMPLE, mini_sample_prompt_comparison.csv)
    is unchanged when --csv is omitted -- backward compatible.
  - New judge_full32_gemini.py: imports build_judge_prompt() and
    extract_json_object() DIRECTLY from judge_mini_sample.py rather than
    redefining them, so the full run is judged under the exact same
    rubric (criteria, CoT field ordering, tone definition, anti-fence
    instruction) as the mini-sample pilot -- only the model administering
    it changes, isolating that as the one real variable. Calls the Gemini
    API (google-genai SDK, client.models.generate_content with
    system_instruction via GenerateContentConfig). Same defensive
    JSON-fence-stripping as the mini-sample judge, reused as-is; worth
    checking on the real run whether Gemini fences its output the same
    way Claude did, since that fix was only verified against Claude's
    actual failure strings.
  - Output: full32_judge_scores_gemini.json (128 entries). Script prints
    two aggregations: (1) per-level mean (quality+strategy_adherence)/2
    across all 32 items, ranked, with the top 2 flagged as xlsx
    candidates; (2) the same means split by correct==True (matched
    signal) vs correct==False (mismatched signal) per level, with the gap
    between them -- this second table is the actual answer to "does
    generation degrade under a wrong strategy signal."
  - Verified: generate_responses.py's load_competition_items() tested
    against a synthetic 4-row CSV mimicking hand_annotation_sample.csv's
    exact columns -- confirms pred_label (not true_label) is used as the
    strategy, and build_prompt() doesn't error for any of the sampled
    strategy names. judge_full32_gemini.py's aggregation logic
    (per-level mean, correct/wrong split, parse-error exclusion)
    verified offline against synthetic scores with a known answer,
    including a deliberately-included parse-error entry to confirm it's
    excluded from means (not silently zeroed) -- all recovered correctly.
    Both scripts py_compile clean. NOT yet run against real API data --
    next step is the user running both commands for real.

Commands (run in order):
    python generate_responses.py --model claude-haiku-4-5 --csv ../outputs/genai/hand_annotation_sample.csv
    python judge_full32_gemini.py --model gemini-2.5-flash

## GenAI judge artifact: Gemini thinking-token truncation (full32 run 1)

Real first run of the 32-item competition: generation succeeded cleanly
(128/128, 0 leaks), but the Gemini judge failed 128/128 -- a 100% failure
rate, worse than the earlier Claude fence bug. Different root cause,
correctly diagnosed before assuming it was the same fence issue reused
from judge_mini_sample.py: the raw failed text was NOT fenced this time
(no ` ``` ` anywhere) -- it was truncated mid-sentence a few words into
the "reasoning" field, e.g. '{"reasoning": "The response effectively'.
Confirmed via the run's own totals: 1103 output tokens across 128 calls
(~8.6 tokens/call) against a 250-token ceiling -- consistent with almost
the entire budget being consumed before any visible text could be
written.

Cause: Gemini 2.5 Flash uses internal "thinking" tokens by default
(extended/hidden reasoning before the visible response), and these count
against `max_output_tokens`. At 250 tokens, hidden thinking alone used up
nearly the whole budget, leaving almost nothing for the actual JSON
output. This is a genuinely different failure mode from the Claude
fence-wrapping bug -- worth distinguishing in the report as two separate,
real LLM-API integration issues caught from live output, not one bug
recurring.

Fixed at the root, not by inflating the token budget alone: added
`thinking_config=types.ThinkingConfig(thinking_budget=0)` to explicitly
disable Gemini's internal thinking for this call -- redundant with the
project's own engineered CoT (the "reasoning" field is already written
first in the required JSON, same technique as judge_mini_sample.py), so
turning off Gemini's hidden thinking doesn't lose any reasoning quality,
it just stops silently spending the token budget on reasoning that never
gets used or reported. `max_output_tokens` also raised 250 -> 500 as a
safety margin on top of that. Verified: GenerateContentConfig with both
the ThinkingConfig and the new ceiling constructs without error; script
py_compile clean. NOT yet re-verified against a real Gemini call (needs
an actual API round-trip to confirm truncation stops, unlike the earlier
Claude fence fix which could be checked entirely offline against the
real failed strings) -- next step is re-running judge_full32_gemini.py
for real.

## GenAI subtask -- combining prompt-competition and kappa/QWK check

Revisited the earlier decision (report_notes.md "GenAI subtask -- Phase 2")
to split the 32-item competition (AI-only scoring, picked "Recommended" by
me) from the assignment's required hand-annotation kappa/QWK check (a
smaller, separate pass planned for later, on only the winning level(s)).
User pushed back -- their original framing ("test the 4 prompts on 32
examples... only after create the excel file with the 2 best prompt
candidates") reads as intending these combined, and I steered toward
splitting them by marking "AI only" Recommended, which wasn't a neutral
call. Corrected: decided to COMBINE -- the user hand-scores all 128
responses (32 items x 4 levels, quality + strategy_adherence, 1-5) now,
matched against Gemini's per-response scores. This satisfies the
assignment's >=30-item kappa/QWK requirement immediately (n=128, well over
minimum) using data already generated, AND picks the winning level(s) from
a human+AI combined score rather than trusting Gemini alone -- more
rigorous given the mini-sample's own finding that human and AI-judge
rankings can disagree substantially (Spearman rho=0.067 there). Tradeoff,
stated plainly to the user before they chose: meaningfully more manual
scoring work now (128 responses) vs. a smaller later pass on just the
winning level(s) -- their call, made with the tradeoff in view.

New scripts:
  - make_human_scoring_template.py: builds
    outputs/genai/full32_human_scoring_template.csv from
    full32_prompt_comparison.csv -- idx/level/strategy/context/response +
    two blank columns. Deliberately does NOT include Gemini's scores --
    scoring must be done BLIND to the AI judge (same principle as the
    mini-sample's blinded quiz), otherwise the kappa/QWK check would just
    measure whether the human copied the AI's numbers, not independent
    agreement.
  - score_full32_kappa.py: joins the user's filled-in
    full32_human_scores.csv with full32_judge_scores_gemini.json on
    (idx, level). Computes, separately for quality and strategy_adherence:
    Cohen's kappa (unweighted) AND quadratic weighted kappa (QWK -- the
    more appropriate measure for an ordinal 1-5 scale, since it penalizes
    a 2-vs-5 disagreement more than a 2-vs-3 one, which plain kappa does
    not). Both implemented directly (no sklearn dependency) for
    auditability. Also reports the per-level human+AI combined winner
    (same human_weight-based combination as combine_mini_sample_scores.py,
    default 0.65, for methodological consistency across judging stages)
    and the correct-vs-wrong-signal breakdown for BOTH human and AI scores
    side by side -- so the "does generation degrade under a wrong signal"
    question gets an answer that isn't solely dependent on trusting
    Gemini's read of it.
  - Rows with no human score yet, or a Gemini PARSE_ERROR, are excluded
    from all analysis (not treated as 0 or silently dropped without
    notice) -- printed as explicit NOTE counts.

Verified: both scripts py_compile clean. Kappa/QWK formulas checked
against: (1) perfect agreement -> kappa=QWK=1.0; (2) a hand-calculable
textbook binary example (15/5/5/15 confusion matrix) -> kappa=0.5,
matching the manual calculation exactly; (3) full independence (every
rating combination equally likely) -> kappa~0.0; (4) maximal opposite-end
disagreement -> both kappa and QWK strongly negative, QWK more so
(penalizes distance). End-to-end pipeline verified against a synthetic
8-row dataset with a deliberately unscored row and a deliberately
AI-parse-error row -- both correctly excluded (counts printed), per-level
ranking and correct/wrong breakdown both recovered correctly by hand
cross-check.

Next steps: (1) user re-runs judge_full32_gemini.py with the thinking-token
fix to get real Gemini scores; (2) user runs make_human_scoring_template.py
and hand-scores all 128 rows (blind to Gemini's scores); (3) user runs
score_full32_kappa.py for the real kappa/QWK numbers and winning level(s).

## Full32 Gemini judge -- real run, thinking-token fix confirmed working

Re-run after the fix: 0/128 parse errors (vs. 128/128 before), confirming
the truncation diagnosis and fix were correct. Real scores:

  level 2: mean=4.984 (n=32)
  level 4: mean=4.844 (n=32)
  level 3: mean=4.781 (n=32)
  level 1: mean=4.484 (n=32)

Correct (matched signal) vs. wrong (mismatched signal) gap, per level:
  level 1: 4.594 vs 4.375 (gap=+0.219)
  level 2: 4.969 vs 5.000 (gap=-0.031)
  level 3: 4.750 vs 4.812 (gap=-0.062)
  level 4: 4.812 vs 4.875 (gap=-0.062)

Real finding, flagged rather than accepted at face value: Gemini's scores
show a strong ceiling effect -- nearly all responses scored 4-5/5, tight
clustering (4.48-4.98 range across ALL 4 levels), and the correct-vs-wrong
gap is near zero or even slightly NEGATIVE for levels 2-4 (better under a
mismatched strategy signal than a matched one, which doesn't have an
obvious causal explanation and more likely reflects judge noise/leniency
than a real effect). This is a plausible instance of judge leniency/ceiling
bias -- a documented LLM-as-judge failure mode distinct from the
self-preference bias flagged earlier (this is Gemini judging Claude's
output, not a same-model issue, so it's a different risk than the one
Gemini was brought in to address). NOT treated as the final word: this is
exactly what the combined human-scoring pass (decided above) is for --
if the user's hand scores show more spread and a clearer correct/wrong
gap than Gemini's did, that's a legitimate, reportable instance of AI-judge
leniency being caught and corrected by the human-in-the-loop step, not a
failure of the pipeline.

Token cost, real: 45,505 input + 9,614 output tokens, 129.4s wall-clock for
128 calls (~1.01s/call avg) -- for the assignment's cost/latency table.

Next: user runs make_human_scoring_template.py (zero API cost -- pure
local file I/O, no tokens spent) to get the blank CSV, hand-scores all 128
rows blind to these Gemini numbers, then runs score_full32_kappa.py.

## Switched hand-scoring format: CSV template -> blind text quiz

User asked for the same readable format the mini-sample's ranking quiz
used, rather than the raw CSV template (make_human_scoring_template.py --
still on disk, no longer the recommended path). Built two new scripts
instead of retrofitting the CSV approach:

  - make_full32_scoring_quiz.py: same blinding mechanism as
    judge_mini_sample.py's write_quiz_file() (shuffle each item's 4
    levels into per-item A-D options, seeded so the shuffle is
    reproducible; blind key written separately, not to be opened before
    scoring) -- but scores each option on quality + strategy_adherence
    (1-5 each) individually rather than producing a 1-4 ranking, since
    this quiz's job is absolute scoring for the kappa/QWK check, not
    preference ordering. Different BLIND_SEED (11) from the mini-sample's
    quiz (7) -- separate quiz, separate shuffle, no reason to couple them.
    Quiz text includes the same tone-definition wording as the judge
    prompt (quality's tone clause) so the human and Gemini are scoring
    against a comparable definition of "quality," not two different
    implicit rubrics.
  - parse_full32_scoring_quiz.py: regex-parses the filled-in quiz text
    (per-item chunks via the "--- Item idx=N" header, then each A-D
    option's two number blanks), uses the blind key to map letters back
    to real levels, and writes full32_human_scores.csv in exactly the
    format score_full32_kappa.py already expects -- no changes needed to
    that script. Blank/unfilled entries are detected and reported by
    count, not silently treated as a score of 0 or dropped without
    notice.

Verified end-to-end on a synthetic 2-item x 4-level dataset: built the
quiz and blind key, filled in known values, confirmed the quiz's shuffled
A-D order really did scramble level order (not accidentally alphabetical),
parsed the filled quiz back, and confirmed every (idx, level) pair
recovered its correct quality/strategy_adherence pair through the
shuffle -- including a deliberately-left-blank entry, correctly detected,
reported by count, and excluded (not miscounted as scored). Both scripts
py_compile clean.

Workflow now: make_full32_scoring_quiz.py -> fill in
full32_scoring_quiz.txt by hand (blind to Gemini's scores) ->
parse_full32_scoring_quiz.py -> score_full32_kappa.py.

## Two real bugs caught from the user's actual filled-in quiz

User hand-scored all 128 responses for real, then asked "how come n=29 on
level 1?" and correctly self-diagnosed the likely cause before I even
looked -- a multi-line response with a leading "# Response" text. Both
turned out to be real, distinct issues, not one:

1. **Parsing bug (parse_full32_scoring_quiz.py)**: the original OPTION_RE
   assumed each response was a single line, requiring "Quality (1-5):" on
   the line immediately after "X. <response>". Three responses (idx=53,
   802, 1288 -- all level 1) span multiple lines because of bug #2 below,
   which silently broke the match for those three options even though the
   user HAD filled in real scores -- miscounted as unscored (29/32 for
   level 1 instead of 32/32). Fixed by abandoning the line-based
   assumption entirely: find each "  X. " option marker's position,
   slice the text between consecutive markers (however many lines that
   spans), and search for the Quality/adherence numbers anywhere in that
   slice. Verified against a synthetic reproduction of the exact real
   failure (a multi-line, header-prefixed response) -- recovers correctly
   -- AND against the user's actual real quiz file end-to-end: 125/128 ->
   128/128 parsed after the fix, confirmed by running the real file
   through both the old and new parser.

2. **Generation artifact (generate_responses.py), root cause of #1**:
   those three responses had the model prepend a markdown header -- "#
   Response" or "# Information" -- followed by a blank line before the
   actual reply, e.g. "# Response\n\nI notice you're asking about
   optimism...". All three are level 1 specifically, which is not a
   coincidence: level 1's prompt is the deliberately "naive" baseline (no
   persona, no strategy definition, and critically no "reply with ONLY
   the response text, nothing else" instruction that levels 2-4 all
   have) -- so level 1 is the one level with nothing telling the model
   not to add a header. Fixed the same way as the earlier word-count leak
   -- a post-processing strip (`strip_leading_header()`, regex-based,
   strips a leading `#{1,6} ...` line plus trailing blank line), NOT by
   adding an instruction to level 1's prompt, which would blur the point
   of the naive baseline (the whole reason level 1 has no such
   instruction). Logged via a new `leading_header_stripped` CSV column,
   same pattern as `meta_commentary_stripped`. Verified against the exact
   3 real leaked strings (all correctly stripped to the intended text)
   plus regression checks (normal text untouched; a genuine "#" in the
   middle of a sentence untouched, not just anything starting with a
   digit-adjacent hash).

   NOT retroactively applied to the already-generated full32_prompt_comparison.csv
   (would mean re-running generation + re-judging + partially re-doing
   the user's already-completed hand-scoring for a cosmetic 3/128 issue
   that doesn't change the substance of those 3 responses) -- fix applies
   to future runs. This run's 3 affected responses are still valid data,
   just visually messier; noted here for transparency rather than quietly
   left unexplained.

Also added, per user request: score_full32_kappa.py now prints quality and
strategy_adherence as a separate per-level breakdown (not just the blended
mean) -- a level can win on one dimension and lose on the other, which the
blended mean alone would hide. Verified: the new breakdown's two numbers
average to exactly the already-verified blended mean from the row above it
in a synthetic re-test.

Next: user re-runs parse_full32_scoring_quiz.py (now fixed, should recover
all 128) and score_full32_kappa.py (now with the added breakdown) for the
real, complete numbers.

## Full32 competition -- FINAL real result (n=128, all real hand scores)

Real, complete run -- 128/128 parsed, 128/128 paired against Gemini's
scores. This is the combined prompt-competition + kappa/QWK result
(per the earlier "combine the two tasks" decision).

Kappa/QWK (human vs. Gemini, n=128):
  quality:            Cohen's kappa=0.013   QWK=0.063
  strategy_adherence: Cohen's kappa=0.033   QWK=0.053

Both very close to zero -- essentially no agreement beyond chance between
the human rater and Gemini on individual response scores, on EITHER
dimension. This is a real, reportable, slightly uncomfortable finding, not
a pipeline failure: it directly confirms the ceiling-effect/leniency
concern flagged when Gemini's scores first came back (report_notes.md
"Full32 Gemini judge -- real run") -- Gemini's mean sits above the
human's mean at every single level (e.g. level 2: human=4.344 vs
Gemini=4.984; level 1: human=3.812 vs Gemini=4.484), consistent with a
lenient judge compressing toward the ceiling rather than genuinely
disagreeing in both directions. Worth stating plainly in the report: this
IS the answer to the assignment's required kappa/QWK check, and a low
value here is itself a valid, defensible result about judge reliability --
not something to explain away.

Per-level means (human_weight=0.65):
  level 2: human_mean=4.344  ai_mean=4.984  combined_mean=0.892  (n=32)
  level 4: human_mean=4.219  ai_mean=4.844  combined_mean=0.859  (n=32)
  level 3: human_mean=4.219  ai_mean=4.781  combined_mean=0.854  (n=32)
  level 1: human_mean=3.812  ai_mean=4.484  combined_mean=0.762  (n=32)

Winner: level 2. Notable and worth highlighting despite the near-zero
kappa above: human and AI RANKINGS of the 4 levels agree exactly (2 > 4 ~
3 > 1, with 4 and 3 close either way) even though item-by-item absolute
agreement is near chance. These are genuinely different kinds of
agreement -- kappa measures whether the same response gets the same
number from both raters; the level ranking is an aggregate over 32 items,
where independent noise on individual responses can average out while
still leaving a real directional signal intact. Not a contradiction, a
useful distinction for the report.

Quality vs. strategy_adherence, separately:
  level 2: h_quality=3.969 h_adherence=4.719 | ai_quality=5.000 ai_adherence=4.969
  level 4: h_quality=3.844 h_adherence=4.594 | ai_quality=4.812 ai_adherence=4.875
  level 3: h_quality=3.781 h_adherence=4.656 | ai_quality=4.844 ai_adherence=4.719
  level 1: h_quality=3.281 h_adherence=4.344 | ai_quality=4.781 ai_adherence=4.188

Real pattern the blended mean would have hidden: the human consistently
rates strategy_adherence noticeably higher than quality at every level
(e.g. level 1: adherence=4.344 vs quality=3.281) -- i.e. the human's read
is that these responses mostly DO execute the right strategy, but aren't
always great as a chat message in their own right. Gemini doesn't show
this gap nearly as clearly. This is exactly the kind of dimension-specific
insight the separate breakdown (added on request) was for.

Correct (matched signal) vs. wrong (mismatched signal) gap:
  level 1: human 4.094 vs 3.531 (gap=0.562) | AI 4.594 vs 4.375 (gap=0.219)
  level 2: human 4.594 vs 4.094 (gap=0.500) | AI 4.969 vs 5.000 (gap=-0.031)
  level 3: human 4.375 vs 4.062 (gap=0.312) | AI 4.750 vs 4.812 (gap=-0.062)
  level 4: human 4.406 vs 4.031 (gap=0.375) | AI 4.812 vs 4.875 (gap=-0.062)

Human gaps are all positive and roughly decreasing with more scaffolding
(0.562 -> 0.500 -> 0.312 -> 0.375, not perfectly monotonic but the
general trend holds) -- a believable, defensible pattern: less-scaffolded
prompts (level 1) are more fragile to a mismatched strategy signal, more
scaffolding buys some robustness. Gemini's gaps are much smaller and
actually NEGATIVE for levels 2-4 (scored slightly BETTER under a wrong
signal, which has no obvious causal story) -- another instance of the
AI judge's near-ceiling scores failing to capture a real effect that the
human's more differentiated scoring did catch.

## Error analysis -- scope, selection, and pulled sample

Re-read the assignment PDF directly (not from memory, not from the
09_notebook's "Next:" placeholder note which was never actually acted
on) -- exact spec (page 2): "An error analysis of ~10 test items you
read by hand where the algorithms got it wrong. With details about the
error vs the ground truth and your thoughts about the reasons." No
per-task split specified, no selection method mandated.

Two decisions, confirmed with the user:
- Scope: 8-class only (not split with 3-class) -- primary/harder task,
  richer error space; 3-class errors are largely inherited from 8-class
  confusions already covered elsewhere in the report.
- Selection: highest-confidence wrong predictions, not a confusion-
  matrix-stratified sample -- most diagnostically interesting (the model
  was confident and still wrong).

src/pull_error_analysis_sample.py: pulls correct==False rows from
test_predictions.csv, sorts by confidence descending, takes top 10.
conversation_id/turn_index/problem_type recovered the same deterministic
way as assemble_final_xlsx.py (re-running preprocessing+splits with
seed=42 reproduces the same test_examples list idx indexes into).
Script deliberately does NOT write the "your thoughts about the reasons"
column -- that's the actual hand-read part the assignment asks for, not
something to script.

Real pulled sample (1,251/1,724 test predictions wrong overall; top 10 by
confidence, all wrong, confidence 0.81-0.88): saved to
outputs/8class/roberta/error_analysis_sample.txt. Visible pattern on
first read, worth confirming/writing up properly: 6/10 items are
conversation-CLOSING turns ("Take care", "Have a good night", "Merry
Christmas", "Enjoy the rest of your day") where the model predicted
"Others" (its learned closing-phrase bucket) but the gold label was
something more specific (Providing Suggestions, Reflection of feelings,
Self-disclosure, Affirmation and Reassurance, Information) -- ties
directly to the greeting/closing lexical analysis already found via
TF-IDF top words and the position-tree baseline (see "TF-IDF
interpretability" section) -- a third, convergent piece of evidence that
closing turns are lexically homogeneous but strategically heterogeneous,
which is a real source of confusion for a model relying on surface
cues. STILL NEEDED: the user's own hand-read reasoning per item (the
"your thoughts" column) -- draft interpretation above is a starting
point, not a substitute.

Correction to the draft diagnosis above: user pointed out my first read
only looked at the model's INPUT (the seeker's preceding context), not
what the supporter actually SAID in the labeled turn. Pulled the real
supporter text (raw ESConv.json, by conversation_id/turn_index) for all
10 items. This splits the sample into two different stories, not one:
- Items 2 (idx=1624) and 4 (idx=987) are real model errors on
  substantive text: #2's supporter text is a textbook paraphrase ("So
  you lost your job that is why you are sad right?") mispredicted as
  Question, plausibly on the trailing "?" as a surface cue. #4 is a
  clear suggestion ("take some time away...") mispredicted as
  Restatement, plausibly because it opens with a reflective line before
  the actual suggestion.
- Items 1, 3, 5-10 are likely ANNOTATION NOISE, not model failure: the
  real supporter text is a bare reciprocal closing ("You too!",
  "Absolutely") with almost no strategy-distinguishing content -- the
  model's "Others" prediction is arguably the more defensible read.
  Items 5 and 9 are actually platform/interface remarks ("press quit",
  "finish from your end"), not support content at all.
This is a stronger, more defensible finding than the surface "closing-
phrase lexical pattern" read, and changes what the written error
analysis should say for 8/10 items.

## Error analysis -- real user interview (final content for Section 7)

Assignment requires "your thoughts about the reasons" -- genuinely the
user's own hand-read judgment, not an AI-drafted stand-in. Conducted a
real item-by-item interview: for each of the 10 items, showed the user
the real seeker context + real supporter text (pulled from ESConv.json,
not just the truncated context) + gold/predicted labels, and recorded
their actual verbatim reasoning. Full Q/A is in the conversation
transcript; final per-item reads now in the report's Section 7.

Real finding surfaced by the user mid-interview, not something I caught
first: too many of the original top-10-by-confidence items were
greeting/closing-turn related. Verified quantitatively against all 1,251
wrong test predictions: pred=Others share is 66.7% in the top 1% most-
confident-wrong, 32.3% in top 5%, 29.6% in top 10%, 20.2% in top 25%, vs
only 11.5% across all wrong predictions. Pure top-N-by-confidence
selection is genuinely biased toward one specific, very-confident
systematic failure mode (closing-turn -> Others) and under-represents
error diversity -- a real limitation of the original selection method,
not just a feeling. Fixed by re-selecting: highest-confidence wrong item
per DISTINCT (true_label, pred_label) pair. This drops idx=1604
(duplicate confusion pair with idx=1317, both Providing Suggestions ->
Others) and adds idx=648 (Reflection of feelings -> Question, conv 466
turn 3, real substantive content, not a closing turn). Final 10:
idx=231, 1624, 1317, 987, 285, 789, 241, 1226, 924, 648.

User's real per-item judgments, sharper than the earlier AI-only draft:
- idx=231: disagrees with gold ("Restatement or Paraphrasing") -- reads
  "Absolutely" answering a direct personal question as self-disclosure,
  thinks the model's prediction is the more defensible one.
- idx=1624: genuine strategy overlap, not a clean single-label case
  ("its actually both... we can use a more complex approach that use 2
  or even more strategies"). Separately critiques the ground-truth
  human supporter's actual response quality -- a professional would
  have explored/unpacked the seeker's statement rather than just
  restating it as a question.
- idx=1317: asked directly whether the gold label might be miscoded.
  Verified against raw ESConv.json: turn 22 genuinely is the
  conversation's last turn, annotation genuinely says "Providing
  Suggestions" for "you too!" -- confirmed real dataset annotation
  noise, not a pipeline bug.
- idx=987: confirms this IS a genuine model error, gold label is
  correct, response is good. Prompted a real, verified gap: none of
  the 4 generation prompt levels include an explicit "understand the
  seeker's message first, then respond" step (chain-of-thought was
  only ever added to the JUDGE prompt, never generation -- checked
  generate_responses.py directly). Logged as an untested "level 5"
  further-work idea in the report, tied to this example and to the
  already-cited two-stage structure of the original paper's own
  generation pipeline.
- idx=285: independently identifies this as a platform/interface
  artifact ("a variant of a system message... informing that he can
  stop answering so the seeker should just type himself"), not real
  support content -- matches the earlier AI-drafted read but in the
  user's own words/reasoning.
- idx=789: "same as the above" -- bare reciprocal closing, model's
  Others call is correct.
- idx=241: explicitly flags this one has NO clear explanation for the
  mislabel (unlike idx=1317/1226 where a literal-reading account is
  plausible) -- honestly reported as unexplained noise rather than
  forcing a tidy story.
- idx=1226: confirms same platform-artifact pattern as idx=285
  ("press quit" -- literal interface instruction, not support content).
- idx=924: "its a greeting" -- despite having more surface content than
  a bare "you too", still judges this as a closing remark, not real
  Affirmation and Reassurance content.
- idx=648 (new): agrees gold ("Reflection of feelings") technically
  fits, but would personally have asked a clarifying question instead
  ("I would personally choose the question strategy to try and
  understand more") -- another genuine overlap case, same theme as
  idx=1624. Also notes the ground-truth response reads as somewhat
  theatrical/overwrought.

New synthesis (sharper than the earlier "8 likely annotation noise"
framing): 4 categories, not 2 -- genuine model error (idx=987); genuine
strategy overlap where multiple labels are defensible (idx=1624,
idx=648); annotation noise where the model's prediction is arguably MORE
correct than gold (idx=231, 1317, 789, 241, 924); platform/interface
artifacts that aren't support content at all (idx=285, 1226). Also
surfaced two real further-work items directly from this process: (1) an
untested "understand-then-respond" prompt level for generation, and (2)
the single-label 8-way framing itself may be too rigid given how often
multiple strategies are genuinely defensible for the same turn --
multi-label/soft-label classification as an alternative worth testing,
noted near the 3-class mapping discussion (Section 3.3).

## src/ cleanup (consolidation pass)

User flagged the GenAI subtask had accumulated too many src/ files.
Categorized all of them: 1 file (make_human_scoring_template.py) was
unambiguously dead -- the CSV-template scoring approach explicitly
abandoned in favor of the blind text-quiz format, never produced any
data that fed a real result. Everything else either an active import
dependency or the direct source of a number/file in the final
deliverables -- not deletable without losing the audit trail for graded
work. User chose to also merge two pairs that did sequential halves of
the same job:
- judge_mini_sample.py + judge_full32_gemini.py -> judge.py
  (--provider claude|gemini)
- make_full32_scoring_quiz.py + parse_full32_scoring_quiz.py -> quiz.py
  (--make / --parse)

Re-verification discipline (this touches code that already produced
submitted numbers, not a free refactor): no live API call was re-run
(would spend real money for a pure refactor); everything else was
checked against real staged data.
- build_judge_prompt()/extract_json_object(): AST-diffed function
  bodies against the originals -- only comment/docstring wording
  changed, all executable logic (prompt string, regex, control flow)
  byte-identical.
- judge.py's mini-sample blind-quiz builder: re-ran offline against the
  real mini_sample_prompt_comparison.csv, diffed against the real
  staged mini_sample_quiz.txt/mini_sample_blind_key.json -- blind key
  identical, quiz template identical except real file has filled-in
  answers where the fresh run has "____".
- quiz.py --make: diffed against the real full32_scoring_blind_key.json
  (JSON-identical) and full32_scoring_quiz.txt (identical after
  blanking the real scores back to "____").
- quiz.py --parse: ran against the real filled-in full32_scoring_quiz.txt
  -- output full32_human_scores.csv byte-identical to the original
  parser's output, 128/128 parsed (same as before).
- Real near-miss caught mid-verification: running `quiz.py --make`
  overwrote the LOCAL staged copy of the user's real filled-in quiz with
  a blank template (only in this session's sandbox, never touched the
  real file on the user's machine) -- re-staged from the user's machine
  to recover it before continuing. No data was actually lost, but a
  reminder to snapshot real files aside before test-running a script
  that writes to their exact path.

Deleted from the user's machine (delete permission requested and
granted for the project folder): make_human_scoring_template.py,
judge_mini_sample.py, judge_full32_gemini.py, make_full32_scoring_quiz.py,
parse_full32_scoring_quiz.py. src/ GenAI-subtask-adjacent file count:
17 -> 13.

## src/ docstring/comment trim (all 16 remaining files)

User flagged that essay-length module docstrings and inline comments were
making the source unreadable -- the design rationale, bug narratives, and
decision history belong in report_notes.md/the final report, not repeated
in every file. Every remaining src/ file's top docstring cut to a few
lines (what the file does, key inputs/outputs, usage), and long inline
comment blocks (multi-line "real observed failure mode" narratives,
per-level design rationale, etc.) trimmed to one-line pointers back to
report_notes.md. No logic touched -- purely comments/docstrings/argparse
help text; also fixed a few now-stale references to the deleted
judge_mini_sample.py/judge_full32_gemini.py/make_full32_scoring_quiz.py/
parse_full32_scoring_quiz.py/make_human_scoring_template.py in error
messages and docstrings (score_full32_kappa.py, combine_mini_sample_scores.py,
generate_responses.py).

Re-verification discipline (same as every other change to code that
already produced submitted numbers): re-ran every script with real
executable logic against the real staged data after trimming and diffed
against the pre-trim results -- select_final_100_sample.py,
score_full32_kappa.py (kappa/QWK numbers exact match), assemble_final_xlsx.py
(400 rows, 0 mismatches), breakdown_final100_quality_adherence.py (same
quality/adherence numbers), pull_error_analysis_sample.py, and the full
test_pipeline_invariants.py suite (all 1,300 conversations, all 10 checks
PASS) -- all identical to before trimming, confirming nothing but
comments changed.

STATUS: this satisfies the assignment's required >=30-item kappa/QWK
check (n=128) and picks the winning prompt level (level 2, both human and
AI agree) for the "2 best prompt candidates" xlsx step. Next: decide the
second candidate (level 4 and level 3 are close -- 4 edges out 3 on both
human and AI combined mean, and on quality specifically, though 3 edges
out 4 on human's adherence) alongside the final generation run
(predicted + gold conditions) and xlsx assembly.

## Final production sample: 100 items (32 hand-scored + 68 new)

Decided levels 2 and 4 as the two winning prompts to carry to the final
run. Cost estimates (real per-call token/latency averages from the
full32 run x target scale, Claude Haiku $1/$5 per MTok + Gemini 2.5
Flash $0.30/$2.50 per MTok, both confirmed via web search) were computed
for 32 / 100 / 500 / 1724 items; user chose 100 as the final scale.

Decision: the 100 is NOT a fresh disconnected sample -- it's the
already-selected 32-item hand-annotation sample (kappa/QWK-validated
above) PLUS 68 new items, so the human-validated subset is a documented
part of the final deliverable rather than a separate throwaway pass.

src/select_final_100_sample.py: excludes the 32 existing idx from the
sampling pool, then allocates the 68 new items PROPORTIONALLY by
true_label frequency (largest-remainder rounding for exact integer
sums) -- proportional, not equal-per-class, matching the original
100-item plan and avoiding under/over-representing rare classes like
Information (5.6% of the full test set). Seed=42 (project-standard).
Output: outputs/genai/final_100_sample.csv, columns idx, context,
true_label, pred_label, correct, confidence, hand_scored (True for the
32, False for the 68 new).

Verified end-to-end against the real files on the user's machine
(test_predictions.csv, hand_annotation_sample.csv): 100 total rows, 100
unique idx (no overlap/dup bug), 32/68 split correct, allocation sums to
68 and roughly tracks real class frequency:
  Question 13, Others 12, Affirmation and Reassurance 11,
  Providing Suggestions 11, Self-disclosure 7, Reflection of feelings 5,
  Restatement or Paraphrasing 5, Information 4.

judge_full32_gemini.py generalized: added --out (default
full32_judge_scores_gemini.json, unchanged) so the final run's judge
output doesn't clobber the already-completed 32-item competition's
scores. --csv was already generalized. py_compile clean.

Ready to run (user-driven, on their machine):
  1. python generate_responses.py --csv outputs/genai/final_100_sample.csv \
       --levels 2,4 --conditions predicted,gold \
       --out final_100_prompt_comparison.csv         (400 generations)
  2. python judge_full32_gemini.py \
       --csv outputs/genai/final_100_prompt_comparison.csv \
       --out final_100_judge_scores_gemini.json        (400 judge calls)
  3. build the real cost/latency table from this actual run's totals
     (the assignment's genuinely required table).

Real run 1 (generation) complete: 400/400 generations (levels 2,4 x
predicted+gold x 100 items), 0 errors, 0 header-strip / word-count-
annotation artifacts. 107,498 input tokens, 16,398 output tokens,
518.7s wall-clock (1.30s/call avg). Output: final_100_prompt_comparison.csv.
Real run 2 (judging) complete: 400/400 judged, 0 parse errors.
141,896 input / 31,010 output tokens (Gemini 2.5 Flash), 398.9s
wall-clock (1.00s/call avg). Output: final_100_judge_scores_gemini.json.

Per-level means (AI judge, n=200 each): level 4=4.793, level 2=4.688 --
level 4 edges out level 2 here (both were the pre-selected finalists
from the full32 hand-scored/kappa-validated pass; that earlier pass had
human+AI both favoring level 2 -- levels 2 and 4 were already close
there too, see "Full32 competition -- FINAL real result"). Matched vs.
mismatched signal gap is again tiny for both levels (2: -0.030,
4: +0.035), consistent with the AI judge's ceiling-effect/low-
differentiation behavior documented earlier -- not read as a real
robustness finding on its own, only the human-scored 32-item subset's
gaps are trusted for that claim.

## Real cost/latency table (assignment-required), from the actual 400-item final run

| stage | model | input tok | output tok | cost | wall-clock | s/call |
|---|---|---|---|---|---|---|
| generation | Claude Haiku 4.5 | 107,498 | 16,398 | $0.1895 | 518.7s | 1.297 |
| judging | Gemini 2.5 Flash | 141,896 | 31,010 | $0.1201 | 398.9s | 0.997 |
| combined | -- | -- | -- | $0.3096 | 917.6s (15.3 min) | -- |

Per-response (gen+judge combined): $0.00077, ~2.3s. Pricing verified via
web search at time of use: Haiku 4.5 $1/$5 per MTok in/out, Gemini 2.5
Flash $0.30/$2.50 per MTok in/out. Scaling projections (linear
extrapolation from these real per-call averages, sequential/no batching):
500 items ~$0.39 / ~19.1 min; 1000 ~$0.77 / ~38.2 min; full 1724-item
test set ~$1.33 / ~65.9 min.

## xlsx assembly -- row layout and word-count decisions, real result

Traced conversation_id/turn_index/problem_type: NOT present in
final_100_sample.csv or test_predictions.csv. Recovered by re-running the
exact deterministic pipeline evaluate_checkpoints.py used to create idx
in the first place -- preprocessing.build_8class_examples() +
splits.apply_conversation_level_split(seed=42) reproduces the identical
1,724-item test_examples list in the same order (idx = position in that
list), so conversation_id/turn_index/problem_type can be looked up
directly. Verified before use, not assumed: context text at
test_examples[0] and [53] matched test_predictions.csv's context column
exactly (real string equality check).

Two design decisions, presented to the user rather than decided
unilaterally (both are things they'll need to defend orally):
1. Row layout: each turn now has 2 generations (predicted-condition,
   gold-condition -- see final production run above). Chose 2 rows per
   turn + an explicit 'condition' column (predicted/gold), beyond the
   spec's literal column list -- necessary because gold_strategy and
   predicted_strategy alone can't disambiguate which row's response was
   generated toward which strategy whenever pred_label==true_label
   (the majority of items), which the assignment's own column list
   doesn't resolve. User confirmed this option.
2. Word-count overage: 11/400 real responses (2.75%, all level 2, both
   conditions) came in 1-5 words over the <=40-word cap (max 45, all
   level 4 responses stayed within cap). Left as generated (not
   truncated), word_count column included so it's visible/auditable.
   User confirmed this option -- also becomes a small, real data point
   for level 2 vs. 4 comparison (level 4's stricter length adherence).

src/assemble_final_xlsx.py: joins final_100_prompt_comparison.csv (400
generation rows) with final_100_judge_scores_gemini.json (400 judge
scores) POSITIONALLY (row i <-> score i) rather than by key -- more
reliable here since idx+level+strategy alone collides for correct==True
items (same strategy value for both conditions' independent
generations). Verified before use: 0/400 idx+level+strategy mismatches
when zipped positionally (real check, not assumed).

Real bug caught before delivery: first draft did `del r["level"]` while
filtering the level-2 subset, which mutated the SAME dict objects still
referenced by the shared all_records list (filter returns references,
not copies) -- corrupted the level-4 pass with a KeyError. Fixed by not
deleting at all (pd.DataFrame(records, columns=COLUMNS) already selects
only the spec'd columns and silently drops "level"). Re-ran clean.

Output: outputs/genai/genai_final_level2.xlsx, genai_final_level4.xlsx
-- 200 rows each (100 predicted-condition + 100 gold-condition), columns
conversation_id, turn_index, problem_type, dialogue_context,
gold_strategy, predicted_strategy, condition, generated_response,
word_count, judged_quality, judged_strategy_adherence. Verified: correct
shape (200x11) both files, 0 nulls in any column, condition split
exactly 100/100, real sample rows spot-checked. Arial font, frozen
header row, auto-sized columns (xlsx skill conventions -- no formulas
needed, pure data table).

STATUS: the 2 required GenAI-subtask output files are done. Remaining
for this subtask: none functionally -- report write-up only (cost/
latency table already built above, prompt-design writeup, kappa/QWK
writeup, all findings already logged through this file).

## Final 100-run quality/adherence breakdown (AI judge only)

src/breakdown_final100_quality_adherence.py: same quality-vs-adherence
separation as score_full32_kappa.py's breakdown, applied to the final
400-item run. AI-only -- the 32-item subset's human scores were
collected against a DIFFERENT set of generations (the earlier 4-level,
predicted-only competition run), so they don't line up row-for-row with
this run's regenerated responses and aren't reused here.

Real bug caught before delivery: first version built a {(idx, level):
condition} dict to join the by-condition split, but idx+level is NOT a
unique key here -- both predicted and gold rows for the same turn share
it, and since generate_responses.py always emits predicted then gold,
gold silently overwrote predicted in the dict for every row (came back
200/0 gold/predicted instead of 100/100). Fixed by reading
gen_rows[i]['condition'] off the same positional pairing (row i <-> score
i) assemble_final_xlsx.py already established as the correct join method
for this run. Verified against the real files: 100/100 predicted/gold
split, and both halves average back to exactly the already-reported
blended means (level 4: (4.815+4.770)/2=4.7925 vs 4.793 reported;
level 2: (4.855+4.520)/2=4.6875 vs 4.688 reported).

Real result (n=400, AI judge only):
  level 4: quality=4.815, adherence=4.770
  level 2: quality=4.855, adherence=4.520
  by condition: level2/predicted q=4.870 a=4.470; level2/gold q=4.840 a=4.570
                level4/predicted q=4.750 a=4.790; level4/gold q=4.880 a=4.750

Level 2 has HIGHER quality than level 4 (4.855 vs 4.815) but noticeably
LOWER strategy_adherence (4.520 vs 4.770) -- the blended mean masks this;
level 4's edge in the combined score is really an adherence story, not a
quality one. Consistent, directionally, with the full32 human-scored
pass, where human raters also rated adherence higher than quality at
every level but did NOT show this particular level-2-adherence dip --
another reminder that this run's AI-only scores (no human check on these
68 new items) carry the same ceiling/low-differentiation caveats already
documented for the Gemini judge.

## Report v2 -- narrative restructure

User wanted a second, differently-structured version of the report: same
underlying facts/numbers as v1, but told as a story (setup -> approach ->
what happened, including real wrong turns -> result -> what we learned),
with a real bibliography and appendix instead of one dense pass through
every stat inline. Motivating example: the split-protocol paragraph in v1
("bootstrap 95% CI... one-sided p=0.2425... paired sign test + Wilcoxon...")
was correct but unreadable and taught nothing on its own.

Re-read the actual ESConv paper PDF in full (not just prior WebFetch
summaries) to find real cross-references for v2. New, verified findings:
- Paper has NO standalone BERT/RoBERTa classifier -- backbones are
  BlenderBot-small (90M) and DialoGPT-small, both generative; strategy
  prediction is one step inside their "Joint" model's autoregressive
  generation, not an isolated classification eval. This is WHY EmoDynamiX,
  not this paper directly, is the right comparison point for our
  macro-F1 numbers -- stated explicitly in v2 (a correction vs. how v1
  implied the comparison).
- Real external validation: our own strategy-label distribution (full
  public ESConv.json, 1,300 conversations, 18,376 labeled utterances)
  matches the paper's own Table 3 (1,053 quality-filtered conversations,
  14,855 utterances) within ~1 point per class across all 8 strategies --
  confirms our parsing/extraction is correct despite using a larger,
  unfiltered release. Real scope caveat logged: 1,300 vs 1,053
  conversations, different releases.
- Paper's own Table 4 (automatic eval): their Joint (predicted strategy)
  scores LOWER than Oracle (gold strategy) on BLEU-2/ROUGE-L/Extrema,
  explicitly attributed to legitimate strategy-mismatch divergence from
  the single reference response, not model failure -- real precedent for
  our own predicted-vs-gold comparison.
- Paper's own Table 5 (human eval): despite lower automatic-metric scores,
  their Joint model still WINS vs Vanilla/Random in real human evaluation
  -- the paper's own evidence that reference-based automatic metrics
  undersell strategy-conditioned generation, used as real justification
  for this project's LLM-judge + human-annotation approach over BLEU/ROUGE.

Process: built the full narrative content as a reviewed markdown draft
first (`ESConv_Report_v2_draft.md`), read it end-to-end before converting,
then built the final docx from that reviewed draft (formatting-only pass,
no content changes) -- deliberately separating "get the content right" from
"make it a Word document" into two steps given the scale of the rewrite.

Also addressed this turn: Section 7 (error analysis) restructured so the
MAIN BODY carries only the key finding (selection-bias catch + 4-category
synthesis), with the full 10-item breakdown (all real, user's own
verbatim-derived reasoning per item, collected via a live interview) moved
to Appendix F. Bibliography added as a real numbered reference list
(8 entries) instead of repeating full citation strings inline throughout.

Output: `ESConv_Report_v2.docx`, 20 pages total (main body reads much
shorter than v1 despite the higher total page count, since the stat-heavy
material moved to the appendix). v1 (`ESConv_Report_v1.docx`) kept as-is,
not retired.
