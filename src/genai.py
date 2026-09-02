"""
src/genai.py

The GenAI subtask's two API stages. They share the ESConv strategy
definitions, so they live together:

  generate : generates the supporter's next response, conditioned on a target
             strategy, via the Claude API. 4 prompt levels of increasing
             detail (1=naive, 2=+persona/definition, 3=+XML structure/
             do-dont/banned phrases, 4=+few-shot) -- see report_notes.md for
             the design rationale and results.
  judge    : LLM-judge scoring, two providers sharing one rubric.
             --provider claude is the mini-sample pilot only (same model as
             the generator, so self-preference bias applies); --provider
             gemini is the independent judge used for every reported score.

Needs: ANTHROPIC_API_KEY for generate and --provider claude;
GEMINI_API_KEY / GOOGLE_API_KEY for --provider gemini.

Usage:
    python genai.py generate --csv final_200_new168.csv --levels 2,4 --out final_200_new168_prompt_comparison.csv
    python genai.py judge --provider gemini --csv final_200_prompt_comparison.csv --out final_200_judge_scores_gemini.json
"""

import argparse
import csv
import re
import time
import sys
from pathlib import Path
import anthropic
import json
import random
from collections import defaultdict


# ========================================================================
# --- from generate_responses.py -----------------------------------
# ========================================================================

# --- Strategy definitions (standard ESConv taxonomy, one line each) ---------

STRATEGY_DEFINITIONS = {
    "Question": "Asking for information to help clarify the seeker's situation, feelings, or experiences.",
    "Restatement or Paraphrasing": "Restating or paraphrasing what the seeker said, to show you understood it.",
    "Reflection of feelings": "Naming or reflecting back the seeker's emotions to show empathetic understanding.",
    "Self-disclosure": "Sharing a relatable experience or feeling of your own to connect with the seeker.",
    "Affirmation and Reassurance": "Affirming the seeker's strengths or effort and offering reassurance or encouragement.",
    "Providing Suggestions": "Offering a concrete suggestion or possible course of action to help the seeker.",
    "Information": "Providing objective, factual information relevant to the seeker's situation.",
    "Others": "General social exchange -- greetings, small talk, or closing remarks not fitting another strategy.",
}

# Strategy-specific Do/Don't guidance for level 3+ prompts.
STRATEGY_GUIDANCE = {
    "Question": "DO: Ask exactly one open-ended question. DON'T: Do not offer any advice or comment before the question. Do not ask a generic question like 'How does that make you feel?'.",
    "Restatement or Paraphrasing": "DO: Reflect the seeker's own words back concisely, to confirm understanding. DON'T: Do not add new advice or opinion. Do not just repeat their sentence verbatim.",
    "Reflection of feelings": "DO: State the specific emotion they are projecting, tied to what they said. DON'T: Do not use the phrase 'I understand'. Do not attempt to fix the problem.",
    "Self-disclosure": "DO: Share one brief, relevant experience of your own that relates. DON'T: Do not make it about yourself at length. Do not overshadow the seeker's situation.",
    "Affirmation and Reassurance": "DO: Affirm something specific the seeker did or felt. DON'T: Do not use generic platitudes like 'you've got this'. Do not dismiss the difficulty of their situation.",
    "Providing Suggestions": "DO: Offer one highly specific, actionable step, phrased collaboratively. DON'T: Do not give a bulleted list or multiple options. Do not use 'Have you tried...' or 'You should...'.",
    "Information": "DO: Give one specific, relevant fact. DON'T: Do not invent facts. Do not give unsolicited advice framed as fact.",
    "Others": "DO: Keep it short -- a greeting, acknowledgment, or closing remark. DON'T: Do not introduce a new topic or strategy here.",
}

# Generic LLM-therapist clichés banned outright at level 3+.
BANNED_PHRASES = ["I understand", "I'm sorry to hear that", "I hear you"]

# 2 real TRAIN-set examples per strategy (never val/test) for level-4
# few-shot, chosen to differ in length/register/topic (guards against
# single-example anchoring -- see report_notes.md).
FEWSHOT_EXAMPLES = {
    "Affirmation and Reassurance": [
        (
            "I do, but often they are not going to get back to what they want. Many people are going to lose their home when safeguards are lifted",
            "But you offer them a better future than what they have currently. It may not be what they wanted, but it helps them in the long run.",
        ),
        (
            "It really is a big decision Thank you for the different perspective",
            "No doubt, but you know in your heart what is right for you.",
        ),
    ],
    "Information": [
        (
            "I could try. It mostly gets to me at the end of the day",
            "Some people can't do what you do because they don't have the heart to give someone else bad news. The reality is though, someone needs to fill that role and you do help people",
        ),
        (
            "Yes, I really think you are correct! thank you so very much for your help today!",
            "The survivor rate from COVID-19 infections is around 99%, so your chances of dying from the virus are still low.",
        ),
    ],
    "Others": [
        (
            "Thanks for the idea. I will give it a try.",
            "I hope it goes well for you. It has been such an awful year, hasn't it?",
        ),
        (
            "Hi",
            "hello there, how are you",
        ),
    ],
    "Providing Suggestions": [
        (
            "That is also true. Sometimes I wonder if it really is for me though I've had to deal with collections before when I was in bad financial condition.",
            "It may not be for you. I think you should think about the pros and cons of keeping your position. It might make things clearer for you.",
        ),
        (
            "have tried that but don't even get an interview",
            "It's a very difficult time to be out of work, I know. I hope that there is something out there for you. Have you tried employment agencies?",
        ),
    ],
    "Question": [
        (
            "I have to deal with many people in hard financial situations and it is upsetting",
            "Do you help your clients to make it to a better financial situation?",
        ),
        (
            "I am having a lot of anxiety about quitting my current job. It is too stressful but pays well",
            "What makes your job stressful for you?",
        ),
    ],
    "Reflection of feelings": [
        (
            "family is not being fully supported compared to when I had full time job. had to get rid of one car and other things just to get by",
            "So you feel as though you have been working hard all your life and now you need help and support and are not getting it",
        ),
        (
            "Seriously! What I am scare of now is how to secure another job",
            "i can feel your pain just by chatting with you",
        ),
    ],
    "Restatement or Paraphrasing": [
        (
            "It just keeps getting better... I applied for the Pandemic Unemployment Assistance and was approved. The next day, I got an email asking me to verify",
            "So you have been getting some assistance?",
        ),
        (
            "I don't know what to do. I want to quit and punch the new manager in the face.",
            "So if I understand correctly, you are upset with your job and your manager?",
        ),
    ],
    "Self-disclosure": [
        (
            "That is also true. Sometimes I wonder if it really is for me though",
            "I've had to deal with collections before when I was in bad financial condition. The person on the other line was really helpful though.",
        ),
        (
            "very well put. seems i am being penalised for working hard, saving and investing? I should have been on more holidays and lived all the way above my means!",
            "I know how you feel .. I sometimes feel I would have been better off if I'd just not bothered working!",
        ),
    ],
}

# --- 3 diverse pilot items from the RoBERTa 8-class test predictions -------
MINI_SAMPLE = [
    {
        "idx": 231,
        "context": "[seeker] I see. Is this something that you've done to make money?",
        "gold_strategy": "Restatement or Paraphrasing",
        "pred_strategy": "Self-disclosure",
    },
    {
        "idx": 757,
        "context": "[seeker] having a rough day. [supporter] I am sorry you are having a rough day.",
        "gold_strategy": "Question",
        "pred_strategy": "Question",
    },
    {
        "idx": 1317,
        "context": "[seeker] Thank you. Enjoy the rest of your day!",
        "gold_strategy": "Providing Suggestions",
        "pred_strategy": "Others",
    },
]

MAX_WORDS = 40


def build_prompt(level, strategy, context):
    """Returns (system_prompt, user_prompt) for one of the 4 detail levels."""
    definition = STRATEGY_DEFINITIONS[strategy]
    guidance = STRATEGY_GUIDANCE[strategy]

    if level == 1:
        # naive baseline: no persona, no strategy definition
        system = "You are a chat assistant."
        user = (
            f"Conversation so far:\n{context}\n\n"
            f"Write your next reply using the \"{strategy}\" strategy. Max {MAX_WORDS} words."
        )
        return system, user

    if level == 2:
        system = (
            "You are a warm, professional, and conversational supporter in an "
            "emotional-support chat conversation, helping someone through a "
            "difficult situation."
        )
        user = (
            f"Conversation so far:\n{context}\n\n"
            f"Strategy to use: \"{strategy}\" -- {definition}\n"
            f"Write your next reply using this strategy. Max {MAX_WORDS} words. "
            f"Reply with ONLY the response text, nothing else."
        )
        return system, user

    if level == 3:
        system = (
            "You are a warm, professional, and conversational supporter in an "
            "emotional-support chat conversation, helping someone through a "
            "difficult situation. You follow a specific support strategy precisely "
            "for each reply."
        )
        banned = ", ".join(f'"{p}"' for p in BANNED_PHRASES)
        user = (
            f"<conversation>\n{context}\n</conversation>\n\n"
            f"<strategy_directive>\n"
            f"Strategy: \"{strategy}\" -- {definition}\n"
            f"{guidance}\n"
            f"Never use these phrases: {banned}.\n"
            f"</strategy_directive>\n\n"
            f"Write your next reply using this strategy. Max {MAX_WORDS} words -- count "
            f"before answering. Reply with ONLY the response text, nothing else."
        )
        return system, user

    if level == 4:
        examples = FEWSHOT_EXAMPLES[strategy]
        system = (
            "You are a warm, professional, and conversational supporter in an "
            "emotional-support chat conversation, helping someone through a "
            "difficult situation. You follow a specific support strategy precisely "
            "for each reply."
        )
        banned = ", ".join(f'"{p}"' for p in BANNED_PHRASES)
        examples_block = "\n".join(
            f"<successful_example>\n"
            f"<context>{ex_context}</context>\n"
            f"<supporter_reply>{ex_response}</supporter_reply>\n"
            f"</successful_example>"
            for ex_context, ex_response in examples
        )
        user = (
            f"<strategy_directive>\n"
            f"Strategy: \"{strategy}\" -- {definition}\n"
            f"{guidance}\n"
            f"Never use these phrases: {banned}.\n"
            f"</strategy_directive>\n\n"
            f"{examples_block}\n\n"
            f"The two examples above show the range of this strategy -- do not copy "
            f"either one's wording or sentence structure; express the strategy in "
            f"your own words for the conversation below.\n\n"
            f"<conversation>\n{context}\n</conversation>\n\n"
            f"Write your own reply using the same strategy for the conversation above. "
            f"Max {MAX_WORDS} words -- count before answering. Reply with ONLY the "
            f"response text, nothing else."
        )
        return system, user

    raise ValueError(f"Unknown level: {level}")


# Strips a self-reported trailing word count the model sometimes appends,
# e.g. "...whatever you're considering? (14 words)" -- see report_notes.md
# "AI was wrong". Logged via the returned was_stripped flag, not silent.
WORD_COUNT_LEAK_PATTERN = re.compile(r"\s*\(\d+\s*words?\)\.?\s*$", re.IGNORECASE)


def strip_meta_commentary(text):
    """Strip a trailing self-reported word-count annotation. Returns
    (cleaned_text, was_stripped)."""
    cleaned = WORD_COUNT_LEAK_PATTERN.sub("", text).strip()
    return cleaned, cleaned != text


# Strips a leading markdown header (e.g. "# Response\n\n") -- a level-1-only
# leak, since level 1 lacks the "reply with ONLY the response text"
# instruction levels 2-4 have. See report_notes.md "AI was wrong".
LEADING_HEADER_PATTERN = re.compile(r"^#{1,6}[ \t]*\S[^\n]*\n+", re.MULTILINE)


def strip_leading_header(text):
    """Strip a leading markdown header line (e.g. '# Response\\n\\n').
    Returns (cleaned_text, was_stripped)."""
    cleaned = LEADING_HEADER_PATTERN.sub("", text, count=1).strip()
    return cleaned, cleaned != text


def generate(client, model, system, user, max_tokens=100):
    call_start = time.time()  # real per-call latency, for the report's cost/time table
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    latency_seconds = time.time() - call_start
    text = response.content[0].text.strip()
    return text, response.usage.input_tokens, response.usage.output_tokens, latency_seconds


def load_competition_items(csv_path):
    """Loads a hand_annotation_sample.csv-format file. Returns items with
    both true_label and pred_label -- main() picks which to target per
    --conditions."""
    items = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            items.append({
                "idx": row["idx"],
                "context": row["context"],
                "true_label": row["true_label"],
                "pred_label": row["pred_label"],
                "correct": row["correct"],
            })
    return items


def main_generate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="claude-haiku-4-5")
    parser.add_argument("--csv", default=None, help="hand_annotation_sample.csv-format file; omit for the 3-item mini-sample.")
    parser.add_argument("--out", default=None, help="Output CSV filename under outputs/genai/.")
    parser.add_argument("--levels", default="1,2,3,4", help="Comma-separated prompt levels, e.g. '2,4'.")
    parser.add_argument("--conditions", default="predicted", help="Comma-separated: 'predicted' and/or 'gold'. Only meaningful with --csv.")
    args = parser.parse_args()

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    using_competition_set = args.csv is not None
    items = load_competition_items(args.csv) if using_competition_set else MINI_SAMPLE
    levels = [int(x) for x in args.levels.split(",")]
    conditions = [c.strip() for c in args.conditions.split(",")] if using_competition_set else [None]
    for c in conditions:
        if c is not None and c not in ("predicted", "gold"):
            raise ValueError(f"--conditions must be 'predicted' and/or 'gold', got {c!r}")

    rows = []
    total_in, total_out = 0, 0
    total_latency = 0.0
    n_stripped = 0
    n_header_stripped = 0
    for item in items:
        for condition in conditions:
            if using_competition_set:
                strategy = item["pred_label"] if condition == "predicted" else item["true_label"]
            else:
                strategy = item["gold_strategy"]
            for level in levels:
                system, user = build_prompt(level, strategy, item["context"])
                text, in_tok, out_tok, latency_s = generate(client, args.model, system, user)
                text, was_stripped = strip_meta_commentary(text)
                if was_stripped:
                    n_stripped += 1
                    print(f"  NOTE: stripped a trailing word-count annotation from "
                          f"idx={item['idx']} level={level}")
                text, was_header_stripped = strip_leading_header(text)
                if was_header_stripped:
                    n_header_stripped += 1
                    print(f"  NOTE: stripped a leading markdown header from "
                          f"idx={item['idx']} level={level}")
                total_in += in_tok
                total_out += out_tok
                total_latency += latency_s
                word_count = len(text.split())  # after stripping, so this reflects the real response
                cond_label = f" condition={condition}" if condition else ""
                print(f"--- idx={item['idx']} level={level}{cond_label} strategy={strategy!r} "
                      f"(words={word_count}, in_tok={in_tok}, out_tok={out_tok}, "
                      f"latency={latency_s:.2f}s) ---")
                print(text)
                print()
                row = {
                    "idx": item["idx"], "level": level, "strategy": strategy,
                    "context": item["context"], "response": text, "word_count": word_count,
                    "input_tokens": in_tok, "output_tokens": out_tok,
                    "latency_seconds": round(latency_s, 3),
                    "meta_commentary_stripped": was_stripped,
                    "leading_header_stripped": was_header_stripped,
                }
                if using_competition_set:
                    row["condition"] = condition
                    row["true_label"] = item["true_label"]
                    row["pred_label"] = item["pred_label"]
                    row["correct"] = item["correct"]
                rows.append(row)

    output_dir = Path(__file__).resolve().parent.parent / "outputs" / "genai"
    output_dir.mkdir(parents=True, exist_ok=True)
    default_name = "full32_prompt_comparison.csv" if using_competition_set else "mini_sample_prompt_comparison.csv"
    out_path = output_dir / (args.out if args.out else default_name)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"=== Done: {len(rows)} generations, {total_in} input tokens, {total_out} output tokens, "
          f"{total_latency:.1f}s total wall-clock ({total_latency/len(rows):.2f}s/call avg), "
          f"{n_stripped} response(s) had a word-count annotation stripped, "
          f"{n_header_stripped} response(s) had a leading markdown header stripped ===")
    print(f"Saved: {out_path}")
    if using_competition_set:
        print(f"NOTE: levels={levels} conditions={conditions}. Judge next with: "
              f"python judge.py --provider gemini --csv {out_path.name}")
    else:
        print("NOTE: mini-sample cost/latency is exploratory only -- the assignment's required "
              "cost/latency table is built from the winning approach's full production run.")


# ========================================================================
# --- from judge.py ------------------------------------------------
# ========================================================================

MINI_SAMPLE_BLIND_SEED = 7  # fixed so re-running reproduces the same shuffle -- judge_mini_sample.py's original value

JUDGE_SYSTEM = (
    "You are an expert evaluator of supportive chat responses, judging "
    "against the ESConv taxonomy of emotional support strategies. You are a "
    "strict grader: you use the full 1-5 range and heavily penalize any "
    "leaked system or formatting artifact in a response."
)


# --- shared: judge prompt + JSON parsing (identical to both originals) -----

def build_judge_prompt(context, strategy, definition, response):
    # "reasoning" is deliberately the FIRST field in the required JSON object,
    # not the last -- chain-of-thought via field order (Zheng et al. 2023,
    # G-Eval/Liu et al. 2023 EMNLP), see report_notes.md.
    #
    # Revised for the 200-item run (v2): (1) an explicit hard cap when a
    # leaked system/formatting artifact is present in the response -- the
    # original rubric only asked about "naturalness" in general terms, which
    # didn't reliably catch the word-count/markdown-header leaks documented
    # in Section 6; (2) explicit per-score anchors on both dimensions, to
    # fight the ceiling-effect clustering the original open-ended 1-5 scale
    # produced (Section 4.3/Appendix G). NOT directly comparable to scores
    # from the original rubric (pilot / 100-item run) -- see report_notes.md.
    user = (
        f"<context>{context}</context>\n"
        f"<target_strategy>{strategy} -- {definition}</target_strategy>\n"
        f"<response>{response}</response>\n\n"
        f"Evaluate this response on two dimensions, each an integer 1-5 (5=best). Use "
        f"the full 1-5 range -- most responses are NOT a 5; reserve 5 for a response you "
        f"cannot meaningfully improve, and 4 for a solid response with at least one real, "
        f"nameable flaw.\n\n"
        f"1. quality: overall naturalness, appropriateness, and helpfulness as a "
        f"supportive chat reply. For tone specifically: does it read like one person "
        f"messaging another in an ongoing conversation, in whatever phrasing achieves "
        f"that -- this is not a preference for shorter or longer replies, or for or "
        f"against opening with an acknowledgment; judge the tone on its own terms for "
        f"this exchange.\n"
        f"   HARD CAP: if the response contains ANY leaked system/formatting artifact -- "
        f"a self-reported word count, a markdown header or code fence, a preamble like "
        f"\"Here's my response:\", a role label like \"[supporter]\", any acknowledgment "
        f"that it is an AI, or text that reads like a platform/interface message (e.g. "
        f"\"press quit\", a session-ending notice) rather than a genuine reply from one "
        f"person to another -- quality CANNOT score above 2, regardless of how good the "
        f"rest of the response otherwise reads. Name the specific artifact in reasoning.\n"
        f"   Quality anchors: 5=natural, no flaws you can point to. 4=solid, one minor "
        f"flaw (slightly stiff phrasing, a bit generic). 3=readable but has a real, "
        f"nameable problem (tonally off, repetitive, mismatched length for the moment). "
        f"2=noticeably broken (any leaked artifact above, or badly mismatched tone). "
        f"1=not a coherent supportive reply at all.\n"
        f"2. strategy_adherence: how well the response actually executes the "
        f"target strategy above (not just whether it's on-topic).\n"
        f"   Adherence anchors: 5=textbook execution of the strategy. 4=clearly the "
        f"right strategy, minor execution slip. 3=recognizable but weak or partial "
        f"execution. 2=only a surface gesture at the strategy. 1=does not execute the "
        f"strategy at all.\n\n"
        f"Reply with ONLY a JSON object, no other text -- no markdown code fences "
        f"(no ```), no commentary before or after. Write the \"reasoning\" field "
        f"FIRST and think through both dimensions there BEFORE assigning scores -- do "
        f"not decide the numbers first and justify them afterward:\n"
        f'{{"reasoning": "<2-3 sentences reasoning through quality and strategy_adherence '
        f'before scoring, naming any artifact found>", "quality": <1-5>, "strategy_adherence": <1-5>}}'
    )
    return JUDGE_SYSTEM, user


def extract_json_object(text):
    """Real observed failure mode (mini-sample run 2, 9/12 judge calls): the
    model wraps its JSON reply in markdown code fences (```json ... ```)
    despite being told not to. Strips a leading/trailing fence if present;
    falls back to the raw text otherwise."""
    stripped = text.strip()
    fence_match = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", stripped, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()
    return stripped


def load_rows(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# --- provider=claude: mini-sample blind ranking quiz + judging -------------

def build_blind_ranking_quiz(rows, seed):
    by_idx = defaultdict(list)
    for row in rows:
        by_idx[row["idx"]].append(row)

    quiz_items = []
    key = {}
    letters = ["A", "B", "C", "D"]
    for idx, group in sorted(by_idx.items(), key=lambda kv: int(kv[0])):
        assert len(group) == 4, f"idx {idx}: expected 4 levels, got {len(group)}"
        rng = random.Random(seed + int(idx))
        shuffled = group[:]
        rng.shuffle(shuffled)
        options, idx_key = {}, {}
        for letter, row in zip(letters, shuffled):
            options[letter] = row["response"]
            idx_key[letter] = row["level"]
        quiz_items.append({
            "idx": idx, "context": group[0]["context"], "strategy": group[0]["strategy"],
            "options": options,
        })
        key[idx] = idx_key
    return quiz_items, key


def write_ranking_quiz_file(quiz_items, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Mini-sample prompt-level comparison -- blinded ranking quiz\n")
        f.write("=" * 60 + "\n")
        f.write("For each item, read the conversation + target strategy, then RANK all\n")
        f.write("4 responses (A/B/C/D) from best (1) to worst (4) at executing that\n")
        f.write("strategy. Options are shuffled per item -- don't assume a fixed order.\n\n")
        for item in quiz_items:
            f.write(f"--- Item idx={item['idx']}  strategy={item['strategy']!r} ---\n")
            f.write(f"Conversation: {item['context']}\n\n")
            for letter in ["A", "B", "C", "D"]:
                f.write(f"  {letter}. {item['options'][letter]}\n")
            f.write("\nYour ranking, best to worst (e.g. \"C, A, D, B\"): ____\n\n")


def run_claude(args, repo_root):
    import anthropic

    csv_path = Path(args.csv) if args.csv else repo_root / "outputs" / "genai" / "mini_sample_prompt_comparison.csv"
    rows = load_rows(csv_path)
    out_dir = repo_root / "outputs" / "genai"
    out_dir.mkdir(parents=True, exist_ok=True)

    quiz_items, key = build_blind_ranking_quiz(rows, MINI_SAMPLE_BLIND_SEED)
    write_ranking_quiz_file(quiz_items, out_dir / "mini_sample_quiz.txt")
    with open(out_dir / "mini_sample_blind_key.json", "w") as f:
        json.dump(key, f, indent=1)
    print(f"Wrote blind quiz: {out_dir / 'mini_sample_quiz.txt'}")
    print(f"Wrote blind key (don't peek before answering): {out_dir / 'mini_sample_blind_key.json'}")

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    scores = []
    total_in, total_out = 0, 0
    total_latency = 0.0
    for row in rows:
        strategy = row["strategy"]
        definition = STRATEGY_DEFINITIONS[strategy]
        system, user = build_judge_prompt(row["context"], strategy, definition, row["response"])
        call_start = time.time()
        response = client.messages.create(
            model=args.model, max_tokens=250, system=system,
            messages=[{"role": "user", "content": user}],
        )
        latency_s = time.time() - call_start
        total_latency += latency_s
        text = response.content[0].text.strip()
        total_in += response.usage.input_tokens
        total_out += response.usage.output_tokens
        try:
            parsed = json.loads(extract_json_object(text))
        except json.JSONDecodeError:
            print(f"WARNING: judge did not return valid JSON for idx={row['idx']} level={row['level']}: {text!r}")
            parsed = {"quality": None, "strategy_adherence": None, "reasoning": f"PARSE_ERROR: {text}"}
        scores.append({
            "idx": row["idx"], "level": row["level"], "strategy": strategy,
            "quality": parsed.get("quality"),
            "strategy_adherence": parsed.get("strategy_adherence"),
            "reasoning": parsed.get("reasoning"),
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "latency_seconds": round(latency_s, 3),
        })
        print(f"  judged idx={row['idx']} level={row['level']}: "
              f"quality={parsed.get('quality')} adherence={parsed.get('strategy_adherence')} "
              f"(latency={latency_s:.2f}s)")

    out_path = out_dir / (args.out if args.out else "mini_sample_judge_scores.json")
    with open(out_path, "w") as f:
        json.dump(scores, f, indent=1)
    print(f"\nWrote judge scores: {out_path}")
    print(f"Judge API usage: {total_in} input tokens, {total_out} output tokens, "
          f"{total_latency:.1f}s total wall-clock ({total_latency/len(rows):.2f}s/call avg)")
    print(f"\nNEXT STEP: read {out_dir / 'mini_sample_quiz.txt'}, create "
          f"{out_dir / 'mini_sample_human_picks.json'} with your rankings "
          f"(see combine_mini_sample_scores.py for the format), then run "
          f"combine_mini_sample_scores.py")


# --- provider=gemini: independent judging of an arbitrary CSV --------------

def run_gemini(args, repo_root):
    from google import genai
    from google.genai import types

    csv_path = Path(args.csv) if args.csv else repo_root / "outputs" / "genai" / "full32_prompt_comparison.csv"
    rows = load_rows(csv_path)
    out_dir = repo_root / "outputs" / "genai"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (args.out if args.out else "full32_judge_scores_gemini.json")

    client = genai.Client()  # reads GEMINI_API_KEY / GOOGLE_API_KEY from env

    scores = []
    total_in, total_out = 0, 0
    total_latency = 0.0
    n_parse_errors = 0
    for i, row in enumerate(rows, start=1):
        strategy = row["strategy"]
        definition = STRATEGY_DEFINITIONS[strategy]
        system, user = build_judge_prompt(row["context"], strategy, definition, row["response"])

        call_start = time.time()
        response = client.models.generate_content(
            model=args.model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                # Gemini 2.5 Flash's internal "thinking" tokens count against
                # max_output_tokens by default -- disabled explicitly (CoT is
                # already engineered via field order above), plus a higher
                # ceiling as a safety margin. See report_notes.md "GenAI
                # judge artifact: Gemini thinking-token truncation".
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                max_output_tokens=500,
            ),
        )
        latency_s = time.time() - call_start
        total_latency += latency_s

        text = (response.text or "").strip()
        in_tok = response.usage_metadata.prompt_token_count or 0
        out_tok = response.usage_metadata.candidates_token_count or 0
        total_in += in_tok
        total_out += out_tok

        try:
            parsed = json.loads(extract_json_object(text))
        except json.JSONDecodeError:
            n_parse_errors += 1
            print(f"WARNING: judge did not return valid JSON for idx={row['idx']} level={row['level']}: {text!r}")
            parsed = {"quality": None, "strategy_adherence": None, "reasoning": f"PARSE_ERROR: {text}"}

        scores.append({
            "idx": row["idx"], "level": row["level"], "strategy": strategy,
            "true_label": row.get("true_label"), "correct": row.get("correct"),
            "quality": parsed.get("quality"),
            "strategy_adherence": parsed.get("strategy_adherence"),
            "reasoning": parsed.get("reasoning"),
            "input_tokens": in_tok, "output_tokens": out_tok,
            "latency_seconds": round(latency_s, 3),
        })
        print(f"  [{i}/{len(rows)}] judged idx={row['idx']} level={row['level']} "
              f"correct={row.get('correct')}: quality={parsed.get('quality')} "
              f"adherence={parsed.get('strategy_adherence')} (latency={latency_s:.2f}s)")

    with open(out_path, "w") as f:
        json.dump(scores, f, indent=1)
    print(f"\nWrote judge scores: {out_path}")
    print(f"Judge API usage: {total_in} input tokens, {total_out} output tokens, "
          f"{total_latency:.1f}s total wall-clock ({total_latency/len(rows):.2f}s/call avg), "
          f"{n_parse_errors} parse error(s)")

    per_level_all = defaultdict(list)
    per_level_correct = defaultdict(list)
    per_level_wrong = defaultdict(list)
    for s in scores:
        if s["quality"] is None or s["strategy_adherence"] is None:
            continue
        mean_score = (s["quality"] + s["strategy_adherence"]) / 2
        per_level_all[s["level"]].append(mean_score)
        if s["correct"] == "True":
            per_level_correct[s["level"]].append(mean_score)
        elif s["correct"] == "False":
            per_level_wrong[s["level"]].append(mean_score)

    print(f"\n=== Per-level mean score (quality+strategy_adherence)/2, all {len(rows)} items ===")
    ranked = sorted(per_level_all.items(), key=lambda kv: -sum(kv[1]) / len(kv[1]))
    for level, vals in ranked:
        print(f"  level {level}: mean={sum(vals) / len(vals):.3f}  (n={len(vals)})")
    if len(ranked) >= 2:
        print(f"\nTop 2 candidates for the assignment's xlsx step: level {ranked[0][0]}, level {ranked[1][0]}")

    print(f"\n=== Same breakdown, split by correct (matched signal) vs wrong (mismatched signal) ===")
    print(f"{'level':>5} {'correct_mean':>13} {'n':>3}   {'wrong_mean':>11} {'n':>3}   {'gap':>6}")
    for level in sorted(per_level_all.keys(), key=int):
        c_vals = per_level_correct.get(level, [])
        w_vals = per_level_wrong.get(level, [])
        c_mean = sum(c_vals) / len(c_vals) if c_vals else float("nan")
        w_mean = sum(w_vals) / len(w_vals) if w_vals else float("nan")
        gap = c_mean - w_mean if c_vals and w_vals else float("nan")
        print(f"{level:>5} {c_mean:>13.3f} {len(c_vals):>3}   {w_mean:>11.3f} {len(w_vals):>3}   {gap:>6.3f}")
    print("\n(gap > 0 means quality/adherence dropped when the requested strategy didn't match\n"
          " the conversation content -- i.e. generation degraded under a wrong signal.)")


def main_judge():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["claude", "gemini"], required=True)
    parser.add_argument("--model", default=None, help="Default: claude-haiku-4-5 or gemini-2.5-flash per --provider.")
    parser.add_argument("--csv", default=None, help="Input prompt-comparison CSV (see per-provider default above).")
    parser.add_argument("--out", default=None, help="Output filename under outputs/genai/ (see per-provider default above).")
    args = parser.parse_args()
    if args.model is None:
        args.model = "claude-haiku-4-5" if args.provider == "claude" else "gemini-2.5-flash"

    repo_root = Path(__file__).resolve().parent.parent
    if args.provider == "claude":
        run_claude(args, repo_root)
    else:
        run_gemini(args, repo_root)


# ========================================================================
# --- dispatcher --------------------------------------------------------
# ========================================================================

COMMANDS = {
    "generate": (main_generate, "Generate responses via the Claude API."),
    "judge": (main_judge, "Score generated responses with an LLM judge."),
}


def main():
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
