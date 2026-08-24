"""Claim-level comparison between a replayed model's Web-search-on ("auto")
and Web-search-off ("none") responses to the same prompt: extracts atomic
claims from each response, then has an LLM judge align and classify the
claim pairs (MATCH/REFINEMENT/CONTRADICTION/UNMATCHED) to characterize what
Web search actually changed about the response, not just whether quality
scores moved.

Claim extraction reuses src.response_generation.claim_extraction.
extract_claims_from_text() (the same extractor response_generation.py's
NLI pipeline uses) rather than a separate duplicated implementation.
Claim comparison is judged by the replayed model's OWN platform's model
(see src.utils.llm_judge.JUDGE_MODEL_BY_PLATFORM and
chat_replayer_evaluation.judge_platform_for_replay_model), not a single
fixed judge -- consistent with every other judge in this codebase.

Lives in src/replays/ (not src/web_search_decision/) alongside
chat_replayer.py/chat_replayer_evaluation.py/extract_replay_artifacts.py:
it operates on chat_replayer.py's replay output, not on the original
web-search-decision conversation data.

Run directly (`python -m src.replays.claim_analysis --help`) for the CLI:
build a claim analysis from a replay file, print a summary, or plot a
multi-model comparison. Requires the relevant provider API key(s) in .env
(see llm_judge.py), and a replay file already produced by
src/replays/chat_replayer.py.
"""

import argparse
import ast
import json
import logging
import os
import re
from collections import Counter
from pathlib import Path

import plotly.graph_objects as go
from dotenv import load_dotenv
from tqdm import tqdm

from src.prompts.evaluator_prompts import (
    SYSTEM_PROMPT_CLAIM_COMPARISON,
    USER_PROMPT_CLAIM_COMPARISON,
)
from src.replays.chat_replayer_evaluation import judge_platform_for_replay_model
from src.replays.extract_replay_artifacts import (
    DEFAULT_MODELS as _REPLAY_DEFAULT_MODELS,
    _has_web_tool_call,
    _infer_provider,
)
from src.response_generation.claim_extraction import extract_claims_from_text as _extract_claims_from_text
from src.utils.figure_style import styler, with_paper_style
from src.utils.common_io import OUTPUT_PATH, load_json, to_json
from src.utils.llm_judge import run_judge


load_dotenv()

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

# The replay model roster (and their actual outputs/replays/<model>.json
# filenames) lives in one place -- extract_replay_artifacts.DEFAULT_MODELS
# -- not duplicated here, so this list can't drift out of sync with which
# replay files actually exist on disk.
MODEL_NAMES = list(_REPLAY_DEFAULT_MODELS)
model_name = MODEL_NAMES[0]

_PLATFORM_DISPLAY_NAME = {
    "chatgpt": "ChatGPT",
    "claude": "Claude",
    "grok": "Grok",
    "deepseek": "Deepseek",
}
MODEL_NAMES_MAP = {
    model_name_value: _PLATFORM_DISPLAY_NAME.get(
        judge_platform_for_replay_model(model_name_value), model_name_value
    )
    for model_name_value in MODEL_NAMES
}
# platform -> its default replay model name (inverse of
# judge_platform_for_replay_model over MODEL_NAMES) -- lets --platform on
# the CLI select the right replay file, not just the judge model.
PLATFORM_TO_MODEL_NAME = {
    judge_platform_for_replay_model(model_name_value): model_name_value
    for model_name_value in MODEL_NAMES
}

REPLAY_PATH = Path(f"{OUTPUT_PATH}/replays/{model_name}.json")
OUTPUT_PATH_CLAIMS = Path(f"{OUTPUT_PATH}/replays/extracted/{model_name}_claims.json")
CACHE_PATH = Path(f"{OUTPUT_PATH}/replays/extracted/{model_name}_claims_cache.json")
PLOT_OUTPUT_DIR = Path(f"{OUTPUT_PATH}/replays/plots")


def _load_replay_json(path):
    """Load a chat_replayer.py results file (JSON, or Python-literal text as
    a fallback) and confirm it parses to a {result_key: result} dict."""
    with open(path) as f:
        raw_text = f.read()

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(raw_text)

    if not isinstance(parsed, dict):
        raise ValueError(
            f"Replay file `{path}` did not parse to a dict; got {type(parsed).__name__}."
        )
    return parsed


def _clean_claims(claims):
    """Normalize a list of claims to short-enough, alphabetic, deduplicated
    strings (each claim may be a plain string or a {"claim"/"text"/
    "statement": ...} dict)."""
    cleaned_claims = []
    seen = set()
    for claim in claims:
        if isinstance(claim, dict):
            claim = (
                claim.get("claim")
                or claim.get("text")
                or claim.get("statement")
                or ""
            )
        claim = str(claim or "").strip(" -*\t\n")
        if len(claim) < 8 or not re.search(r"[A-Za-z]", claim):
            continue
        if claim in seen:
            continue
        seen.add(claim)
        cleaned_claims.append(claim)
    return cleaned_claims


def _normalize_claim_comparison_judgment(payload):
    """Normalize a claim-comparison judge's parsed JSON to a consistent
    shape: {"category": str, "explanation": str, "alignments": [...]}, with
    each alignment's "relation" mapped through
    _normalize_claim_level_relation_label and older/alternate key names
    ("no_web_claim"/"web_claim") folded into "claim_a"/"claim_b". Returns {}
    if the payload has nothing usable."""
    if not isinstance(payload, dict):
        return {}

    category = str(payload.get("category", "") or "").strip()
    explanation = str(payload.get("explanation", "") or "").strip()

    normalized_alignments = []
    for item in payload.get("alignments", []) or []:
        if not isinstance(item, dict):
            continue
        relation = str(item.get("relation", "") or "").strip()
        if relation == "NEW_CLAIM":
            relation = "UNMATCHED"
        normalized_item = {
            "claim_a": str(
                item.get("claim_a", item.get("no_web_claim", "")) or ""
            ).strip(),
            "relation": relation,
            "claim_b": str(
                item.get("claim_b", item.get("web_claim", "")) or ""
            ).strip(),
        }
        if (
            normalized_item["claim_a"]
            or normalized_item["relation"]
            or normalized_item["claim_b"]
        ):
            normalized_alignments.append(normalized_item)

    normalized_payload = {
        "category": category,
        "explanation": explanation,
        "alignments": normalized_alignments,
    }
    if not category and not explanation and not normalized_alignments:
        return {}
    return normalized_payload


def _normalize_claim_level_relation_label(label):
    """Map the legacy "NEW_CLAIM" relation label to its current name,
    "UNMATCHED" (see SYSTEM_PROMPT_CLAIM_COMPARISON); passes any other
    label through unchanged."""
    label = str(label or "").strip()
    if label == "NEW_CLAIM":
        return "UNMATCHED"
    return label


def extract_claims_from_text(text):
    """Extract a deduplicated list of atomic factual claims from `text`
    (typically one response's final answer). Thin wrapper around
    src.response_generation.claim_extraction.extract_claims_from_text --
    the same extractor response_generation.py's NLI pipeline uses -- so
    claim extraction isn't duplicated between the two pipelines. Returns
    [] on empty input or extraction failure."""
    return _extract_claims_from_text(text)


def compare_claim_sets(user_query, claims_without_web, claims_with_web, platform="chatgpt"):
    """Have `platform`'s own judge model (see llm_judge.JUDGE_MODEL_BY_PLATFORM)
    align and classify two claim sets for the same prompt
    (SYSTEM_PROMPT_CLAIM_COMPARISON) -- e.g. the claims extracted from a
    response generated without vs. with Web search. `platform` should be
    the replayed model's own platform (see judge_platform_for_replay_model)
    so a model's claim changes are judged by that model's own family, not
    a single fixed judge. Returns {"raw_output": str, "judgment":
    normalized_dict, "error": str}; on judge failure, judgment is {} and
    error explains why."""
    user_prompt = USER_PROMPT_CLAIM_COMPARISON.format(
        user_query=str(user_query or "").strip(),
        claims_without_web=json.dumps(
            _clean_claims(claims_without_web), ensure_ascii=False, indent=2
        ),
        claims_with_web=json.dumps(
            _clean_claims(claims_with_web), ensure_ascii=False, indent=2
        ),
    )

    try:
        judge_result = run_judge(
            platform,
            system_prompt=SYSTEM_PROMPT_CLAIM_COMPARISON,
            user_prompt=user_prompt,
            temperature=0,
        )
    except Exception as exc:
        logger.warning("Claim comparison judge failed: %s", exc)
        return {"judgment": {}, "error": str(exc)}

    normalized_judgment = _normalize_claim_comparison_judgment(
        judge_result["parsed_judgment"]
    )
    return {
        "raw_output": judge_result["raw_judgment"],
        "judgment": normalized_judgment,
        "error": "",
    }


def _sample_has_auto_web_call(row, model_name_value):
    """True if a replayed sample's "auto" (Web-search-allowed) response
    actually invoked Web search -- build_claim_analysis() only compares
    samples where it did, since otherwise "auto" and "none" wouldn't differ
    in the way being studied."""
    auto_payload = row.get("auto") or {}
    response = auto_payload.get("response") or {}
    provider = _infer_provider(model_name_value, row)
    return _has_web_tool_call(provider, response)


def _load_cache(cache_path):
    """Load the claim-extraction cache (text -> claims) at `cache_path`,
    or {} if it doesn't exist / isn't a dict yet."""
    cache = load_json(cache_path)
    if isinstance(cache, dict):
        return cache
    return {}


def _extract_claims_cached(text, cache, cache_dirty_state):
    """extract_claims_from_text(text), cached by exact text match in
    `cache` so re-running build_claim_analysis() doesn't re-spend API calls
    on responses already seen. Sets cache_dirty_state["dirty"] = True on a
    cache miss so the caller knows to persist the updated cache."""
    text = str(text or "").strip()
    if not text:
        return []

    if text in cache:
        return _clean_claims(cache[text])

    claims = extract_claims_from_text(text)
    cache[text] = claims
    cache_dirty_state["dirty"] = True
    return claims


def build_claim_analysis(
    replay_path=REPLAY_PATH,
    output_path=OUTPUT_PATH_CLAIMS,
    cache_path=CACHE_PATH,
    limit=None,
    model_name_value=model_name,
    platform=None,
):
    """For every replayed sample at `replay_path` whose "auto" response
    actually called Web search, extract claims from both its "none" and
    "auto" responses and have a judge compare them. Writes the per-sample
    results to `output_path` and returns them; also persists the claim-
    extraction cache to `cache_path` if it changed. `limit`, if given, caps
    how many qualifying samples are processed (useful for a quick check
    before a full, costly run). `platform` is the judge platform for
    compare_claim_sets (default: derived from `model_name_value` via
    judge_platform_for_replay_model, i.e. the replayed model's own
    platform).
    """
    platform = platform or judge_platform_for_replay_model(model_name_value)
    replay_data = _load_replay_json(replay_path)
    cache = _load_cache(cache_path)
    cache_dirty_state = {"dirty": False}

    selected_items = [
        (result_key, row)
        for result_key, row in replay_data.items()
        if isinstance(row, dict)
        and not row.get("skipped_replay")
        and row.get("auto")
        and row.get("none")
        and _sample_has_auto_web_call(row, model_name_value)
    ]

    if limit is not None:
        selected_items = selected_items[:limit]

    results = {}
    for result_key, row in tqdm(selected_items, desc="Extracting claims"):
        auto_text = str((row.get("auto") or {}).get("output_text", "") or "")
        none_text = str((row.get("none") or {}).get("output_text", "") or "")
        user_prompt = row.get("user_prompt")
        none_claims = _extract_claims_cached(none_text, cache, cache_dirty_state)
        auto_claims = _extract_claims_cached(auto_text, cache, cache_dirty_state)
        comparison = compare_claim_sets(
            user_prompt,
            none_claims,
            auto_claims,
            platform=platform,
        )

        results[result_key] = {
            "sample_idx": row.get("sample_idx"),
            "sample_source": row.get("sample_source"),
            "conv_id": row.get("conv_id"),
            "turn_id": row.get("turn_id"),
            "user_prompt": user_prompt,
            "none": {
                "final_response": none_text,
                "claims": none_claims,
            },
            "auto": {
                "final_response": auto_text,
                "claims": auto_claims,
            },
            "claim_comparison_judge": comparison,
        }

    to_json(results, output_path, indent=2)
    if cache_dirty_state["dirty"]:
        to_json(cache, cache_path, indent=2)

    logger.info("Saved %s samples to %s", len(results), output_path)
    return results


def print_claim_comparison_summary(input_path=OUTPUT_PATH_CLAIMS):
    """Print the claim-level relation distribution (MATCH/REFINEMENT/
    CONTRADICTION/UNMATCHED) and response-level category distribution
    (SAME_CLAIMS/UPDATED_OR_SPECIFIED/CORRECTED/NEW_CLAIMS) for a single
    model's build_claim_analysis() output at `input_path`."""
    records = load_json(input_path)
    if not isinstance(records, dict) or not records:
        print(f"No claim analysis records found at {input_path}.")
        return

    relation_counter = Counter()
    category_counter = Counter()
    total_alignments = 0
    total_responses = 0

    for sample in records.values():
        if not isinstance(sample, dict):
            continue

        judge_payload = sample.get("claim_comparison_judge") or {}
        judgment = judge_payload.get("judgment") or {}
        if not isinstance(judgment, dict):
            continue

        category = str(judgment.get("category", "") or "").strip()
        if category:
            category_counter[category] += 1
            total_responses += 1

        for alignment in judgment.get("alignments", []) or []:
            if not isinstance(alignment, dict):
                continue
            relation = _normalize_claim_level_relation_label(
                alignment.get("relation", "")
            )
            if not relation:
                continue
            relation_counter[relation] += 1
            total_alignments += 1

    print("Claim-level relation summary")
    print(f"Total aligned web claims: {total_alignments}")
    for relation, count in sorted(
        relation_counter.items(), key=lambda item: (-item[1], item[0])
    ):
        percentage = (100.0 * count / total_alignments) if total_alignments else 0.0
        print(f"  {relation}: {count} ({percentage:.2f}%)")

    print()
    print("Response-level category summary")
    print(f"Total judged responses: {total_responses}")
    for category, count in sorted(
        category_counter.items(), key=lambda item: (-item[1], item[0])
    ):
        percentage = (100.0 * count / total_responses) if total_responses else 0.0
        print(f"  {category}: {count} ({percentage:.2f}%)")


def _claims_output_path_for_model(model_name_value):
    """Where build_claim_analysis() would have written results for
    `model_name_value` under its default output_path."""
    return Path(f"{OUTPUT_PATH}/replays/extracted/{model_name_value}_claims.json")


def load_multi_model_claim_analysis_results(model_names=None):
    """Load build_claim_analysis() output for each of `model_names`
    (default: MODEL_NAMES, all four platforms) into
    {model_name: records}. Raises FileNotFoundError only if none of them
    have results yet; otherwise warns and skips the missing ones."""
    model_names = model_names or MODEL_NAMES
    loaded_results = {}
    missing_paths = []

    for model_name_value in model_names:
        input_path = _claims_output_path_for_model(model_name_value)
        records = load_json(input_path)
        if not isinstance(records, dict):
            missing_paths.append(str(input_path))
            continue
        loaded_results[model_name_value] = records

    if not loaded_results:
        raise FileNotFoundError(
            "No claim-analysis result files were found. Missing paths: "
            + ", ".join(missing_paths)
        )

    if missing_paths:
        logger.warning(
            "Skipping missing or unreadable claim-analysis files: %s",
            ", ".join(missing_paths),
        )

    return loaded_results


def _normalize_counter_to_percentages(counter, total):
    """Convert a {label: count} Counter to {label: percent_of_total}."""
    normalized = {}
    for key, count in counter.items():
        normalized[key] = (100.0 * count / total) if total else 0.0
    return normalized


def _collapse_counter_to_allowed_labels(counter, allowed_labels, other_label="OTHER"):
    """Re-bucket a Counter's keys down to `allowed_labels`, merging every
    other key's count into `other_label`."""
    collapsed_counter = Counter()
    allowed_set = set(allowed_labels)
    for label, count in counter.items():
        normalized_label = label if label in allowed_set else other_label
        collapsed_counter[normalized_label] += count
    return collapsed_counter


def _claim_level_relation_distribution(records):
    """Count claim-alignment relations (MATCH/REFINEMENT/CONTRADICTION/
    UNMATCHED) across every sample in one model's build_claim_analysis()
    output. Returns (Counter, total_alignments)."""
    relation_counter = Counter()
    total = 0

    for sample in records.values():
        if not isinstance(sample, dict):
            continue
        judgment = ((sample.get("claim_comparison_judge") or {}).get("judgment") or {})
        if not isinstance(judgment, dict):
            continue
        for alignment in judgment.get("alignments", []) or []:
            if not isinstance(alignment, dict):
                continue
            relation = _normalize_claim_level_relation_label(
                alignment.get("relation", "")
            )
            if not relation:
                continue
            relation_counter[relation] += 1
            total += 1

    return relation_counter, total


def _response_level_category_distribution(records):
    """Count response-level judge categories (SAME_CLAIMS/
    UPDATED_OR_SPECIFIED/CORRECTED/NEW_CLAIMS) across every sample in one
    model's build_claim_analysis() output. Returns (Counter, total_responses)."""
    category_counter = Counter()
    total = 0

    for sample in records.values():
        if not isinstance(sample, dict):
            continue
        judgment = ((sample.get("claim_comparison_judge") or {}).get("judgment") or {})
        if not isinstance(judgment, dict):
            continue
        category = str(judgment.get("category", "") or "").strip()
        if not category:
            continue
        category_counter[category] += 1
        total += 1

    return category_counter, total


def _stacked_bar_figure(distributions_by_model, category_order, color_map, title, yaxis_title):
    """A 100%-stacked bar chart, one bar per model in
    `distributions_by_model`, segmented by `category_order` (each a
    {category: percent} dict from _normalize_counter_to_percentages)."""
    model_order = list(distributions_by_model.keys())
    model_display_order = [
        MODEL_NAMES_MAP.get(model_name_value, model_name_value)
        for model_name_value in model_order
    ]
    fig = go.Figure()
    display_name_map = {
        "MATCH": "Match",
        "REFINEMENT": "Refinement",
        "CONTRADICTION": "Contradiction",
        "UNMATCHED": "Unmatched",
        "SAME_CLAIMS": "Same Claims",
        "UPDATED_OR_SPECIFIED": "Updated/Specified",
        "CORRECTED": "Corrected",
        "NEW_CLAIMS": "New Claims",
        "OTHER": "Other",
    }

    for category in category_order:
        y_values = [
            distributions_by_model[model_name_value].get(category, 0.0)
            for model_name_value in model_order
        ]
        display_name = display_name_map.get(category, category.replace("_", " ").title())
        fig.add_trace(
            go.Bar(
                name=display_name,
                x=model_display_order,
                y=y_values,
                marker_color=color_map.get(category),
                text=[
                    f"{value:.1f}%"
                    if value >= 4.0
                    else ""
                    for value in y_values
                ],
                textposition="inside",
                textfont=dict(color="white"),
                hovertemplate=(
                    "Model: %{x}<br>"
                    f"{yaxis_title}: %{{y:.2f}}%<br>"
                    f"Label: {display_name}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=title,
        barmode="stack",
        yaxis_title=yaxis_title,
        xaxis_title="Model",
        # legend_title_text="Label",
        showlegend=True,
    )
    fig.update_yaxes(range=[0, 100], ticksuffix="%")
    paper_fig = with_paper_style(fig, config=styler(22, 22), legend_pos=(0.9, 1.3))
    paper_fig.update_layout(showlegend=True)
    return paper_fig


def plot_multi_model_claim_comparison_summaries(
    model_names=None,
    output_dir=PLOT_OUTPUT_DIR,
):
    """Build and save (PDF, under `output_dir`) two stacked-bar charts
    comparing `model_names` (default MODEL_NAMES): claim-level relation
    distribution, and response-level category distribution. Returns the
    figures and output paths."""
    model_results = load_multi_model_claim_analysis_results(model_names=model_names)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    relation_order = ["MATCH", "REFINEMENT", "CONTRADICTION", "UNMATCHED"]
    relation_color_map = {
        "MATCH": "#2E8B57",
        "REFINEMENT": "#E69F00",
        "CONTRADICTION": "#D55E00",
        "UNMATCHED": "#4C78A8",
    }
    category_order = [
        "SAME_CLAIMS",
        "UPDATED_OR_SPECIFIED",
        "CORRECTED",
        "NEW_CLAIMS",
    ]
    category_color_map = {
        "SAME_CLAIMS": "#2E8B57",
        "UPDATED_OR_SPECIFIED": "#E69F00",
        "CORRECTED": "#D55E00",
        "NEW_CLAIMS": "#4C78A8",
    }

    relation_distributions = {}
    category_distributions = {}
    for model_name_value, records in model_results.items():
        relation_counter, relation_total = _claim_level_relation_distribution(records)
        category_counter, category_total = _response_level_category_distribution(records)
        relation_counter = _collapse_counter_to_allowed_labels(
            relation_counter,
            relation_order,
        )
        category_counter = _collapse_counter_to_allowed_labels(
            category_counter,
            category_order,
        )
        relation_distributions[model_name_value] = _normalize_counter_to_percentages(
            relation_counter,
            relation_total,
        )
        category_distributions[model_name_value] = _normalize_counter_to_percentages(
            category_counter,
            category_total,
        )

    claim_level_fig = _stacked_bar_figure(
        distributions_by_model=relation_distributions,
        category_order=relation_order,
        color_map=relation_color_map,
        title="Claim-Level Relation Distribution by Model",
        yaxis_title="Rate of Claims",
    )
    response_level_fig = _stacked_bar_figure(
        distributions_by_model=category_distributions,
        category_order=category_order,
        color_map=category_color_map,
        title="Response-Level Category Distribution by Model",
        yaxis_title="Rate of Responses",
    )

    claim_level_pdf = output_dir / "claim_comparison_relations_by_model.pdf"
    response_level_pdf = output_dir / "claim_comparison_categories_by_model.pdf"
    try:
        claim_level_fig.write_image(claim_level_pdf, format="pdf")
    except Exception as exc:
        logger.warning("Could not write claim-level comparison PDF: %s", exc)
    try:
        response_level_fig.write_image(response_level_pdf, format="pdf")
    except Exception as exc:
        logger.warning("Could not write response-level comparison PDF: %s", exc)

    logger.info("Saved claim-level plot to %s", claim_level_pdf)
    logger.info("Saved response-level plot to %s", response_level_pdf)
    return {
        "claim_level_figure": claim_level_fig,
        "response_level_figure": response_level_fig,
        "claim_level_pdf": claim_level_pdf,
        "response_level_pdf": response_level_pdf,
    }


def main():
    """CLI entry point: build a claim analysis (default), or --print-summary
    / --plot-multi-model-summary against results already built."""
    parser = argparse.ArgumentParser(
        description=(
            "Extract claims from replay final responses for samples where "
            "auto invoked web search, pairing auto and none outputs."
        )
    )
    parser.add_argument(
        "-n",
        "--limit",
        type=int,
        default=None,
        help="Maximum number of qualifying replay samples to process.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Replayed model name (see MODEL_NAMES) to process -- default: "
        "PLATFORM_TO_MODEL_NAME[--platform] if --platform is given, else "
        f"{model_name!r}. Also determines the judge platform (via "
        "judge_platform_for_replay_model) unless --platform is also given.",
    )
    parser.add_argument(
        "--platform",
        default=None,
        choices=["chatgpt", "claude", "grok", "deepseek"],
        help="Which platform to process: selects that platform's replay "
        "model (PLATFORM_TO_MODEL_NAME) -- and hence its replay/output/"
        "cache paths -- unless --model-name is also given explicitly, in "
        "which case --platform only overrides the judge platform for "
        "compare_claim_sets.",
    )
    parser.add_argument(
        "--replay-path",
        default=None,
        help="Path to the replay JSON file (default: derived from --model-name).",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Path to write the extracted claims JSON (default: derived "
        "from --model-name).",
    )
    parser.add_argument(
        "--cache-path",
        default=None,
        help="Path to the claim extraction cache JSON (default: derived "
        "from --model-name).",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print relation/category summary from the saved output file and exit.",
    )
    parser.add_argument(
        "--plot-multi-model-summary",
        action="store_true",
        help="Load the four model result files and generate stacked summary plots.",
    )
    args = parser.parse_args()

    if args.model_name:
        model_name_value = args.model_name
    elif args.platform:
        model_name_value = PLATFORM_TO_MODEL_NAME.get(args.platform)
        if model_name_value is None:
            parser.error(f"No default replay model configured for platform {args.platform!r}.")
    else:
        model_name_value = model_name

    replay_path = Path(args.replay_path) if args.replay_path else Path(
        f"{OUTPUT_PATH}/replays/{model_name_value}.json"
    )
    output_path = Path(args.output_path) if args.output_path else _claims_output_path_for_model(
        model_name_value
    )
    cache_path = Path(args.cache_path) if args.cache_path else Path(
        f"{OUTPUT_PATH}/replays/extracted/{model_name_value}_claims_cache.json"
    )

    if args.print_summary:
        print_claim_comparison_summary(input_path=output_path)
        return

    if args.plot_multi_model_summary:
        plot_multi_model_claim_comparison_summaries()
        return

    build_claim_analysis(
        replay_path=replay_path,
        output_path=output_path,
        cache_path=cache_path,
        limit=args.limit,
        model_name_value=model_name_value,
        platform=args.platform,
    )


if __name__ == "__main__":
    main()
