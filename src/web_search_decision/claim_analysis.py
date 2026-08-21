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
from openai import OpenAI
from tqdm import tqdm

from src.prompts.evaluator_prompts import (
    SYSTEM_PROMPT_CLAIM_COMPARISON,
    SYSTEM_PROMPT_CLAIM_EXTRACTION,
    USER_PROMPT_CLAIM_COMPARISON,
    USER_PROMPT_CLAIM_EXTRACTION,
)
from src.replays.extract_replay_artifacts import _has_web_tool_call, _infer_provider
from src.utils.paper import styler, with_paper_style
from src.utils.utils import OUTPUT_PATH, load_json, to_json


load_dotenv()

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

# model_name = "gpt-5.3-chat-latest" 
# model_name = "claude-sonnet-4-6" 
# model_name = "grok-4.3" 
model_name = "deepseek-v4-flash"
MODEL_NAMES = [
    "gpt-5.3-chat-latest",
    "claude-sonnet-4-6",
    "grok-4.3",
    "deepseek-v4-flash",
]

MODEL_NAMES_MAP = {
    "gpt-5.3-chat-latest": "ChatGPT",
    "claude-sonnet-4-6": "Claude",
    "grok-4.3": "Grok",
    "deepseek-v4-flash": "Deepseek"
}

REPLAY_PATH = Path(f"{OUTPUT_PATH}/replays/{model_name}.json")
OUTPUT_PATH_CLAIMS = Path(f"{OUTPUT_PATH}/replays/extracted/{model_name}_claims.json")
CACHE_PATH = Path(f"{OUTPUT_PATH}/replays/extracted/{model_name}_claims_cache.json")
PLOT_OUTPUT_DIR = Path(f"{OUTPUT_PATH}/replays/plots")
CLAIM_EXTRACTION_MODEL = os.getenv("CLAIM_ANALYSIS_MODEL")
CLAIM_COMPARISON_JUDGE_MODEL = os.getenv("CLAIM_COMPARISON_JUDGE_MODEL")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _load_replay_json(path):
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


def _coerce_claim_list(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ["claims", "claim_list", "items", "sentences", "chunks"]:
            value = payload.get(key)
            if isinstance(value, list):
                return value
    return []


def _clean_claims(claims):
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


def _extract_first_json_array(text):
    if not isinstance(text, str):
        return None
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or start >= end:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed


def _extract_first_json_object(text):
    if not isinstance(text, str):
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or start >= end:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed


def _normalize_claim_comparison_judgment(payload):
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
    label = str(label or "").strip()
    if label == "NEW_CLAIM":
        return "UNMATCHED"
    return label


def extract_claims_from_text(text):
    text = str(text or "").strip()
    if not text:
        return []

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_CLAIM_EXTRACTION},
        {
            "role": "user",
            "content": USER_PROMPT_CLAIM_EXTRACTION.format(text=text),
        },
    ]

    response_text = ""
    try:
        response = client.chat.completions.create(
            model=CLAIM_EXTRACTION_MODEL,
            messages=messages,
            # max_tokens=CLAIM_EXTRACTION_MAX_OUTPUT_TOKENS,
            temperature=0.0,
        )
        response_text = response.choices[0].message.content or ""
    except Exception as exc:
        logger.warning("Claim extraction failed: %s", exc)
        return []

    parsed_payload = None
    try:
        parsed_payload = json.loads(response_text)
    except json.JSONDecodeError:
        parsed_payload = _extract_first_json_array(response_text)

    claims = _coerce_claim_list(parsed_payload)
    return _clean_claims(claims)


def compare_claim_sets(user_query, claims_without_web, claims_with_web):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_CLAIM_COMPARISON},
        {
            "role": "user",
            "content": USER_PROMPT_CLAIM_COMPARISON.format(
                user_query=str(user_query or "").strip(),
                claims_without_web=json.dumps(
                    _clean_claims(claims_without_web), ensure_ascii=False, indent=2
                ),
                claims_with_web=json.dumps(
                    _clean_claims(claims_with_web), ensure_ascii=False, indent=2
                ),
            ),
        },
    ]

    raw_text = ""
    try:
        response = client.chat.completions.create(
            model=CLAIM_COMPARISON_JUDGE_MODEL,
            messages=messages,
            temperature=0.0,
        )
        raw_text = response.choices[0].message.content or ""
    except Exception as exc:
        logger.warning("Claim comparison judge failed: %s", exc)
        return {"judgment": {}, "error": str(exc)}

    parsed = None
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        parsed = _extract_first_json_object(raw_text)

    normalized_judgment = _normalize_claim_comparison_judgment(parsed)
    return {
        "raw_output": raw_text,
        "judgment": normalized_judgment,
        "error": "",
    }


def _sample_has_auto_web_call(row):
    auto_payload = row.get("auto") or {}
    response = auto_payload.get("response") or {}
    provider = _infer_provider("gpt-5.3-chat-latest", row)
    return _has_web_tool_call(provider, response)


def _load_cache(cache_path):
    cache = load_json(cache_path)
    if isinstance(cache, dict):
        return cache
    return {}


def _extract_claims_cached(text, cache, cache_dirty_state):
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
):
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
        and _sample_has_auto_web_call(row)
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
    return Path(f"{OUTPUT_PATH}/replays/extracted/{model_name_value}_claims.json")


def load_multi_model_claim_analysis_results(model_names=None):
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
    normalized = {}
    for key, count in counter.items():
        normalized[key] = (100.0 * count / total) if total else 0.0
    return normalized


def _collapse_counter_to_allowed_labels(counter, allowed_labels, other_label="OTHER"):
    collapsed_counter = Counter()
    allowed_set = set(allowed_labels)
    for label, count in counter.items():
        normalized_label = label if label in allowed_set else other_label
        collapsed_counter[normalized_label] += count
    return collapsed_counter


def _claim_level_relation_distribution(records):
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

    claim_level_html = output_dir / "claim_comparison_relations_by_model.html"
    response_level_html = output_dir / "claim_comparison_categories_by_model.html"
    claim_level_pdf = output_dir / "claim_comparison_relations_by_model.pdf"
    response_level_pdf = output_dir / "claim_comparison_categories_by_model.pdf"
    claim_level_fig.write_html(claim_level_html)
    response_level_fig.write_html(response_level_html)
    try:
        claim_level_fig.write_image(claim_level_pdf, format="pdf")
    except Exception as exc:
        logger.warning("Could not write claim-level comparison PDF: %s", exc)
    try:
        response_level_fig.write_image(response_level_pdf, format="pdf")
    except Exception as exc:
        logger.warning("Could not write response-level comparison PDF: %s", exc)

    logger.info("Saved claim-level plot to %s", claim_level_html)
    logger.info("Saved response-level plot to %s", response_level_html)
    return {
        "claim_level_figure": claim_level_fig,
        "response_level_figure": response_level_fig,
        "claim_level_html": claim_level_html,
        "response_level_html": response_level_html,
        "claim_level_pdf": claim_level_pdf,
        "response_level_pdf": response_level_pdf,
    }


def main():
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
        "--replay-path",
        default=str(REPLAY_PATH),
        help="Path to the replay JSON file.",
    )
    parser.add_argument(
        "--output-path",
        default=str(OUTPUT_PATH_CLAIMS),
        help="Path to write the extracted claims JSON.",
    )
    parser.add_argument(
        "--cache-path",
        default=str(CACHE_PATH),
        help="Path to the claim extraction cache JSON.",
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

    if args.print_summary:
        print_claim_comparison_summary(input_path=Path(args.output_path))
        return

    if args.plot_multi_model_summary:
        plot_multi_model_claim_comparison_summaries()
        return

    build_claim_analysis(
        replay_path=Path(args.replay_path),
        output_path=Path(args.output_path),
        cache_path=Path(args.cache_path),
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
