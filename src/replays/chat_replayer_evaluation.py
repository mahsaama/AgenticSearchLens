"""LLM-judge scoring of invitro replay results (src/replays/chat_replayer.py's
output): rates each replayed response's factuality, completeness, and
relevance on a 5-point Likert scale, both with Web search on ("auto") and
forced off ("none").

Each replayed response is judged by its OWN platform's model (see
src.utils.llm_judge.JUDGE_MODEL_BY_PLATFORM) -- a deepseek-v4-flash reply
is judged by DeepSeek's own judge model, not a single fixed evaluator,
consistent with every other judge in this codebase
(query_reformulations.query_specificity_evaluation,
entailment_analysis.evaluate_claim_factuality). The judge platform is
inferred from the replayed model's name (chat_replayer._infer_provider_
from_model), not passed in separately.

Run directly (`python -m src.replays.chat_replayer_evaluation`) to score
the models configured in the __main__ block below; import
`evaluate_replay_results()` to drive it programmatically. Requires the
relevant provider API key(s) in .env (see llm_judge.py).
"""

from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from src.prompts.evaluator_prompts import (
    SYSTEM_PROMPT_COMPLETENESS_5LIKERT,
    SYSTEM_PROMPT_FACTUALITY_5LIKERT,
    SYSTEM_PROMPT_RELEVANCE_5LIKERT,
    USER_PROMPT_5LIKERT,
)
from src.replays.chat_replayer import _infer_provider_from_model
from src.utils.common_io import OUTPUT_PATH, load_json, to_json
from src.utils.llm_judge import judge_model_for_platform, run_judge

load_dotenv()

# chat_replayer._infer_provider_from_model returns "openai" for ChatGPT;
# llm_judge.JUDGE_MODEL_BY_PLATFORM keys it as "chatgpt".
_PROVIDER_TO_JUDGE_PLATFORM = {
    "openai": "chatgpt",
    "claude": "claude",
    "grok": "grok",
    "deepseek": "deepseek",
}


def judge_platform_for_replay_model(replay_model):
    """Which platform's own judge model should score a response replayed
    by `replay_model` (e.g. "deepseek-v4-flash" -> "deepseek")."""
    provider = _infer_provider_from_model(replay_model)
    return _PROVIDER_TO_JUDGE_PLATFORM.get(provider, "chatgpt")


def _web_call_count(result):
    """True if a replay result's raw API response shows the replayed model
    actually searched (as opposed to just being allowed to), across both
    response shapes chat_replayer.py's providers can produce: an OpenAI/
    Grok Responses-API `output` list containing a `web_search_call` item,
    or a Claude/DeepSeek Messages-API `content` list containing a
    `server_tool_use`/`web_search_tool_result` block."""
    if not isinstance(result, dict):
        return False
    response = result.get("response", {})
    if not isinstance(response, dict):
        return False

    output_items = response.get("output", [])
    if isinstance(output_items, list) and any(
        isinstance(item, dict) and item.get("type") == "web_search_call"
        for item in output_items
    ):
        return True

    content_blocks = response.get("content", [])
    if isinstance(content_blocks, list) and any(
        isinstance(block, dict)
        and block.get("type") in ("server_tool_use", "web_search_tool_result")
        for block in content_blocks
    ):
        return True

    return False


def evaluate_replay_results(replay_model, data, filename, save_every=5):
    """Score every replay result in `data` (a {result_key: replay_result}
    dict, as produced by chat_replayer.replayer()) using the judge model
    for `replay_model`'s own platform (see judge_platform_for_replay_model/
    llm_judge.JUDGE_MODEL_BY_PLATFORM), for both the "auto" (Web search
    allowed) and "none" (Web search forced off) replay modes.

    Writes/resumes from
    outputs/<platform>/metadata/preference_evaluation/<filename>.{json,csv,pkl},
    saving incrementally every `save_every` scored rows so an interrupted
    run can pick back up without re-scoring what's already done. Returns
    the final results as a DataFrame.
    """
    platform = judge_platform_for_replay_model(replay_model)
    judge_model = judge_model_for_platform(platform)
    response_modes = ["auto", "none"]

    likert_metrics = {
        "factuality_5likert": SYSTEM_PROMPT_FACTUALITY_5LIKERT,
        "completeness_5likert": SYSTEM_PROMPT_COMPLETENESS_5LIKERT,
        "relevance_5likert": SYSTEM_PROMPT_RELEVANCE_5LIKERT,
    }

    def run_eval(system_prompt, user_prompt):
        """Send one judge request (Web search on, so the judge can verify
        claims against current sources) using `platform`'s own model, and
        return the raw text alongside its parsed JSON verdict.

        max_tokens is higher than run_judge's own default: the same token
        budget has to cover both the judge's own web-search tool-use turns
        and its final Likert-JSON answer, and the replayed responses being
        fact-checked here can be long real chat outputs -- 1024 was
        occasionally getting exhausted mid-search, before any final text.
        """
        return run_judge(
            platform,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            require_web_search=True,
            max_tokens=4096,
        )

    output_dir = Path(f"{OUTPUT_PATH}/replays/preference_evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json_path = output_dir / f"{filename}.json"

    def load_existing_records():
        """Previously-scored rows for this (platform, filename), if any --
        lets a re-run resume instead of re-scoring everything."""
        if not output_json_path.exists():
            return []
        existing = load_json(output_json_path)
        if isinstance(existing, list):
            return existing
        if isinstance(existing, dict):
            return list(existing.values())
        return []

    def save_records():
        """Write the current `records` to .csv/.pkl/.json and return them
        as a DataFrame."""
        results_df = pd.DataFrame(records)
        results_df.to_csv(output_dir / f"{filename}.csv", index=False)
        results_df.to_pickle(output_dir / f"{filename}.pkl")
        to_json(records, output_dir / f"{filename}.json")
        return results_df

    records = load_existing_records()
    records_by_result_key = {
        record.get("result_key"): record
        for record in records
        if isinstance(record, dict) and record.get("result_key")
    }

    print(f"[{platform}] Judge model: {judge_model}. Evaluating {len(data)} prompts ...")
    for result_key, results in tqdm(data.items(), total=len(data)):
        prompt = results.get("user_prompt", result_key)
        row = records_by_result_key.get(result_key, {})
        if not isinstance(row, dict):
            row = {}
        row.update(
            {
                "prompt": row.get("prompt", prompt),
                "result_key": result_key,
                "Prompt_with_history": row.get("Prompt_with_history", results.get("prompt")),
                "sample_source": row.get("sample_source", results.get("sample_source")),
                "conv_id": row.get("conv_id", results.get("conv_id")),
                "turn_id": row.get("turn_id", results.get("turn_id")),
                "judge_platform": platform,
                "judge_model": judge_model,
            }
        )

        try:
            for mode in response_modes:
                mode_score_fields = [
                    f"{mode}_factuality_5likert_score",
                    f"{mode}_completeness_5likert_score",
                    f"{mode}_relevance_5likert_score",
                ]
                if all(row.get(field) not in (None, "") for field in mode_score_fields):
                    continue

                if mode == "invivo":
                    if filename != "invivo":
                        continue
                    else:
                        response_text = results["invivo_response"]
                        row[f"{mode}_called_web"] = (
                            str(results.get("sample_source", "")).strip().lower()
                            == "web"
                        )
                else:
                    response_text = results[mode]["output_text"]
                    row[f"{mode}_called_web"] = _web_call_count(results.get(mode, {}))
                row[f"{mode}_output_text"] = response_text

                for metric_name, system_prompt in likert_metrics.items():
                    eval_result = run_eval(
                        system_prompt=system_prompt,
                        user_prompt=USER_PROMPT_5LIKERT.format(
                            user_query=prompt,
                            response=response_text,
                        ),
                    )
                    parsed = eval_result["parsed_judgment"]
                    row[f"{mode}_{metric_name}_score"] = parsed.get("score")
                    row[f"{mode}_{metric_name}_reasoning"] = parsed.get("reasoning")
        except Exception as e:
            print(prompt, e)
            row["evaluation_status"] = "failed"
            row["evaluation_error"] = str(e)
        else:
            row["evaluation_status"] = "ok"
            row["evaluation_error"] = ""

        records_by_result_key[result_key] = row
        records = list(records_by_result_key.values())
        if save_every and len(records) % save_every == 0:
            save_records()

    return save_records()


if __name__ == "__main__":
    replay_models = [
        "gpt-4.1-mini-2025-04-14",
        # "grok-4.3",
        # "claude-sonnet-4-6",
        # "deepseek-v4-flash",
    ]
    for replay_model in replay_models:
        judge_platform = judge_platform_for_replay_model(replay_model)
        print(f"Replayer model: {replay_model} (judged by {judge_platform})")
        data = load_json(
            f"{OUTPUT_PATH}/replays/{replay_model}.json"
        )
        evaluate_replay_results(replay_model, data, replay_model)
