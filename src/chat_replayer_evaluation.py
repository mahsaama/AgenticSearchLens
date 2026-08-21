import os
from dotenv import load_dotenv

load_dotenv()
from pathlib import Path
import json
import ast
import argparse
import logging
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from src.utils import *
from src.evaluator_prompts import *
import random


def openai_inference(model_name, data, filename, temperature=0.0, save_every=5):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    # response_modes = ["none", "required", "auto", "invivo"]
    response_modes = ["auto", "none"]

    likert_metrics = {
        "factuality_5likert": SYSTEM_PROMPT_FACTUALITY_5LIKERT,
        "completeness_5likert": SYSTEM_PROMPT_COMPLETENESS_5LIKERT,
        "relevance_5likert": SYSTEM_PROMPT_RELEVANCE_5LIKERT,
    }

    def parse_eval_json(text):
        text = (text or "").strip()
        if not text:
            return {}

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and start < end:
            candidate = text[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                try:
                    return ast.literal_eval(candidate)
                except (ValueError, SyntaxError):
                    return {}
        return {}

    def run_eval(system_prompt, user_prompt):
        msg = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = client.responses.create(
            model=model_name,
            tools=[{"type": "web_search"}],
            tool_choice="required",
            input=msg,
        )
        raw_text = response.output_text
        return {
            "raw_judgment": raw_text,
            "parsed_judgment": parse_eval_json(raw_text),
        }

    def _web_call_count(result):
        if not isinstance(result, dict):
            return False
        response = result.get("response", {})
        if not isinstance(response, dict):
            return False
        output_items = response.get("output", [])
        if not isinstance(output_items, list):
            return False
        return any(
            1
            for item in output_items
            if isinstance(item, dict) and item.get("type") == "web_search_call"
        )

    output_dir = Path(
        f"{OUTPUT_PATH}/metadata/preference_evaluation/{model_name}/{temperature}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json_path = output_dir / f"{filename}.json"

    def load_existing_records():
        if not output_json_path.exists():
            return []
        existing = load_json(output_json_path)
        if isinstance(existing, list):
            return existing
        if isinstance(existing, dict):
            return list(existing.values())
        return []

    def save_records():
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
    # # 1. Set the seed for reproducibility
    # random.seed(42)

    # # 2. Separate into "web" and "non-web" dictionaries
    # web_dict = {k: v for k, v in data.items() if k.startswith("web")}
    # non_web_dict = {k: v for k, v in data.items() if k.startswith("non_web")}

    # # 3. Sample 100 items from each group
    # web_sample = dict(random.sample(list(web_dict.items()), 100))
    # non_web_sample = dict(random.sample(list(non_web_dict.items()), 100))

    # # 4. Merge the two samples into one final dictionary
    # data = {**web_sample, **non_web_sample}

    print(f"Evaluating {len(data)} prompts ...")
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
            }
        )

        try:
            # query_type_eval = run_eval(
            #     system_prompt=SYSTEM_PROMPT_QUERY_TYPE,
            #     user_prompt=USER_PROMPT_QUERY_TYPE.format(user_query=prompt),
            # )
            # query_type_parsed = query_type_eval["parsed_judgment"]
            # row["query_type"] = query_type_parsed.get("type")
            # row["query_type_reasoning"] = query_type_parsed.get("reasoning")
            # row["query_type_raw_judgment"] = query_type_eval["raw_judgment"]

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

                # for metric_name, system_prompt in binary_metrics.items():
                #     eval_result = run_eval(
                #         system_prompt=system_prompt,
                #         user_prompt=USER_PROMPT_BINARY.format(
                #             user_query=prompt,
                #             response=response_text,
                #         ),
                #     )
                #     parsed = eval_result["parsed_judgment"]
                #     row[f"{mode}_{metric_name}_score"] = parsed.get("score")
                #     row[f"{mode}_{metric_name}_reasoning"] = parsed.get("reasoning")
                #     # row[f"{mode}_{metric_name}_raw_judgment"] = eval_result[
                #     #     "raw_judgment"
                #     # ]

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
                    # row[f"{mode}_{metric_name}_raw_judgment"] = eval_result[
                    #     "raw_judgment"
                    # ]
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
    # evaluator_models = ["gpt-5.4-mini", "gpt-5.4-nano-2026-03-17"]
    evaluator_models = ["gpt-5.6-luna"]
    # replay_models = ["gpt-4o-mini", "invivo"]
    replay_models = [
        # "gpt-5-mini-2025-08-07",
        # "gpt-5.3-chat-latest",
        # "grok-4.3",
        # "claude-sonnet-4-6",
        "deepseek-v4-flash",
    ]
    for eval_model in evaluator_models:
        print(f"Evaluator: {eval_model}")
        for replay_model in replay_models:
            print(f"Replayer model: {replay_model}")
            data = load_json(
                f"{OUTPUT_PATH}/replays/{replay_model}.json"
            )
            openai_inference(eval_model, data, replay_model)
