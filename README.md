# 🔎 Web Search Tool Calling In AI Chatbots

When a chatbot decides to search the web, formulates a query, and writes an
answer grounded in what it found — this repo is about that whole chain, end
to end, across the four platforms that actually do it in production.

## 🎯 Motivation

1. AI agents powering chatbots are increasingly relying on tools, particularly Web search.
2. Web search is a complex task:
   - deciding when parametric knowledge is enough vs. when a tool should be called,
   - formulating search queries from user intent,
   - reformulating queries as new evidence is retrieved,
   - generating final responses grounded in both model knowledge and retrieved results.
3. We analyze longitudinal data from four popular chatbot platforms:
   - 🤖 ChatGPT
   - 🟠 Claude
   - ⚡ Grok
   - 🐳 DeepSeek
4. Implications:
   - (a) designers of AI agents,
   - (b) designers of Web search tools for AI agents,
   - (c) end users of chat platforms.

## 🔄 Web Search Life Cycle

![Life cycle of agentic Web search: User Prompt → Web Search Decision → Query Formulation → Response Generation](docs/images/lifecycle.png)

## 📂 Repository Layout

- `src/web_search_decision/extraction.py`: parses raw ChatGPT/Claude/Grok/DeepSeek exports into a per-turn summary dataframe (one unified module, `--platform` picks which).
- `src/utils/chatgpt_conversation_utils.py` / `src/utils/other_platforms_parsing_utils.py`: the conversation-tree parsing and topic-lookup helpers the two extraction modules above build on.
- `src/utils/common_io.py`: shared path constants and JSON read/write helpers.
- `src/utils/topic_classifier.py`: dependency-free keyword-based topic classification, used when no topic-annotation file is available (see below).
- `src/utils/llm_judge.py`: the shared multi-provider LLM-judge client — every judged metric (factuality, specificity, claim comparison, ...) is scored by *that platform's own model*, not one fixed judge for everyone.
- `src/web_search_decision/web_tool_invocation.py`: analyses focused on Web-call decisions and trends.
- `src/query_formulation/query_reformulations.py`: query evolution and reformulation analyses.
- `src/response_generation/source_selection.py`: retrieved/cited source analyses.
- `src/response_generation/response_generation.py`: response grounding and quality analyses -- the orchestrator; imports and runs the three modules below plus its own raw response/sources extraction and embedding-similarity checks. Fully platform-generalized: claims, entailment, and factuality all run per-platform now.
- `src/response_generation/web_content_fetch.py`: fetches and caches cited/retrieved source URLs' raw text (requests → Wikipedia API → Playwright fallback).
- `src/response_generation/claim_extraction.py`: extracts atomic claims from a response's text, with a content-hash cache.
- `src/response_generation/entailment_analysis.py`: NLI entailment scoring and the factuality/grounding-source analysis built on it (the bulk of §5.2's figures/tables).
- `src/response_generation/hallucinated_url_detection.py`: checks cited URLs for reachability/fabrication — domain-level retrieved-vs-cited matching, `--platform` picks the source, or feed it any DataFrame directly.
- `src/replays/chat_replayer.py` / `src/replays/chat_replayer_evaluation.py`: invitro replay (re-querying prompts via platform APIs) and LLM-judge scoring of the replayed responses.
- `src/replays/extract_replay_artifacts.py`: post-processes replay outputs into the artifacts/plots used across the analyses above, plus its own Tranco-rank and hallucination checks for replay data.
- `src/replays/claim_analysis.py`: claim-level comparison between a replayed model's Web and no-Web responses — `--platform` picks the model, judged by that platform's own model.
- `src/prompts/evaluator_prompts.py`: prompt templates used by the LLM-judge evaluations.
- `src/utils/figure_style.py`: shared Plotly styling for figures.
- `outputs/`: generated analysis artifacts.
- `data/`: your local copies of exported chat data (see below).

## ⚙️ Setup

This project uses Python and common data-science dependencies, including test
tooling (pytest), all listed in `requirements.txt`. A virtual environment keeps
these isolated from anything else on your machine:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

🔁 In every new terminal session after that, just reactivate it (from the repo
root) before running anything -- no need to recreate it or reinstall:

```bash
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

You'll see `(.venv)` show up in your prompt once it's active; run `deactivate`
to leave it.

One-time extras some modules need at runtime:

```bash
# response_generation.py scrapes cited URLs via Playwright
playwright install chromium

# query_reformulations.py tokenizes text with NLTK
python -m nltk.downloader punkt punkt_tab
```

🔑 Copy `.env_sample` to `.env` and fill in your keys (placeholders for every
API key and optional override are already there, alongside its built-in default):

```bash
cp .env_sample .env
```

```bash
OPENAI_API_KEY=...       # LLM-judge evaluations (factuality/completeness/relevance,
                          # claim extraction, entailment, query-specificity, ...) and
                          # invitro replay of OpenAI models
ANTHROPIC_API_KEY=...    # invitro replay of Claude models
XAI_API_KEY=...          # invitro replay of Grok models
DEEPSEEK_API_KEY=...     # invitro replay of DeepSeek models
```

Everything above is only needed if you plan to run the LLM-judge evaluations or the
invitro replay (`src/replays/chat_replayer.py`). The extraction and descriptive-analysis
scripts work without any API key.

## 📥 Exporting Your Own Data

Because of ERB restrictions we cannot share the original donated dataset (see
**Ethical Considerations** in the paper). To run this pipeline on your own chat
history, request a data export from each platform and drop it under `data/`. Each
platform's export contains **your own conversations only** — nothing donated by
anyone else is required to run the code.

Every platform uses the same layout: `data/<platform>/user_0/conversations.json`.
That's also the `user_<i>/conversations.json` shape all four extraction paths expect
if you ever combine exports from several accounts under the same platform (`user_0`,
`user_1`, ...).

| Platform | Where to export | Raw file you get | Where it goes |
|---|---|---|---|
| 🤖 ChatGPT | Settings → Data controls → Export data (emailed download link) | `conversations.json` | `data/chatgpt/user_0/conversations.json` |
| 🟠 Claude | Settings → Privacy → Export data | `conversations.json` | `data/claude/user_0/conversations.json` |
| 🐳 DeepSeek | Settings → Data export | `conversations.json` | `data/deepseek/user_0/conversations.json` |
| ⚡ Grok | Export your conversation history | `prod-grok-backend.json` | rename to `conversations.json`, then `data/grok/user_0/conversations.json` |

Notes:

- ⚠️ **Grok's export file is not named `conversations.json`** — it comes down as
  `prod-grok-backend.json`. Rename it to `conversations.json` and nest it under
  `user_0/`, e.g.:

  ```bash
  mkdir -p data/grok/user_0
  mv ~/Downloads/prod-grok-backend.json data/grok/user_0/conversations.json
  ```

- Exact export menu wording changes over time and by account type; look for
  "export my data" / "download your data" under each platform's privacy or account
  settings if the paths above have moved.
- 🔒 Exported data can contain personal or sensitive information — treat your
  `data/` and `outputs/` directories as private.

## 🚀 Running The Pipeline / Evaluating Your Data

**1. Extract raw exports into summary dataframes.** This parses the JSON exports
above into per-turn dataframes (`data_summary.*` and `web_data_summary.*`, in
parquet/pickle/csv), under `outputs/<platform>/metadata/` -- same layout for
all four platforms:

```bash
python -m src.web_search_decision.extraction --platform chatgpt    # ChatGPT
python -m src.web_search_decision.extraction --platform claude     # Claude
python -m src.web_search_decision.extraction --platform grok       # Grok
python -m src.web_search_decision.extraction --platform deepseek   # DeepSeek
```

Load the results back with:

```python
from src.web_search_decision.extraction import load_whole_data_from_file, load_web_data_from_file

df = load_whole_data_from_file("pkl", platform="chatgpt")  # platform defaults to "chatgpt"
web_df = load_web_data_from_file("pkl", platform="chatgpt")
```

**2. Run the descriptive analyses.** Each of these reads the extracted dataframes
and writes figures/tables under `outputs/<module_name>/` (each loops over all four
platforms itself — no `--platform` flag needed here):

```bash
python -m src.web_search_decision.web_tool_invocation     # §3: search-calling decisions & trends
python -m src.query_formulation.query_reformulations      # §4: querying strategies & reformulation
python -m src.response_generation.source_selection        # §4.3/§5.1: retrieved/cited source bias
python -m src.response_generation.response_generation     # §5.2: response grounding & entailment
```

**3. (Optional) invitro replay + LLM-judge evaluation.** To reproduce the
API-based replay and quality scoring (needs the provider API keys above):

```bash
python -m src.replays.chat_replayer               # replay prompts through each platform's API
python -m src.replays.chat_replayer_evaluation     # score replayed responses (factuality/
                                                    # completeness/relevance) with an LLM judge
python -m src.replays.claim_analysis --platform chatgpt              # claim-level Web vs. no-Web comparison, one platform
python -m src.replays.claim_analysis --plot-multi-model-summary      # ...and compare across all four
python -m src.response_generation.hallucinated_url_detection --platform chatgpt  # cited-URL reachability check
```

`src/replays/extract_replay_artifacts.py` is invoked internally by the scripts above to
turn replay output into the artifacts consumed by the analyses in step 2 — you generally
don't need to run it directly.

## 🧩 A Few Things Worth Knowing

- **No topic-annotation file?** That's expected — the paper's hand-labeled dataset
  doesn't ship with this repo. Extraction falls back to `topic_classifier.py`, a
  small dependency-free keyword classifier, instead of labeling everything "Other".
- **Every output lives under `outputs/<platform>/metadata/`**, ChatGPT included —
  there's no flat `outputs/metadata/` anymore.
- **Replay refuses to run without a PII-safety annotation first.** `chat_replayer.py`
  won't send unscreened personal data to a provider API — see Appendix C.1 of the
  paper, or generate one yourself with `chat_replayer.generate_pii_safety_annotations()`.
- **One rough edge, left as-is rather than papered over:** `web_tool_invocation.py`'s
  tool-intent judge references two prompts (`SYSTEM_PROMPT_TOOL_INTENT`/
  `USER_PROMPT_TOOL_INTENT`) that don't exist yet in `evaluator_prompts.py` — add
  them before calling `classify_web_call_tool_intent_from_thoughts()`.

## 📝 Notes

- Raw data paths in some scripts are machine-specific and may require local edits.
- Analysis outputs are written under `outputs/` by default.
