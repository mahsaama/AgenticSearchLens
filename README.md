# Web Search Tool Calling In AI Chatbots

This repository studies how modern AI chatbots decide to call Web search tools and how those calls shape the final response.

## Motivation

1. AI agents powering chatbots are increasingly relying on tools, particularly Web search.
2. Web search is a complex task:
   - deciding when parametric knowledge is enough vs. when a tool should be called,
   - formulating search queries from user intent,
   - reformulating queries as new evidence is retrieved,
   - generating final responses grounded in both model knowledge and retrieved results.
3. We analyze longitudinal data from four popular chatbot platforms:
   - ChatGPT
   - Grok
   - Claude
   - DeepSeek
4. Implications:
   - (a) designers of AI agents,
   - (b) designers of Web search tools for AI agents,
   - (c) end users of chat platforms.

## Web Search Life Cycle

```mermaid
flowchart LR
    A[User Prompt] --> B[Web Search Decision]
    B --> C[Query Formulation]
    C --> D[Response Generation]
```

## Repository Layout

- `src/web_tool_invocation.py`: analyses focused on Web-call decisions and trends.
- `src/query_reformulations.py`: query evolution and reformulation analyses.
- `src/source_selection.py`: retrieved/cited source analyses.
- `src/response_generation.py`: response grounding and quality analyses.
- `outputs/`: generated analysis artifacts.

## Setup

This project uses Python and common data-science dependencies (see `requirements.txt`).

```bash
pip install -r requirements.txt
```

## Dummy Dataset For Development

Due to the ERB restrictions we cannot share the data (not even the snippet of the original data). To support testing and later pipeline stages we have generated a dummy dataset.


```python
from data_extraction import load_whole_data_from_file, load_web_data_from_file

df = load_whole_data_from_file("pkl")
web_df = load_web_data_from_file("pkl")
```

## Notes

- Raw data paths in some scripts are machine-specific and may require local edits.
- Analysis outputs are written under `outputs/` by default.
