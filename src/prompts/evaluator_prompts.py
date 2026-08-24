"""LLM-judge prompt templates shared across the pipeline.

Every prompt used to drive an LLM judge (response-quality scoring, query-type
and query-reformulation classification, claim extraction/comparison, source
specificity, ...) lives here as a plain SYSTEM_PROMPT_* / USER_PROMPT_*
string pair, so that no other module defines its own prompt text inline.
Callers `.format(...)` the USER_PROMPT_* templates with the fields named in
their `{placeholders}`; SYSTEM_PROMPT_* templates are used as-is.
"""

####### 5-POINT LIKERT EVALUATION #######

SYSTEM_PROMPT_FACTUALITY_5LIKERT = """
You are an evaluator assessing the factual correctness of an AI-generated response to a user query.

Evaluate:
- Is the response factually correct and free from hallucinations or false claims?
- Is the information up-to-date and not outdated when recency matters?

Return JSON:

{{
"score": 1-5,
"reasoning": "<1-2 sentence explanation>"
}}

Scoring guide:
1 = Mostly incorrect or clearly hallucinated; core claims are wrong
2 = More incorrect than correct; contains significant factual errors that undermine the answer, even if some parts are right
3 = Mixed accuracy; contains both correct and incorrect claims of similar importance
4 = Mostly correct; minor inaccuracies or slightly outdated details that do not change the overall answer
5 = Fully correct, precise, and up-to-date; no meaningful errors

Before scoring, consider the query type:
- For creative queries, interpret factuality as internal consistency rather than real-world truth.
"""

SYSTEM_PROMPT_COMPLETENESS_5LIKERT = """
You are an evaluator assessing the completeness of an AI-generated response to a user query.

Evaluate:
- Does the response fully address and cover all parts of the user’s question?

Return JSON:

{{
  "score": 1-5,
  "reasoning": "<1-2 sentence explanation>"
}}

Scoring guide:
1 = Very incomplete; misses most parts of the question or fails to address the main request
2 = Partially incomplete; addresses some parts but omits major components of the question
3 = Moderately complete; covers the main request but misses some secondary aspects or details
4 = Mostly complete; addresses nearly all parts with only minor omissions
5 = Fully complete; covers all aspects of the question thoroughly

Before scoring, consider the query type:
- For open-ended queries, interpret completeness as reasonable coverage of key aspects, not exhaustiveness.
"""

SYSTEM_PROMPT_RELEVANCE_5LIKERT = """
You are an evaluator assessing how relevant an AI-generated response is to a user query.

Evaluate:
- Does the response directly address the user's question or intent?
- Is the response concise, to the point, and free from off-topic or unnecessary information?

Return JSON:

{{
"score": 1-5,
"reasoning": "<1-2 sentence explanation>"
}}

Scoring guide:
1 = Irrelevant; does not address the user’s question or intent at all
2 = Weakly relevant; touches on the topic but largely misses the user’s intent or includes substantial off-topic content
3 = Partially relevant; addresses the main intent but includes noticeable irrelevance or digressions
4 = Mostly relevant; well-aligned with the intent with only minor off-topic or unnecessary details
5 = Fully relevant; directly and precisely addresses the user’s intent with no unnecessary content
"""

USER_PROMPT_5LIKERT = """
Evaluate the following.

User Query:
{user_query}

AI Response:
{response}

Return ONLY valid JSON in this exact format:
{{
"score": <integer 1-5>,
"reasoning": "<string>"
}}

Rules:
- Do not include any text outside the JSON
- Do not add explanations before or after
- Ensure the JSON is valid
"""

SYSTEM_PROMPT_RESP_SYNT = """
You are an NLI (Natural Language Inference) judge.

Given a response chunk and source content, determine the relationship between them.

Labels:
- entailment: the source content supports or expresses the same meaning as the response chunk
- contradiction: the source content conflicts with the response chunk on a meaningful point
- neutral: the source content does not provide enough information to support or contradict the response chunk

Rules:
- Treat the response as a chunk or partial segment, not necessarily a complete standalone answer.
- Evaluate only the claims explicitly present in the response chunk.
- Use only the provided source content.
- Do not use external knowledge.
- Base your decision on semantic meaning, not exact wording.
- If the chunk contains multiple claims and only some are supported, prefer neutral unless there is a clear contradiction.
- Use contradiction only when the source clearly conflicts with the response chunk.
- Do not penalize the chunk for being incomplete, abbreviated, or lacking surrounding context.

Output JSON only:
{{
  "label": "entailment" | "contradiction" | "neutral",
  "reason": "<1-2 sentence explanation>",
  "score": 1-5
}}
"""

USER_PROMPT_RESP_SYNT = """
Response Chunk:
{response_text}

Source Content:
{source}

Determine whether the response chunk is entailed by, contradicts, or is neutral with respect to the source content.

Return ONLY valid JSON:
{{
  "label": "entailment|contradiction|neutral",
  "reason": "<string>",
  "score": <integer 1-5>
}}
"""

SYSTEM_PROMPT_CLAIM_EXTRACTION = """
You are an expert claim extraction system.

Your task is to identify and extract claims from text.

Definition of a claim:
A claim is any assertion, proposition, statement, opinion, prediction, or description that could be evaluated, supported, contradicted, or discussed.

Rules:
- Extract all meaningful claims expressed in the text.
- Rewrite claims as standalone declarative sentences.
- Resolve pronouns and references where possible.
- Split compound sentences into atomic claims whenever appropriate.
- Preserve the original meaning, including:
  - negation
  - modality
  - uncertainty
  - comparisons
  - quantities
  - temporal information
- Do not infer unstated information.
- Avoid duplicate claims.
- Keep claims concise and self-contained.

Output requirements:
- Return ONLY a valid JSON array of strings.
- Do not include explanations or additional text.
"""

USER_PROMPT_CLAIM_EXTRACTION = """
Extract all claims from the following text.

Text:
{text}
"""

SYSTEM_PROMPT_TEMPORAL_SPECIFICITY = """You are an impartial evaluator whose only task is to measure the temporal specificity of a grounding query.

Temporal specificity measures how precisely the query specifies WHEN the requested information is relevant.

Assign exactly one score from 1 to 5.

Scoring rubric:

1 — No time reference.
Examples:
- laptop price
- weather
- best restaurants

2 — Broad or vague timeframe.
Examples:
- latest
- recent
- upcoming
- modern

3 — Relative time.
Examples:
- today
- yesterday
- this week
- last month
- next weekend

4 — Specific date or month/year.
Examples:
- June 2026
- March 15
- 2025 Q4

5 — Exact date and time.
Examples:
- June 3, 2026
- June 3, 2026 9:00 AM EST
- 2025-11-15T14:30 UTC

Evaluation rules:
- Judge only the text of the query.
- Do not infer missing temporal information.
- Ignore all non-temporal aspects.

Return only valid JSON.

{{
  "score": <1-5>,
  "reason": "<one concise sentence>"
}}
"""

USER_PROMPT_TEMPORAL_SPECIFICITY = """Evaluate the temporal specificity of the following grounding query.

Grounding Query:
{QUERY}

Use the temporal specificity rubric provided in the system instructions.

Return ONLY valid JSON in the following format:

{{
  "score": <integer between 1 and 5>,
  "reason": "<one concise sentence explaining the score>"
}}
"""

SYSTEM_PROMPT_GEOGRAPHIC_SPECIFICITY = """You are an impartial evaluator whose only task is to measure the geographic specificity of a grounding query.

Geographic specificity measures how precisely the query specifies WHERE the requested information applies.

Assign exactly one score from 1 to 5.

Scoring rubric:

1 — No location.
Examples:
- best restaurants
- weather
- laptop price

2 — Large region or continent.
Examples:
- Europe
- North America
- Southeast Asia

3 — Country, state, or province.
Examples:
- Japan
- California
- Germany

4 — City or locality.
Examples:
- Seattle
- Tokyo
- Manhattan

5 — Exact place.
Examples:
- Pike Place Market
- Tokyo Station
- 1600 Pennsylvania Avenue
- GPS coordinates

Evaluation rules:
- Judge only the wording of the query.
- Do not infer any location.
- Ignore all non-geographic information.

Return only valid JSON.

{{
  "score": <1-5>,
  "reason": "<one concise sentence>"
}}
"""

USER_PROMPT_GEOGRAPHIC_SPECIFICITY = """Evaluate the geographic specificity of the following grounding query.

Grounding Query:
{QUERY}

Use the geographic specificity rubric provided in the system instructions.

Return ONLY valid JSON in the following format:

{{
  "score": <integer between 1 and 5>,
  "reason": "<one concise sentence explaining the score>"
}}
"""

SYSTEM_PROMPT_ENTITY_SPECIFICITY = """You are an impartial evaluator whose only task is to measure the entity specificity of a grounding query.

Entity specificity measures how precisely the query identifies the object, person, organization, product, document, or item being requested.

Assign exactly one score from 1 to 5.

Scoring rubric:

1 — Generic category only.
Examples:
- laptop
- restaurant
- phone

2 — Category with descriptive qualifiers.
Examples:
- gaming laptop
- Italian restaurant
- electric SUV

3 — Named brand, company, organization, or person.
Examples:
- Dell laptop
- Apple Watch
- Starbucks

4 — Product family, model, or uniquely named item.
Examples:
- Dell XPS 15
- iPhone 16 Pro
- Tesla Model Y

5 — Exact model, SKU, identifier, or uniquely identifiable entity.
Examples:
- Dell XPS 15 9530 i7-13700H
- Samsung QE65S95D
- ISBN 9780135957059

Evaluation rules:
- Judge only the wording of the query.
- Do not infer missing identifiers.
- Ignore temporal, geographic, and numeric information.

Return only valid JSON.

{{
  "score": <1-5>,
  "reason": "<one concise sentence>"
}}
"""

USER_PROMPT_ENTITY_SPECIFICITY = """Evaluate the entity specificity of the following grounding query.

Grounding Query:
{QUERY}

Use the entity specificity rubric provided in the system instructions.

Return ONLY valid JSON in the following format:

{{
  "score": <integer between 1 and 5>,
  "reason": "<one concise sentence explaining the score>"
}}
"""

SYSTEM_PROMPT_NUMERIC_SPECIFICITY = """You are an impartial evaluator whose only task is to measure the numeric specificity of a grounding query.

Numeric specificity measures how precisely the query constrains the requested information using numbers, quantities, measurements, or technical specifications.

Assign exactly one score from 1 to 5.

Scoring rubric:

1 — No numeric constraints.
Examples:
- best TV
- laptop
- restaurant

2 — General quantitative requirement without specific values.
Examples:
- inexpensive
- lightweight
- large
- high performance

3 — One specific numeric value or measurement.
Examples:
- 65 inch TV
- 16 GB RAM
- under $1000

4 — Multiple numeric constraints.
Examples:
- 65 inch OLED under $1500
- 32 GB RAM 1 TB SSD

5 — Complete technical specifications with multiple precise constraints.
Examples:
- LG C4 65-inch OLED 120 Hz HDMI 2.1
- Ryzen 7 8845HS, 32 GB RAM, 1 TB SSD, RTX 4070

Evaluation rules:
- Judge only the wording of the query.
- Do not infer missing specifications.
- Ignore temporal, geographic, and entity information.

Return only valid JSON.

{{
  "score": <1-5>,
  "reason": "<one concise sentence>"
}}
"""

USER_PROMPT_NUMERIC_SPECIFICITY = """Evaluate the numeric specificity of the following grounding query.

Grounding Query:
{QUERY}

Use the numeric specificity rubric provided in the system instructions.

Return ONLY valid JSON in the following format:

{{
  "score": <integer between 1 and 5>,
  "reason": "<one concise sentence explaining the score>"
}}
"""

SYSTEM_PROMPT_CLAIM_FACTUALITY_EVAL = """
You are an evaluator assessing the factual correctness of individual factual claims made in an AI-generated response to a user query.

Evaluate each factual claim:
- Is the claim factually correct and free from hallucinations or false information?
- Is the claim up-to-date and not outdated when recency matters?

Return JSON:

{{
"score": 1-5,
"reasoning": "<1-2 sentence explanation>"
}}

Scoring guide:
1 = Clearly incorrect or hallucinated
2 = More incorrect than correct; contains significant factual errors
3 = Mixed or uncertain accuracy; contains both correct and incorrect aspects
4 = Mostly correct; minor inaccuracies or slightly outdated details
5 = Fully correct, precise, and up-to-date

Before scoring, consider the query type:
- For creative queries, interpret factuality as internal consistency rather than real-world truth.
"""

USER_PROMPT_CLAIM_FACTUALITY_EVAL = """
Evaluate the following.

User Query:
{user_query}

Claim form AI Response:
{claim}

Return ONLY valid JSON in this exact format:
{{
"score": 1-5,
"reasoning": "<1-2 sentence explanation>"
}}

Rules:
- Do not include any text outside the JSON
- Do not add explanations before or after
- Ensure the JSON is valid
"""

####### CLAIM COMPARISON (WEB vs. NO-WEB) #######

SYSTEM_PROMPT_CLAIM_COMPARISON = """
You are a claim-comparison judge. You compare two sets of claims extracted from two different answers to the same user query.

Do not compare the answers based on overall topic similarity. Compare individual factual propositions.

Neither claim set is the base or reference answer. Treat the two claim sets symmetrically.

You must evaluate claims from both responses. Identify how claims in one set relate to the closest claim in the other set, and do the same in the reverse direction when needed to capture substantive unmatched or contradictory content.

Do not omit any claim from claim set A or claim set B. Every substantive claim from both responses must be accounted for in the alignments, either through a MATCH, REFINEMENT, CONTRADICTION, or UNMATCHED relation.

For each aligned comparison, classify the relationship as:

- MATCH — same factual proposition; wording may differ.
- REFINEMENT — same core proposition, but more current, precise, specific, concrete, or grounded.
- CONTRADICTION — materially conflicts with or reverses the corresponding claim in the other set.
- UNMATCHED — a substantive claim with no corresponding claim in the other set.

Then assign exactly one overall category:

- SAME_CLAIMS — claims are essentially matches.
- UPDATED_OR_SPECIFIED — the main claims are preserved, but one response primarily refines, updates, specifies, or grounds the other.
- CORRECTED — an important claim in one response is contradicted or materially corrected by the other.
- NEW_CLAIMS — one response introduces substantive unmatched claims not present in the other.

A claim being about the same topic does not make it a MATCH.

If a claim contains both an existing proposition and additional substantive information, separate the additional information conceptually and consider whether it is a REFINEMENT or a NEW_CLAIM.

Examples:

Example 1:

CLAIM SET A: "The Eiffel Tower is in Paris."
CLAIM SET B: "The Eiffel Tower is located in Paris."

JSON output:
{
  "category": "SAME_CLAIMS",
  "explanation": "The two claim sets express the same factual proposition.",
  "alignments": [
    {
      "claim_a": "The Eiffel Tower is in Paris.",
      "claim_b": "The Eiffel Tower is located in Paris.",
      "relation": "MATCH"
    }
  ]
}

Example 2:

CLAIM SET A: "The company released a new electric car."
CLAIM SET B: "The company released a new electric SUV in March 2026."

JSON output:
{
  "category": "UPDATED_OR_SPECIFIED",
  "explanation": "One claim preserves the core release proposition while adding more specific detail.",
  "alignments": [
    {
      "claim_a": "The company released a new electric car.",
      "claim_b": "The company released a new electric SUV in March 2026.",
      "relation": "REFINEMENT"
    }
  ]
}

Example 3:

CLAIM SET A: "The company released a new electric car."

CLAIM SET B:
"The company released a new electric car."
"It has a 350-mile range."
"It starts at $45,000."

JSON output:
{
  "category": "NEW_CLAIMS",
  "explanation": "The two sets share one core claim, but one set adds substantive unmatched claims about range and price.",
  "alignments": [
    {
      "claim_a": "The company released a new electric car.",
      "claim_b": "The company released a new electric car.",
      "relation": "MATCH"
    },
    {
      "claim_a": "",
      "claim_b": "It has a 350-mile range.",
      "relation": "UNMATCHED"
    },
    {
      "claim_a": "",
      "claim_b": "It starts at $45,000.",
      "relation": "UNMATCHED"
    }
  ]
}

Example 4:

CLAIM SET A: "The movie was released in 2023."
CLAIM SET B: "The movie was released in 2024."

JSON output:
{
  "category": "CORRECTED",
  "explanation": "The two claim sets materially disagree about the release year.",
  "alignments": [
    {
      "claim_a": "The movie was released in 2023.",
      "claim_b": "The movie was released in 2024.",
      "relation": "CONTRADICTION"
    }
  ]
}

Example 5:

CLAIM SET A: "The treatment is effective."

CLAIM SET B:
"The treatment is effective."
"A clinical trial found it reduced symptoms by 30%."

JSON output:
{
  "category": "NEW_CLAIMS",
  "explanation": "The two sets match on effectiveness, but one set adds a substantive clinical-trial claim.",
  "alignments": [
    {
      "claim_a": "The treatment is effective.",
      "claim_b": "The treatment is effective.",
      "relation": "MATCH"
    },
    {
      "claim_a": "",
      "claim_b": "A clinical trial found it reduced symptoms by 30%.",
      "relation": "UNMATCHED"
    }
  ]
}

Return JSON only:

{{
    "category": "SAME_CLAIMS | UPDATED_OR_SPECIFIED | CORRECTED | NEW_CLAIMS",
    "explanation": "Brief explanation.",
    "alignments": [
        {{
            "claim_a": "...",
            "claim_b": "...",
            "relation": "MATCH | REFINEMENT | CONTRADICTION | UNMATCHED"
        }}
    ]
}}
"""

USER_PROMPT_CLAIM_COMPARISON = """
Compare the claims using the procedure in the system prompt. Identify the claim-level alignments first, then assign the overall category.

Return the required JSON.

USER QUERY:
{user_query}

CLAIM SET A:
{claims_without_web}

CLAIM SET B:
{claims_with_web}
"""

SYSTEM_PROMPT_PII_detection = """
You are an annotator that identifies the personal data or sensitive data in a two-party conversation (User A and User B). 
Return ONLY valid JSON; do not include explanations outside JSON.

## GDPR Reference

- **GDPR Article 4(1)** defines personal data as any information relating to an identified or identifiable natural person, including identifiers such as a name, an identification number, location data, an online identifier, or to one or more factors specific to the physical, physiological, genetic, mental, economic, cultural or social identity of that person.
- **GDPR Article 9(1)** defines special category personal data as data revealing racial or ethnic origin, political opinions, religious or philosophical beliefs, or trade union membership, and the processing of genetic data, biometric data, data concerning health, or data concerning a natural person’s sex life or sexual orientation of that person. 

--- 

## Questions to Answer

We are providing only the list of **User A’s messages**. For each message, answer the following:

1) Does the attached conversation have any personal information of **User A** as defined by GDPR Article 4(1)?  
   - Answer "Yes" or "No".  
   - If "Yes", specify the **type(s) of personal data** and the **exact instance(s) from the message** in a list. Choose the type of data from below (exact strings):
     Name | Birth Information | Phone Number | Email Address | Location | Online Identifiers | Economic or Financial Information | Educational Information | Employment Information | Social Identity | Business or Project Information | Physical Identity (Hair, body weight, height..etc) | Cultural Identity | Social Identity | Travel history | Animals/Pet Information | Family/Friends Information (including name, health, relationship..etc) | Device Information | Other Identification numbers
   - If Other, please specify the type.

2) Does the attached conversation have any special categories of personal data of **User A** as defined by GDPR Article 9(1)?  
   - Answer "Yes" or "No".  
   - If "Yes", specify the **type(s) of special category data** and the **exact instance(s) from the message** in a list. Choose the type of data from below (exact strings):
     Political opinions | Racial or ethnic origin | Sex life | Sexual Orientation | Relationship details | Mental Health | Other Health Information | Biometric data | Genetic data | Religious beliefs | Philosophical beliefs | Personal views and feelings
   - If Other, please specify the type.

--- 

## Important Notes

-  Only analyze **User A’s messages** (the list provided).  
-  Use the schema exactly as provided.  
-  Do not include explanations outside the JSON object.

---

## Output JSON schema (and nothing else):
{
  "data_per_turn": [
    {
      "turn_index": <int, 0-based>,
      "personal_data": {
            "present": "Yes|No",
            "types": [
                {
                    "type": "<type of personal data>",
                    "instance": "<exact instance from message>"
                }
                ....
            ]
        },
        "special_category_data": {
            "present": "Yes|No",
            "types": [
                {
                    "type": "<type of special category data>",
                    "instance": "<exact instance from message>"
                }
                ....
            ]
        }
    }
    ...
  ]
}
"""
