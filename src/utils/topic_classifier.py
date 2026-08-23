"""Dependency-free topic classification, used when no topic-annotation file
is available.

data_extraction.py and data_extraction_other_cai.py normally look up each
conversation's topic from a research-only annotation file (see
chatgpt_conversation_utils.load_topics / other_platforms_parsing_utils.load_topics)
that only exists for the paper's own donated dataset -- it will not exist
for anyone running this pipeline on their own export. Rather than label
every conversation "Other" in that (common) case, `classify_topic` derives a
topic directly from the conversation's own text: a simple, transparent
keyword match against the same topic taxonomy
(web_tool_invocation.ALL_TOPICS) the rest of the pipeline already expects
topic values to be drawn from. No API calls, no ML model -- just
observing what the conversation is actually about.

This is a heuristic, not a replacement for the paper's LLM-annotated
labels: it is meant to make the pipeline meaningfully topic-aware out of
the box, not to reproduce the paper's own topic annotations exactly.
"""

import re
from collections import Counter

# Keyword lists are intentionally simple substrings (checked against
# lowercased text), one list per topic in the taxonomy every other analysis
# module already expects (see web_tool_invocation.ALL_TOPICS).
# "Misc" has no keywords -- like "Other", it's a fallback only, never a match.
_TOPIC_KEYWORDS = {
    "Travel": ["travel", "trip", "flight", "vacation", "hotel", "itinerary",
               "destination", "passport", "visa", "tourist", "tourism", "airbnb", "best places"],
    "Cars": ["car ", "vehicle", "engine", "mileage", "tire", "dealership",
             "suv", "sedan", "mechanic", "driving", "car’s", "car's"],
    "Mobile phones": ["phone", "iphone", "android", "smartphone", "samsung",
                       "pixel phone", "app store"],
    "Gift Suggestion": ["gift", "present for", "what should i get", "birthday present"],
    "Fashion": ["fashion", "outfit", "clothing", "wardrobe", "what to wear", "dress code"],
    "Household Work": ["chores", "cleaning", "laundry", "home repair",
                        "plumbing", "furniture", "declutter"],
    "Event Planning": ["planning a wedding", "planning a party", "event planning",
                        "guest list", "venue for"],
    "Weather and Climate": ["weather", "forecast", "temperature today",
                             "climate change", "rain", "snow", "storm"],
    "Finance": ["finance", "stock market", "invest", "budget", "tax ",
                "mortgage", "bank account", "crypto", "salary", "401k"],
    "Energy": ["electricity bill", "solar panel", "power grid", "fuel efficiency",
               "battery life"],
    "Politics & History": ["election", "politics", "government policy",
                            "president", "historical", "history of", "war of"],
    "Games": ["video game", "playstation", "xbox", "steam ", "level up",
              "gaming", "board game"],
    "Music": ["music", "song ", "album", "playlist", "concert", "lyrics"],
    "Military": ["military", "army", "navy", "air force", "warfare", "weapon system"],
    "Health": ["symptom", "doctor", "medicine", "diagnosis", "treatment for",
               "disease", "medical"],
    "Mental Health": ["anxiety", "depression", "therapy", "mental health",
                       "stressed", "counseling", "therapist"],
    "Law": ["legal advice", "lawsuit", "contract clause", "attorney", "lawyer", "court case"],
    "Security & Privacy": ["security", "privacy", "password", "hacked",
                            "encryption", "data breach", "vpn"],
    "Scientific Writing": ["research paper", "citation", "journal article",
                            "manuscript", "thesis", "literature review"],
    "Books": ["book recommendation", "novel", "which book", "author of",
              "reading list", "book club"],
    "Programming": ["code", "python", "javascript", "function", "bug in",
                     "programming", "software", "compile", "debug", "api "],
    "AI": ["artificial intelligence", "machine learning", "large language model",
           "llm ", "neural network", " ai ", "chatgpt", "gpt-", "openai"],
    "Science": ["biology", "chemistry", "experiment", "hypothesis", "scientific"],
    "Physics": ["physics", "quantum", "relativity", "gravity", "thermodynamics"],
    "Astrology": ["astrology", "horoscope", "zodiac", "star sign"],
    "Religion": ["religion", "bible", "quran", "church", "prayer", "faith in god"],
    "Gender and Diversity": ["gender identity", "transgender", "non-binary", "nonbinary",
                              "gender pronouns", "lgbtq", "diversity and inclusion",
                              "gender equality", "gender discrimination"],
    "Animals/Pets information": ["my dog", "my cat", "pet ", "veterinarian", "dog breed", "cat breed"],
    "Art": ["painting", "sculpture", "art history", "artist "],
    "Drinks": ["cocktail", "coffee recipe", "tea recipe", "wine pairing", "beer "],
    "Languages": ["translate", "translation", "grammar", "vocabulary", "learn spanish",
                  "learn french", "how do you say"],
    "Time conversion": ["time zone", "convert time", "utc", "gmt", "what time is it in"],
    "Math": ["equation", "algebra", "calculus", "solve for x", "derivative", "integral"],
    "Troubleshooting": ["not working", "troubleshoot", "error message", "fix this",
                         "won't start", "broken"],
    "Cooking": ["recipe", "how to cook", "ingredient", "bake ", "meal prep"],
    "Social Media Content Generation": ["instagram post", "tweet about", "caption for",
                                         "content for social media", "linkedin post"],
    "Email Drafting": ["draft an email", "write an email", "email to my",
                        "subject line"],
    "Job Search": ["resume", "cv for", "job interview", "career advice",
                    "job application"],
    "Art Generation": ["generate an image", "draw me", "image of a", "create an image"],
    "Roleplay": ["roleplay", "pretend you are", "act as ", "in character as"],
}

_COMPILED_TOPIC_PATTERNS = {
    topic: [re.compile(re.escape(kw)) for kw in keywords]
    for topic, keywords in _TOPIC_KEYWORDS.items()
}

DEFAULT_TOPIC = "Other"


def classify_topic(text):
    """Return the best-matching topic label for `text`, or DEFAULT_TOPIC if
    nothing matches.

    `text` is typically a conversation's first user message (or a short
    concatenation of its early messages) -- enough to gauge what the
    conversation is about without needing the whole history.
    """
    if not text:
        return DEFAULT_TOPIC

    haystack = f" {str(text).lower()} "
    hits = Counter()
    for topic, patterns in _COMPILED_TOPIC_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(haystack):
                hits[topic] += 1

    if not hits:
        return DEFAULT_TOPIC

    # Most keyword hits wins; ties broken by taxonomy order (dict insertion
    # order), matching _TOPIC_KEYWORDS's own ordering above.
    best_count = max(hits.values())
    for topic in _TOPIC_KEYWORDS:
        if hits.get(topic) == best_count:
            return topic
    return DEFAULT_TOPIC  # unreachable, but keeps this function total
