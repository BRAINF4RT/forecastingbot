# Metaculus Fully-Free AI Forecasting Bot

Automated Metaculus forecasting bot using free OpenRouter models and free web research.

## Architecture

```text
Metaculus
   |
   v
OpenRouter/free query generator
   |
   +--> generated queries
   +--> ORIGINAL QUESTION (always searched)
   |
   v
DDGS multi-backend search
   |
   v
Trafilatura
   |
   +--> BeautifulSoup
   |
   +--> DDGS indexed-snippet fallback
   |
   v
Research summarisation
Nemotron -> Laguna -> openrouter/free
   |
   v
Forecast reasoning
Nemotron -> Laguna -> openrouter/free
   |
   v
Structured parsing
Nex-N2.5-Pro -> Nex-N2.5-Mini
   |
   v
Metaculus
```

## Supported question types

- Binary
- Multiple-choice
- Numeric
- Date
- Conditional

Conditional questions are forecast as four linked binary components: parent, child, child given parent=YES, and child given parent=NO.

## Model fallbacks

Main reasoning and research:

```text
nvidia/nemotron-3-ultra-550b-a55b:free
    -> poolside/laguna-s-2.1:free
    -> openrouter/free
```

Search-query generation:

```text
openrouter/free
```

Reasoning is explicitly disabled for query generation.

Structured parsing deliberately remains separate:

```text
openrouter/nex-agi/nex-n2.5-pro:free
    -> openrouter/nex-agi/nex-n2.5-mini:free
```

## Web research

The scraper has three extraction layers:

1. Trafilatura full-page extraction.
2. BeautifulSoup direct HTML extraction.
3. **DDGS indexed-snippet extraction** as the final fallback.

The DDGS fallback searches for the exact target URL and then the page title. This allows the bot to recover indexed text when the live page is blocked, JavaScript-only, rate-limited, or otherwise unavailable.

Metaculus URLs are never scraped.

## Search-query behaviour

The original question is **not merely a fallback**.

For every normal research attempt:

```text
query generator output
    +
original Metaculus question
    |
    v
DDGS
```

If a retry is needed, newly generated queries are combined with the original question again. Deterministic queries are only used after those attempts fail.

## Run

```bash
poetry install
cp .env.template .env
```

Set:

```text
METACULUS_TOKEN=...
OPENROUTER_API_KEY=...
```

Test:

```bash
poetry run python main.py --mode test_questions
```

Tournament:

```bash
poetry run python main.py --mode tournament
```

## Free-only design

No paid LLM or paid web-search provider is required by the active pipeline. Free model availability can change, so the OpenRouter free roster should be checked periodically.

## Development

This repository was developed with assistance from **ChatGPT and Claude** for architecture, implementation, debugging, research and troubleshooting.

Forecasts are automated model outputs and do not represent the author's personal views.
