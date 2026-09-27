BRAINF4RT Metaculus Forecasting Bot

A Metaculus forecasting bot based on Metaculus/metac-bot-template, built around free OpenRouter models and a free DDGS web-research pipeline.

Architecture

Metaculus question
        |
        v
Deterministic targeted search queries
  - exact original question (ALWAYS included)
  - entity/topic query + latest news
  - entity/topic query + official data
  - resolution-focused variants
        |
        v
DDGS multi-backend search
        |
        v
Source ranking + dead-page detection
        |
        v
Scraping fallback chain
  1. Trafilatura
  2. BeautifulSoup
  3. DDGS indexed snippet
  4. original search-result snippet as final evidence layer
        |
        v
Research summarisation / forecast reasoning
  Nemotron 3 Ultra :free
        -> Laguna S 2.1 :free
        -> Qwen3.8 27B :free
        |
        v
forecasting_tools structured parsing
  Qwen3.8 27B :free
        -> Gemma 4 31B :free
        |
        v
Metaculus

All model endpoints used by the runtime are free OpenRouter endpoints. The bot does not require a paid search provider.

Supported question types

The bot implements all five question types supported by the current forecasting-tools template:

Binary

Multiple choice

Numeric

Date

Conditional

Conditional forecasts follow forecasting-tools semantics, including parent/child handling and conditional Yes/No child forecasts.

Search and scraping

The research system deliberately searches the full original Metaculus question, even when additional deterministic queries are generated. It no longer falls back to searching only a shortened form of the question.

Search candidates are ranked with lightweight lexical relevance, but the filter is intentionally permissive. When too few candidates meet the relevance threshold, lower-scoring candidates are retained so the research summariser can make the final relevance decision.

Dead-page protection remains enabled for genuine 404/410 responses and common soft-404 pages.

Metaculus URLs are allowed. There is no longer any URL/hostname filter that removes URLs containing or hosted on metaculus.com.

For pages that cannot be directly retrieved, the scraper searches DDGS for the exact URL and page title, then uses an indexed search-result snippet. A confirmed HTTP 404/410 is never resurrected from stale search data.

Model configuration

Forecast/research chain

nvidia/nemotron-3-ultra-550b-a55b:free
→ poolside/laguna-s-2.1:free
→ qwen/qwen3.8-27b:free

Structured parser chain

qwen/qwen3.8-27b:free
→ google/gemma-4-31b-it:free

The parser uses dedicated structured-output-capable models rather than relying on the research/reasoning models for JSON conversion.

The model configuration is validated at startup to catch accidental model drift.

Concurrency and retries

The bot keeps free-endpoint pressure low:

One active forecasting question at a time.

One direct OpenRouter request at a time for research/reasoning.

One structured-output request at a time.

Each direct OpenRouter model gets bounded retries before the next fallback model is attempted.

Async semaphores are loop-aware, so repeated runs in tests do not reuse an asyncio primitive from an old event loop.

Run modes

poetry run python main.py --mode tournament
poetry run python main.py --mode metaculus_cup
poetry run python main.py --mode test_questions

tournament forecasts the current Fall FutureEval tournament plus MiniBench.

metaculus_cup forecasts the current Metaculus Cup without skipping previously forecast questions.

test_questions forecasts the official bot-testing-area (including binary, multiple-choice, numeric, date, and conditional examples) and does not publish the resulting forecasts.

Setup

poetry install
cp .env.template .env

Set:

METACULUS_TOKEN=...
OPENROUTER_API_KEY=...
OPENROUTER_MODEL=nvidia/nemotron-3-ultra-550b-a55b:free

The checked-in requirements.txt is intentionally left unchanged.

GitHub Actions

The existing workflows can continue to invoke main.py directly. The important behavior change is that test_questions is now a dry run, while tournament modes publish as before.

Development credits

This bot was developed with assistance from ChatGPT and Claude for code review, debugging, architecture work, and documentation.

The project is based on the official Metaculus bot template:
https://github.com/Metaculus/metac-bot-template
