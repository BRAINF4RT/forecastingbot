# BRAINF4RT Metaculus Forecasting Bot

A fully automated Metaculus forecasting bot built around **free OpenRouter models**, deterministic web research, resilient web scraping, and `forecasting-tools`.

The bot is designed for Metaculus forecasting tournaments and currently targets the **Fall FutureEval 2026 tournament**, the current MiniBench, and the Metaculus Cup.

The entire LLM inference pipeline uses OpenRouter's `:free` model endpoints. There are **no paid LLM APIs or paid search APIs required**.

> **Status:** Active development
> **Primary target:** Metaculus Fall FutureEval 2026
> **Language:** Python 3.11+
> **Inference:** OpenRouter free models
> **Framework:** `forecasting-tools`

---

## What makes this bot different?

This bot deliberately avoids relying on an LLM to invent web-search queries.

Instead, it uses a deterministic query-generation strategy:

```text
Metaculus question
        │
        ├── Original question verbatim
        ├── First 8 words
        └── First 6 words + "latest news"
        │
        ▼
   DDGS web search
        │
        ├── Brave
        ├── Google
        ├── Bing
        ├── DuckDuckGo
        ├── Yahoo
        └── Wikipedia
        │
        ▼
 Source ranking + relevance filtering
        │
        ▼
    Web scraping
        │
        ├── Trafilatura
        ├── BeautifulSoup
        └── DDGS indexed snippets
        │
        ▼
 Research summary
        │
        ▼
 Forecast/reasoning LLMs
        │
        ├── Nemotron 3 Ultra
        ├── Laguna S 2.1
        └── Qwen3.8 27B
        │
        ▼
 Structured-output parser
        │
        ├── Qwen3.8 27B
        ├── Gemma 4 31B
        └── Nemotron 3 Ultra
        │
        ▼
 forecasting-tools
        │
        ▼
 Metaculus
```

The original Metaculus question is **always included as the first search query**. Search-query generation does not use an LLM, and there is no automatic LLM query rewriting or speculative query expansion.

---

# Features

### 🧠 Multi-model forecasting

The bot uses a fallback chain of free OpenRouter models for research summarisation and forecasting reasoning:

1. `nvidia/nemotron-3-ultra-550b-a55b:free`
2. `poolside/laguna-s-2.1:free`
3. `qwen/qwen3.8-27b:free`

If a model fails, the next model is tried automatically.

The parser uses a separate chain because the reasoning models do not reliably provide the structured-output behaviour required by `forecasting-tools`:

1. `qwen/qwen3.8-27b:free`
2. `google/gemma-4-31b-it:free`
3. `nvidia/nemotron-3-ultra-550b-a55b:free`

These model assignments are explicitly validated during startup so that the bot does not silently drift onto a paid or unintended model.

---

### 🌐 Free web research

Research is performed without paid search APIs.

The bot uses `DDGS` and searches multiple available backends:

* Brave
* Google
* Bing
* DuckDuckGo
* Yahoo
* Wikipedia

Search results are deduplicated, ranked for relevance, and then scraped where possible.

---

### 🕸️ Resilient scraping

A source is processed through several fallback layers:

```text
HTTP request
     │
     ▼
Trafilatura
     │
     ├── success → use extracted article
     │
     ▼
BeautifulSoup
     │
     ├── success → use extracted paragraphs
     │
     ▼
DDGS indexed snippet
     │
     ▼
Original search-engine snippet
```

This means the bot can still obtain useful evidence from websites that:

* block direct scraping
* return bot-check pages
* fail to extract cleanly
* have temporarily inaccessible pages
* expose useful information only through search-engine indexes

Confirmed HTTP `404`/`410` pages are rejected rather than being treated as evidence. The scraper also detects several classes of soft-dead pages.

---

### 🎯 Relevance filtering

Search results are not simply dumped into the LLM.

The scraper calculates a deterministic relevance score using:

* meaningful token overlap
* title matches
* body matches
* matching word pairs/bigrams

Titles receive greater weight than body text.

Sources below the relevance threshold can be discarded when enough stronger candidates are available.

---

### 🔗 Metaculus URLs are allowed

The bot **does not filter out Metaculus URLs**.

This is intentional. A Metaculus page can itself contain relevant information, forecasts, question context, or other material useful to the research process.

---

# Supported question types

The current bot handles:

| Question type   | Forecast format                    |
| --------------- | ---------------------------------- |
| Binary          | Probability                        |
| Multiple choice | Probability for every option       |
| Numeric         | Percentile distribution            |
| Date            | Date percentile distribution       |
| Conditional     | Parent/child conditional forecasts |

The conditional implementation uses the existing `forecasting-tools` semantics and can reuse an existing valid parent/child forecast where appropriate.

---

# Forecasting pipeline

## 1. Metaculus question retrieval

`forecasting-tools` obtains the question from Metaculus.

The bot preserves the important question metadata, including:

* question text
* question type
* background information
* resolution criteria
* fine print
* options
* units
* numeric bounds
* date bounds
* conditional structure
* Metaculus URL

The complete question context is made available to the research summarisation model.

---

## 2. Deterministic search-query generation

For a question with more than six words, up to three queries are generated:

```text
1. Exact question text

2. First 8 words

3. First 6 words + "latest news"
```

The first query is the **exact original question**.

For example:

```text
Will country X join organisation Y before 2030?
```

could produce:

```text
Will country X join organisation Y before 2030?

Will country X join organisation Y before

Will country X join organisation latest news
```

No LLM is involved in this stage.

---

## 3. Parallel web searching

The generated queries are searched concurrently.

Each query can retrieve multiple sources, which are then deduplicated and ranked.

The research pipeline currently requests up to four results per query.

---

## 4. Source extraction

Each selected source is passed through the scraping pipeline.

The resulting `ScrapedSource` records metadata such as:

* search query
* title
* URL
* original snippet
* extracted content
* extraction method
* search backend
* relevance score

This information is retained when the research is formatted for the LLM.

---

## 5. Research summarisation

The retrieved material is passed to the research model.

The research model receives the **complete Metaculus question context**, not just the title.

It is explicitly instructed to:

* identify information relevant to the exact question
* focus on the resolution criteria
* preserve important dates and numbers
* identify named sources
* preserve uncertainty
* identify meaningful disagreement
* avoid inventing facts
* avoid producing the actual forecast

The research model therefore acts as a research assistant rather than the final forecaster.

The final research context contains both the summary and a portion of the raw retrieved research.

---

# 6. Forecast reasoning

The forecasting model receives:

* the Metaculus question
* background
* resolution criteria
* fine print
* research
* relevant bounds/options/units
* current date

The forecasting prompts are adapted to the question type.

### Binary

The model produces a probability between 0 and 100%.

It is instructed to consider:

* time remaining
* status quo
* Yes scenario
* No scenario
* current research

The parsed probability is constrained to the range `0.01–0.99` before being returned to `forecasting-tools`.

### Multiple choice

The model produces a probability for every available option.

The parser is explicitly instructed to:

* use only valid option names
* remove accidental `"Option"` prefixes
* retain options with `0%`

### Numeric

The model produces a percentile distribution:

```text
Percentile 10
Percentile 20
Percentile 40
Percentile 60
Percentile 80
Percentile 90
```

The prompt explicitly distinguishes question bounds from actual expert/market evidence. Question metadata is not allowed to masquerade as an independent forecast.

### Date

Date questions use the same percentile structure, but dates are converted into timestamps for `forecasting-tools`.

The model is instructed to keep the percentiles chronologically ordered.

### Conditional

Conditional questions are decomposed into:

```text
Parent
Child
Child | Parent = YES
Child | Parent = NO
```

Existing valid parent/child forecasts can be reused rather than unnecessarily re-forecasting the same component.

---

# 7. Structured-output parsing

The reasoning model is not required to return perfect machine-readable JSON.

Instead:

```text
LLM reasoning
      │
      ▼
forecasting-tools structure_output()
      │
      ▼
Typed prediction
```

A dedicated parser LLM converts the forecaster's final response into the appropriate `forecasting-tools` data structure.

The parser chain is:

```text
Qwen3.8 27B
      ↓
Gemma 4 31B
      ↓
Nemotron 3 Ultra
```

The parser uses a low temperature and validation sampling to reduce formatting errors.

---

# Model architecture

There are effectively **two LLM systems** in the bot.

## Research / forecasting chain

Used for:

* research summarisation
* forecast reasoning

```text
Nemotron 3 Ultra :free
        ↓
Laguna S 2.1 :free
        ↓
Qwen3.8 27B :free
```

These requests are made directly against the OpenRouter chat-completions API using `httpx`.

---

## Structured-output parser chain

Used only for converting generated reasoning into `forecasting-tools` structured predictions.

```text
Qwen3.8 27B :free
        ↓
Gemma 4 31B :free
        ↓
Nemotron 3 Ultra :free
```

This path uses the `forecasting-tools`/LiteLLM interface because `structure_output()` requires a compatible `GeneralLlm`.

---

# Failure handling

Free model endpoints are inherently less predictable than paid endpoints, so the bot has several layers of resilience.

## OpenRouter retries

Retryable HTTP statuses include:

```text
408
409
425
429
500
502
503
504
```

Requests use bounded exponential backoff, up to three attempts per model.

---

## Model fallback

If a model continues failing after its retries:

```text
Nemotron
   ↓ failure
Laguna
   ↓ failure
Qwen
   ↓ failure
question fails
```

The same principle is used for structured-output parsing, with its own parser-specific chain.

---

## Concurrency limiting

The direct OpenRouter client currently allows one OpenRouter request at a time:

```python
_OPENROUTER_CONCURRENCY = 1
```

The `forecasting-tools` LLM path has its own semaphore with a concurrency of two.

These are separate concurrency pools because the bot has two different OpenRouter access paths.

The conservative limits are intentional: free OpenRouter endpoints can become heavily rate-limited, and increasing concurrency can turn throughput improvements into a large number of `429` responses.

---

# Repository structure

```text
forecastingbot/
│
├── .claude/
│   └── skills/
│       └── review-bot/
│
├── .github/
│   └── workflows/
│       └── GitHub Actions workflows
│
├── clients/
│   └── openrouter_helper.py
│
├── research/
│   ├── pipeline.py
│   └── scraper.py
│
├── bot.py
├── bot_helpers.py
├── main.py
├── main_with_no_framework.py
│
├── .env.template
├── .gitignore
├── DEPENDENCIES.md
├── poetry.lock
├── pyproject.toml
├── requirements.txt
└── README.md
```

The core components are:

| File                           | Purpose                                                                      |
| ------------------------------ | ---------------------------------------------------------------------------- |
| `main.py`                      | CLI entry point and tournament orchestration                                 |
| `bot.py`                       | Main `ForecastBot` implementation and question-type forecasting              |
| `bot_helpers.py`               | Environment checks, logging helpers and run summaries                        |
| `clients/openrouter_helper.py` | Direct OpenRouter client, retries and model fallback                         |
| `research/pipeline.py`         | Deterministic query generation and research orchestration                    |
| `research/scraper.py`          | DDGS search, source ranking, scraping and fallback extraction                |
| `main_with_no_framework.py`    | Standalone/reference implementation without the normal framework entry point |
| `.env.template`                | Environment-variable template                                                |
| `DEPENDENCIES.md`              | Dependency/integration documentation                                         |
| `.github/workflows/`           | Automated GitHub Actions execution                                           |

---

# Running the bot

## Requirements

* Python 3.11+
* Poetry **or** pip
* A Metaculus API token
* An OpenRouter API key
* Internet access

The project declares Python `^3.11` and depends on `forecasting-tools`, `ddgs`, `trafilatura`, `beautifulsoup4`, `httpx`, `python-dotenv`, and related packages.

---

## Environment variables

Create a `.env` file from the supplied template:

```bash
cp .env.template .env
```

At minimum:

```env
METACULUS_TOKEN=your_metaculus_token
OPENROUTER_API_KEY=your_openrouter_api_key
```

The bot performs explicit startup validation and refuses to run if the required credentials or expected model configuration are missing.

---

# Installation

## Poetry

```bash
poetry lock
poetry install
```

Then:

```bash
cp .env.template .env
```

Fill in the credentials and run:

```bash
poetry run python main.py --mode test_questions
```

---

## pip

```bash
pip install -r requirements.txt
```

Then:

```bash
cp .env.template .env
```

and:

```bash
python main.py --mode test_questions
```

---

# Run modes

## Tournament

```bash
poetry run python main.py --mode tournament
```

This runs the configured Fall FutureEval 2026 tournament and the current MiniBench.

Previously forecast questions are skipped in this mode.

---

## Metaculus Cup

```bash
poetry run python main.py --mode metaculus_cup
```

This runs the current Metaculus Cup.

Unlike the normal tournament mode, previously forecast questions are not skipped.

---

## Test questions

```bash
poetry run python main.py --mode test_questions
```

This runs against Metaculus':

```text
bot-testing-area
```

The test mode does **not publish forecasts**.

The testing tournament contains examples of the supported question structures, allowing the CI/test run to exercise:

* binary
* multiple choice
* numeric
* date
* conditional

forecasting paths.

---

# GitHub Actions

The repository includes GitHub Actions automation for running the bot remotely.

The intended setup is to provide the required credentials as repository secrets:

```text
METACULUS_TOKEN
OPENROUTER_API_KEY
```

This allows tournament runs to occur without storing credentials in the repository.

The bot itself also performs configuration validation before starting, including checking that the configured model chain matches the expected free models.

---

# Free-only design

One of the main goals of this project is keeping the complete AI forecasting pipeline accessible without paid inference services.

The bot currently uses:

### LLM inference

OpenRouter `:free` models:

```text
nvidia/nemotron-3-ultra-550b-a55b:free
poolside/laguna-s-2.1:free
qwen/qwen3.8-27b:free
google/gemma-4-31b-it:free
```

### Search

`DDGS`:

```text
Brave
Google
Bing
DuckDuckGo
Yahoo
Wikipedia
```

### Extraction

```text
requests
Trafilatura
BeautifulSoup
DDGS indexed snippets
```

There is therefore no dependency on:

* paid OpenAI inference
* paid Anthropic inference
* paid Google inference
* paid Perplexity search
* paid Google Search APIs
* paid news APIs

The OpenRouter API key is still required for access to the free model endpoints.

> **Important:** "free" refers to the model endpoints selected by this repository. Availability and rate limits of free OpenRouter models can change independently of this project.

---

# Why deterministic search queries?

Earlier versions of the bot experimented with LLM-generated search queries.

The current implementation intentionally does not do this.

The research pipeline guarantees that the original Metaculus question is searched exactly as written before adding two deterministic variants.

This has several advantages:

* the original question cannot accidentally disappear
* query generation is reproducible
* the bot cannot hallucinate search terms
* query generation does not consume an additional LLM call
* debugging is substantially easier
* behaviour is less dependent on whichever model happens to be available

The research system therefore separates **retrieval mechanics** from **LLM reasoning**.

---

# Why two separate OpenRouter paths?

The bot deliberately has two different ways of talking to OpenRouter.

### Direct client

`clients/openrouter_helper.py`

Used for:

* research summarisation
* forecast reasoning

It uses `httpx` directly and implements its own retry/fallback system.

### forecasting-tools LLM interface

`bot.py`

Used for:

* structured prediction parsing

This is required because `forecasting-tools.structure_output()` expects a `GeneralLlm` compatible implementation.

Keeping these systems separate allows the bot to use one model chain for reasoning and a different chain optimized for structured output.

---

# Current limitations

### Free-model availability

All inference models are free OpenRouter endpoints.

They can therefore experience:

* rate limiting
* temporary provider failures
* overloaded providers
* model availability changes

The bot has retries and fallbacks, but it cannot guarantee that every free endpoint will always be available.

---

### Sequential request pressure

The bot intentionally keeps OpenRouter concurrency conservative.

Increasing concurrency may increase throughput, but can also substantially increase `429` responses from free providers.

---

### Model roster can change

OpenRouter's free model roster is not permanent.

The repository currently locks itself to specific model IDs and validates those IDs at startup.

If a free model disappears or changes availability, the configuration will need to be updated deliberately rather than silently switching models.

---

### No paid fallback

If all configured free models fail, the question fails rather than silently switching to a paid model.

This is intentional.

---

# Configuration philosophy

The project intentionally prefers explicit configuration over hidden framework defaults.

`OpenRouterForecastBot` explicitly configures the LLM purposes it uses so `forecasting-tools` cannot silently select a different default model.

Likewise, `main.py` checks that the expected model IDs are still configured before starting.

This is particularly important for a competition bot where accidentally switching from a free model to a paid model would violate the project's design goal.

---

# Based on Metaculus' bot template

This repository began as a fork of the official:

**Metaculus `metac-bot-template`**

The current implementation retains the `forecasting-tools` architecture and `ForecastBot` interface, while substantially replacing the research, model, fallback, scraping, and orchestration layers.

Official framework:

https://github.com/Metaculus/forecasting-tools

Metaculus bot template:

https://github.com/Metaculus/metac-bot-template

---

# Development

The project is primarily Python and uses Poetry for dependency management.

Core dependencies include:

```text
Python 3.11+
forecasting-tools
httpx
ddgs
trafilatura
beautifulsoup4
requests
python-dotenv
```

See:

```text
pyproject.toml
requirements.txt
DEPENDENCIES.md
```

for the current dependency definitions.

---

# Credits

This project was developed from the Metaculus bot template and has been substantially modified for this forecasting system.

Development and debugging assistance was provided by:

* **ChatGPT**
* **Claude**

Both were used as coding/research assistants during the development of the bot's forecasting pipeline, fallback architecture, web-research system, and reliability improvements.

---

# License

See the repository's license and the licenses of its upstream dependencies for applicable terms.
