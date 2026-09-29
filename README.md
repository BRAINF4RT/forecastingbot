# BRAINF4RT Metaculus Forecasting Bot

A fully automated Metaculus forecasting bot built around **free OpenRouter models**, deterministic web research, resilient web scraping, and `forecasting-tools`. It began as a fork of the official [Metaculus `metac-bot-template`](https://github.com/Metaculus/metac-bot-template) and has been substantially rewritten.

**Metaculus profile:** [metaculus.com/accounts/profile/277439](https://www.metaculus.com/accounts/profile/277439/)

The bot targets the **Fall FutureEval 2026 tournament**, the current MiniBench, and the Metaculus Cup.

> **Language:** Python 3.11+
> **Inference:** OpenRouter free models
> **Framework:** `forecasting-tools`
> **Paid APIs required:** none for the primary pipeline (see [Free-only design](#free-only-design))

---

## Contents

- [What makes this bot different](#what-makes-this-bot-different)
- [Supported question types](#supported-question-types)
- [Forecasting pipeline](#forecasting-pipeline)
- [Model configuration](#model-configuration)
- [Failure handling](#failure-handling)
- [Repository structure](#repository-structure)
- [Running the bot](#running-the-bot)
- [Run modes](#run-modes)
- [GitHub Actions](#github-actions)
- [Reviewing bot performance](#reviewing-bot-performance)
- [Optional integrations](#optional-integrations)
- [Free-only design](#free-only-design)
- [Known issues](#known-issues)
- [Credits](#credits)

---

## What makes this bot different

The research pipeline deliberately avoids using an LLM to invent web-search queries. Query generation is **fixed and deterministic**:

```text
Metaculus question
        │
        ├── Original question, verbatim
        ├── First 8 words (only if the question is >6 words)
        └── First 6 words + "latest news" (only if the question is >6 words)
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
 Research summary (LLM)
        │
        ▼
 Forecast reasoning (LLM)
        │
        ▼
 Structured-output parsing (LLM)
        │
        ▼
 forecasting-tools → Metaculus
```

The original question is always query #1, exactly as written. There is no LLM-driven query rewriting, expansion, or retry-query generation — see [`research/pipeline.py`](research/pipeline.py).

---

## Supported question types

| Question type    | Forecast format                    |
| ----------------- | ----------------------------------- |
| Binary            | Probability                         |
| Multiple choice   | Probability for every option        |
| Numeric           | Percentile distribution             |
| Date              | Date percentile distribution        |
| Conditional       | Parent/child conditional forecasts  |

The conditional implementation uses `forecasting-tools`' own semantics and can reuse an existing valid parent/child forecast where appropriate.

---

## Forecasting pipeline

### 1. Question retrieval

`forecasting-tools`' `MetaculusClient` fetches the question and its full metadata (text, type, background, resolution criteria, fine print, options, units, numeric/date bounds, conditional structure, URL).

### 2. Deterministic search-query generation — [`research/pipeline.py`](research/pipeline.py)

`build_search_queries()` produces up to three queries, described above. `resolution_criteria`, `background`, and a `retry` flag remain in its signature for backward compatibility but are not used — there is no deterministic-or-otherwise query expansion beyond the three fixed rules.

### 3. Parallel web search — [`research/scraper.py`](research/scraper.py)

The deduplicated queries are searched concurrently via `DDGS` across six backends (Brave, Google, Bing, DuckDuckGo, Yahoo, Wikipedia). Up to 4 results per query are requested.

### 4. Relevance filtering + extraction

Each candidate is scored with a deterministic relevance function (meaningful-token overlap, title vs. body weighting, bigram matches). Confirmed `404`/`410` and other soft-dead pages are rejected rather than treated as evidence. Surviving sources are scraped through a fallback chain:

```text
HTTP request → Trafilatura → BeautifulSoup → DDGS indexed snippet
```

Metaculus URLs are **not** filtered out of research results — a Metaculus page can itself be useful context.

### 5. Research summarisation

`clients/openrouter_helper.py`'s `summarize_research()` receives the full question context (not just the title) and is instructed to stick to facts relevant to the resolution criteria, preserve dates/numbers/sources, and avoid producing the actual forecast itself. The final research context returned to the forecaster combines this summary with a slice of the raw retrieved text.

### 6. Forecast reasoning

`generate_forecast_reasoning()` (also in `clients/openrouter_helper.py`) receives the question, background, resolution criteria, fine print, research, and any relevant bounds/options/units, with prompts adapted per question type (binary probability, per-option probabilities, percentile distributions, etc.).

### 7. Structured-output parsing

The reasoning model's free-text output is converted into a typed `forecasting-tools` prediction via `structure_output()`, using a dedicated **parser** LLM chain (see below) rather than requiring the reasoning model itself to emit clean JSON.

---

## Model configuration

There are two separate places models are configured, because two different code paths talk to OpenRouter.

### Direct OpenRouter client — `clients/openrouter_helper.py`

Used for research summarisation and forecast reasoning, called directly via `httpx` (not through `forecasting-tools`):

```text
PRIMARY_MODEL  = nvidia/nemotron-3-ultra-550b-a55b:free
FALLBACK_MODEL = poolside/laguna-s-2.1:free
THIRD_MODEL    = qwen/qwen3.8-27b:free
```

This client implements its own retry/backoff for retryable HTTP statuses (`408, 409, 425, 429, 500, 502, 503, 504`), and limits itself to `_OPENROUTER_CONCURRENCY = 1` concurrent request.

### `forecasting-tools` LLM chain — `bot.py`

`forecasting-tools` requires a `GeneralLlm`-compatible interface, so `bot.py` wraps the above models (with an `openrouter/` LiteLLM routing prefix) plus a dedicated parser chain, via the custom `FallbackGeneralLlm` class:

```text
default / summarizer / researcher:
    openrouter/nvidia/nemotron-3-ultra-550b-a55b:free
        → openrouter/poolside/laguna-s-2.1:free
            → qwen/qwen3.8-27b:free   (see "Known issues" below)

parser:
    openrouter/qwen/qwen3.8-27b:free
        → openrouter/google/gemma-4-31b-it:free
            → openrouter/nvidia/nemotron-3-ultra-550b-a55b:free
```

The parser chain is intentionally separate: the reasoning models don't reliably produce the structured output `forecasting-tools.structure_output()` needs, so parsing is delegated to a different model rotation optimized for that.

`FallbackGeneralLlm.invoke()`:

- Tries each model in the chain in order.
- If **every** model in the chain fails and all failures are rate-limit errors specifically, the whole chain is retried with linear backoff (up to 3 attempts total).
- If any failure isn't a rate-limit error, or the retries are exhausted, it raises with every model's error attached.
- Structured-output validation sampling is set to 1 (`_structure_output_validation_samples`), since each sample is a full extra parser call and the parser models share OpenRouter's free-tier rate-limit pool.

Both `main.py` (`validate_openrouter_configuration()`) and `bot.py` explicitly assert the exact model IDs at startup, so the bot fails loudly rather than silently drifting onto an unintended (or paid) model.

### Unused/experimental client — `clients/hf_vibethinker.py`

A separate client exists for calling `WeiboAI/VibeThinker-3B` via Hugging Face Inference Providers (routed through Featherless AI), with its own retry/backoff logic and an `HF_TOKEN` requirement. **It is not currently wired into `bot.py`** — `bot.py` imports `generate_forecast_reasoning` from `clients/openrouter_helper.py`, not from this module. It's present as an alternate forecasting-brain option, not part of the active pipeline.

---

## Failure handling

- **Per-request retries:** the direct OpenRouter client retries retryable HTTP statuses with bounded exponential backoff (up to 3 attempts per model).
- **Model fallback:** each chain (research/reasoning and parser) tries its models in order.
- **Whole-chain retry on rate limiting:** if every model in a chain is rate-limited simultaneously (common on OpenRouter's shared free-tier pool), the entire chain is retried with backoff rather than failing immediately.
- **Concurrency limits:** deliberately conservative — 1 concurrent request on the direct OpenRouter client, a small semaphore (`_GENERAL_LLM_SEMAPHORE`) on the `forecasting-tools` LLM path. Free endpoints get heavily rate-limited under higher concurrency, so this trades some throughput for reliability.
- **No paid fallback:** if every configured free model fails, the question fails. This is intentional — the bot does not silently switch to a paid model.

---

## Repository structure

```text
forecastingbot/
│
├── .claude/
│   └── skills/
│       └── review-bot/            # Claude Code skill for post-hoc performance review
│
├── .github/
│   └── workflows/
│       ├── run_bot_on_tournament.yaml     # Fall FutureEval 2026 + MiniBench, every 20 min
│       ├── run_bot_on_metaculus_cup.yaml  # Metaculus Cup, every 2 days
│       ├── test_bot.yaml                  # Manual-only, dry-run against bot-testing-area
│       └── review_bot.yaml                # Weekly scoring review (off by default)
│
├── clients/
│   ├── openrouter_helper.py       # Direct OpenRouter client: retries, fallback, summarisation
│   ├── hf_vibethinker.py          # Alternate HF-based reasoning client (not currently wired in)
│   └── test                       # stray placeholder file, not a real test suite
│
├── research/
│   ├── pipeline.py                # Deterministic query generation + research orchestration
│   ├── scraper.py                 # DDGS search, relevance scoring, scraping fallback chain
│   └── test                       # stray placeholder file, not a real test suite
│
├── integrations/
│   ├── README.md                  # Optional third-party integrations (see below)
│   └── main_lightningrod_eval.py  # LightningRod SDK example: news → questions → eval
│
├── bot.py                         # OpenRouterForecastBot, FallbackGeneralLlm, per-type forecasting
├── bot_helpers.py                 # Env checks, logging setup, run-summary banners
├── main.py                        # CLI entry point, mode dispatch, startup model validation
├── main_with_no_framework.py      # Standalone reference bot (OpenAI + AskNews/Perplexity), not the active pipeline
│
├── .env.template
├── .gitignore
├── DEPENDENCIES.md                # Notes on adding this bot's deps to a fresh template fork
├── poetry.lock
├── pyproject.toml
├── requirements.txt               # pip alternative to Poetry
└── README.md
```

| File                              | Purpose                                                              |
| ---------------------------------- | ---------------------------------------------------------------------- |
| `main.py`                          | CLI entry point, mode dispatch, startup model-configuration validation |
| `bot.py`                           | `OpenRouterForecastBot`, `FallbackGeneralLlm`, per-question-type forecasting logic |
| `bot_helpers.py`                   | Environment checks, noisy-dependency silencing, run banners            |
| `clients/openrouter_helper.py`     | Direct OpenRouter client: retries, model fallback, research summarisation, forecast reasoning |
| `clients/hf_vibethinker.py`        | Alternate HF Inference Providers client (currently unused by `bot.py`) |
| `research/pipeline.py`             | Deterministic query generation + research orchestration                |
| `research/scraper.py`              | `DDGS` search, source ranking/relevance, scraping fallback chain       |
| `main_with_no_framework.py`        | Standalone reference implementation, not part of the active bot        |
| `integrations/`                    | Optional third-party tooling (LightningRod SDK, bot-review)            |
| `.claude/skills/review-bot/`       | Claude Code skill that drives read-only post-hoc performance review    |
| `.env.template`                    | Environment-variable template                                          |
| `DEPENDENCIES.md`                  | How to add this bot's deps to a fresh `metac-bot-template` fork        |
| `.github/workflows/`               | GitHub Actions automation                                              |

---

## Running the bot

### Requirements

- Python 3.11+
- Poetry **or** pip
- A Metaculus API token
- An OpenRouter API key

### Environment variables

```bash
cp .env.template .env
```

At minimum:

```env
METACULUS_TOKEN=your_metaculus_token
OPENROUTER_API_KEY=your_openrouter_api_key
```

`main.py` validates both the credentials and the exact expected model configuration at startup, and exits before making any API calls if either is wrong.

### Installation — Poetry

```bash
poetry lock
poetry install
cp .env.template .env   # then fill in credentials
poetry run python main.py --mode test_questions
```

### Installation — pip

```bash
pip install -r requirements.txt
cp .env.template .env   # then fill in credentials
python main.py --mode test_questions
```

---

## Run modes

```bash
poetry run python main.py --mode <mode>
```

| Mode | What it does | Publishes? | Skips already-forecasted questions? |
| ---- | ------------- | :--------: | :----------------------------------: |
| `tournament` (default) | Fall FutureEval 2026 + current MiniBench | ✅ | ✅ |
| `metaculus_cup` | Current Metaculus Cup | ✅ | ❌ |
| `test_questions` | `bot-testing-area` (all supported question types) | ❌ (dry run) | ❌ |

`test_questions` mode is meant to exercise every supported question-type path (binary, multiple choice, numeric, date, conditional) in one CI run — it currently forecasts every question the API returns for `bot-testing-area`, not a single sampled question.

---

## GitHub Actions

| Workflow | Trigger | Purpose |
| -------- | ------- | ------- |
| `run_bot_on_tournament.yaml` | `schedule: */20 * * * *` + manual | Main tournament forecasting run |
| `run_bot_on_metaculus_cup.yaml` | `schedule: 0 0 */2 * *` (every 2 days) + manual | Metaculus Cup forecasting run |
| `test_bot.yaml` | Manual only | Dry-run against `bot-testing-area`, no publishing |
| `review_bot.yaml` | `schedule: 0 6 * * 1` (weekly) + manual | Read-only performance review; **off by default** — set repo variable `REVIEW_BOT_ENABLED=true` to enable the schedule |

Required repository secrets: `METACULUS_TOKEN`, `OPENROUTER_API_KEY`. `test_bot.yaml` additionally references several optional integration secrets (`PERPLEXITY_API_KEY`, `EXA_API_KEY`, `ASKNEWS_CLIENT_ID`, `ASKNEWS_SECRET`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) that the core DDGS-based research path does not actually require.

> **Note on scheduling reliability:** GitHub's `schedule` trigger is best-effort, not real-time — very frequent cron intervals (like every 20 minutes) can be delayed or dropped for hours during platform load, especially on public repos. If a tournament deadline is time-sensitive, consider triggering `run_bot_on_tournament.yaml`'s `workflow_dispatch` externally (e.g. via a third-party cron service calling the GitHub Actions `dispatches` API) rather than relying on the `schedule` trigger alone.

---

## Reviewing bot performance

A read-only, no-LLM-spend way to see how the bot's *already-resolved* forecasts scored, via the optional `metaculus-bot-review` package:

```bash
poetry install --with integrations
poetry run bot-review review --tournament <slug-or-id> --output review.json --summary review.md
poetry run bot-review review --resolved-since 30
```

`review.md` gives rank, questions scored, and best/worst questions. `review.json` adds per-question detail including every forecaster's prediction on every run. Pull specific reasoning without reading a whole report:

```bash
poetry run bot-review show <POST_ID> --section research
poetry run bot-review show <POST_ID> --forecaster R1:F3
```

Two things build on this:

- **`.github/workflows/review_bot.yaml`** — runs it weekly and attaches `review.json`/`review.md` to the run. Off by default (see the workflow table above).
- **`.claude/skills/review-bot/SKILL.md`** — a Claude Code skill that drives the whole diagnose-and-write-up loop. It's strictly read-only: it will not run the bot or change code without asking first.

---

## Optional integrations

`poetry install --with integrations` pulls in an optional dependency group not needed for core forecasting:

- **[LightningRod SDK](integrations/main_lightningrod_eval.py)** — generates forecasting questions from news sources (e.g. Google News) for benchmarking/training purposes. Needs `LIGHTNINGROD_API_KEY`. Unrelated to the bot's own forecasting pipeline.
- **[metaculus-bot-review](https://github.com/LouisP96/metaculus-bot-review)** — see [Reviewing bot performance](#reviewing-bot-performance) above.

See [`integrations/README.md`](integrations/README.md) for details.

---

## Free-only design

The core forecasting pipeline uses only free-tier services:

**LLM inference** (OpenRouter `:free` models):
```text
nvidia/nemotron-3-ultra-550b-a55b:free
poolside/laguna-s-2.1:free
qwen/qwen3.8-27b:free
google/gemma-4-31b-it:free
```

**Search:** `DDGS` across Brave, Google, Bing, DuckDuckGo, Yahoo, Wikipedia — no paid search API.

**Extraction:** `requests`, Trafilatura, BeautifulSoup, DDGS indexed snippets.

No paid OpenAI, Anthropic, Google, Perplexity, or news-API inference/search is required for the core pipeline. An OpenRouter API key is still required to access the free model endpoints, and free-tier availability/rate limits can change independently of this repository.

`main_with_no_framework.py` and the optional integrations (LightningRod, AskNews, Perplexity) are exceptions to this — they're reference material and opt-in tooling, not part of the default pipeline.

---

## Known issues

- **`THIRD_MODEL` third-fallback prefix:** `THIRD_MODEL` in `clients/openrouter_helper.py` is defined bare (`qwen/qwen3.8-27b:free`, no `openrouter/` prefix) for use by the direct-httpx client, which is correct for that path. However, `bot.py`'s `default`/`summarizer`/`researcher` LLM purposes pass this same bare value straight through as their `third_model` to `FallbackGeneralLlm`, which goes through LiteLLM — LiteLLM requires the `openrouter/` prefix to route correctly. If both the primary and fallback models in those three chains are rate-limited simultaneously, the third-fallback call will fail with `litellm.BadRequestError: LLM Provider NOT provided` instead of succeeding. This was already fixed specifically for the **parser** chain (`PARSER_THIRD_LLM` is explicitly set to the fully-qualified `openrouter/nvidia/...` string with a comment explaining why), but the same fix hasn't yet been applied to the `default`/`summarizer`/`researcher` chains.
- **`test_questions` mode forecasts every question in `bot-testing-area`**, not a single one — if `bot-testing-area` grows, so does the runtime and API-call volume of `test_bot.yaml`.
- **`run_bot_on_tournament.yaml` has no `concurrency` group**, unlike `run_bot_on_metaculus_cup.yaml` and `review_bot.yaml`. If a run is still in progress when the next scheduled trigger fires, they can run concurrently.

---

## Credits

Forked from the official [Metaculus `metac-bot-template`](https://github.com/Metaculus/metac-bot-template), which retains the `forecasting-tools` ([Metaculus/forecasting-tools](https://github.com/Metaculus/forecasting-tools)) architecture and `ForecastBot` interface. The research, model, fallback, scraping, and orchestration layers have been substantially rewritten for this bot's free-only, deterministic-query design.

Development and debugging assistance was provided by ChatGPT and Claude, used as coding/research assistants throughout.

## License

See the repository's license and the licenses of its upstream dependencies for applicable terms.
