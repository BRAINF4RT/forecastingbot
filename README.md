README.md
markdown
# forecastingbot

A Metaculus forecasting bot forked from [Metaculus/metac-bot-template](https://github.com/Metaculus/metac-bot-template),
running entirely on **free** OpenRouter models with no paid API keys.

## How it works

Metaculus question
|
v
Deterministic search-query construction
(the question text, an 8-word fragment, a "latest news" variant --
no LLM is involved in building queries)
|
v
DDGS web search + scrape
(Trafilatura -> BeautifulSoup -> DDGS's own snippet as a last resort)
|
v
Research summarization Forecast reasoning
nvidia/nemotron-3-ultra-550b :free nvidia/nemotron-3-ultra-550b :free
-> poolside/laguna-s-2.1 :free -> poolside/laguna-s-2.1 :free
-> qwen/qwen3.8-27b :free -> qwen/qwen3.8-27b :free
(clients/openrouter_helper.py, direct httpx calls, 3-model fallback chain)
| |
+------------------+-------------------+
|
v
forecasting_tools structured-output parsing
nvidia/nemotron-3-ultra-550b :free
-> poolside/laguna-s-2.1 :free
(bot.py's FallbackGeneralLlm, via forecasting_tools/LiteLLM,
2-model fallback -- this is the only forecasting_tools LLM
purpose this bot configures or uses)
|
v
Metaculus


Two independent code paths talk to OpenRouter, and it's worth understanding
why:

- **Research summarization and forecast reasoning** (`clients/openrouter_helper.py`)
  make raw `httpx` calls directly against OpenRouter's chat-completions
  endpoint, with their own retry/backoff logic and a 3-model fallback chain
  (Nemotron → Laguna → Qwen).
- **Structured-output parsing** (turning a forecaster's prose into a
  `BinaryPrediction` / `PredictedOptionList` / percentile list) goes through
  `forecasting_tools`' own LLM-purpose system (`bot.py`'s
  `FallbackGeneralLlm`, via LiteLLM), because `structure_output()` needs a
  `GeneralLlm`-compatible object. This path only has a 2-model fallback
  (Nemotron → Laguna).

Each path rate-limits itself with its own `asyncio.Semaphore`. These are
separate pools that don't coordinate, so the practical concurrency cap on
requests to OpenRouter's free endpoints at any moment is the sum of both
semaphores (currently 1 + 1 = 2). If you raise either, remember you're
raising total concurrent free-tier pressure, not just one code path's.

## Files

| File | Purpose |
|---|---|
| `main.py` | CLI entry point. Validates env vars and the expected model configuration, builds the bot, dispatches `--mode tournament / metaculus_cup / test_questions`. |
| `bot.py` | `OpenRouterForecastBot(ForecastBot)` — binary / multiple-choice / numeric forecasting logic, question-level concurrency limiting, and the `FallbackGeneralLlm` wrapper used only for structured-output parsing. |
| `bot_helpers.py` | Env-var validation, `.env` placeholder detection, startup/result banners, noisy-dependency suppression. |
| `clients/openrouter_helper.py` | Direct OpenRouter client (`httpx`-based) used for research summarization and forecast reasoning. Owns the 3-model fallback chain and its own request semaphore. |
| `research/pipeline.py` | Builds deterministic search queries from the question, runs the DDGS-backed scraper, and calls `summarize_research()` to produce the research brief `bot.py` uses. |
| `pyproject.toml` / `requirements.txt` | Dependencies — notably `httpx` (used directly by `clients/openrouter_helper.py`, not just pulled in transitively), `ddgs`, `trafilatura`, `beautifulsoup4`, `forecasting-tools`. |
| `.env.template` | Copy to `.env` and fill in `METACULUS_TOKEN` and `OPENROUTER_API_KEY`. |
| `.github/workflows/` | CI: a manual smoke test against `bot-testing-area`, a scheduled tournament run, and a scheduled Metaculus Cup run. |

## Models

The bot is intentionally locked to specific free OpenRouter models, and
`main.py`'s `validate_openrouter_configuration()` will refuse to start if
`bot.py` or `clients/openrouter_helper.py` don't match what it expects —
this is a deliberate guardrail against accidentally drifting onto a paid
model.

- **Primary:** `nvidia/nemotron-3-ultra-550b-a55b:free`
- **Fallback:** `poolside/laguna-s-2.1:free`
- **Third fallback** (research/reasoning path only): `qwen/qwen3.8-27b:free`

If you change a model anywhere, update it in **both** `bot.py` /
`clients/openrouter_helper.py` and the `EXPECTED_*` constants in `main.py`,
or startup validation will (correctly) reject the mismatch.

## Setup

1. **Set repository secrets** (`Settings → Secrets and variables → Actions
   → New repository secret`):
   - `METACULUS_TOKEN` — from <https://www.metaculus.com/futureeval/participate/>
   - `OPENROUTER_API_KEY` — from <https://openrouter.ai> (free account; no
     billing needed, every model used here is `:free`)

2. **Enable Actions**, then run `Actions → Test Bot → Run workflow` to
   smoke-test. This currently forecasts exactly one hardcoded subquestion
   (a numeric question, `metaculus.com/questions/43322`, unpacked from a
   group question) — see **Known limitations** below.

3. **Local install:**
```bash
   poetry lock
   poetry install
   cp .env.template .env   # then fill in your real keys
   poetry run python main.py --mode test_questions
```
   or with plain pip:
```bash
   pip install -r requirements.txt
   cp .env.template .env
   python main.py --mode test_questions
```

4. **Run modes:**
   - `--mode tournament` — forecasts the live `fall-futureeval-2026`
     tournament plus MiniBench, skipping questions already forecasted.
   - `--mode metaculus_cup` — forecasts the Metaculus Cup, always
     re-forecasting (doesn't skip previously-forecasted questions).
   - `--mode test_questions` — forecasts the one hardcoded smoke-test
     subquestion.

## Known limitations / things worth knowing

- **`test_questions` mode only exercises the numeric code path.** It
  targets one specific numeric subquestion, so a CI-green "Test Bot" run
  doesn't tell you `_run_forecast_on_binary` or
  `_run_forecast_on_multiple_choice` are actually working. If you want
  fuller coverage, point it at the `bot-testing-area` tournament instead
  (which has a mix of question types) — trade-off is a slower, less
  deterministic smoke test.
- **Binary/multiple-choice/numeric only.** There's no override for date or
  conditional questions. The live AIB tournament doesn't use them, but if
  you ever point this bot at a Metaculus question elsewhere on the site
  that does, that specific question will fail (caught individually, since
  every run uses `return_exceptions=True` — it won't take down the rest of
  the batch, but it also won't get a forecast).
- **Free-tier variability.** All three models are `:free` OpenRouter
  endpoints, which get rate-limited and occasionally deprioritized under
  load. Every OpenRouter call already retries with exponential backoff
  (`clients/openrouter_helper.py`) and/or falls back to a second/third
  model, but if every model in the chain is down at once, that question's
  forecast fails for the run and gets picked up again next scheduled
  trigger (tournament mode skips only questions it has *already
  succeeded* on).
- **`_max_concurrent_questions = 1` on `OpenRouterForecastBot`.** This
  bot processes one question fully (research → reasoning → parsing)
  before starting the next, rather than researching/forecasting several
  questions in parallel. This is deliberate: with the small OpenRouter
  concurrency limits above, letting many questions race for the same 1-2
  request slots just produces queuing and timeouts without actually
  speeding anything up. If you want more throughput and are willing to
  risk more `429`s, raise `_max_concurrent_questions` in `bot.py` and the
  two semaphores together, not just one of them.
