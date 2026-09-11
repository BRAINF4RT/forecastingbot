# VibeThinker Metaculus Bot

A Metaculus forecasting bot built on the official
[Metaculus/metac-bot-template](https://github.com/Metaculus/metac-bot-template)
scaffolding (`ForecastBot`, `MetaculusClient`, prediction aggregation, etc.
from the `forecasting-tools` package), with a fully custom brain and
research pipeline:

| Task                              | Model / tool                                             |
|------------------------------------|------------------------------------------------------------|
| **Forecast reasoning** (main brain) | `WeiboAI/VibeThinker-3B` via Hugging Face Inference Providers → Featherless AI |
| Search-query generation            | `nex-agi/nex-n2.5-mini:free` on OpenRouter (never the main brain) |
| Research summarization              | `nex-agi/nex-n2.5-pro:free` on OpenRouter                  |
| Parsing brain output → structured prediction | `nex-agi/nex-n2.5-pro:free` on OpenRouter (via `structure_output`) |
| Web search                          | DDGS                                                       |
| Page scraping                       | trafilatura → BeautifulSoup fallback → DDGS snippet fallback |

Model choices verified live against OpenRouter's `/api/v1/models` catalog on
2026-09-11 — both Nex-N2.5 variants are free, have a 262K context window,
and (unlike most of the other free-tier options checked) natively support
`structured_outputs`, which matters most for the parser slot since a bad
parse there corrupts the whole forecast. The free roster rotates, though —
re-check https://openrouter.ai/models?max_price=0 periodically.

## Why the brain never generates search queries

`clients/hf_vibethinker.py` exposes exactly one function,
`generate_forecast_reasoning`, and it's only ever called from `bot.py`'s
`_run_forecast_on_*` methods. `research/pipeline.py` (which does query
generation, scraping, and summarization) only imports
`clients/openrouter_helper.py` — it has no import path to the HF client at
all, so the main brain structurally cannot be used for research.

## Architecture

```
Metaculus question
      │
      ▼
bot.run_research()
      │
      ▼
research/pipeline.py
  1. openrouter_helper.generate_search_queries()   <- free OpenRouter model
  2. research/scraper.py: for each query:
        DDGS.text()  →  trafilatura.extract()
                      →  (fallback) requests + BeautifulSoup
                      →  (fallback) DDGS result snippet itself
  3. openrouter_helper.summarize_research()        <- free OpenRouter model
      │
      ▼
bot._run_forecast_on_binary/multiple_choice/numeric()
  - builds prompt (question + research brief)
  - hf_vibethinker.generate_forecast_reasoning()   <- VibeThinker-3B (main brain)
  - structure_output(..., model=parser)            <- free OpenRouter model
      │
      ▼
ForecastBot aggregates + posts to Metaculus
```

## Setup

This is meant to drop into a fork of the official
[Metaculus/metac-bot-template](https://github.com/Metaculus/metac-bot-template),
which uses **Poetry**. In your fork:

1. **Copy these files in**: `bot.py`, `clients/`, `research/` are new;
   `main.py` and `.env.template` replace the template's own copies.
   `bot_helpers.py` from the template is kept as-is (unchanged).

2. **Add the extra dependencies** — see `DEPENDENCIES.md`:
   ```bash
   poetry add httpx trafilatura beautifulsoup4 ddgs
   ```

3. **Get your API keys**
   - `METACULUS_TOKEN` — create at https://metaculus.com/aib
   - `HF_TOKEN` — a Hugging Face **fine-grained** token with the
     "Make calls to Inference Providers" permission:
     https://huggingface.co/settings/tokens
   - `OPENROUTER_API_KEY` — https://openrouter.ai/keys

4. **Copy `.env.template` to `.env`** and fill in the values. Three
   OpenRouter model env vars control the helper tasks:
   `OPENROUTER_QUERY_MODEL` (search-query generation),
   `OPENROUTER_HELPER_MODEL` (research summarization), and
   `OPENROUTER_PARSER_MODEL` (structured-output parsing) — all default to
   free Nex-N2.5 variants. OpenRouter's free roster rotates, so check
   https://openrouter.ai/models?max_price=0 and update these if calls start
   failing with a pricing error.

5. **Smoke test locally**
   ```bash
   poetry run python main.py --mode test_questions
   ```
   This forecasts on whatever's open in Metaculus's `bot-testing-area`
   tournament, which covers all question types.

6. **Run for real**
   ```bash
   poetry run python main.py --mode tournament
   ```

## Running on GitHub Actions

1. Push this repo to GitHub (your fork of the template).
2. Settings → Secrets and variables → Actions → add `METACULUS_TOKEN`,
   `HF_TOKEN`, `OPENROUTER_API_KEY` as repository secrets.
3. Enable Actions, then run the **Test Bot** workflow manually first to
   confirm everything posts correctly to Metaculus.
4. Enable **Forecast on new AI tournament questions** (every 20 min) and,
   if you want it, **Forecast on Metaculus Cup** (every 2 days).

## Notes and caveats

- **VibeThinker-3B is a small reasoning model** and tends to produce long
  chain-of-thought before its final answer — `max_tokens` for the brain
  call defaults to 4000 and the HTTP timeout to 180s. Raise these in
  `clients/hf_vibethinker.py` if you see truncated reasoning in the logs.
- **Featherless AI (free/serverless tier) can rate-limit or cold-start.**
  Both HF and OpenRouter calls retry with exponential backoff on HTTP 429.
- Date and conditional question types aren't implemented in `bot.py` (only
  binary, multiple-choice, and numeric) — the official template's `main.py`
  has reference implementations for those if you want to add them.
- `_max_concurrent_questions` isn't set here; the free-tier backends are
  rate-limit-sensitive, so you may want to add a semaphore (see the
  official template's `_concurrency_limiter` pattern) if you run into
  concurrent-request errors.
