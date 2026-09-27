BRAINF4RT Metaculus Forecasting Bot

A Metaculus forecasting bot based on the current Metaculus/metac-bot-template, with a fully free OpenRouter reasoning stack and a custom DDGS web-research pipeline.

Architecture

Metaculus question
        |
        +--> deterministic search-query builder
        |       +--> FULL ORIGINAL QUESTION (always searched)
        |       +--> entity/topic queries
        |       +--> resolution-focused queries
        |
        v
DDGS web research
        |
        +--> HTTP fetch
        +--> Trafilatura
        +--> BeautifulSoup
        +--> DDGS indexed-snippet fallback
        +--> original DDGS search snippet fallback
        |
        v
Research-analysis LLM
        |
        +--> Nemotron 3 Ultra :free
        +--> Laguna S 2.1 :free
        +--> Qwen 3.8 27B :free
        |
        v
Template-style forecasting prompt
        |
        +--> Nemotron 3 Ultra :free
        +--> Laguna S 2.1 :free
        +--> Qwen 3.8 27B :free
        |
        v
Structured parser
        |
        +--> Qwen 3.8 27B :free
        +--> Gemma 4 31B IT :free
        |
        v
Metaculus

All OpenRouter calls in the research/reasoning path use free endpoints. No paid provider is required for the bot's normal forecasting workflow.

Question types

The bot implements the five question types supported by the current forecasting-tools template:

Binary

Multiple choice

Numeric

Date

Conditional

Conditional questions follow the current forecasting-tools structure of a parent question, child question, and the two child forecasts conditional on the parent resolving Yes or No.

Forecasting prompts

The forecasting prompts have been brought back in line with the current Metaculus bot-template prompts rather than using the previous custom forecasting wording.

This includes the template's instructions around:

status-quo reasoning

time remaining

Yes/No scenarios for binary questions

moderate probability on multiple-choice alternatives

numeric units, monotonic percentile ordering and wide 90/10 intervals

chronological date percentiles and UTC timestamps when hours matter

conditional-child handling

The parser instructions likewise follow the current template's parsing guidance.

The bot retains its own model/fallback and research infrastructure around those prompts.

Researcher question context

Every research-analysis LLM receives the actual Metaculus question information before it sees the web research.

For a normal question this includes:

Question type

Metaculus page URL

Exact question text

Background information

Resolution criteria

Fine print

Multiple-choice options when applicable

Numeric units and bounds when applicable

Date bounds when applicable

For conditional questions it also receives the full parent/child structure, including the Yes-conditional and No-conditional child questions.

This is deliberate: the researcher is instructed to judge evidence against the exact Metaculus resolution criteria, not merely against a topic or search query.

Search behavior

The exact original question is always included as a search query, alongside the additional deterministic queries generated from question entities, background and resolution criteria.

The search stage runs queries in parallel. The pipeline makes a second attempt with alternate deterministic query variants when the first pass produces no usable research.

Search candidates use lightweight lexical ranking, but the filter is intentionally permissive so useful sources are not thrown away solely because their title has weak word overlap with the question. The research-analysis LLM performs the final relevance judgement.

Metaculus URLs

Metaculus URLs are allowed. There is no hostname or URL rule excluding metaculus.com.

A Metaculus page can therefore:

appear in DDGS results,

survive source aggregation,

be fetched directly,

fall through to the DDGS indexed snippet path if direct extraction fails, and

be included in the research passed to the researcher/forecaster.

Scraping fallback chain

For each result the scraper attempts:

HTTP request
    -> Trafilatura extraction
    -> BeautifulSoup extraction
    -> DDGS targeted indexed snippet lookup
    -> original DDGS search-result snippet

Confirmed HTTP 404/410 responses and clearly identified dead-page content are discarded. Transient network failures are not treated as proof that a page is dead.

Model configuration

Research and forecast reasoning

nvidia/nemotron-3-ultra-550b-a55b:free
    -> poolside/laguna-s-2.1:free
    -> qwen/qwen3.8-27b:free

A failure of one model causes the next model in the chain to be attempted.

Structured parsing

qwen/qwen3.8-27b:free
    -> google/gemma-4-31b-it:free

These models are kept separate from the reasoning chain because structured-output compatibility can differ between free OpenRouter endpoints.

Installation

poetry install
cp .env.template .env

Set at least:

METACULUS_TOKEN=...
OPENROUTER_API_KEY=...
OPENROUTER_MODEL=nvidia/nemotron-3-ultra-550b-a55b:free

The application validates the pinned model configuration at startup so an accidental model change does not silently alter the intended architecture.

Running

Tournament

poetry run python main.py --mode tournament

Runs the Fall 2026 FutureEval tournament plus MiniBench and publishes forecasts.

Metaculus Cup

poetry run python main.py --mode metaculus_cup

Runs against the current Metaculus Cup target and publishes forecasts.

Test questions

poetry run python main.py --mode test_questions

Uses the official bot-testing-area, which is intended to exercise all supported question types. Test mode does not publish forecasts.

Configuration files

bot.py contains the ForecastBot implementation and current template prompts.

clients/openrouter_helper.py contains the direct OpenRouter request path and the three-model fallback chain.

research/pipeline.py contains deterministic query generation, parallel web searches and research summarisation orchestration.

research/scraper.py contains DDGS search, source ranking, dead-page detection and the scraping fallback chain.

main.py contains environment/model validation and run-mode dispatch.

bot_helpers.py contains startup/environment helpers.

Testing

CI should at minimum run:

poetry run python main.py --mode test_questions

Python syntax for the modified source files can also be checked with:

python -m py_compile bot.py main.py bot_helpers.py clients/openrouter_helper.py research/pipeline.py research/scraper.py

Development credit

This project was developed with assistance from ChatGPT and Claude for code review, implementation, debugging, prompt comparison and repository documentation.

The forecasting logic and final configuration remain the project owner's responsibility.
