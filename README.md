# Metaculus Fully-Free AI Forecasting Bot

An experimental AI forecasting bot for [Metaculus](https://www.metaculus.com/) built around **free AI models, free web research, automated reasoning, structured forecast parsing, and automated tournament submission**.

The bot is designed primarily for the **Fall 2026 FutureEval Bot Tournament**.

> **The core goal of this project is simple: build a competitive automated forecasting system without paying for AI or research APIs.**

The project uses free OpenRouter models for every LLM stage and free/public web-search infrastructure for research.

> **Development note:** This repository was developed with assistance from **ChatGPT and Claude**. Both were used for code generation, debugging, architecture design, research, troubleshooting, and implementation throughout development.

---

# Why This Bot?

Most modern AI forecasting systems can become expensive very quickly.

A typical forecasting pipeline might use:

* A paid reasoning model
* A paid web-search API
* A paid research/answer engine
* A separate paid structured-output model
* Multiple paid fallback providers

This project deliberately takes a different approach.

## Everything is free

The bot is designed so that **the entire AI and web-research pipeline uses free services**.

There is no paid LLM required for:

* Search query generation
* Forecast reasoning
* Forecast fallbacks
* Structured parsing
* Research summarisation

There is also no paid research provider required for:

* Search
* Webpage retrieval
* Webpage extraction

Instead, the bot combines free OpenRouter models with DDGS and direct webpage extraction.

---

# The Core Architecture

```text
                         METACULUS
                             │
                             ▼
                    Question + Metadata
                             │
                             ▼
              ┌──────────────────────────┐
              │  SEARCH QUERY GENERATION │
              │                          │
              │ Gemma 4 31B IT :free     │
              │ Reasoning: DISABLED      │
              └────────────┬─────────────┘
                           │
                           ▼
              ┌──────────────────────────┐
              │      WEB RESEARCH        │
              │                          │
              │ DDGS                     │
              │ ├─ Brave                 │
              │ ├─ Google                │
              │ ├─ Bing                  │
              │ ├─ DuckDuckGo            │
              │ ├─ Yahoo                 │
              │ └─ Wikipedia             │
              │                          │
              │ Trafilatura              │
              │ BeautifulSoup fallback   │
              └────────────┬─────────────┘
                           │
                           ▼
              ┌──────────────────────────┐
              │   FORECAST REASONING     │
              │                          │
              │ Nemotron 3 Ultra :free   │
              └────────────┬─────────────┘
                           │
                     FAILURE / OUTAGE
                           │
                           ▼
              ┌──────────────────────────┐
              │ Laguna S 2.1 :free       │
              │ FALLBACK                 │
              └────────────┬─────────────┘
                           │
                           ▼
              ┌──────────────────────────┐
              │ STRUCTURED PARSING       │
              │                          │
              │ Nex-N2.5-Pro :free       │
              └────────────┬─────────────┘
                           │
                     FAILURE / OUTAGE
                           │
                           ▼
              ┌──────────────────────────┐
              │ Nex-N2.5-Mini :free      │
              │ FALLBACK                 │
              └────────────┬─────────────┘
                           │
                           ▼
                       METACULUS
                     Forecast Submit
```

The system deliberately separates:

1. **What should I research?**
2. **What does the research say?**
3. **What probability should I assign?**
4. **How do I turn that forecast into valid API data?**

Each stage has its own role.

---

# What Makes This Bot Different?

## 1. The entire AI stack is free

The current LLM architecture is:

| Function           | Primary                                  | Fallback                     | Cost     |
| ------------------ | ---------------------------------------- | ---------------------------- | -------- |
| Search queries     | `google/gemma-4-31b-it:free`             | Original question            | **Free** |
| Forecast reasoning | `nvidia/nemotron-3-ultra-550b-a55b:free` | `poolside/laguna-s-2.1:free` | **Free** |
| Structured parsing | `nex-agi/nex-n2.5-pro:free`              | `nex-agi/nex-n2.5-mini:free` | **Free** |

No paid model is intentionally required by the architecture.

The bot does **not** rely on:

* OpenAI API
* Anthropic API
* Perplexity API
* Paid Google Gemini API
* Paid OpenRouter models
* Hugging Face inference
* Featherless AI
* AskNews
* Exa
* Other paid research APIs

The project instead attempts to get as much forecasting capability as possible from the free models currently available through OpenRouter.

### Why?

This makes the project accessible to people who cannot afford to spend hundreds or thousands of dollars on API calls.

It also makes long-running experimentation much more practical.

---

# 2. Free web research

The research system does not use a commercial research API.

Instead, it uses **DDGS** to access multiple search backends:

```text
Brave
Google
Bing
DuckDuckGo
Yahoo
Wikipedia
```

Search results are then processed using:

```text
Trafilatura
```

with:

```text
BeautifulSoup
```

as a fallback.

The research process is therefore:

```text
Question
   │
   ▼
Gemma-generated queries
   │
   ▼
DDGS
   │
   ├── Brave
   ├── Google
   ├── Bing
   ├── DuckDuckGo
   ├── Yahoo
   └── Wikipedia
   │
   ▼
Search results
   │
   ▼
Direct webpage retrieval
   │
   ▼
Trafilatura
   │
   └── failure
          ▼
      BeautifulSoup
   │
   ▼
Research
```

There is no requirement for a paid search subscription.

---

# 3. Multiple fallback layers

Free services are inherently less predictable than paid enterprise APIs.

Free model providers can experience:

* Rate limits
* Temporary outages
* Provider overload
* HTTP errors
* Slow responses
* Capacity restrictions

The bot is therefore designed around **redundancy**.

Instead of:

```text
Model fails
   │
   ▼
Forecast fails
```

the bot attempts:

```text
Model fails
   │
   ▼
Fallback model
   │
   ▼
Continue forecasting
```

This philosophy is applied throughout the system.

---

# Forecast Reasoning

The main forecasting model is:

```text
nvidia/nemotron-3-ultra-550b-a55b:free
```

Nemotron receives the forecasting question along with relevant information such as:

* Question wording
* Resolution criteria
* Background information
* Fine print
* Current date
* Research results
* Forecasting instructions

It then produces the actual forecasting reasoning and proposed prediction.

---

# Forecasting Fallback

If Nemotron cannot produce a usable response, the bot falls back to:

```text
poolside/laguna-s-2.1:free
```

The architecture is:

```text
                Forecast question
                       │
                       ▼
             ┌──────────────────┐
             │ Nemotron 3 Ultra │
             │ PRIMARY          │
             └────────┬─────────┘
                      │
                Request fails
                      │
                      ▼
             ┌──────────────────┐
             │ Laguna S 2.1     │
             │ FALLBACK         │
             └────────┬─────────┘
                      │
                      ▼
                Continue
```

A fallback can be triggered by failures such as:

* Provider unavailable
* Temporary API failure
* Model outage
* Request failure
* Other errors preventing a usable forecast

The purpose is to make the bot resilient to temporary free-model instability.

---

# Structured Forecast Parsing

The forecasting model is responsible for reasoning.

It is **not** responsible for producing the final API structure perfectly.

A separate model handles structured parsing.

Primary parser:

```text
nex-agi/nex-n2.5-pro:free
```

Fallback:

```text
nex-agi/nex-n2.5-mini:free
```

The pipeline is:

```text
Nemotron / Laguna
       │
       ▼
Forecast reasoning
       │
       ▼
Nex-N2.5-Pro
       │
       │ failure
       ▼
Nex-N2.5-Mini
       │
       ▼
Structured forecast
       │
       ▼
forecasting-tools
       │
       ▼
Metaculus
```

The parser is therefore not intended to "re-forecast" the question.

Its primary purpose is to convert the model's forecast into the exact structured representation required by the Metaculus forecasting tools.

---

# Search Query Generation

The first LLM stage uses:

```text
google/gemma-4-31b-it:free
```

The model generates targeted queries based on:

* The question
* Resolution criteria
* Background information

Reasoning is explicitly disabled for this model.

The intention is to use a relatively capable model to create useful search queries without wasting additional reasoning capacity on a task that does not require it.

If query generation completely fails, the research pipeline can fall back to using the original question text as the search query.

```text
Gemma
  │
  ├── success ──► Generated queries
  │
  └── failure ──► Original question
                         │
                         ▼
                    Web research
```

---

# Web Search Fallbacks

The search system attempts multiple DDGS backends.

Current backends include:

```text
Brave
Google
Bing
DuckDuckGo
Yahoo
Wikipedia
```

If one backend fails, the system can continue using other available backends.

Conceptually:

```text
                   Search query
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       Brave        Google        Bing
          │            │            │
          └────────────┼────────────┘
                       │
                 Other backends
                       │
                       ▼
                 Search results
```

This is particularly useful because individual search providers can intermittently return:

* 403 errors
* 429 rate limits
* 5xx errors
* Timeouts
* Empty results

The system is designed not to treat one search provider failing as equivalent to the entire research system failing.

---

# Webpage Extraction Fallback

After finding a relevant webpage, the bot attempts to extract its useful text.

Primary extraction:

```text
Trafilatura
```

Fallback:

```text
BeautifulSoup
```

Therefore:

```text
Webpage
   │
   ▼
Trafilatura
   │
   ├── success ──► Extracted content
   │
   └── failure
          │
          ▼
     BeautifulSoup
          │
          ▼
     Extracted content
```

Where possible, search snippets can also be retained when full webpage extraction fails.

---

# Metaculus Research Restriction

The external research scraper deliberately blocks Metaculus URLs.

Blocked:

```text
metaculus.com
*.metaculus.com
```

This is an important part of the tournament-oriented design.

The research system should not scrape Metaculus pages and use information such as Community Prediction as research evidence.

The scraper therefore checks URLs before retrieval and also checks redirects so that a non-Metaculus URL cannot simply redirect into Metaculus and bypass the restriction.

```text
Search result
     │
     ▼
Is URL Metaculus?
     │
   ┌─┴─┐
  YES  NO
   │    │
   ▼    ▼
BLOCK  FETCH
```

Metaculus API access through `forecasting-tools` is still used for normal bot functionality, including:

* Retrieving questions
* Retrieving question metadata
* Submitting forecasts

The restriction applies specifically to the **external research/scraping pipeline**.

---

# Supported Forecast Types

The bot supports the major question types used by the target forecasting environment.

## Binary

Example:

```text
Will event X happen before date Y?
```

The bot produces a probability such as:

```text
37%
```

---

## Multiple Choice

The bot produces a probability distribution over the available options.

Example:

```text
Option A: 45%
Option B: 30%
Option C: 20%
Option D: 5%
```

The probabilities are converted into the appropriate `forecasting-tools` structure.

---

## Numeric

Numeric questions are handled using percentile distributions.

The bot reasons about the likely distribution of possible outcomes and produces percentile values that can be converted into a `NumericDistribution`.

---

## Date

Date questions use a similar percentile-distribution approach, but with date-specific bounds.

Date questions are handled separately from ordinary numeric questions because their bounds and data types differ.

---

# Forecasting Pipeline

For each question, the complete process is approximately:

```text
1. Retrieve question from Metaculus
             │
             ▼
2. Read question + resolution criteria
             │
             ▼
3. Generate search queries with Gemma
             │
             ▼
4. Search the web with DDGS
             │
             ▼
5. Filter blocked URLs
             │
             ▼
6. Retrieve relevant webpages
             │
             ▼
7. Extract content with Trafilatura
             │
             ▼
8. Fall back to BeautifulSoup if required
             │
             ▼
9. Assemble research
             │
             ▼
10. Forecast with Nemotron
             │
             │ failure
             ▼
       Forecast with Laguna
             │
             ▼
11. Parse forecast with Nex-N2.5-Pro
             │
             │ failure
             ▼
       Parse with Nex-N2.5-Mini
             │
             ▼
12. Convert to forecasting-tools object
             │
             ▼
13. Submit forecast to Metaculus
```

---

# Error Handling Philosophy

The bot is designed around **graceful degradation**.

A single component failing should not unnecessarily terminate the entire forecasting run.

For example:

```text
Question 1 → successful
Question 2 → Nemotron → Laguna → successful
Question 3 → successful
Question 4 → Nex Pro → Nex Mini → successful
Question 5 → search backend failure → other backend → successful
Question 6 → complete failure
Question 7 → successful
```

The failure of Question 6 should not prevent Question 7 from being processed.

This is especially important for automated tournament operation.

---

# Concurrency

Free AI endpoints can become unreliable if too many requests are sent simultaneously.

The bot therefore limits concurrent LLM requests.

The current general LLM concurrency limit is:

```text
2
```

This intentionally sacrifices some theoretical throughput for improved reliability.

The architecture is roughly:

```text
                 Question queue
                       │
                ┌──────┴──────┐
                ▼             ▼
            Request 1     Request 2
                │             │
                ▼             ▼
             Primary        Primary
                │             │
             fallback      fallback
                │             │
                └──────┬──────┘
                       ▼
                  Next questions
```

This is particularly useful when working with free OpenRouter providers that may have limited capacity.

---

# Fall 2026 FutureEval

The bot is configured to target the:

**Fall 2026 FutureEval Bot Tournament**

Tournament ID:

```text
fall-futureeval-2026
```

The tournament page currently lists:

* **Start:** September 28, 2026
* **End:** January 6, 2027
* **Prize pool:** $50,000

The bot explicitly targets the Fall 2026 tournament rather than relying solely on a generic "current AI competition" identifier.

This reduces the risk of the bot accidentally targeting a different seasonal tournament when the Metaculus competition changes.

---

# Running the Bot

## Test Questions

Before using tournament mode, use the Metaculus bot-testing tournament:

```bash
poetry run python main.py --mode test_questions
```

This is the recommended mode for testing code changes.

---

## Tournament Mode

Run:

```bash
poetry run python main.py --mode tournament
```

This runs the configured tournament forecasting process.

---

## Metaculus Cup

Run:

```bash
poetry run python main.py --mode metaculus_cup
```

This runs the configured Metaculus Cup mode.

---

# Installation

## Requirements

The project currently targets:

```text
Python 3.11
Poetry
```

The main dependencies include:

* `forecasting-tools`
* `python-dotenv`
* `requests`
* `httpx`
* `ddgs`
* `trafilatura`
* `beautifulsoup4`
* `numpy`

---

## Clone the Repository

```bash
git clone https://github.com/BRAINF4RT/forecastingbot.git
cd forecastingbot
```

---

## Install Dependencies

```bash
poetry install
```

Run commands through Poetry:

```bash
poetry run python main.py --mode test_questions
```

---

# Environment Configuration

Copy the environment template:

```bash
cp .env.template .env
```

The bot requires:

```env
METACULUS_TOKEN=your_metaculus_token
OPENROUTER_API_KEY=your_openrouter_api_key
```

These credentials should **never be committed to GitHub**.

For GitHub Actions, configure them as repository secrets.

---

# GitHub Actions

The repository includes an automated GitHub Actions workflow for tournament operation.

The workflow:

* Runs on Ubuntu
* Uses Python 3.11
* Installs Poetry
* Installs dependencies
* Runs the forecasting bot automatically
* Can be manually triggered
* Uses GitHub repository secrets
* Runs repeatedly during tournament operation

The required secrets are:

```text
METACULUS_TOKEN
OPENROUTER_API_KEY
```

The workflow runs:

```bash
poetry run python main.py --mode tournament
```

This allows the bot to operate without requiring a local computer to remain running.

---

# Repository Structure

```text
forecastingbot/
│
├── bot.py
│   └── Main forecasting implementation
│
├── main.py
│   └── Application entry point and tournament selection
│
├── bot_helpers.py
│   └── Startup, environment and reporting helpers
│
├── clients/
│   └── openrouter_helper.py
│       └── OpenRouter API/model handling
│
├── research/
│   ├── pipeline.py
│   │   └── Research orchestration
│   │
│   └── scraper.py
│       └── DDGS search and webpage extraction
│
├── .github/
│   └── workflows/
│       └── run_bot_on_tournament.yaml
│           └── Automated forecasting workflow
│
├── .env.template
│   └── Environment configuration template
│
├── pyproject.toml
│   └── Poetry dependencies and project configuration
│
└── README.md
    └── Project documentation
```

---

# Model Stack

The complete current model stack is:

| Stage                | Model                                    | Purpose                            | Fallback          |
| -------------------- | ---------------------------------------- | ---------------------------------- | ----------------- |
| Query generation     | `google/gemma-4-31b-it:free`             | Generate targeted research queries | Original question |
| Forecasting          | `nvidia/nemotron-3-ultra-550b-a55b:free` | Main forecasting reasoning         | Laguna            |
| Forecasting fallback | `poolside/laguna-s-2.1:free`             | Backup forecasting reasoning       | None              |
| Parser               | `nex-agi/nex-n2.5-pro:free`              | Structured forecast extraction     | Nex Mini          |
| Parser fallback      | `nex-agi/nex-n2.5-mini:free`             | Backup structured extraction       | None              |

All models are accessed through OpenRouter.

All are intended to be **free endpoints**.

---

# Why Use Different Models?

The bot intentionally does not use one model for everything.

Different tasks have different requirements.

### Gemma

Used for:

```text
"What should I search for?"
```

It does not need to perform the entire forecasting task.

### Nemotron

Used for:

```text
"Given the question, evidence and resolution criteria,
what probability should I forecast?"
```

This is the primary reasoning task.

### Laguna

Used when Nemotron cannot complete the task.

### Nex-N2.5

Used for:

```text
"Convert this forecast into the exact structured format
required by the forecasting API."
```

This separation allows each model to focus on a narrower responsibility.

---

# Cost Philosophy

The project is deliberately optimised around:

```text
$0 AI/research cost
        │
        ▼
Free model redundancy
        │
        ▼
Free search redundancy
        │
        ▼
Automated forecasting
```

There are trade-offs.

Free services can have:

* Lower availability
* Rate limits
* Provider overload
* Variable latency
* Model replacement
* Changing free-tier policies

The bot attempts to compensate through redundancy rather than simply switching to paid services.

The philosophy is:

> **Use multiple free components intelligently instead of paying for one expensive component to do everything.**

---

# Comparison to Other Metaculus Bots

The Metaculus ecosystem contains bots using sophisticated commercial research infrastructure.

Some approaches can use services such as:

* AskNews
* Perplexity
* Exa
* OpenAI
* Gemini
* Other commercial APIs

These can be extremely useful, but they may introduce:

* API costs
* Monthly subscriptions
* Credit limits
* Per-request charges
* Paid search quotas

This project intentionally avoids those dependencies.

Its approach is:

```text
              THIS PROJECT

          Free LLM
             │
             ▼
       Free metasearch
             │
             ▼
      Free web scraping
             │
             ▼
          Free LLM
             │
             ▼
      Free parser model
             │
             ▼
          Metaculus
```

The objective is not to claim that free models are inherently better than paid models.

Instead, the project asks:

> **How competitive can an automated forecasting system become when its AI and research pipeline costs essentially nothing?**

---

# Reliability Through Redundancy

The project combines several independent fallback mechanisms:

```text
                 ┌───────────────────────┐
                 │ Search query failure  │
                 └───────────┬───────────┘
                             ▼
                     Original question


                 ┌───────────────────────┐
                 │ Search backend fails  │
                 └───────────┬───────────┘
                             ▼
                    Another backend


                 ┌───────────────────────┐
                 │ Trafilatura fails     │
                 └───────────┬───────────┘
                             ▼
                       BeautifulSoup


                 ┌───────────────────────┐
                 │ Nemotron fails       │
                 └───────────┬───────────┘
                             ▼
                         Laguna


                 ┌───────────────────────┐
                 │ Nex Pro fails        │
                 └───────────┬───────────┘
                             ▼
                        Nex Mini
```

This is particularly important because the project intentionally operates on free infrastructure.

---

# Development

The recommended development process is:

## 1. Make a change

Modify the relevant source file.

## 2. Run the testing tournament

```bash
poetry run python main.py --mode test_questions
```

## 3. Inspect the logs

Look for:

* Search query generation
* Search backend results
* Scraping results
* Forecast reasoning
* Model fallback events
* Parser fallback events
* Forecast submission
* Question-specific failures

## 4. Only then use tournament mode

```bash
poetry run python main.py --mode tournament
```

---

# Known Trade-offs

A completely free forecasting stack has unavoidable limitations.

## Provider availability

Free OpenRouter models can become unavailable or overloaded.

This is why the bot has model fallbacks.

## Search reliability

Search engines can rate-limit automated requests or return inconsistent results.

This is why multiple DDGS backends are used.

## Web scraping

Some websites actively prevent automated retrieval.

This is why the scraper has multiple extraction methods.

## Model changes

Free model endpoints can be changed or removed by providers.

The model configuration may therefore need to be updated over time.

---

# Tournament Compliance

This project is intended to be used as an automated forecasting bot.

The external research scraper is explicitly prevented from accessing Metaculus pages.

This means the research pipeline should not use:

* Community Prediction
* Metaculus question-page content
* Other information obtained by scraping Metaculus

as external research.

The bot still interacts with Metaculus through the official forecasting tooling for legitimate operational purposes.

Tournament rules can change, so users should always check the current rules before entering a competition.

---

# Security

Never commit:

```text
METACULUS_TOKEN
OPENROUTER_API_KEY
```

to the repository.

Use:

* `.env` locally
* GitHub Actions Secrets in CI

The `.env` file should remain outside version control.

---

# Disclaimer

This is an experimental AI forecasting system.

The forecasts are generated automatically and should not be treated as guaranteed predictions or professional advice.

The project makes no claim that using larger or more expensive models necessarily produces better forecasts.

Actual forecasting performance should be evaluated using calibration, accuracy, scoring, and tournament results.

---

# Acknowledgements

This project builds upon the Metaculus forecasting ecosystem and the `forecasting-tools` framework.

Development was assisted by both **ChatGPT** and **Claude**.

### ChatGPT

Used for:

* Code generation
* Debugging
* Architecture design
* Research
* Troubleshooting
* API investigation
* Tournament compatibility investigation
* Refactoring
* Documentation

### Claude

Used for:

* Code generation
* Debugging
* Architecture design
* Research
* Troubleshooting
* Implementation assistance

The final repository contains custom modifications and architecture rather than being an unmodified copy of the original Metaculus template.

---

# Project Status

**Experimental — actively developed**

The project is being developed with the goal of testing how far a **fully free AI forecasting pipeline** can perform in competitive forecasting environments.

The architecture will continue to evolve as:

* Free models improve
* New free OpenRouter models become available
* Search reliability improves
* Forecasting performance is measured
* Tournament requirements change
* Metaculus APIs evolve
* Better fallback strategies are developed

---

## TL;DR

This project is an attempt to build a **competitive, automated Metaculus forecasting bot for $0 in AI and research costs**.

It uses:

```text
Gemma 4 31B
     │
     ▼
Free web search
     │
     ▼
Free webpage scraping
     │
     ▼
Nemotron 3 Ultra
     │
     └──► Laguna S 2.1 fallback
     │
     ▼
Nex-N2.5-Pro
     │
     └──► Nex-N2.5-Mini fallback
     │
     ▼
Metaculus
```

No paid reasoning API.

No paid search API.

No paid research provider.

No paid structured-output model.

Just free models, free research infrastructure, redundancy, and automated forecasting.
