# Model Architecture

The bot uses multiple specialised models and **fallbacks at each critical stage**.

The purpose of the fallback system is to prevent a temporary failure of one model or service from causing an entire forecasting run to fail.

The architecture is:

```text
                         Metaculus
                             │
                             ▼
                    Question + Metadata
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Search Query        │
                  │ Generation          │
                  │                     │
                  │ Gemma 4 31B IT      │
                  │ reasoning: OFF      │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Web Research        │
                  │                     │
                  │ DDGS                │
                  │ Trafilatura         │
                  │ BeautifulSoup       │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Forecast Reasoning  │
                  │                     │
                  │ Nemotron 3 Ultra    │
                  └──────────┬──────────┘
                             │
                       API/model failure
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Laguna S 2.1        │
                  │ fallback            │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Structured Parsing  │
                  │                     │
                  │ Nex-N2.5-Pro        │
                  └──────────┬──────────┘
                             │
                       API/model failure
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Nex-N2.5-Mini       │
                  │ fallback            │
                  └──────────┬──────────┘
                             │
                             ▼
                         Metaculus
```

---

## Search Query Generation

The bot first determines what information should be searched for.

Search queries are generated using:

```text
google/gemma-4-31b-it:free
```

Reasoning is deliberately disabled for this stage.

The generated queries are then passed to the web research pipeline.

There is intentionally **no second LLM used as a query-generation fallback**. If query generation fails completely, the research pipeline falls back to searching the original question text rather than allowing the entire forecast to fail.

```text
Gemma 4 31B
     │
     │ success
     ▼
Generated search queries
     
     │ failure
     ▼
Original question text
     │
     ▼
Web research
```

---

# Forecast Reasoning Fallback

Forecast reasoning is the most important model stage, so it has an explicit primary/fallback system.

### Primary

```text
nvidia/nemotron-3-ultra-550b-a55b:free
```

Nemotron 3 Ultra is the primary forecasting model because it is intended to perform the deeper reasoning required to turn the question, resolution criteria, and research into a probabilistic forecast.

### Fallback

```text
poolside/laguna-s-2.1:free
```

Laguna S 2.1 is used when the primary Nemotron request fails.

A failure can include things such as:

* Provider unavailable
* HTTP/API failure
* Temporary model outage
* Request failure
* Other errors that prevent the primary model from returning a usable response

The bot does **not** simply abandon the question when Nemotron fails.

Instead:

```text
                    Forecast question
                           │
                           ▼
                ┌─────────────────────┐
                │ Nemotron 3 Ultra     │
                │ PRIMARY              │
                └──────────┬──────────┘
                           │
                    successful?
                      /          \
                    YES           NO
                     │             │
                     ▼             ▼
                 Continue       Laguna S 2.1
                                  FALLBACK
                                     │
                                     ▼
                                  Continue
```

Both models are accessed through OpenRouter.

This is particularly important when using free model endpoints, where temporary provider overload can occur.

---

# Structured Parser Fallback

The forecasting model produces reasoning and a forecast, but Metaculus requires structured prediction data.

The bot therefore uses a separate model to convert the forecast into the required structured format.

### Primary parser

```text
nex-agi/nex-n2.5-pro:free
```

### Fallback parser

```text
nex-agi/nex-n2.5-mini:free
```

The parser fallback works similarly to the forecasting fallback:

```text
             Forecast reasoning
                    │
                    ▼
          ┌───────────────────┐
          │ Nex-N2.5-Pro      │
          │ PRIMARY PARSER    │
          └─────────┬─────────┘
                    │
              successful?
                /       \
              YES        NO
               │          │
               ▼          ▼
          Structured   Nex-N2.5-Mini
           forecast      FALLBACK
                           │
                           ▼
                    Structured forecast
```

The parser models are **not responsible for deciding the forecast from scratch**.

Their job is to take the forecasting model's output and convert it into the exact structure required by `forecasting_tools`.

This separation reduces the amount of work the forecasting model has to do and makes structured-output failures easier to recover from.

---

# Web Research Fallbacks

The web research system also has multiple layers of fallback.

## Search engines

The bot uses DDGS to query several search backends:

```text
Brave
Google
Bing
DuckDuckGo
Yahoo
Wikipedia
```

A backend failing does not necessarily prevent the other search backends from being attempted.

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
                  More backends
                       │
                       ▼
                  Search results
```

This is useful because individual search providers can intermittently return errors, rate-limit requests, or become unavailable.

---

## Webpage extraction

Once a search result has been found, the bot attempts to extract the useful page content.

The preferred extraction method is:

```text
Trafilatura
```

If Trafilatura cannot successfully extract the page, the scraper falls back to:

```text
BeautifulSoup
```

The process is:

```text
Search result
     │
     ▼
Trafilatura
     │
     │ extraction failure
     ▼
BeautifulSoup
     │
     ▼
Extracted webpage content
```

If neither method succeeds, the result can still retain its search snippet where available.

---

# Why Multiple Fallbacks?

The bot is designed around the assumption that **AI forecasting infrastructure is unreliable**.

A single request can fail because:

* A model provider is overloaded
* A free model reaches a provider limit
* A search engine returns an error
* A webpage blocks automated requests
* A webpage cannot be parsed
* A model returns malformed structured output
* A temporary network failure occurs

Without fallbacks, one of these failures could prevent an entire question from being forecast.

With the fallback architecture, the system attempts to degrade gracefully:

```text
                    Ideal
                      │
                      ▼
              Nemotron forecast
                      │
                      │ failure
                      ▼
                Laguna forecast
                      │
                      ▼
                Nex Pro parser
                      │
                      │ failure
                      ▼
               Nex Mini parser
                      │
                      ▼
                 Submission
```

The goal is therefore **not to hide failures**, but to recover from failures wherever a reasonable alternative exists.

---

# Current Model Stack

| Function                | Primary                                         | Fallback                     |
| ----------------------- | ----------------------------------------------- | ---------------------------- |
| Search query generation | `google/gemma-4-31b-it:free`                    | Original question text       |
| Forecast reasoning      | `nvidia/nemotron-3-ultra-550b-a55b:free`        | `poolside/laguna-s-2.1:free` |
| Structured parsing      | `nex-agi/nex-n2.5-pro:free`                     | `nex-agi/nex-n2.5-mini:free` |
| Web search              | Brave / Google / Bing / DDG / Yahoo / Wikipedia | Other available backends     |
| Web extraction          | Trafilatura                                     | BeautifulSoup                |

All of the LLMs in the current architecture are accessed through **OpenRouter**.

The project intentionally uses free model endpoints wherever possible.

---

# Error Handling

The bot is designed to allow individual failures without unnecessarily terminating an entire forecasting run.

Forecast reports are collected with exceptions returned rather than immediately terminating the complete tournament run.

For example, if one question fails:

```text
Question 1 → successful
Question 2 → successful
Question 3 → Nemotron fails → Laguna succeeds
Question 4 → successful
Question 5 → parser fails → Nex Mini succeeds
Question 6 → failed
```

the bot can continue processing the remaining questions instead of treating Question 6 as a reason to stop the entire tournament run.

Failures are logged so they can be investigated after the run.

---

# Concurrency Protection

Free model endpoints can be particularly sensitive to high request volumes.

The bot therefore limits concurrent LLM requests.

The current general LLM concurrency limit is:

```text
2
```

This means the bot deliberately avoids sending a large burst of simultaneous requests to OpenRouter.

The concurrency limit works alongside the fallback system:

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
        failure?       failure?
             │             │
             ▼             ▼
          Fallback       Fallback
```

The intention is to improve reliability and reduce provider overload while still allowing multiple questions to be processed concurrently.

---

# Tournament Resilience

The combination of:

* Multiple search providers
* Multiple webpage extraction methods
* Primary/fallback forecasting models
* Primary/fallback parsing models
* Limited concurrency
* Per-question exception handling

means that the bot is designed to remain operational even when individual components temporarily fail.

This is particularly useful for tournament operation, where the bot may run automatically for long periods through GitHub Actions.

The fallback architecture does **not** guarantee that every question will succeed. If all available models, search providers, and extraction methods fail, the question can still fail. The purpose is to reduce avoidable failures rather than eliminate them entirely.
