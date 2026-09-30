"""
OpenRouter Metaculus Forecast Bot.

LLM architecture:
    Metaculus question
        |
        +--> deterministic targeted search queries
        |       +--> original question (always included)
        |       +--> entity/topic + current-data variants
        |       +--> resolution-focused variants
        |
        v
    DDGS web research
        |
        +--> Trafilatura
        +--> BeautifulSoup
        +--> DDGS indexed snippet
        +--> original search-result snippet
        |
        v
    Forecast/research reasoning
        +--> Nemotron 3 Ultra :free
        +--> Laguna S 2.1 :free
        +--> Qwen3.8 27B :free
        |
        v
    forecasting_tools structured parsing
        +--> Qwen3.8 27B :free
        +--> Gemma 4 31B :free
        +--> Nemotron 3 Ultra :free
        |
        v
    Metaculus

The dedicated parser models are used because the research/reasoning models do
not reliably advertise native structured-output support on their free
OpenRouter endpoints.
"""

from __future__ import annotations

import asyncio
import logging
import weakref
from datetime import datetime, timezone
from typing import Any

from forecasting_tools import (
    BinaryPrediction,
    BinaryQuestion,
    ConditionalPrediction,
    ConditionalQuestion,
    DateQuestion,
    DatePercentile,
    ForecastBot,
    GeneralLlm,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    PredictedOptionList,
    PredictionAffirmed,
    PredictionTypes,
    ReasonedPrediction,
    clean_indents,
    structure_output,
)

from clients.openrouter_helper import (
    FALLBACK_MODEL,
    PRIMARY_MODEL,
    THIRD_MODEL,
    generate_forecast_reasoning,
)

from forecasting_tools import FreeSearcher


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

PRIMARY_LLM = f"openrouter/{PRIMARY_MODEL}"
FALLBACK_LLM = f"openrouter/{FALLBACK_MODEL}"

# Preferred parser models.
#
# These are deliberately kept separate from the main reasoning chain because
# the parser is used by forecasting_tools' structured-output machinery.
PARSER_PRIMARY_LLM = "openrouter/qwen/qwen3.8-27b:free"
PARSER_FALLBACK_LLM = "openrouter/google/gemma-4-31b-it:free"

# IMPORTANT:
# Keep this explicitly provider-qualified.
#
# Previously the final parser fallback could become a bare:
#
#     qwen/qwen3.8-27b:free
#
# which LiteLLM interpreted without a provider and rejected with:
#
#     LLM Provider NOT provided
#
# Nemotron is already known to work through the OpenRouter endpoint.
PARSER_THIRD_LLM = "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free"

# Same fix, applied to the main research/reasoning chain's third fallback.
#
# THIRD_MODEL from clients/openrouter_helper.py is correctly bare (no
# "openrouter/" prefix) for that module's direct-httpx OpenRouter client,
# but the default/summarizer/researcher LLM purposes below go through
# LiteLLM, which requires the "openrouter/" prefix to route correctly --
# otherwise it fails with "LLM Provider NOT provided" exactly like the
# parser chain used to.
RESEARCH_THIRD_LLM = f"openrouter/{THIRD_MODEL}"


# Maximum number of forecasting_tools / LiteLLM calls allowed concurrently.
#
# The free OpenRouter models are subject to shared provider limits. A single
# concurrent call is intentionally conservative and greatly reduces the
# chance of several requests hitting the same shared limit simultaneously.
_GENERAL_LLM_CONCURRENCY = 1


# When every model in a chain returns a rate-limit error, back off and retry
# the complete chain rather than immediately failing the question.
_RATE_LIMIT_MAX_RETRIES = 3
_RATE_LIMIT_BACKOFF_SECONDS = 8


class _LoopBoundSemaphore:
    """
    Keep a separate semaphore for each asyncio event loop.

    forecasting_tools can create and destroy event loops in different
    execution paths. A normal asyncio.Semaphore created at module import time
    can become attached to the wrong loop, so this wrapper creates one lazily
    per loop.
    """

    def __init__(self, value: int) -> None:
        self._value = value
        self._semaphores: weakref.WeakKeyDictionary[
            asyncio.AbstractEventLoop,
            asyncio.Semaphore,
        ] = weakref.WeakKeyDictionary()

    def _get(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()

        semaphore = self._semaphores.get(loop)

        if semaphore is None:
            semaphore = asyncio.Semaphore(self._value)
            self._semaphores[loop] = semaphore

        return semaphore

    async def __aenter__(self) -> None:
        await self._get().acquire()

    async def __aexit__(self, *_: object) -> None:
        self._get().release()


_GENERAL_LLM_SEMAPHORE = _LoopBoundSemaphore(
    _GENERAL_LLM_CONCURRENCY
)


def _is_rate_limit_error(exc: Exception) -> bool:
    """Return True when an exception looks like an HTTP/OpenRouter 429."""
    name = type(exc).__name__.lower()
    text = str(exc).lower()

    return (
        "ratelimit" in name
        or "rate limit" in text
        or "rate-limited" in text
        or "429" in text
        or "temporarily rate" in text
    )


class FallbackGeneralLlm(GeneralLlm):
    """
    GeneralLlm wrapper with:

        primary
            -> fallback
                -> final fallback

    If every configured model is rate-limited, the complete chain is retried
    with exponential-ish backoff.
    """

    def __init__(
        self,
        *,
        primary_model: str,
        fallback_model: str,
        third_model: str | None,
        temperature: float,
        timeout: float,
    ) -> None:

        super().__init__(
            model=primary_model,
            temperature=temperature,
            timeout=timeout,
            allowed_tries=1,
        )

        self._models = [
            primary_model,
            fallback_model,
        ]

        if third_model:
            self._models.append(third_model)

        self._fallback_llms = [
            GeneralLlm(
                model=fallback_model,
                temperature=temperature,
                timeout=timeout,
                allowed_tries=1,
            )
        ]

        if third_model:
            self._fallback_llms.append(
                GeneralLlm(
                    model=third_model,
                    temperature=temperature,
                    timeout=timeout,
                    allowed_tries=1,
                )
            )

    async def _try_chain_once(
        self,
        prompt: str,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Any | None, list[tuple[str, Exception]]]:
        """
        Try every configured model exactly once.

        Returns:
            (result, errors)

        result is None when all models fail.
        """

        errors: list[tuple[str, Exception]] = []

        # Primary model.
        try:
            return (
                await super().invoke(
                    prompt,
                    *args,
                    **kwargs,
                ),
                errors,
            )

        except Exception as exc:
            errors.append(
                (
                    self._models[0],
                    exc,
                )
            )

            if len(self._models) > 1:
                logger.warning(
                    "Primary model failed: %s. Falling back to %s.",
                    self._models[0],
                    self._models[1],
                )

        # Remaining fallback models.
        for index, (model, llm) in enumerate(
            zip(
                self._models[1:],
                self._fallback_llms,
            ),
            start=1,
        ):

            try:
                result = await llm.invoke(
                    prompt,
                    *args,
                    **kwargs,
                )

                logger.info(
                    "Fallback model succeeded: %s",
                    model,
                )

                return result, errors

            except Exception as exc:
                errors.append(
                    (
                        model,
                        exc,
                    )
                )

                if index < len(self._fallback_llms):
                    next_model = self._models[index + 1]

                    logger.warning(
                        "Fallback model failed: %s. Falling back to %s.",
                        model,
                        next_model,
                    )

        return None, errors

    async def invoke(
        self,
        prompt: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Invoke the model chain.

        The whole chain is retried only when every model in the current attempt
        failed specifically because of rate limiting.
        """

        all_errors: list[tuple[str, Exception]] = []

        async with _GENERAL_LLM_SEMAPHORE:

            for attempt in range(
                1,
                _RATE_LIMIT_MAX_RETRIES + 1,
            ):

                result, errors = await self._try_chain_once(
                    prompt,
                    *args,
                    **kwargs,
                )

                all_errors.extend(errors)

                if result is not None:
                    return result

                all_rate_limited = (
                    bool(errors)
                    and all(
                        _is_rate_limit_error(exc)
                        for _, exc in errors
                    )
                )

                if (
                    all_rate_limited
                    and attempt < _RATE_LIMIT_MAX_RETRIES
                ):
                    wait = _RATE_LIMIT_BACKOFF_SECONDS * attempt

                    logger.warning(
                        "All %d configured models were rate-limited "
                        "(attempt %d/%d). Retrying the whole chain "
                        "after %ds.",
                        len(self._models),
                        attempt,
                        _RATE_LIMIT_MAX_RETRIES,
                        wait,
                    )

                    await asyncio.sleep(wait)
                    continue

                break

        details = "\n".join(
            f"{model}: {error!r}"
            for model, error in all_errors
        )

        raise RuntimeError(
            "All configured LLMs failed.\n" + details
        ) from (
            all_errors[-1][1]
            if all_errors
            else RuntimeError(
                "No models were configured."
            )
        )


# ---------------------------------------------------------------------------
# FreeSearcher research pipeline
# ---------------------------------------------------------------------------
#
# Replaces research/pipeline.py's run_research_pipeline(). Both the query
# planner and the evidence condenser use the exact same three-model fallback
# chain as the main forecaster (PRIMARY_LLM -> FALLBACK_LLM ->
# RESEARCH_THIRD_LLM), via the FallbackGeneralLlm class defined above, so
# research gets the same rate-limit backoff/retry behavior as reasoning does.
#
# If query planning fails entirely (all three models down), FreeSearcher
# falls back to the same deterministic query rules research/pipeline.py
# already used (original question, first 8 words, first 6 words + "latest
# news"), then still gathers Google News / GDELT / DDGS news / Wikipedia /
# Polymarket / Manifold on top regardless of whether planning succeeded.

_FREE_SEARCHER_CONDENSER_LLM = FallbackGeneralLlm(
    primary_model=PRIMARY_LLM,
    fallback_model=FALLBACK_LLM,
    third_model=RESEARCH_THIRD_LLM,
    temperature=0.15,
    timeout=240,
)

_FREE_SEARCHER_PLANNER_LLM = FallbackGeneralLlm(
    primary_model=PRIMARY_LLM,
    fallback_model=FALLBACK_LLM,
    third_model=RESEARCH_THIRD_LLM,
    temperature=0.3,
    timeout=240,
)

_free_searcher = FreeSearcher(
    llm=_FREE_SEARCHER_CONDENSER_LLM,
    planner=_FREE_SEARCHER_PLANNER_LLM,
    num_queries=3,
)


class OpenRouterForecastBot(ForecastBot):
    """
    Metaculus forecasting bot.

    Every forecasting_tools LLM purpose is explicitly configured so that the
    framework never silently falls back to an unavailable default.
    """

    # Every validation sample causes another structured parsing call.
    #
    # One validation sample is sufficient for this free-tier setup and avoids
    # unnecessarily multiplying requests against the same provider pools.
    _structure_output_validation_samples = 1

    def _llm_config_defaults(self) -> dict[str, GeneralLlm]:
        """
        Explicitly configure every forecasting_tools LLM purpose.

        This is important because forecasting_tools can request:
            - default
            - summarizer
            - researcher
            - parser

        Leaving one of these absent results in:
            Unknown llm requested from llm dict for purpose: 'summarizer'
        """

        return {
            "default": FallbackGeneralLlm(
                primary_model=PRIMARY_LLM,
                fallback_model=FALLBACK_LLM,
                third_model=RESEARCH_THIRD_LLM,
                temperature=0.15,
                timeout=240,
            ),
            "summarizer": FallbackGeneralLlm(
                primary_model=PRIMARY_LLM,
                fallback_model=FALLBACK_LLM,
                third_model=RESEARCH_THIRD_LLM,
                temperature=0.10,
                timeout=240,
            ),
            "researcher": FallbackGeneralLlm(
                primary_model=PRIMARY_LLM,
                fallback_model=FALLBACK_LLM,
                third_model=RESEARCH_THIRD_LLM,
                temperature=0.10,
                timeout=240,
            ),
            "parser": FallbackGeneralLlm(
                primary_model=PARSER_PRIMARY_LLM,
                fallback_model=PARSER_FALLBACK_LLM,
                third_model=PARSER_THIRD_LLM,
                temperature=0.0,
                timeout=240,
            ),
        }

    # -----------------------------------------------------------------------
    # RESEARCH
    # -----------------------------------------------------------------------

    async def run_research(
        self,
        question: MetaculusQuestion,
    ) -> str:
        """
        Run the complete research pipeline.

        The researcher receives:
            - exact question text
            - resolution criteria
            - background
            - fine print
            - question type
            - URL
            - options where relevant
            - numeric/date bounds where relevant
            - conditional structure where relevant
        """

        logger.info(
            "Starting research for %s",
            question.page_url,
        )

        research = await _free_searcher.research(
            question_text=question.question_text,
            resolution_criteria=(
                question.resolution_criteria or ""
            ),
            background=(
                question.background_info or ""
            ),
            fine_print=(
                question.fine_print or ""
            ),
            question_context=(
                self._format_research_question_context(
                    question
                )
            ),
        )

        logger.info(
            "Research for %s:\n%s",
            question.page_url,
            research[:1000],
        )

        return research

    def _format_research_question_context(
        self,
        question: MetaculusQuestion,
    ) -> str:
        """Build complete question metadata for the research summarizer."""

        sections = [
            f"Question type: {type(question).__name__}",
            f"Metaculus question URL: {question.page_url}",
            f"Question: {question.question_text}",
            f"Background: {question.background_info or ''}",
            f"Resolution criteria: {question.resolution_criteria or ''}",
            f"Fine print: {question.fine_print or ''}",
        ]

        if isinstance(
            question,
            MultipleChoiceQuestion,
        ):
            sections.append(
                f"Options: {question.options}"
            )

        if isinstance(
            question,
            NumericQuestion,
        ):
            sections.extend(
                [
                    (
                        "Units for answer: "
                        f"{question.unit_of_measure or 'Not stated'}"
                    ),
                    (
                        "Lower bound: "
                        f"{question.lower_bound}"
                    ),
                    (
                        "Upper bound: "
                        f"{question.upper_bound}"
                    ),
                    (
                        "Open lower bound: "
                        f"{question.open_lower_bound}"
                    ),
                    (
                        "Open upper bound: "
                        f"{question.open_upper_bound}"
                    ),
                ]
            )

        if isinstance(
            question,
            DateQuestion,
        ):
            sections.extend(
                [
                    (
                        "Lower date bound: "
                        f"{question.lower_bound}"
                    ),
                    (
                        "Upper date bound: "
                        f"{question.upper_bound}"
                    ),
                    (
                        "Open lower bound: "
                        f"{question.open_lower_bound}"
                    ),
                    (
                        "Open upper bound: "
                        f"{question.open_upper_bound}"
                    ),
                ]
            )

        if isinstance(
            question,
            ConditionalQuestion,
        ):
            sections.extend(
                [
                    "Conditional structure:",
                    (
                        "PARENT:\n"
                        + self._format_research_question_context(
                            question.parent
                        )
                    ),
                    (
                        "CHILD:\n"
                        + self._format_research_question_context(
                            question.child
                        )
                    ),
                    (
                        "CHILD CONDITIONAL ON PARENT = YES:\n"
                        + self._format_research_question_context(
                            question.question_yes
                        )
                    ),
                    (
                        "CHILD CONDITIONAL ON PARENT = NO:\n"
                        + self._format_research_question_context(
                            question.question_no
                        )
                    ),
                ]
            )

        return clean_indents(
            "\n".join(sections)
        )

    # -----------------------------------------------------------------------
    # BINARY
    # -----------------------------------------------------------------------

    async def _run_forecast_on_binary(
        self,
        question: BinaryQuestion,
        research: str,
    ) -> ReasonedPrediction[float]:

        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:

            {question.question_text}

            Question background:

            {question.background_info}

            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:

            {question.resolution_criteria}

            {question.fine_print}

            Your research assistant says:

            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:

            (a) The time left until the outcome to the question is known.

            (b) The status quo outcome if nothing changed.

            (c) A brief description of a scenario that results in a No outcome.

            (d) A brief description of a scenario that results in a Yes outcome.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

            {self._get_conditional_disclaimer_if_necessary(question)}

            The last thing you write is your final answer as:
            "Probability: ZZ%"
            where ZZ is 0-100.
            """
        )

        reasoning = await generate_forecast_reasoning(
            prompt
        )

        logger.info(
            "Forecast reasoning for %s: %s",
            question.page_url,
            reasoning[:1000],
        )

        binary_prediction: BinaryPrediction = (
            await structure_output(
                reasoning,
                BinaryPrediction,
                model=self._parser_llm(),
                num_validation_samples=(
                    self._structure_output_validation_samples
                ),
            )
        )

        decimal_pred = max(
            0.01,
            min(
                0.99,
                binary_prediction.prediction_in_decimal,
            ),
        )

        return ReasonedPrediction(
            prediction_value=decimal_pred,
            reasoning=reasoning,
        )

    # -----------------------------------------------------------------------
    # MULTIPLE CHOICE
    # -----------------------------------------------------------------------

    async def _run_forecast_on_multiple_choice(
        self,
        question: MultipleChoiceQuestion,
        research: str,
    ) -> ReasonedPrediction[PredictedOptionList]:

        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:

            {question.question_text}

            The options are:
            {question.options}

            Background:

            {question.background_info}

            Resolution criteria:

            {question.resolution_criteria}

            {question.fine_print}

            Your research assistant says:

            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:

            (a) The time left until the outcome to the question is known.

            (b) The status quo outcome if nothing changed.

            (c) A description of a scenario that results in an unexpected outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}

            You write your rationale remembering that:

            (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and

            (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

            The last thing you write is your final probabilities for the N options in this exact order:

            {question.options}

            Format the final answer as:

            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )

        reasoning = await generate_forecast_reasoning(
            prompt
        )

        logger.info(
            "Forecast reasoning for %s: %s",
            question.page_url,
            reasoning[:1000],
        )

        parsing_instructions = clean_indents(
            f"""
            Make sure that all option names are one of the following:

            {question.options}

            The text you are parsing may prepend these options with some
            variation of "Option". Remove that prefix if it is not actually
            part of the option name.

            A 0% probability is valid. Do not omit an option simply because
            its probability is zero.
            """
        )

        predicted_option_list: PredictedOptionList = (
            await structure_output(
                text_to_structure=reasoning,
                output_type=PredictedOptionList,
                model=self._parser_llm(),
                num_validation_samples=(
                    self._structure_output_validation_samples
                ),
                additional_instructions=(
                    parsing_instructions
                ),
            )
        )

        return ReasonedPrediction(
            prediction_value=predicted_option_list,
            reasoning=reasoning,
        )

    # -----------------------------------------------------------------------
    # NUMERIC
    # -----------------------------------------------------------------------

    async def _run_forecast_on_numeric(
        self,
        question: NumericQuestion,
        research: str,
    ) -> ReasonedPrediction[NumericDistribution]:

        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(
                question
            )
        )

        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:

            {question.question_text}

            Background:

            {question.background_info}

            Resolution criteria:

            {question.resolution_criteria}

            Fine print:

            {question.fine_print}

            Units for answer:
            {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}

            Your research assistant says:

            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            QUESTION BOUNDS:
            {lower_bound_message}
            {upper_bound_message}

            IMPORTANT INTERPRETATION OF QUESTION BOUNDS:

            If a bound is an open/question-creator bound, treat it only as
            question metadata or a soft constraint. Do NOT describe it as an
            expert forecast, market expectation, polling result, or independent
            evidence.

            Formatting Instructions:

            - Give your answer in the requested units.
            - Never use scientific notation.
            - Percentile 10 must be lower than percentile 20, and so on.
            - Use sufficiently wide tails to account for unknown unknowns.

            Before answering you write:

            (a) The time left until the outcome to the question is known.

            (b) The outcome if nothing changed.

            (c) The outcome if the current trend continued.

            (d) The expectations of experts and markets, using ONLY actual
                expert forecasts, market prices, polling/forecasting data,
                surveys, or other direct evidence found in the research.

                Do NOT treat the question creator's numeric/date bounds as
                expert or market expectations.

                If no direct expert or market expectation exists in the
                research, explicitly say that no direct evidence was found
                rather than inferring an expectation from the question bounds.

            (e) A brief description of an unexpected scenario that results in
                a low outcome.

            (f) A brief description of an unexpected scenario that results in
                a high outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}

            The last thing you write is your final answer as:

            Percentile 10: XX
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX
            """
        )

        reasoning = await generate_forecast_reasoning(
            prompt
        )

        logger.info(
            "Forecast reasoning for %s: %s",
            question.page_url,
            reasoning[:1000],
        )

        parsing_instructions = clean_indents(
            f"""
            The text given to you is trying to give a forecast distribution
            for a numeric question.

            Numeric question:

            {question.question_text}

            Units:

            {question.unit_of_measure}

            When parsing:

            - Values must use the correct units.
            - Convert scientific notation to ordinary numbers.
            - Preserve the requested percentile labels.
            - Only use percentile values explicitly supported by the model's
              final answer.
            - Do not invent missing percentile values.
            - The question's numeric bounds are metadata, not expert or market
              forecasts.
            - Do not add a bound to the parsed output merely because it appears
              in the question metadata.
            """
        )

        percentile_list: list[Percentile] = (
            await structure_output(
                reasoning,
                list[Percentile],
                model=self._parser_llm(),
                additional_instructions=(
                    parsing_instructions
                ),
                num_validation_samples=(
                    self._structure_output_validation_samples
                ),
            )
        )

        prediction = NumericDistribution.from_question(
            percentile_list,
            question,
        )

        return ReasonedPrediction(
            prediction_value=prediction,
            reasoning=reasoning,
        )

    # -----------------------------------------------------------------------
    # DATE
    # -----------------------------------------------------------------------

    async def _run_forecast_on_date(
        self,
        question: DateQuestion,
        research: str,
    ) -> ReasonedPrediction[NumericDistribution]:

        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(
                question
            )
        )

        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:

            {question.question_text}

            Background:

            {question.background_info}

            Resolution criteria:

            {question.resolution_criteria}

            Fine print:

            {question.fine_print}

            Your research assistant says:

            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            QUESTION DATE BOUNDS:
            {lower_bound_message}
            {upper_bound_message}

            IMPORTANT INTERPRETATION OF QUESTION BOUNDS:

            If a bound is an open/question-creator bound, treat it only as
            question metadata or a soft constraint. Do NOT describe it as an
            expert forecast, market expectation, polling result, or independent
            evidence.

            Formatting Instructions:

            - This is a date question.
            - Dates must be YYYY-MM-DD.
            - If hours matter, use UTC:
              YYYY-MM-DDTHH:MM:SSZ
            - Percentile dates must be monotonically increasing.
            - P10 must be the earliest and P90 the latest.

            Before answering you write:

            (a) The time left until the outcome to the question is known.

            (b) The outcome if nothing changed.

            (c) The outcome if the current trend continued.

            (d) The expectations of experts and markets, using ONLY actual
                expert forecasts, market prices, polling/forecasting data,
                surveys, or other direct evidence found in the research.

                Do NOT treat the question creator's date bounds as expert or
                market expectations.

                If no direct expert or market expectation exists, explicitly
                say that no direct evidence was found rather than inferring an
                expectation from the question bounds.

            (e) A brief description of an unexpected scenario that results in
                an early outcome.

            (f) A brief description of an unexpected scenario that results in
                a late outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}

            The last thing you write is your final answer as:

            Percentile 10: YYYY-MM-DD
            Percentile 20: YYYY-MM-DD
            Percentile 40: YYYY-MM-DD
            Percentile 60: YYYY-MM-DD
            Percentile 80: YYYY-MM-DD
            Percentile 90: YYYY-MM-DD
            """
        )

        reasoning = await generate_forecast_reasoning(
            prompt
        )

        logger.info(
            "Forecast reasoning for %s: %s",
            question.page_url,
            reasoning[:1000],
        )

        parsing_instructions = clean_indents(
            f"""
            The text given to you is trying to give a forecast distribution
            for a date question.

            Date question:

            {question.question_text}

            When parsing:

            - Parse percentile values as dates.
            - Use ISO format YYYY-MM-DD when possible.
            - If the target schema requires a numeric value, convert the date
              to a Unix timestamp in UTC.
            - Preserve the requested percentile labels.
            - Only use dates explicitly supported by the model's final answer.
            - Do not invent missing percentile values.
            - The question's date bounds are metadata, not expert or market
              forecasts.
            """
        )

        date_percentile_list: list[DatePercentile] = (
            await structure_output(
                reasoning,
                list[DatePercentile],
                model=self._parser_llm(),
                additional_instructions=(
                    parsing_instructions
                ),
                num_validation_samples=(
                    self._structure_output_validation_samples
                ),
            )
        )

        percentile_list = [
            Percentile(
                percentile=percentile.percentile,
                value=(
                    percentile.value.replace(
                        tzinfo=timezone.utc
                    )
                    if percentile.value.tzinfo is None
                    else percentile.value
                ).timestamp(),
            )
            for percentile in date_percentile_list
        ]

        prediction = NumericDistribution.from_question(
            percentile_list,
            question,
        )

        return ReasonedPrediction(
            prediction_value=prediction,
            reasoning=reasoning,
        )

    # -----------------------------------------------------------------------
    # CONDITIONAL
    # -----------------------------------------------------------------------

    async def _run_forecast_on_conditional(
        self,
        question: ConditionalQuestion,
        research: str,
    ) -> ReasonedPrediction[ConditionalPrediction]:
        """
        Forecast conditional-question components using the existing
        forecasting_tools semantics.
        """

        parent_info, full_research = (
            await self._get_question_prediction_info(
                question.parent,
                research,
                "parent",
            )
        )

        child_info, full_research = (
            await self._get_question_prediction_info(
                question.child,
                full_research,
                "child",
            )
        )

        yes_info, full_research = (
            await self._get_question_prediction_info(
                question.question_yes,
                full_research,
                "yes",
            )
        )

        no_info, full_research = (
            await self._get_question_prediction_info(
                question.question_no,
                full_research,
                "no",
            )
        )

        full_reasoning = clean_indents(
            f"""
            ## Parent Question Reasoning
            {parent_info.reasoning}

            ## Child Question Reasoning
            {child_info.reasoning}

            ## Child Conditional on Parent = YES
            {yes_info.reasoning}

            ## Child Conditional on Parent = NO
            {no_info.reasoning}
            """
        )

        full_prediction = ConditionalPrediction(
            parent=parent_info.prediction_value,  # type: ignore
            child=child_info.prediction_value,  # type: ignore
            prediction_yes=yes_info.prediction_value,  # type: ignore
            prediction_no=no_info.prediction_value,  # type: ignore
        )

        return ReasonedPrediction(
            reasoning=full_reasoning,
            prediction_value=full_prediction,
        )

    async def _get_question_prediction_info(
        self,
        question: MetaculusQuestion,
        research: str,
        question_type: str,
    ) -> tuple[ReasonedPrediction[Any], str]:
        """
        Reuse open parent/child forecasts where appropriate.
        """

        from forecasting_tools.data_models.data_organizer import (
            DataOrganizer,
        )

        previous_forecasts = (
            question.previous_forecasts
        )

        if (
            question_type in {"parent", "child"}
            and previous_forecasts
            and question_type
            not in getattr(
                self,
                "force_reforecast_in_conditional",
                set(),
            )
        ):

            previous_forecast = (
                previous_forecasts[-1]
            )

            current_utc_time = (
                datetime.now(timezone.utc)
            )

            if (
                previous_forecast.timestamp_end is None
                or previous_forecast.timestamp_end
                > current_utc_time
            ):

                readable = (
                    DataOrganizer.get_readable_prediction(
                        previous_forecast
                    )
                )

                return (
                    ReasonedPrediction(
                        prediction_value=PredictionAffirmed(),
                        reasoning=(
                            f"Already existing "
                            f"{question_type} forecast "
                            f"reaffirmed at {readable}."
                        ),
                    ),
                    research,
                )

        info = await self._make_prediction(
            question,
            research,
        )

        full_research = (
            self._add_reasoning_to_research(
                research,
                info,
                question_type,
            )
        )

        return info, full_research

    def _add_reasoning_to_research(
        self,
        research: str,
        reasoning: ReasonedPrediction[Any],
        question_type: str,
    ) -> str:
        from forecasting_tools.data_models.data_organizer import (
            DataOrganizer,
        )

        label = question_type.title()

        return clean_indents(
            f"""
            {research}

            ---

            ## {label} Question Information

            You have previously forecasted the {label} Question to the value:

            {DataOrganizer.get_readable_prediction(
                reasoning.prediction_value
            )}

            This is relevant information for your current forecast, but it is
            NOT your current forecast.

            Do not use this reasoning to re-forecast that component directly.

            Reasoning:

            {reasoning.reasoning}
            """
        )

    def _get_conditional_disclaimer_if_necessary(
        self,
        question: MetaculusQuestion,
    ) -> str:
        if question.conditional_type not in {
            "yes",
            "no",
        }:
            return ""

        return clean_indents(
            """
            As you are given a conditional question with a parent and child,
            you are to only forecast the CHILD question, given the parent
            question's resolution.

            You never re-forecast the parent question under any circumstances.

            You use probabilistic reasoning, strongly considering the parent
            question's resolution, to forecast the child question.
            """
        )

    # -----------------------------------------------------------------------
    # HELPERS
    # -----------------------------------------------------------------------

    def _create_upper_and_lower_bound_messages(
        self,
        question: NumericQuestion | DateQuestion,
    ) -> tuple[str, str]:
        """
        Create human-readable bound messages for numeric and date questions.

        NumericQuestion has nominal_upper_bound / nominal_lower_bound.

        DateQuestion does not have those attributes, so its upper_bound /
        lower_bound fields are used directly.
        """

        if isinstance(
            question,
            NumericQuestion,
        ):

            upper_bound_number = (
                question.nominal_upper_bound
                if question.nominal_upper_bound
                is not None
                else question.upper_bound
            )

            lower_bound_number = (
                question.nominal_lower_bound
                if question.nominal_lower_bound
                is not None
                else question.lower_bound
            )

            unit_of_measure = (
                question.unit_of_measure
                if question.unit_of_measure
                else "units"
            )

        elif isinstance(
            question,
            DateQuestion,
        ):

            upper_bound_number = (
                question.upper_bound
                .date()
                .isoformat()
            )

            lower_bound_number = (
                question.lower_bound
                .date()
                .isoformat()
            )

            unit_of_measure = "date"

        else:
            raise TypeError(
                "Unsupported question type for bound messages: "
                f"{type(question).__name__}"
            )

        if question.open_upper_bound:

            upper_bound_message = (
                f"The question creator set a soft upper bound at "
                f"{upper_bound_number} {unit_of_measure}. "
                f"This is question metadata, not an expert "
                f"or market forecast."
            )

        else:

            upper_bound_message = (
                f"The outcome cannot be higher than "
                f"{upper_bound_number} "
                f"{unit_of_measure}."
            )

        if question.open_lower_bound:

            lower_bound_message = (
                f"The question creator set a soft lower bound at "
                f"{lower_bound_number} {unit_of_measure}. "
                f"This is question metadata, not an expert "
                f"or market forecast."
            )

        else:

            lower_bound_message = (
                f"The outcome cannot be lower than "
                f"{lower_bound_number} "
                f"{unit_of_measure}."
            )

        return (
            upper_bound_message,
            lower_bound_message,
        )

    def _parser_llm(self) -> GeneralLlm:
        """
        Return the explicitly configured structured-output parser.

        Parser chain:

            Qwen3.8 27B
                ->
            Gemma 4 31B
                ->
            Nemotron 3 Ultra

        All three model identifiers are explicitly provider-qualified.
        """

        return self.get_llm(
            "parser",
            "llm",
        )
