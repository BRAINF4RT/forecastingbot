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
    ForecastBot,
    GeneralLlm,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    DatePercentile,
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
from research.pipeline import run_research_pipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

PRIMARY_LLM = f"openrouter/{PRIMARY_MODEL}"
FALLBACK_LLM = f"openrouter/{FALLBACK_MODEL}"

# Preferred parser models. Qwen and Gemma are tried first because they are
# intended for structured extraction. Nemotron is a final cross-provider
# fallback so a temporary shared free-tier 429 does not kill a forecast.
PARSER_PRIMARY_LLM = "openrouter/qwen/qwen3.8-27b:free"
PARSER_FALLBACK_LLM = "openrouter/google/gemma-4-31b-it:free"
PARSER_THIRD_LLM = PRIMARY_LLM

# Maximum number of forecasting-tools / LiteLLM calls allowed at once.
# Keeping this at one avoids simultaneous requests consuming the same free
# provider pool and makes parser fallback deterministic.
_GENERAL_LLM_CONCURRENCY = 1

# Backoff used when every parser model in one chain attempt is rate-limited.
_RATE_LIMIT_MAX_RETRIES = 3
_RATE_LIMIT_BACKOFF_SECONDS = 8


class _LoopBoundSemaphore:
    """Keep a separate semaphore for each asyncio event loop."""

    def __init__(self, value: int) -> None:
        self._value = value
        self._semaphores: weakref.WeakKeyDictionary[
            asyncio.AbstractEventLoop, asyncio.Semaphore
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


_GENERAL_LLM_SEMAPHORE = _LoopBoundSemaphore(_GENERAL_LLM_CONCURRENCY)


def _is_rate_limit_error(exc: Exception) -> bool:
    """Return True for LiteLLM/OpenRouter rate-limit failures."""
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
    """GeneralLlm wrapper with primary -> fallback -> final fallback."""

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
        self._models = [primary_model, fallback_model] + (
            [third_model] if third_model else []
        )
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

    async def invoke(self, prompt: str, *args: Any, **kwargs: Any) -> Any:
        """Try the configured chain, retrying the whole chain after all-model 429s."""
        all_errors: list[tuple[str, Exception]] = []

        async with _GENERAL_LLM_SEMAPHORE:
            for attempt in range(1, _RATE_LIMIT_MAX_RETRIES + 1):
                errors: list[tuple[str, Exception]] = []

                try:
                    return await super().invoke(prompt, *args, **kwargs)
                except Exception as exc:
                    errors.append((self._models[0], exc))
                    logger.warning(
                        "Primary model failed: %s. Falling back to %s.",
                        self._models[0],
                        self._models[1],
                    )

                for index, (model, llm) in enumerate(
                    zip(self._models[1:], self._fallback_llms),
                    start=1,
                ):
                    try:
                        result = await llm.invoke(prompt, *args, **kwargs)
                        logger.info("Fallback model succeeded: %s", model)
                        return result
                    except Exception as exc:
                        errors.append((model, exc))

                        if index < len(self._fallback_llms):
                            logger.warning(
                                "Fallback model failed: %s. Falling back to %s.",
                                model,
                                self._models[index + 1],
                            )

                all_errors.extend(errors)

                if (
                    errors
                    and all(_is_rate_limit_error(exc) for _, exc in errors)
                    and attempt < _RATE_LIMIT_MAX_RETRIES
                ):
                    wait = _RATE_LIMIT_BACKOFF_SECONDS * attempt

                    logger.warning(
                        "All %d configured models were rate-limited "
                        "(attempt %d/%d). Retrying the chain after %ds.",
                        len(errors),
                        attempt,
                        _RATE_LIMIT_MAX_RETRIES,
                        wait,
                    )

                    await asyncio.sleep(wait)
                    continue

                break

        details = "\n".join(
            f"{model}: {error!r}" for model, error in all_errors
        )

        raise RuntimeError(
            "All configured LLMs failed.\n" + details
        ) from (
            all_errors[-1][1]
            if all_errors
            else RuntimeError("no models configured")
        )


class OpenRouterForecastBot(ForecastBot):
    """
    Metaculus forecasting bot.

    Every forecasting_tools LLM purpose is explicitly configured so that
    ForecastBot never silently falls back to its own defaults.
    """

    # structure_output performs additional validation calls. Keeping this at
    # one is much friendlier to OpenRouter's free-tier provider limits.
    _structure_output_validation_samples = 1

    def _llm_config_defaults(self) -> dict[str, GeneralLlm]:
        """Configure only the forecasting_tools purpose this bot actually uses."""
        return {
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
        """Run web research with complete Metaculus question context visible to the researcher LLMs."""
        logger.info("Starting research for %s", question.page_url)

        research = await run_research_pipeline(
            question_text=question.question_text,
            resolution_criteria=question.resolution_criteria or "",
            background=question.background_info or "",
            fine_print=question.fine_print or "",
            question_context=self._format_research_question_context(question),
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
        sections = [
            f"Question type: {type(question).__name__}",
            f"Metaculus question URL: {question.page_url}",
            f"Question: {question.question_text}",
            f"Background: {question.background_info or ''}",
            f"Resolution criteria: {question.resolution_criteria or ''}",
            f"Fine print: {question.fine_print or ''}",
        ]

        if isinstance(question, MultipleChoiceQuestion):
            sections.append(f"Options: {question.options}")

        if isinstance(question, NumericQuestion):
            sections.extend(
                [
                    f"Units for answer: {question.unit_of_measure or 'Not stated'}",
                    f"Lower bound: {question.lower_bound}",
                    f"Upper bound: {question.upper_bound}",
                    f"Open lower bound: {question.open_lower_bound}",
                    f"Open upper bound: {question.open_upper_bound}",
                ]
            )

        if isinstance(question, DateQuestion):
            sections.extend(
                [
                    f"Lower date bound: {question.lower_bound}",
                    f"Upper date bound: {question.upper_bound}",
                    f"Open lower bound: {question.open_lower_bound}",
                    f"Open upper bound: {question.open_upper_bound}",
                ]
            )

        if isinstance(question, ConditionalQuestion):
            sections.extend(
                [
                    "Conditional structure:",
                    "PARENT:\n"
                    + self._format_research_question_context(question.parent),
                    "CHILD:\n"
                    + self._format_research_question_context(question.child),
                    "CHILD CONDITIONAL ON PARENT = YES:\n"
                    + self._format_research_question_context(
                        question.question_yes
                    ),
                    "CHILD CONDITIONAL ON PARENT = NO:\n"
                    + self._format_research_question_context(
                        question.question_no
                    ),
                ]
            )

        return clean_indents("\n".join(sections))

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

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )

        reasoning = await generate_forecast_reasoning(prompt)

        logger.info(
            "Forecast reasoning for %s: %s",
            question.page_url,
            reasoning[:1000],
        )

        binary_prediction: BinaryPrediction = await structure_output(
            reasoning,
            BinaryPrediction,
            model=self._parser_llm(),
            num_validation_samples=self._structure_output_validation_samples,
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

            The options are: {question.options}


            Background:

            {question.background_info}
            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:

            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:

            (a) The time left until the outcome to the question is known.

            (b) The status quo outcome if nothing changed.

            (c) A description of an scenario that results in an unexpected outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}

            You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

            The last thing you write is your final probabilities for the N options in this order {question.options} as:

            Option_A: Probability_A

            Option_B: Probability_B

            ...

            Option_N: Probability_N
            """
        )

        reasoning = await generate_forecast_reasoning(prompt)

        logger.info(
            "Forecast reasoning for %s: %s",
            question.page_url,
            reasoning[:1000],
        )

        parsing_instructions = clean_indents(
            f"""
            Make sure that all option names are one of the following:
            {question.options}
            The text you are parsing may prepend these options with some variation of "Option" which you should remove if not part of the option names I just gave you.
            Additionally, you may sometimes need to parse a 0% probability. Please do not skip options with 0% but rather make it an entry in your final list with 0% probability.
            """
        )

        predicted_option_list: PredictedOptionList = await structure_output(
            text_to_structure=reasoning,
            output_type=PredictedOptionList,
            model=self._parser_llm(),
            num_validation_samples=self._structure_output_validation_samples,
            additional_instructions=parsing_instructions,
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
            self._create_upper_and_lower_bound_messages(question)
        )

        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}
            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}

            Your research assistant says:

            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - Please notice the units requested and give your answer in these units (e.g. whether you represent a number as 1,000,000 or 1 million).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there. The value for percentile 10 should always be less than the value for percentile 20, and so on.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets, using only actual expert forecasts, market prices, polling/forecasting data, or other direct evidence found in the research.
                Do NOT treat the question creator's numeric/date bounds as expert or market expectations. They are question metadata / soft constraints, not independent forecasts. If no direct expert or market expectation is available, explicitly say so instead of inferring one from the bounds.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.
            {self._get_conditional_disclaimer_if_necessary(question)}

            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 10: XX (lowest number value)
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX (highest number value)
            "
            """
        )

        reasoning = await generate_forecast_reasoning(prompt)

        logger.info(
            "Forecast reasoning for %s: %s",
            question.page_url,
            reasoning[:1000],
        )

        parsing_instructions = clean_indents(
            f"""
            The text given to you is trying to give a forecast distribution for a numeric question.
            - This text is trying to answer the numeric question: "{question.question_text}".
            - When parsing the text, please make sure to give the values (the ones assigned to percentiles) in terms of the correct units.
            - The units for the forecast are: {question.unit_of_measure}
            - Your work will be shown publicly with these units stated verbatim after the numbers your parse.
            - The question's numeric bounds are metadata for interpreting the forecast. They are not expert or market forecasts. Do not add, invent, or reclassify them as evidence when parsing.
            - Preserve a creator-supplied bound only when the final answer explicitly assigns that value to a percentile.
            - If the answer doesn't give the answer in the correct units, you should parse it in the right units. For instance if the answer gives numbers as $500,000,000 and units are "B $" then you should parse the answer as 0.5 (since $500,000,000 is $0.5 billion).
            - If percentiles are not explicitly given (e.g. only a single value is given) please don't return a parsed output, but rather indicate that the answer is not explicitly given in the text.
            - Turn any values that are in scientific notation into regular numbers.
            """
        )

        percentile_list: list[Percentile] = await structure_output(
            reasoning,
            list[Percentile],
            model=self._parser_llm(),
            additional_instructions=parsing_instructions,
            num_validation_samples=self._structure_output_validation_samples,
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
            self._create_upper_and_lower_bound_messages(question)
        )

        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - This is a date question, and as such, the answer must be expressed in terms of dates.
            - The dates must be written in the format of YYYY-MM-DD. If hours matter, please append the date with the hour in UTC and military time: YYYY-MM-DDTHH:MM:SSZ.No other formatting is allowed.
            - Always start with a lower date chronologically and then increase from there.
            - Do NOT forget this. The dates must be written in chronological order starting at the earliest time at percentile 10 and increasing from there.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets, using only actual expert forecasts, market prices, polling/forecasting data, or other direct evidence found in the research.
                Do NOT treat the question creator's numeric/date bounds as expert or market expectations. They are question metadata / soft constraints, not independent forecasts. If no direct expert or market expectation is available, explicitly say so instead of inferring one from the bounds.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.
            {self._get_conditional_disclaimer_if_necessary(question)}

            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 10: YYYY-MM-DD (oldest date)
            Percentile 20: YYYY-MM-DD
            Percentile 40: YYYY-MM-DD
            Percentile 60: YYYY-MM-DD
            Percentile 80: YYYY-MM-DD
            Percentile 90: YYYY-MM-DD (newest date)
            "
            """
        )

        reasoning = await generate_forecast_reasoning(prompt)

        logger.info(
            "Forecast reasoning for %s: %s",
            question.page_url,
            reasoning[:1000],
        )

        parsing_instructions = clean_indents(
            f"""
            The text given to you is trying to give a forecast distribution for a date question.
            - This text is trying to answer the question: "{question.question_text}".
            - The question's date bounds are metadata for interpreting the forecast. They are not expert or market forecasts. Do not add, invent, or reclassify them as evidence when parsing.
            - Preserve a creator-supplied bound only when the final answer explicitly assigns that date to a percentile.
            - The output is given as dates/times please format it into a valid datetime parsable string. Assume midnight UTC if no hour is given.
            - If percentiles are not explicitly given (e.g. only a single value is given) please don't return a parsed output, but rather indicate that the answer is not explicitly given in the text.
            """
        )

        date_percentile_list: list[DatePercentile] = await structure_output(
            reasoning,
            list[DatePercentile],
            model=self._parser_llm(),
            additional_instructions=parsing_instructions,
            num_validation_samples=self._structure_output_validation_samples,
        )

        percentile_list = [
            Percentile(
                percentile=percentile.percentile,
                value=(
                    percentile.value.replace(tzinfo=timezone.utc)
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
        """Forecast the parent/child components using forecasting-tools semantics."""
        parent_info, full_research = await self._get_question_prediction_info(
            question.parent,
            research,
            "parent",
        )
        child_info, full_research = await self._get_question_prediction_info(
            question.child,
            full_research,
            "child",
        )
        yes_info, full_research = await self._get_question_prediction_info(
            question.question_yes,
            full_research,
            "yes",
        )
        no_info, full_research = await self._get_question_prediction_info(
            question.question_no,
            full_research,
            "no",
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
        """Reuse open parent/child forecasts where appropriate."""
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        previous_forecasts = question.previous_forecasts

        if (
            question_type in {"parent", "child"}
            and previous_forecasts
            and question_type not in getattr(
                self,
                "force_reforecast_in_conditional",
                set(),
            )
        ):
            previous_forecast = previous_forecasts[-1]
            current_utc_time = datetime.now(timezone.utc)

            if (
                previous_forecast.timestamp_end is None
                or previous_forecast.timestamp_end > current_utc_time
            ):
                readable = DataOrganizer.get_readable_prediction(
                    previous_forecast
                )

                return (
                    ReasonedPrediction(
                        prediction_value=PredictionAffirmed(),
                        reasoning=(
                            f"Already existing {question_type} forecast "
                            f"reaffirmed at {readable}."
                        ),
                    ),
                    research,
                )

        info = await self._make_prediction(question, research)

        full_research = self._add_reasoning_to_research(
            research,
            info,
            question_type,
        )

        return info, full_research

    def _add_reasoning_to_research(
        self,
        research: str,
        reasoning: ReasonedPrediction[Any],
        question_type: str,
    ) -> str:
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        label = question_type.title()

        return clean_indents(
            f"""
            {research}

            ---
            ## {label} Question Information

            You have previously forecasted the {label} Question to the value: 
            {DataOrganizer.get_readable_prediction(reasoning.prediction_value)}

            This is relevant information for your current forecast, but it is NOT
            your current forecast. Do not use this reasoning to re-forecast that
            component directly.

            Reasoning:
            {reasoning.reasoning}
            """
        )

    def _get_conditional_disclaimer_if_necessary(
        self,
        question: MetaculusQuestion,
    ) -> str:
        if question.conditional_type not in ["yes", "no"]:
            return ""

        return clean_indents(
            """
            As you are given a conditional question with a parent and child, you are to only forecast the **CHILD** question, given the parent question's resolution.
            You never re-forecast the parent question under any circumstances, but you use probabilistic reasoning, strongly considering the parent question's resolution, to forecast the child question.
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

        DateQuestion does NOT have those attributes, so its actual
        upper_bound / lower_bound fields are used directly.
        """

        if isinstance(question, NumericQuestion):
            upper_bound_number = (
                question.nominal_upper_bound
                if question.nominal_upper_bound is not None
                else question.upper_bound
            )

            lower_bound_number = (
                question.nominal_lower_bound
                if question.nominal_lower_bound is not None
                else question.lower_bound
            )

            unit_of_measure = (
                question.unit_of_measure
                if question.unit_of_measure
                else "units"
            )

        elif isinstance(question, DateQuestion):
            upper_bound_number = question.upper_bound.date().isoformat()
            lower_bound_number = question.lower_bound.date().isoformat()
            unit_of_measure = "date"

        else:
            raise TypeError(
                "Unsupported question type for bound messages: "
                f"{type(question).__name__}"
            )

        if question.open_upper_bound:
            upper_bound_message = (
                f"The question creator set a soft upper bound at "
                f"{upper_bound_number} {unit_of_measure}. This is question "
                f"metadata, not an expert or market forecast."
            )
        else:
            upper_bound_message = (
                f"The outcome cannot be higher than "
                f"{upper_bound_number} {unit_of_measure}."
            )

        if question.open_lower_bound:
            lower_bound_message = (
                f"The question creator set a soft lower bound at "
                f"{lower_bound_number} {unit_of_measure}. This is question "
                f"metadata, not an expert or market forecast."
            )
        else:
            lower_bound_message = (
                f"The outcome cannot be lower than "
                f"{lower_bound_number} {unit_of_measure}."
            )

        return upper_bound_message, lower_bound_message

    def _parser_llm(self) -> GeneralLlm:
        """
        Return the explicitly configured parser.

        The parser itself uses:
            Qwen3.8 27B -> Gemma 4 31B -> Nemotron 3 Ultra

        Qwen and Gemma are preferred; Nemotron is the final fallback for
        transient free-tier rate limits.
        """

        return self.get_llm("parser", "llm")
