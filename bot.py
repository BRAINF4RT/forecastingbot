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

# Dedicated parser models: these need reliable native structured-output
# support, which Nemotron/Laguna do not advertise. Verified live on
# OpenRouter's free tier -- re-check https://openrouter.ai/models?max_price=0
# periodically since the free roster rotates.
PARSER_PRIMARY_LLM = "openrouter/qwen/qwen3.8-27b:free"
PARSER_FALLBACK_LLM = "openrouter/google/gemma-4-31b-it:free"

# Maximum number of forecasting-tools / LiteLLM calls allowed at once.
#
# This is separate from the direct OpenRouter semaphore in
# clients/openrouter_helper.py.
_GENERAL_LLM_CONCURRENCY = 2


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
        self._models = [primary_model, fallback_model] + ([third_model] if third_model else [])
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
        errors: list[tuple[str, Exception]] = []

        async with _GENERAL_LLM_SEMAPHORE:
            try:
                return await super().invoke(prompt, *args, **kwargs)
            except Exception as exc:
                errors.append((self._models[0], exc))
                logger.warning(
                    "Primary model failed: %s. Falling back to %s.",
                    self._models[0], self._models[1],
                )

            for model, llm in zip(self._models[1:], self._fallback_llms):
                try:
                    result = await llm.invoke(prompt, *args, **kwargs)
                    logger.info("Fallback model succeeded: %s", model)
                    return result
                except Exception as exc:
                    errors.append((model, exc))
                    if model != self._models[-1]:
                        logger.warning(
                            "Fallback model failed: %s. Falling back to %s.",
                            model, self._models[self._models.index(model) + 1],
                        )

        details = "\n".join(f"{model}: {error!r}" for model, error in errors)
        raise RuntimeError("All configured LLMs failed.\n" + details) from errors[-1][1]


class OpenRouterForecastBot(ForecastBot):
    """
    Metaculus forecasting bot.

    Every forecasting_tools LLM purpose is explicitly configured so that
    ForecastBot never silently falls back to its own defaults.
    """

    _structure_output_validation_samples = 2

    def _llm_config_defaults(self) -> dict[str, GeneralLlm]:
        """Configure only the forecasting_tools purpose this bot actually uses."""
        return {
            "parser": FallbackGeneralLlm(
                primary_model=PARSER_PRIMARY_LLM,
                fallback_model=PARSER_FALLBACK_LLM,
                third_model=None,
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
        logger.info(
            "Starting research for %s",
            question.page_url,
        )

        research = await run_research_pipeline(
            question_text=question.question_text,
            resolution_criteria=question.resolution_criteria or "",
            background=question.background_info or "",
        )

        logger.info(
            "Research for %s:\n%s",
            question.page_url,
            research[:1000],
        )

        return research

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
            You are a professional probabilistic forecaster.

            Your task is to forecast the probability of the following
            binary event.

            QUESTION:
            {question.question_text}

            QUESTION BACKGROUND:
            {question.background_info}

            RESOLUTION CRITERIA:
            {question.resolution_criteria}

            FINE PRINT:
            {question.fine_print}

            RESEARCH:
            {research}

            TODAY:
            {datetime.now().strftime("%Y-%m-%d")}

            Before giving your final probability, carefully consider:
            (a) How much time remains until the outcome is known.
            (b) The status quo if nothing significant changes.
            (c) A plausible scenario producing a NO outcome.
            (d) A plausible scenario producing a YES outcome.
            (e) Base rates and historical precedent.
            (f) Important evidence supporting each side.
            (g) Important uncertainties and unknowns.

            Good forecasters generally put substantial weight on the
            status quo because the world often changes more slowly than
            people expect.

            Do not blindly follow the research. Evaluate its quality.

            {self._get_conditional_disclaimer_if_necessary(question)}

            The final line MUST be exactly:

            Probability: ZZ%

            where ZZ is your probability from 0 to 100.
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
            You are a professional probabilistic forecaster.

            QUESTION:
            {question.question_text}

            OPTIONS:
            {question.options}

            BACKGROUND:
            {question.background_info}

            RESOLUTION CRITERIA:
            {question.resolution_criteria}

            FINE PRINT:
            {question.fine_print}

            RESEARCH:
            {research}

            TODAY:
            {datetime.now().strftime("%Y-%m-%d")}

            Before producing probabilities, consider:
            (a) The time remaining.
            (b) The status quo outcome.
            (c) The most likely option.
            (d) Why each alternative could occur.
            (e) Base rates and historical precedent.
            (f) Unexpected scenarios.
            (g) Whether the research contains conflicting evidence.

            Do not assign probability merely because an option sounds
            plausible.

            Probabilities should reflect your actual assessment.

            Give a probability to EVERY option.

            The final answer must contain the options in exactly this order:

            {question.options}

            Use:

            Option_A: Probability_A
            Option_B: Probability_B
            ...

            {self._get_conditional_disclaimer_if_necessary(question)}

            Probabilities should sum to approximately 100%.
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
            The valid option names are:

            {question.options}

            When parsing the answer:
            - Every valid option must appear.
            - Use exactly the supplied option names.
            - Remove prefixes such as "Option" if they are not part of
              the actual option name.
            - Preserve 0% probabilities.
            - Do not invent options.
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
            You are a professional probabilistic forecaster.

            QUESTION:
            {question.question_text}

            BACKGROUND:
            {question.background_info}

            RESOLUTION CRITERIA:
            {question.resolution_criteria}

            FINE PRINT:
            {question.fine_print}

            UNITS:
            {question.unit_of_measure if question.unit_of_measure else "Not stated; infer carefully."}

            RESEARCH:
            {research}

            TODAY:
            {datetime.now().strftime("%Y-%m-%d")}

            QUESTION LOWER BOUND:
            {lower_bound_message}

            QUESTION UPPER BOUND:
            {upper_bound_message}

            Consider:
            (a) The time remaining.
            (b) The current value or status quo.
            (c) Historical base rates.
            (d) Current trends.
            (e) Expert and market expectations where available.
            (f) A plausible low-outcome scenario.
            (g) A plausible high-outcome scenario.
            (h) Unknown unknowns.

            Be appropriately uncertain.

            Formatting requirements:
            - Use the requested units.
            - Never use scientific notation.
            - Percentile values must increase monotonically.
            - Do not invent unsupported precision.

            {self._get_conditional_disclaimer_if_necessary(question)}

            Your final answer MUST contain exactly:

            Percentile 10: XX
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX

            where XX is a numerical value in the requested units.
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
            The numeric question is:

            {question.question_text}

            Units:
            {question.unit_of_measure}

            When parsing:
            - Values must be expressed in the correct units.
            - Convert scientific notation to ordinary numbers.
            - Only use percentile values explicitly supported by the
              model's final answer.
            - Preserve the requested percentile labels.
            - Do not invent missing percentile values.
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
            You are a professional probabilistic forecaster.

            QUESTION:
            {question.question_text}

            BACKGROUND:
            {question.background_info}

            RESOLUTION CRITERIA:
            {question.resolution_criteria}

            FINE PRINT:
            {question.fine_print}

            RESEARCH:
            {research}

            TODAY:
            {datetime.now().strftime("%Y-%m-%d")}

            QUESTION LOWER BOUND:
            {lower_bound_message}

            QUESTION UPPER BOUND:
            {upper_bound_message}

            Consider:
            (a) The time remaining until the outcome is known.
            (b) The status quo / current trajectory.
            (c) Historical base rates for similar events.
            (d) Expert and market expectations where available.
            (e) A plausible early-outcome scenario.
            (f) A plausible late-outcome scenario.
            (g) Unknown unknowns.

            Be appropriately uncertain. Good forecasters usually need wider
            date ranges than their first instinct suggests.

            Formatting requirements:
            - Dates must be in ISO format: YYYY-MM-DD.
            - Percentile dates must increase monotonically
              (P10 earliest, P90 latest).
            - Do not invent unsupported precision.

            {self._get_conditional_disclaimer_if_necessary(question)}

            Your final answer MUST contain exactly:

            Percentile 10: YYYY-MM-DD
            Percentile 20: YYYY-MM-DD
            Percentile 40: YYYY-MM-DD
            Percentile 60: YYYY-MM-DD
            Percentile 80: YYYY-MM-DD
            Percentile 90: YYYY-MM-DD
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
            The question is a DATE question:

            {question.question_text}

            When parsing:
            - Parse each percentile value as an ISO date (YYYY-MM-DD).
            - If the target schema requires a numeric value, convert the
              parsed date to a Unix timestamp (seconds since epoch, UTC).
            - Only use percentile values explicitly supported by the
              model's final answer.
            - Preserve the requested percentile labels.
            - Do not invent missing percentile values.
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
            question.parent, research, "parent"
        )
        child_info, full_research = await self._get_question_prediction_info(
            question.child, full_research, "child"
        )
        yes_info, full_research = await self._get_question_prediction_info(
            question.question_yes, full_research, "yes"
        )
        no_info, full_research = await self._get_question_prediction_info(
            question.question_no, full_research, "no"
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
                self, "force_reforecast_in_conditional", set()
            )
        ):
            previous_forecast = previous_forecasts[-1]
            current_utc_time = datetime.now(timezone.utc)
            if (
                previous_forecast.timestamp_end is None
                or previous_forecast.timestamp_end > current_utc_time
            ):
                readable = DataOrganizer.get_readable_prediction(previous_forecast)
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
            research, info, question_type
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
        if getattr(question, "conditional_type", None) not in {"yes", "no"}:
            return ""
        return clean_indents(
            """
            This is a conditional child question. Forecast ONLY the child outcome
            under the stated parent resolution; do not independently re-forecast
            the parent inside this component.
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
                f"The question creator thinks the number is likely "
                f"not higher than {upper_bound_number} "
                f"{unit_of_measure}."
            )
        else:
            upper_bound_message = (
                f"The outcome cannot be higher than "
                f"{upper_bound_number} {unit_of_measure}."
            )

        if question.open_lower_bound:
            lower_bound_message = (
                f"The question creator thinks the number is likely "
                f"not lower than {lower_bound_number} "
                f"{unit_of_measure}."
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
            Qwen3.8 27B -> Gemma 4 31B

        These free endpoints advertise structured-output support, unlike the
        current Nemotron Ultra and Laguna endpoints.
        """

        return self.get_llm("parser", "llm")
