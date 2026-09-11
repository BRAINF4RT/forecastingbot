"""
OpenRouter Metaculus Forecast Bot.

LLM architecture:

    Search query generation
        |
        +--> Google Gemma 4 31B IT :free
             reasoning = OFF
        |
        v
    DDGS web research
        |
        v
    Nemotron 3 Ultra :free
        |
        +--> Laguna S 2.1 :free on failure
        |
        v
    Forecast reasoning
        |
        +--> Laguna S 2.1 :free on failure
        |
        v
    forecasting_tools structured parsing
        |
        +--> Laguna S 2.1 :free on failure
        |
        v
    Metaculus
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

from forecasting_tools import (
    BinaryPrediction,
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    PredictedOptionList,
    ReasonedPrediction,
    clean_indents,
    structure_output,
)

from clients.openrouter_helper import (
    FALLBACK_MODEL,
    PRIMARY_MODEL,
    generate_forecast_reasoning,
)
from research.pipeline import run_research_pipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

PRIMARY_LLM = f"openrouter/{PRIMARY_MODEL}"
FALLBACK_LLM = f"openrouter/{FALLBACK_MODEL}"

# Maximum number of forecasting-tools / LiteLLM calls allowed at once.
#
# This is separate from the direct OpenRouter semaphore in
# clients/openrouter_helper.py.
_GENERAL_LLM_CONCURRENCY = 2

_GENERAL_LLM_SEMAPHORE = asyncio.Semaphore(
    _GENERAL_LLM_CONCURRENCY
)


class FallbackGeneralLlm(GeneralLlm):
    """
    GeneralLlm wrapper with explicit primary -> fallback behaviour.

    Primary:
        Nemotron 3 Ultra

    Fallback:
        Laguna S 2.1

    The wrapper deliberately uses allowed_tries=1 on each underlying model.
    The wrapper itself controls the fallback so that we don't get multiple
    layers of hidden retries.

    It also shares a semaphore across all instances to prevent a burst of
    simultaneous requests against free OpenRouter endpoints.
    """

    def __init__(
        self,
        *,
        primary_model: str,
        fallback_model: str,
        temperature: float,
        timeout: float,
    ) -> None:
        super().__init__(
            model=primary_model,
            temperature=temperature,
            timeout=timeout,
            allowed_tries=1,
        )

        self._primary_model_name = primary_model
        self._fallback_model_name = fallback_model

        self._fallback_llm = GeneralLlm(
            model=fallback_model,
            temperature=temperature,
            timeout=timeout,
            allowed_tries=1,
        )

    async def invoke(
        self,
        prompt: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Try Nemotron once, then Laguna once.

        The forecasting_tools GeneralLlm itself has allowed_tries=1 so this
        wrapper is the component responsible for model failover.
        """

        primary_error: Exception | None = None

        async with _GENERAL_LLM_SEMAPHORE:
            try:
                logger.debug(
                    "Calling primary forecasting LLM: %s",
                    self._primary_model_name,
                )

                return await super().invoke(
                    prompt,
                    *args,
                    **kwargs,
                )

            except Exception as exc:
                primary_error = exc

                logger.warning(
                    "Primary forecasting model failed: %s. "
                    "Falling back to %s.",
                    self._primary_model_name,
                    self._fallback_model_name,
                )

            try:
                result = await self._fallback_llm.invoke(
                    prompt,
                    *args,
                    **kwargs,
                )

                logger.info(
                    "Fallback forecasting model succeeded: %s",
                    self._fallback_model_name,
                )

                return result

            except Exception as fallback_error:
                raise RuntimeError(
                    "Both forecasting LLMs failed.\n"
                    f"Primary ({self._primary_model_name}): "
                    f"{primary_error!r}\n"
                    f"Fallback ({self._fallback_model_name}): "
                    f"{fallback_error!r}"
                ) from fallback_error


class OpenRouterForecastBot(ForecastBot):
    """
    Metaculus forecasting bot.

    All forecasting_tools LLM purposes explicitly use the same
    Nemotron -> Laguna fallback wrapper.

    This prevents forecasting_tools from silently selecting OpenAI models
    when a purpose is not manually configured.
    """

    _structure_output_validation_samples = 1

    def _llm_config_defaults(self) -> dict[str, GeneralLlm]:
        """
        Explicitly configure every forecasting_tools LLM purpose.

        This is important because ForecastBot otherwise fills missing purposes
        with its own defaults such as GPT-4o / GPT-4o-mini / search-preview.
        """

        return {
            "default": FallbackGeneralLlm(
                primary_model=PRIMARY_LLM,
                fallback_model=FALLBACK_LLM,
                temperature=0.15,
                timeout=240,
            ),
            "summarizer": FallbackGeneralLlm(
                primary_model=PRIMARY_LLM,
                fallback_model=FALLBACK_LLM,
                temperature=0.10,
                timeout=240,
            ),
            "researcher": FallbackGeneralLlm(
                primary_model=PRIMARY_LLM,
                fallback_model=FALLBACK_LLM,
                temperature=0.10,
                timeout=240,
            ),
            "parser": FallbackGeneralLlm(
                primary_model=PRIMARY_LLM,
                fallback_model=FALLBACK_LLM,
                temperature=0.0,
                timeout=240,
            ),
        }

    ##################################### RESEARCH #####################################

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

    ##################################### BINARY #####################################

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

    ##################################### MULTIPLE CHOICE #####################################

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

    ##################################### NUMERIC #####################################

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

    ##################################### HELPERS #####################################

    def _create_upper_and_lower_bound_messages(
        self,
        question: NumericQuestion,
    ) -> tuple[str, str]:

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

        unit_of_measure = question.unit_of_measure

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
            Nemotron -> Laguna
        """

        return self.get_llm("parser", "llm")
