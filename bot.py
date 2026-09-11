
"""
OpenRouter-only Metaculus forecasting bot.

LLM architecture
----------------

Primary:
    nvidia/nemotron-3-ultra-550b-a55b:free

Fallback:
    poolside/laguna-s-2.1:free

Both models are accessed exclusively through OpenRouter.

The forecasting-tools framework has four LLM purposes:

    default
    summarizer
    researcher
    parser

All four are explicitly configured to use the same
Nemotron -> Laguna fallback system.

There are intentionally no active:
    - VibeThinker
    - Hugging Face
    - Featherless
    - OpenAI
    - Anthropic
    - Perplexity
    - AskNews
    - Exa

LLM dependencies in this bot.

Web research is performed separately through:
    research/pipeline.py
    research/scraper.py
    DDGS
    trafilatura
    BeautifulSoup
"""

from __future__ import annotations

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

from clients.openrouter_helper import generate_forecast_reasoning
from research.pipeline import run_research_pipeline


logger = logging.getLogger(__name__)


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

PRIMARY_MODEL = "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free"
FALLBACK_MODEL = "openrouter/poolside/laguna-s-2.1:free"


class FallbackGeneralLlm(GeneralLlm):
    """
    GeneralLlm with an OpenRouter-only fallback.

    Primary:
        Nemotron 3 Ultra Free

    Fallback:
        Laguna S 2.1 Free

    The class deliberately subclasses GeneralLlm so it remains compatible
    with forecasting-tools functions such as structure_output().
    """

    PRIMARY_MODEL = PRIMARY_MODEL
    FALLBACK_MODEL = FALLBACK_MODEL

    def __init__(
        self,
        *,
        temperature: float | int | None = None,
        timeout: float | int = 180,
        allowed_tries: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=self.PRIMARY_MODEL,
            temperature=temperature,
            timeout=timeout,
            allowed_tries=allowed_tries,
            **kwargs,
        )

        self._fallback_llm = GeneralLlm(
            model=self.FALLBACK_MODEL,
            temperature=temperature,
            timeout=timeout,
            allowed_tries=allowed_tries,
            **kwargs,
        )

    async def invoke(
        self,
        prompt,
        system_prompt: str | None = None,
    ) -> str:
        """
        Try Nemotron first.

        If Nemotron fails for any reason, use Laguna.

        This covers:
            - provider unavailable
            - HTTP 5xx
            - rate limiting
            - timeouts
            - upstream failures
            - LiteLLM provider errors
            - model-specific failures
        """

        try:
            result = await super().invoke(
                prompt,
                system_prompt=system_prompt,
            )

            logger.info(
                "LLM request succeeded with Nemotron 3 Ultra."
            )

            return result

        except Exception as primary_error:
            logger.warning(
                "Nemotron 3 Ultra failed. "
                "Falling back to Laguna S 2.1. Error: %s",
                primary_error,
            )

        try:
            result = await self._fallback_llm.invoke(
                prompt,
                system_prompt=system_prompt,
            )

            logger.warning(
                "LLM fallback succeeded with Laguna S 2.1."
            )

            return result

        except Exception as fallback_error:
            logger.error(
                "Both Nemotron 3 Ultra and Laguna S 2.1 failed. "
                "Fallback error: %s",
                fallback_error,
            )

            raise RuntimeError(
                "Both OpenRouter free models failed. "
                f"Nemotron primary error: {primary_error!r}. "
                f"Laguna fallback error: {fallback_error!r}."
            ) from fallback_error


def _make_fallback_llm(
    *,
    temperature: float,
    timeout: float = 180,
) -> FallbackGeneralLlm:
    """
    Construct an independent fallback LLM instance.

    Separate instances prevent mutable LiteLLM configuration from being
    accidentally shared between forecasting-tools purposes.
    """

    return FallbackGeneralLlm(
        temperature=temperature,
        timeout=timeout,
        allowed_tries=1,
    )


# ============================================================================
# FORECAST BOT
# ============================================================================


class OpenRouterForecastBot(ForecastBot):
    """
    Metaculus forecasting bot using OpenRouter-only free models.

    Every forecasting-tools LLM purpose is explicitly configured:

        default    -> Nemotron -> Laguna
        summarizer -> Nemotron -> Laguna
        researcher -> Nemotron -> Laguna
        parser     -> Nemotron -> Laguna
    """

    _structure_output_validation_samples = 2

    # Free providers can become overloaded. Keeping question-level
    # concurrency at one avoids sending a large burst of simultaneous
    # requests to the same free provider.
    _max_concurrent_questions = 1

    @classmethod
    def _llm_config_defaults(
        cls,
    ) -> dict[str, str | GeneralLlm | None]:
        """
        Override forecasting-tools' provider defaults.

        This is critical.

        forecasting-tools normally sees OPENROUTER_API_KEY and defaults to
        GPT-4o / GPT-4o-mini / GPT-4o-search-preview.

        We explicitly replace all of those defaults.
        """

        return {
            "default": _make_fallback_llm(
                temperature=0.15,
                timeout=240,
            ),
            "summarizer": _make_fallback_llm(
                temperature=0.10,
                timeout=180,
            ),
            "researcher": _make_fallback_llm(
                temperature=0.10,
                timeout=180,
            ),
            "parser": _make_fallback_llm(
                temperature=0.0,
                timeout=180,
            ),
        }

    # ========================================================================
    # RESEARCH
    # ========================================================================

    async def run_research(
        self,
        question: MetaculusQuestion,
    ) -> str:
        """
        Run the custom DDGS/trafilatura research pipeline.

        The research pipeline itself uses OpenRouter through
        clients.openrouter_helper.py for query generation and summarisation.
        """

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
            "Research completed for %s. Characters: %d",
            question.page_url,
            len(research),
        )

        return research

    # ========================================================================
    # BINARY QUESTIONS
    # ========================================================================

    async def _run_forecast_on_binary(
        self,
        question: BinaryQuestion,
        research: str,
    ) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            You are a professional probabilistic forecaster.

            Forecast the probability of the following binary event.

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

            Carefully consider:

            (a) How much time remains until the outcome is known.
            (b) The status quo if nothing significant changes.
            (c) A plausible scenario producing a NO outcome.
            (d) A plausible scenario producing a YES outcome.
            (e) Base rates and historical precedent.
            (f) Important evidence supporting each side.
            (g) Important uncertainties and unknowns.
            (h) Whether the research sources are reliable and relevant.

            Good forecasting requires calibrated uncertainty.
            Do not blindly follow the research.

            The final line MUST be exactly:

            Probability: ZZ%

            where ZZ is your probability from 0 to 100.
            """
        )

        reasoning = await generate_forecast_reasoning(prompt)

        logger.info(
            "Forecast reasoning generated for binary question %s",
            question.page_url,
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

        logger.info(
            "Binary prediction for %s: %.4f",
            question.page_url,
            decimal_pred,
        )

        return ReasonedPrediction(
            prediction_value=decimal_pred,
            reasoning=reasoning,
        )

    # ========================================================================
    # MULTIPLE CHOICE QUESTIONS
    # ========================================================================

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

            Carefully consider:

            (a) The time remaining.
            (b) The status quo outcome.
            (c) The most likely option.
            (d) Why each alternative could occur.
            (e) Base rates and historical precedent.
            (f) Unexpected scenarios.
            (g) Conflicting evidence.
            (h) The quality of the research.

            Give a probability to EVERY option.

            Do not assign probability merely because an option sounds
            plausible. Probabilities should reflect your actual assessment.

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
            "Forecast reasoning generated for multiple-choice question %s",
            question.page_url,
        )

        parsing_instructions = clean_indents(
            f"""
            The valid option names are:

            {question.options}

            Parsing requirements:

            - Every valid option must appear.
            - Use exactly the supplied option names.
            - Remove prefixes such as "Option" if they are not part of the
              actual option name.
            - Preserve 0% probabilities.
            - Do not invent options.
            - Probabilities should represent the model's stated forecast.
            """
        )

        predicted_option_list: PredictedOptionList = await structure_output(
            text_to_structure=reasoning,
            output_type=PredictedOptionList,
            model=self._parser_llm(),
            num_validation_samples=self._structure_output_validation_samples,
            additional_instructions=parsing_instructions,
        )

        logger.info(
            "Multiple-choice prediction generated for %s",
            question.page_url,
        )

        return ReasonedPrediction(
            prediction_value=predicted_option_list,
            reasoning=reasoning,
        )

    # ========================================================================
    # NUMERIC QUESTIONS
    # ========================================================================

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
            (i) Whether the research contains conflicting evidence.

            Be appropriately uncertain.

            Formatting requirements:

            - Use the requested units.
            - Never use scientific notation.
            - Percentile values must increase monotonically.
            - Do not invent unsupported precision.
            - Use a reasonably wide 10th-90th percentile range.
            - Do not make the distribution artificially narrow.

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
            "Numeric forecast reasoning generated for %s",
            question.page_url,
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
            - Only use percentile values explicitly supported by the model's
              final answer.
            - Preserve the requested percentile labels.
            - Do not invent missing percentile values.
            - Ensure percentile values are monotonically increasing.
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

        logger.info(
            "Numeric prediction generated for %s",
            question.page_url,
        )

        return ReasonedPrediction(
            prediction_value=prediction,
            reasoning=reasoning,
        )

    # ========================================================================
    # HELPERS
    # ========================================================================

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

        unit_of_measure = question.unit_of_measure or ""

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
        Return the parser LLM.

        The parser itself is also Nemotron -> Laguna because the parser
        purpose is explicitly configured in _llm_config_defaults().
        """

        return self.get_llm(
            "parser",
            "llm",
        )
