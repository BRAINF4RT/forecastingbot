"""
OpenRouter Nemotron Metaculus Forecast Bot.

Architecture:

    Metaculus question
          |
          v
    research/pipeline.py
          |
          +--> Nemotron 3 Ultra -> search queries
          |
          +--> DDGS -> web pages
          |
          +--> Nemotron 3 Ultra -> research summary
          |
          v
    Nemotron 3 Ultra -> forecast reasoning
          |
          v
    forecasting_tools.structure_output
          |
          +--> Nemotron 3 Ultra
          |
          v
    ForecastBot -> Metaculus

The entire LLM pipeline uses the free OpenRouter model:

    nvidia/nemotron-3-ultra-550b-a55b:free
"""

from __future__ import annotations

import logging
from datetime import datetime

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


class OpenRouterForecastBot(ForecastBot):
    """
    Metaculus forecasting bot powered entirely by Nemotron 3 Ultra
    through OpenRouter's free tier.
    """

    _structure_output_validation_samples = 2

    ##################################### RESEARCH #####################################

    async def run_research(
        self,
        question: MetaculusQuestion,
    ) -> str:
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

    ##################################### BINARY QUESTIONS #####################################

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
            "Nemotron reasoning for %s: %s",
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

    ##################################### MULTIPLE CHOICE QUESTIONS #####################################

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
            plausible. Probabilities should reflect your actual assessment.

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
            "Nemotron reasoning for %s: %s",
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

    ##################################### NUMERIC QUESTIONS #####################################

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
            "Nemotron reasoning for %s: %s",
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
        Return the same Nemotron model used for all other LLM operations.
        """

        return self.get_llm("parser", "llm")
