"""
VibeThinkerForecastBot

Architecture:
  - Main brain: WeiboAI/VibeThinker-3B, called via Hugging Face Inference
    Providers (Featherless AI backend) in clients/hf_vibethinker.py. This
    model ONLY writes forecast reasoning/predictions -- it is never used
    for search-query generation.
  - Helper model: a free OpenRouter model (clients/openrouter_helper.py),
    used for (a) search-query generation, (b) research summarization, and
    (c) parsing the main brain's free-text reasoning into a structured
    prediction (via forecasting_tools.structure_output).
  - Research: DDGS search -> trafilatura -> BeautifulSoup fallback -> DDGS
    snippet fallback. See research/pipeline.py and research/scraper.py.

This subclasses ForecastBot from the official Metaculus bot template
(https://github.com/Metaculus/metac-bot-template), which handles fetching
questions, posting forecasts, and aggregating multiple runs. Only the
research step and the three per-question-type forecast methods are
overridden here.
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

from clients import hf_vibethinker
from research.pipeline import run_research_pipeline

logger = logging.getLogger(__name__)


class VibeThinkerForecastBot(ForecastBot):
    """Metaculus forecasting bot whose reasoning brain is VibeThinker-3B."""

    _structure_output_validation_samples = 2

    ##################################### RESEARCH #####################################

    async def run_research(self, question: MetaculusQuestion) -> str:
        research = await run_research_pipeline(
            question_text=question.question_text,
            resolution_criteria=question.resolution_criteria or "",
            background=question.background_info or "",
        )
        logger.info("Research for %s:\n%s", question.page_url, research[:1000])
        return research

    ##################################### BINARY QUESTIONS #####################################

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
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

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )
        reasoning = await hf_vibethinker.generate_forecast_reasoning(prompt)
        logger.info("VibeThinker reasoning for %s: %s", question.page_url, reasoning[:500])

        binary_prediction: BinaryPrediction = await structure_output(
            reasoning,
            BinaryPrediction,
            model=self._parser_llm(),
            num_validation_samples=self._structure_output_validation_samples,
        )
        decimal_pred = max(0.01, min(0.99, binary_prediction.prediction_in_decimal))
        return ReasonedPrediction(prediction_value=decimal_pred, reasoning=reasoning)

    ##################################### MULTIPLE CHOICE QUESTIONS #####################################

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
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
            (c) A description of a scenario that results in an unexpected outcome.

            You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )
        reasoning = await hf_vibethinker.generate_forecast_reasoning(prompt)
        logger.info("VibeThinker reasoning for %s: %s", question.page_url, reasoning[:500])

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
        return ReasonedPrediction(prediction_value=predicted_option_list, reasoning=reasoning)

    ##################################### NUMERIC QUESTIONS #####################################

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = self._create_upper_and_lower_bound_messages(
            question
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
            - Please notice the units requested and give your answer in these units.
            - Never use scientific notation.
            - Always start with a smaller number and then increase from there. The value for percentile 10 should always be less than the value for percentile 20, and so on.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

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
        reasoning = await hf_vibethinker.generate_forecast_reasoning(prompt)
        logger.info("VibeThinker reasoning for %s: %s", question.page_url, reasoning[:500])

        parsing_instructions = clean_indents(
            f"""
            The text given to you is trying to give a forecast distribution for a numeric question.
            - This text is trying to answer the numeric question: "{question.question_text}".
            - When parsing the text, please make sure to give the values (the ones assigned to percentiles) in terms of the correct units.
            - The units for the forecast are: {question.unit_of_measure}
            - As an example, someone else guessed that the answer will be between {question.lower_bound} {question.unit_of_measure} and {question.upper_bound} {question.unit_of_measure}, so the numbers parsed from an answer like this would be verbatim "{question.lower_bound}" and "{question.upper_bound}".
            - If the answer doesn't give the answer in the correct units, you should parse it in the right units.
            - If percentiles are not explicitly given, please don't return a parsed output, but rather indicate that the answer is not explicitly given in the text.
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
        prediction = NumericDistribution.from_question(percentile_list, question)
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion
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
                f"The question creator thinks the number is likely not higher "
                f"than {upper_bound_number} {unit_of_measure}."
            )
        else:
            upper_bound_message = (
                f"The outcome can not be higher than {upper_bound_number} {unit_of_measure}."
            )
        if question.open_lower_bound:
            lower_bound_message = (
                f"The question creator thinks the number is likely not lower "
                f"than {lower_bound_number} {unit_of_measure}."
            )
        else:
            lower_bound_message = (
                f"The outcome can not be lower than {lower_bound_number} {unit_of_measure}."
            )
        return upper_bound_message, lower_bound_message

    ##################################### HELPERS #####################################

    def _parser_llm(self) -> GeneralLlm:
        """The free OpenRouter model used to parse VibeThinker-3B's free-text
        reasoning into a structured prediction. Never the main brain."""
        return self.get_llm("parser", "llm")
