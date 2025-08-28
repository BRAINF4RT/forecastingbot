import argparse
import asyncio
import logging
import time
import random
from datetime import datetime
from typing import Literal
from duckduckgo_search import DDGS
ddgs = DDGS()

def search_internet(query: str, max_results: int = 50, batch_size: int = 5):
    all_results = []
    seen_urls = set()
    modifiers = [" future", " recent", " analysis", " report", " news", " study", " trend", " update", "data", " stat",]
    try:
        while len(all_results) < max_results:
            modifier = random.choice(modifiers)
            var_query = f"{query} {modifier}"
            results = ddgs.text(var_query, max_results=batch_size)
            for r in results:
                if "body" in r and r["href"] not in seen_urls:
                    all_results.append(r)
                    seen_urls.add(r["href"])
            if not results or len(results) == 0:
                break
            time.sleep(1)
        return all_results[:max_results]
    except Exception as e:
        return all_results

from forecasting_tools import (
    AskNewsSearcher,
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusApi,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    BinaryPrediction,
    PredictedOptionList,
    ReasonedPrediction,
    SmartSearcher,
    clean_indents,
    structure_output,
)

logger = logging.getLogger(__name__)

async def generate_search_query(question: MetaculusQuestion, model: str) -> str:
    prompt = f"""
    You are an AI assistant tasked with generating concise, effective search queries for research.

    Given a Metaculus prediction question, create a short search query (MAX 25 words) that captures the key
    entities, relevant numbers, and concepts. Avoid copying the question word-for-word. Focus on what
    someone would type in a search engine to find information that could help answer the question. Try to
    avoid words in your output that could bring up irrelevent information.

    Question Title:
    {question.question_text}

    Resolution Criteria:
    {question.resolution_criteria}

    More info:
    {question.fine_print}

    Return ONLY the final search query.
    """

    llm = GeneralLlm(
        model=model,
        temperature=0.2,
        timeout=20,
        allowed_tries=2,
    )
    query = await llm.invoke(prompt)
    query = query.strip() 
    logger.info(f"Generated search query for question {question.page_url}: {query}")
    return query

async def get_combined_response_openrouter(prompt: str, query: str, model: str):
    search_results = search_internet(query)
    search_content = "\n".join([result['body'] for result in search_results])

    full_prompt = f"""{prompt}

    Additional Internet Search Results:
    {search_content}
    """

    llm = GeneralLlm(
        model=model,
        temperature=0.2,
        timeout=40,
        allowed_tries=2,
    )
    response = await llm.invoke(full_prompt)
    return response


class FallTemplateBot2025(ForecastBot):


    _max_concurrent_questions = (
        1  
    )
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            research = ""
            researcher = self.get_llm("researcher")

            prompt = clean_indents(
                f"""
                You are an assistant to a superforecaster.
                The superforecaster will give you a question they intend to forecast on.
                To be a great assistant, you generate a very detailed rundown of:
                1. The most relevant news and most relevant information from searches. 
                2. Historical precedents: past events, case studies, or reference classes that are related to this question. 
                   - Identify how often similar events have occurred in the past.
                   - Highlight similarities and differences between past cases and the present one.
                Try to diversify your sources, but also ensure that they are reputable.
                Tell the forecaster what YOU think the question will resolve as and why, however you do not produce forecasts yourself.
                Your output prioritises quality information and it can be as large as it needs to be, as long as it gets all the relevent information across.

                Question:
                {question.question_text}

                This question's outcome will be determined by the specific criteria below:
                {question.resolution_criteria}

                {question.fine_print}
                """
            )

            if isinstance(researcher, GeneralLlm):
                research = await researcher.invoke(prompt)
            elif not researcher or researcher == "None":
                research = ""
            else:
                research_results = []
                for _ in range(3):
                    search_query = await generate_search_query(question, model=self.get_llm("querier"))
                    logger.info(f"Using search query for question {question.page_url}: {search_query}")
                    result = await get_combined_response_openrouter(
                        prompt,
                        search_query,
                        model=self.get_llm("researcher")
                    )
                    research_results.append(result)
                    await asyncio.sleep(3)
                research = "\n\n".join(research_results)
            return research
            
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


            The compiled information of your multiple research assistants says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A brief description of a scenario that results in a No outcome.
            (d) A brief description of a scenario that results in a Yes outcome.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.
            Keep in mind that if you put extra weight on a prediction and your prediction is correct, you will score better. However if your prediction is wrong, you will be penalised harder for adding that confidence.

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        binary_prediction: BinaryPrediction = await structure_output(
            reasoning, BinaryPrediction, model=self.get_llm("parser", "llm")
        )
        decimal_pred = max(0.01, min(0.99, binary_prediction.prediction_in_decimal))

        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {decimal_pred}"
        )
        return ReasonedPrediction(prediction_value=decimal_pred, reasoning=reasoning)

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


            The compiled information of your multiple research assistants says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A description of an scenario that results in an unexpected outcome.

            You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.
            Keep in mind that if you put extra weight on a prediction and your prediction is correct, you will score better. However if your prediction is wrong, you will be penalised harder for adding that confidence.

            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )
        parsing_instructions = clean_indents(
            f"""
            Make sure that all option names are one of the following:
            {question.options}
            The text you are parsing may prepend these options with some variation of "Option" which you should remove if not part of the option names I just gave you.
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        predicted_option_list: PredictedOptionList = await structure_output(
            text_to_structure=reasoning,
            output_type=PredictedOptionList,
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
        )
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {predicted_option_list}"
        )
        return ReasonedPrediction(
            prediction_value=predicted_option_list, reasoning=reasoning
        )

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
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

            The compiled information of your multiple research assistants says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - Please notice the units requested (e.g. whether you represent a number as 1,000,000 or 1 million).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.
            Keep in mind that if you put extra weight on a prediction and your prediction is correct, you will score better. However if your prediction is wrong, you will be penalised harder for adding that confidence.

            The last thing you write is your final answer as:
            "
            Percentile 10: XX
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX
            "
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        percentile_list: list[Percentile] = await structure_output(
            reasoning, list[Percentile], model=self.get_llm("parser", "llm")
        )
        prediction = NumericDistribution.from_question(percentile_list, question)
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}"
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion
    ) -> tuple[str, str]:
        if question.nominal_upper_bound is not None:
            upper_bound_number = question.nominal_upper_bound
        else:
            upper_bound_number = question.upper_bound
        if question.nominal_lower_bound is not None:
            lower_bound_number = question.nominal_lower_bound
        else:
            lower_bound_number = question.lower_bound

        if question.open_upper_bound:
            upper_bound_message = f"The question creator thinks the number is likely not higher than {upper_bound_number}."
        else:
            upper_bound_message = (
                f"The outcome can not be higher than {upper_bound_number}."
            )

        if question.open_lower_bound:
            lower_bound_message = f"The question creator thinks the number is likely not lower than {lower_bound_number}."
        else:
            lower_bound_message = (
                f"The outcome can not be lower than {lower_bound_number}."
            )
        return upper_bound_message, lower_bound_message


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False
    logging.getLogger("openai.agents").setLevel(logging.ERROR)
    logging.getLogger("forecasting_tools.ai_models.model_tracker").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(
        description="Run the Q1TemplateBot forecasting system"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "metaculus_cup", "test_questions", "market_pulse",],
        default="tournament",
        help="Specify the run mode (default: tournament)",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "metaculus_cup", "test_questions", "market_pulse",] = args.mode
    assert run_mode in [
        "tournament",
        "metaculus_cup",
        "test_questions",
        "market_pulse",
    ], "Invalid run mode"

    template_bot = FallTemplateBot2025(
        research_reports_per_question=2,
        predictions_per_research_report=5,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
         llms={  
                 "default": GeneralLlm(
                 model="openrouter/deepseek/deepseek-r1-0528",
                 temperature=0.2,
                 timeout=40,
                 allowed_tries=2,
             ),
             "summarizer": "openrouter/openai/gpt-oss-20b",
             "researcher": "openrouter/openai/gpt-oss-120b",  
             "parser": "openrouter/openai/gpt-oss-20b",
             "querier": "openrouter/openai/gpt-oss-20b",
         },
    )         
    if run_mode == "tournament":
        seasonal_tournament_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                MetaculusApi.CURRENT_AI_COMPETITION_ID, return_exceptions=True
            )
        )
        minibench_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                MetaculusApi.CURRENT_MINIBENCH_ID, return_exceptions=True
            )
        )
        forecast_reports = seasonal_tournament_reports + minibench_reports
    elif run_mode == "metaculus_cup":
        template_bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                MetaculusApi.CURRENT_METACULUS_CUP_ID, return_exceptions=True
            )
        )
    elif run_mode == "market_pulse":
        MP25Q3_TOURNAMENT_ID = 32773
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                MP25Q3_TOURNAMENT_ID, return_exceptions=True
            )
        )       
    elif run_mode == "test_questions":
        EXAMPLE_QUESTIONS = [
            #"https://www.metaculus.com/questions/39109/which-party-will-lead-tasmania/",
            #"https://www.metaculus.com/questions/39110/practice-what-will-be-the-score-ratio-of-the-highest-performing-bot-compared-to-the-top-5-participants-in-the-summer-2025-metaculus-cup/",
            #"https://www.metaculus.com/questions/39056/practice-will-shigeru-ishiba-cease-to-be-prime-minister-of-japan-before-september-2025/",
            #"https://www.metaculus.com/questions/39055/community-prediction-of-this-question-divided-by-2/",
            #"https://www.metaculus.com/questions/578/human-extinction-by-2100/",  # Human Extinction - Binary
            "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",  # Age of Oldest Human - Numeric
            #"https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",  # Number of New Leading AI Labs - Multiple Choice
            #"https://www.metaculus.com/c/diffusion-community/38880/how-many-us-labor-strikes-due-to-ai-in-2029/",  # Number of US Labor Strikes Due to AI in 2029 - Discrete
        ]
        template_bot.skip_previously_forecasted_questions = False
        questions = [
            MetaculusApi.get_question_by_url(question_url)
            for question_url in EXAMPLE_QUESTIONS
        ]
        forecast_reports = asyncio.run(
            template_bot.forecast_questions(questions, return_exceptions=True)
        )
    template_bot.log_report_summary(forecast_reports)
