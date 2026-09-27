"""Entry point for the OpenRouter Metaculus forecasting bot."""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
from typing import Literal

import dotenv

from bot_helpers import (
    check_environment,
    print_run_summary_banner,
    print_startup_banner,
    silence_noisy_dependencies,
)

silence_noisy_dependencies()

from forecasting_tools import MetaculusClient  # noqa: E402
from bot import (  # noqa: E402
    FALLBACK_LLM,
    PARSER_FALLBACK_LLM,
    PARSER_PRIMARY_LLM,
    PRIMARY_LLM,
    THIRD_MODEL,
    OpenRouterForecastBot,
)

dotenv.load_dotenv()

logger = logging.getLogger(__name__)

FALL_FUTUREEVAL_2026_ID = "fall-futureeval-2026"
FALL_FUTUREEVAL_2026_URL = (
    "https://www.metaculus.com/tournament/fall-futureeval-2026/"
)

EXPECTED_PRIMARY_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
EXPECTED_FALLBACK_MODEL = "poolside/laguna-s-2.1:free"
EXPECTED_THIRD_MODEL = "qwen/qwen3.8-27b:free"
EXPECTED_PARSER_PRIMARY_MODEL = "qwen/qwen3.8-27b:free"
EXPECTED_PARSER_FALLBACK_MODEL = "google/gemma-4-31b-it:free"


def _bare_model(model: str) -> str:
    return model.removeprefix("openrouter/")


def validate_openrouter_configuration() -> None:
    if not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is not set.")

    configured_primary = os.getenv(
        "OPENROUTER_MODEL",
        EXPECTED_PRIMARY_MODEL,
    )
    if configured_primary != EXPECTED_PRIMARY_MODEL:
        raise RuntimeError(
            "Unexpected primary model. "
            f"Expected {EXPECTED_PRIMARY_MODEL}; found {configured_primary}."
        )

    actual = {
        "primary": _bare_model(PRIMARY_LLM),
        "fallback": _bare_model(FALLBACK_LLM),
        "third fallback": THIRD_MODEL,
        "parser primary": _bare_model(PARSER_PRIMARY_LLM),
        "parser fallback": _bare_model(PARSER_FALLBACK_LLM),
    }
    expected = {
        "primary": EXPECTED_PRIMARY_MODEL,
        "fallback": EXPECTED_FALLBACK_MODEL,
        "third fallback": EXPECTED_THIRD_MODEL,
        "parser primary": EXPECTED_PARSER_PRIMARY_MODEL,
        "parser fallback": EXPECTED_PARSER_FALLBACK_MODEL,
    }

    for name, value in actual.items():
        if value != expected[name]:
            raise RuntimeError(
                f"Internal {name} model mismatch: "
                f"expected {expected[name]}, found {value}"
            )

    logger.info(
        "Forecast/research chain: %s -> %s -> %s",
        PRIMARY_LLM,
        FALLBACK_LLM,
        THIRD_MODEL,
    )
    logger.info(
        "Parser chain: %s -> %s",
        PARSER_PRIMARY_LLM,
        PARSER_FALLBACK_LLM,
    )
    logger.info(
        "Search query generation: deterministic; original question always included."
    )
    logger.info(
        "Scraper chain: HTTP -> Trafilatura -> BeautifulSoup -> DDGS indexed snippet."
    )
    logger.info("Metaculus URLs are not filtered from research results.")


def create_bot(*, publish_reports_to_metaculus: bool) -> OpenRouterForecastBot:
    return OpenRouterForecastBot(
        research_reports_per_question=1,
        predictions_per_research_report=3,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=publish_reports_to_metaculus,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
    )


def _dedupe_questions(questions: list) -> list:
    """
    get_all_open_questions_from_tournament() fetches questions with
    group_question_mode="unpack_subquestions", which can hand back the
    same question (same id_of_question / page_url) more than once for
    certain group/conditional questions. That silently doubles the
    research + forecast work (and the free-tier API calls) for that
    question. Dedupe by id_of_question before forecasting.
    """
    deduped = []
    seen_ids: set[int] = set()
    for question in questions:
        question_id = question.id_of_question
        if question_id is not None and question_id in seen_ids:
            logger.warning(
                "Skipping duplicate question returned by "
                "get_all_open_questions_from_tournament: %s (id=%s)",
                question.page_url, question_id,
            )
            continue
        if question_id is not None:
            seen_ids.add(question_id)
        deduped.append(question)

    if len(deduped) != len(questions):
        logger.warning(
            "%d questions returned, %d after deduplication.",
            len(questions), len(deduped),
        )

    return deduped


async def _run_forecasting_async(
    bot: OpenRouterForecastBot,
    run_mode: Literal["tournament", "metaculus_cup", "test_questions"],
) -> list:
    client = MetaculusClient()

    async def _forecast_tournament_deduped(tournament_id: int | str) -> list:
        questions = client.get_all_open_questions_from_tournament(tournament_id)
        deduped = _dedupe_questions(questions)
        return await bot.forecast_questions(deduped, return_exceptions=True)

    if run_mode == "tournament":
        seasonal = await _forecast_tournament_deduped(FALL_FUTUREEVAL_2026_ID)
        minibench = await _forecast_tournament_deduped(
            client.CURRENT_MINIBENCH_ID
        )
        return seasonal + minibench

    if run_mode == "metaculus_cup":
        bot.skip_previously_forecasted_questions = False
        return await _forecast_tournament_deduped(client.CURRENT_METACULUS_CUP_ID)

    bot.skip_previously_forecasted_questions = False

    # The official bot-testing-area contains examples of all supported
    # question types. Test mode only needs to confirm the pipeline runs
    # end-to-end, so it forecasts a single question rather than every
    # example in the tournament -- this keeps CI runs fast and avoids
    # burning through the free-tier rate limits on every push.
    questions = client.get_all_open_questions_from_tournament("bot-testing-area")
    deduped = _dedupe_questions(questions)

    if not deduped:
        logger.warning("bot-testing-area returned no open questions.")
        return []

    single_question = deduped[:1]
    logger.info(
        "test_questions mode: forecasting 1/%d available question(s): %s",
        len(deduped), single_question[0].page_url,
    )
    return await bot.forecast_questions(single_question, return_exceptions=True)


def run_forecasting(
    bot: OpenRouterForecastBot,
    run_mode: Literal["tournament", "metaculus_cup", "test_questions"],
) -> list:
    """Run the selected mode inside exactly one asyncio event loop."""
    return asyncio.run(_run_forecasting_async(bot, run_mode))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Run the OpenRouter Metaculus forecasting bot"
    )
    parser.add_argument(
        "--mode",
        choices=["tournament", "metaculus_cup", "test_questions"],
        default="tournament",
    )
    args = parser.parse_args()

    check_environment(strict=True)
    missing = [
        variable
        for variable in ("METACULUS_TOKEN", "OPENROUTER_API_KEY")
        if not os.getenv(variable)
    ]
    if missing:
        raise RuntimeError(
            "Missing required environment variables: " + ", ".join(missing)
        )

    validate_openrouter_configuration()

    publish = args.mode != "test_questions"
    logger.info("Selected mode: %s", args.mode)
    logger.info("Publishing enabled: %s", publish)
    logger.info("Research original question + deterministic targeted variants: ON")
    logger.info("Metaculus source URLs: ALLOWED")

    print_startup_banner(args.mode, will_publish=publish)
    bot = create_bot(publish_reports_to_metaculus=publish)
    reports = run_forecasting(bot, args.mode)
    bot.log_report_summary(reports)

    tournament_urls = {
        "tournament": FALL_FUTUREEVAL_2026_URL,
        "test_questions": "https://www.metaculus.com/tournament/bot-testing-area/",
    }
    print_run_summary_banner(
        reports,
        will_publish=publish,
        tournament_url=tournament_urls.get(args.mode),
    )


if __name__ == "__main__":
    main()
