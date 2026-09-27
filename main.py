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
    PRIMARY_LLM,
    OpenRouterForecastBot,
)
from clients.openrouter_helper import THIRD_MODEL  # noqa: E402

dotenv.load_dotenv()

logger = logging.getLogger(__name__)

FALL_FUTUREEVAL_2026_ID = "fall-futureeval-2026"
FALL_FUTUREEVAL_2026_URL = (
    "https://www.metaculus.com/tournament/fall-futureeval-2026/"
)
TEST_QUESTION_URL = "https://www.metaculus.com/questions/43322/"

EXPECTED_PRIMARY_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
EXPECTED_FALLBACK_MODEL = "poolside/laguna-s-2.1:free"
EXPECTED_THIRD_MODEL = "qwen/qwen3.8-27b:free"


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
            f"Expected {EXPECTED_PRIMARY_MODEL}; "
            f"found {configured_primary}."
        )

    actual = {
        "primary": PRIMARY_LLM.removeprefix("openrouter/"),
        "fallback": FALLBACK_LLM.removeprefix("openrouter/"),
        "third fallback": THIRD_MODEL,
    }

    expected = {
        "primary": EXPECTED_PRIMARY_MODEL,
        "fallback": EXPECTED_FALLBACK_MODEL,
        "third fallback": EXPECTED_THIRD_MODEL,
    }

    for name, value in actual.items():
        if value != expected[name]:
            raise RuntimeError(
                f"Internal {name} model mismatch: "
                f"expected {expected[name]}, found {value}"
            )

    logger.info(
        "Forecast chain: %s -> %s -> %s",
        PRIMARY_LLM,
        FALLBACK_LLM,
        THIRD_MODEL,
    )
    logger.info(
        "Search queries: deterministic rule-based construction; "
        "no LLM query generator."
    )
    logger.info(
        "Research uses the full question, first-8-word fragment, "
        "and latest-news variant."
    )
    logger.info(
        "Scraper chain: Trafilatura -> BeautifulSoup -> "
        "DDGS indexed snippet."
    )


def create_bot() -> OpenRouterForecastBot:
    return OpenRouterForecastBot(
        # Was 2 -- this made the bot run research (and the full forecast
        # chain) TWICE per question, doubling LLM calls, doubling free-tier
        # rate-limit exposure, and roughly doubling wall-clock time. One
        # research pass is enough; predictions_per_research_report already
        # gives you multiple forecast samples off that single research pass.
        research_reports_per_question=1,
        predictions_per_research_report=3,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
    )


def run_forecasting(
    bot: OpenRouterForecastBot,
    run_mode: Literal[
        "tournament",
        "metaculus_cup",
        "test_questions",
    ],
) -> list:
    client = MetaculusClient()

    if run_mode == "tournament":
        seasonal = asyncio.run(
            bot.forecast_on_tournament(
                FALL_FUTUREEVAL_2026_ID,
                return_exceptions=True,
            )
        )

        minibench = asyncio.run(
            bot.forecast_on_tournament(
                client.CURRENT_MINIBENCH_ID,
                return_exceptions=True,
            )
        )

        return seasonal + minibench

    if run_mode == "metaculus_cup":
        bot.skip_previously_forecasted_questions = False

        return asyncio.run(
            bot.forecast_on_tournament(
                client.CURRENT_METACULUS_CUP_ID,
                return_exceptions=True,
            )
        )

    # -----------------------------------------------------------------------
    # TEST MODE
    #
    # Question 43322 is a group question. We explicitly unpack it and then
    # forecast only the first returned subquestion so this mode performs
    # exactly one forecast target.
    # -----------------------------------------------------------------------
    bot.skip_previously_forecasted_questions = False

    questions = client.get_question_by_url(
        TEST_QUESTION_URL,
        group_question_mode="unpack_subquestions",
    )

    if not questions:
        raise RuntimeError(
            f"No subquestions were returned for {TEST_QUESTION_URL}"
        )

    question = questions[0]

    logger.info(
        "Test mode: forecasting exactly one subquestion: %s",
        question.page_url,
    )

    return asyncio.run(
        bot.forecast_questions(
            [question],
            return_exceptions=True,
        )
    )


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
        choices=[
            "tournament",
            "metaculus_cup",
            "test_questions",
        ],
        default="tournament",
    )

    args = parser.parse_args()

    check_environment(strict=True)

    missing = [
        variable
        for variable in (
            "METACULUS_TOKEN",
            "OPENROUTER_API_KEY",
        )
        if not os.getenv(variable)
    ]

    if missing:
        raise RuntimeError(
            "Missing required environment variables: "
            + ", ".join(missing)
        )

    validate_openrouter_configuration()

    logger.info("=" * 60)
    logger.info("Fully-free OpenRouter forecasting configuration")
    logger.info("Primary: %s", PRIMARY_LLM)
    logger.info("Fallback: %s", FALLBACK_LLM)
    logger.info("Third fallback: %s", THIRD_MODEL)
    logger.info(
        "Query generation: deterministic; no query-generation LLM"
    )
    logger.info(
        "Research queries: verbatim question + "
        "derived deterministic variants"
    )
    logger.info(
        "Scraper: Trafilatura -> BeautifulSoup -> DDGS snippet"
    )
    logger.info("FutureEval: %s", FALL_FUTUREEVAL_2026_ID)

    if args.mode == "test_questions":
        logger.info(
            "Test question: %s",
            TEST_QUESTION_URL,
        )

    logger.info("=" * 60)

    publish = True

    print_startup_banner(
        args.mode,
        will_publish=publish,
    )

    bot = create_bot()

    reports = run_forecasting(
        bot,
        args.mode,
    )

    bot.log_report_summary(reports)

    urls = {
        "tournament": FALL_FUTUREEVAL_2026_URL,
        "metaculus_cup": (
            "https://www.metaculus.com/"
            "tournament/metaculus-cup-fall-2026/"
        ),
        "test_questions": TEST_QUESTION_URL,
    }

    print_run_summary_banner(
        reports,
        will_publish=publish,
        tournament_url=urls.get(args.mode),
    )


if __name__ == "__main__":
    main()
