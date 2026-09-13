"""
Entry point for the OpenRouter Metaculus forecasting bot.

Usage:

    python main.py --mode tournament
    python main.py --mode test_questions
    python main.py --mode metaculus_cup

LLM architecture:

    Query generation:
        google/gemma-4-31b-it:free
        reasoning disabled

    Forecasting:
        nvidia/nemotron-3-ultra-550b-a55b:free
        ->
        poolside/laguna-s-2.1:free
"""

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
from clients.openrouter_helper import QUERY_MODEL  # noqa: E402

dotenv.load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tournament configuration
# ---------------------------------------------------------------------------

# Explicitly target the Fall 2026 FutureEval tournament instead of relying
# on forecasting_tools.CURRENT_AI_COMPETITION_ID to rotate automatically.
FALL_FUTUREEVAL_2026_ID = "fall-futureeval-2026"

FALL_FUTUREEVAL_2026_URL = (
    "https://www.metaculus.com/tournament/"
    "fall-futureeval-2026/"
)


# These are intentionally hard-coded so an environment variable cannot
# accidentally turn the bot into a paid model.
EXPECTED_PRIMARY_MODEL = (
    "nvidia/nemotron-3-ultra-550b-a55b:free"
)

EXPECTED_FALLBACK_MODEL = (
    "poolside/laguna-s-2.1:free"
)

EXPECTED_QUERY_MODEL = (
    "google/gemma-4-31b-it:free"
)


def validate_openrouter_configuration() -> None:
    """
    Validate that the bot is using exactly the intended models.
    """

    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set."
        )

    configured_primary = os.getenv(
        "OPENROUTER_MODEL",
        EXPECTED_PRIMARY_MODEL,
    )

    if configured_primary != EXPECTED_PRIMARY_MODEL:
        raise RuntimeError(
            "This bot is intentionally locked to the free Nemotron model.\n"
            f"Expected: {EXPECTED_PRIMARY_MODEL}\n"
            f"Found:    {configured_primary}"
        )

    actual_primary = PRIMARY_LLM.removeprefix(
        "openrouter/"
    )

    actual_fallback = FALLBACK_LLM.removeprefix(
        "openrouter/"
    )

    if actual_primary != EXPECTED_PRIMARY_MODEL:
        raise RuntimeError(
            "Internal primary-model configuration mismatch:\n"
            f"Expected: {EXPECTED_PRIMARY_MODEL}\n"
            f"Found:    {actual_primary}"
        )

    if actual_fallback != EXPECTED_FALLBACK_MODEL:
        raise RuntimeError(
            "Internal fallback-model configuration mismatch:\n"
            f"Expected: {EXPECTED_FALLBACK_MODEL}\n"
            f"Found:    {actual_fallback}"
        )

    if QUERY_MODEL != EXPECTED_QUERY_MODEL:
        raise RuntimeError(
            "Internal query-model configuration mismatch:\n"
            f"Expected: {EXPECTED_QUERY_MODEL}\n"
            f"Found:    {QUERY_MODEL}"
        )

    logger.info(
        "OpenRouter configuration validated."
    )

    logger.info(
        "Primary LLM: %s",
        PRIMARY_LLM,
    )

    logger.info(
        "Fallback LLM: %s",
        FALLBACK_LLM,
    )

    logger.info(
        "Query-generation LLM: openrouter/%s",
        QUERY_MODEL,
    )

    logger.info(
        "Query-generation reasoning: DISABLED"
    )


def create_bot() -> OpenRouterForecastBot:
    """
    Construct the forecasting bot.

    Notice that we do NOT pass a partial `llms` dictionary here.

    OpenRouterForecastBot._llm_config_defaults() explicitly configures
    default, summarizer, researcher and parser, preventing
    forecasting_tools from silently inserting OpenAI models.
    """

    return OpenRouterForecastBot(
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
    """
    Run the selected Metaculus forecasting mode.
    """

    client = MetaculusClient()

    if run_mode == "tournament":

        logger.info(
            "Targeting Fall 2026 FutureEval tournament: %s",
            FALL_FUTUREEVAL_2026_ID,
        )

        seasonal_reports = asyncio.run(
            bot.forecast_on_tournament(
                FALL_FUTUREEVAL_2026_ID,
                return_exceptions=True,
            )
        )

        minibench_reports = asyncio.run(
            bot.forecast_on_tournament(
                client.CURRENT_MINIBENCH_ID,
                return_exceptions=True,
            )
        )

        return seasonal_reports + minibench_reports

    if run_mode == "metaculus_cup":

        bot.skip_previously_forecasted_questions = False

        return asyncio.run(
            bot.forecast_on_tournament(
                client.CURRENT_METACULUS_CUP_ID,
                return_exceptions=True,
            )
        )

    # test_questions

    bot.skip_previously_forecasted_questions = False

    return asyncio.run(
        bot.forecast_on_tournament(
            "bot-testing-area",
            return_exceptions=True,
        )
    )


def main() -> None:

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s - %(name)s - "
            "%(levelname)s - %(message)s"
        ),
    )

    parser = argparse.ArgumentParser(
        description=(
            "Run the OpenRouter Metaculus forecasting bot"
        )
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "tournament",
            "metaculus_cup",
            "test_questions",
        ],
        default="tournament",
        help="What to forecast on.",
    )

    args = parser.parse_args()

    run_mode: Literal[
        "tournament",
        "metaculus_cup",
        "test_questions",
    ] = args.mode

    check_environment(strict=True)

    required = [
        "METACULUS_TOKEN",
        "OPENROUTER_API_KEY",
    ]

    missing = [
        variable
        for variable in required
        if not os.getenv(variable)
    ]

    if missing:
        raise RuntimeError(
            "Missing required environment variables: "
            + ", ".join(missing)
        )

    validate_openrouter_configuration()

    logger.info(
        "=" * 60
    )

    logger.info(
        "OpenRouter-only forecasting configuration"
    )

    logger.info(
        "Primary:  %s",
        PRIMARY_LLM,
    )

    logger.info(
        "Fallback: %s",
        FALLBACK_LLM,
    )

    logger.info(
        "Queries:  openrouter/%s",
        QUERY_MODEL,
    )

    logger.info(
        "Gemma query reasoning: OFF"
    )

    logger.info(
        "Direct OpenRouter concurrency limit: 2"
    )

    logger.info(
        "Forecasting-tools LLM concurrency limit: 2"
    )

    logger.info(
        "No VibeThinker/HuggingFace/Featherless/OpenAI "
        "LLM route is enabled."
    )

    logger.info(
        "FutureEval tournament: %s",
        FALL_FUTUREEVAL_2026_ID,
    )

    logger.info(
        "=" * 60
    )

    publish_to_metaculus = True

    print_startup_banner(
        run_mode,
        will_publish=publish_to_metaculus,
    )

    bot = create_bot()

    tournament_urls = {
        "tournament": FALL_FUTUREEVAL_2026_URL,
        "metaculus_cup": (
            "https://www.metaculus.com/tournament/"
            "metaculus-cup-fall-2026/"
        ),
        "test_questions": (
            "https://www.metaculus.com/tournament/"
            "bot-testing-area/"
        ),
    }

    logger.info(
        "Running Metaculus %s.",
        run_mode,
    )

    forecast_reports = run_forecasting(
        bot,
        run_mode,
    )

    bot.log_report_summary(
        forecast_reports
    )

    print_run_summary_banner(
        forecast_reports,
        will_publish=publish_to_metaculus,
        tournament_url=tournament_urls.get(
            run_mode
        ),
    )


if __name__ == "__main__":
    main()
