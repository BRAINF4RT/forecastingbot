"""
Entry point for the OpenRouter-only Metaculus forecasting bot.

LLM routing:

    Primary:
        nvidia/nemotron-3-ultra-550b-a55b:free

    Fallback:
        poolside/laguna-s-2.1:free

Usage:

    python main.py --mode tournament

    python main.py --mode metaculus_cup

    python main.py --mode test_questions
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


# ============================================================================
# STARTUP
# ============================================================================

silence_noisy_dependencies()

from forecasting_tools import MetaculusClient  # noqa: E402

from bot import (  # noqa: E402
    FALLBACK_MODEL,
    PRIMARY_MODEL,
    OpenRouterForecastBot,
)


dotenv.load_dotenv()

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

OPENROUTER_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"

# The bot is intentionally restricted to these two free OpenRouter models.
EXPECTED_PRIMARY_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
EXPECTED_FALLBACK_MODEL = "poolside/laguna-s-2.1:free"


TOURNAMENT_URLS = {
    "tournament": (
        "https://www.metaculus.com/tournament/"
        "summer-futureeval-2026/"
    ),
    "metaculus_cup": (
        "https://www.metaculus.com/tournament/"
        "metaculus-cup-summer-2025/"
    ),
    "test_questions": (
        "https://www.metaculus.com/tournament/"
        "bot-testing-area/"
    ),
}


# ============================================================================
# ENVIRONMENT VALIDATION
# ============================================================================


def validate_openrouter_configuration() -> None:
    """
    Validate the OpenRouter-only model configuration.

    This deliberately fails early if someone accidentally configures the
    repository to use another model.
    """

    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not configured."
        )

    configured_model = os.getenv(
        "OPENROUTER_MODEL",
        OPENROUTER_MODEL,
    )

    if configured_model != EXPECTED_PRIMARY_MODEL:
        raise RuntimeError(
            "This bot is intentionally locked to the free Nemotron "
            "3 Ultra OpenRouter model.\n\n"
            f"Expected:\n"
            f"  {EXPECTED_PRIMARY_MODEL}\n\n"
            f"Configured:\n"
            f"  {configured_model}"
        )

    if PRIMARY_MODEL != f"openrouter/{EXPECTED_PRIMARY_MODEL}":
        raise RuntimeError(
            "Internal primary model configuration is incorrect.\n"
            f"Expected: openrouter/{EXPECTED_PRIMARY_MODEL}\n"
            f"Found: {PRIMARY_MODEL}"
        )

    if FALLBACK_MODEL != f"openrouter/{EXPECTED_FALLBACK_MODEL}":
        raise RuntimeError(
            "Internal fallback model configuration is incorrect.\n"
            f"Expected: openrouter/{EXPECTED_FALLBACK_MODEL}\n"
            f"Found: {FALLBACK_MODEL}"
        )

    logger.info(
        "OpenRouter configuration validated."
    )

    logger.info(
        "Primary LLM: %s",
        PRIMARY_MODEL,
    )

    logger.info(
        "Fallback LLM: %s",
        FALLBACK_MODEL,
    )


# ============================================================================
# ARGUMENTS
# ============================================================================


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the OpenRouter-only Metaculus forecasting bot "
            "using Nemotron 3 Ultra with Laguna S 2.1 fallback."
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

    return parser.parse_args()


# ============================================================================
# BOT CONSTRUCTION
# ============================================================================


def create_bot() -> OpenRouterForecastBot:
    """
    Create the forecasting bot.

    We intentionally DO NOT pass an `llms=` dictionary here.

    OpenRouterForecastBot._llm_config_defaults() provides all four purposes:

        default
        summarizer
        researcher
        parser

    Each purpose uses:

        Nemotron -> Laguna
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


# ============================================================================
# FORECAST DISPATCH
# ============================================================================


def run_forecasting(
    bot: OpenRouterForecastBot,
    run_mode: Literal[
        "tournament",
        "metaculus_cup",
        "test_questions",
    ],
) -> list:
    """
    Dispatch the bot to the selected tournament.
    """

    client = MetaculusClient()

    if run_mode == "tournament":
        logger.info(
            "Running primary AI competition tournament."
        )

        seasonal_reports = asyncio.run(
            bot.forecast_on_tournament(
                client.CURRENT_AI_COMPETITION_ID,
                return_exceptions=True,
            )
        )

        logger.info(
            "Running MiniBench."
        )

        minibench_reports = asyncio.run(
            bot.forecast_on_tournament(
                client.CURRENT_MINIBENCH_ID,
                return_exceptions=True,
            )
        )

        return seasonal_reports + minibench_reports

    if run_mode == "metaculus_cup":
        logger.info(
            "Running Metaculus Cup."
        )

        bot.skip_previously_forecasted_questions = False

        return asyncio.run(
            bot.forecast_on_tournament(
                client.CURRENT_METACULUS_CUP_ID,
                return_exceptions=True,
            )
        )

    logger.info(
        "Running Metaculus bot-testing-area."
    )

    bot.skip_previously_forecasted_questions = False

    return asyncio.run(
        bot.forecast_on_tournament(
            "bot-testing-area",
            return_exceptions=True,
        )
    )


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s - %(name)s - "
            "%(levelname)s - %(message)s"
        ),
    )

    args = parse_arguments()

    run_mode: Literal[
        "tournament",
        "metaculus_cup",
        "test_questions",
    ] = args.mode

    # Standard forecasting-tools environment validation.
    check_environment(strict=True)

    # Our own stricter validation.
    validate_openrouter_configuration()

    publish_to_metaculus = True

    print_startup_banner(
        run_mode,
        will_publish=publish_to_metaculus,
    )

    logger.info(
        "============================================================"
    )

    logger.info(
        "OpenRouter-only forecasting configuration"
    )

    logger.info(
        "Primary:  %s",
        PRIMARY_MODEL,
    )

    logger.info(
        "Fallback: %s",
        FALLBACK_MODEL,
    )

    logger.info(
        "No VibeThinker/HuggingFace/Featherless/OpenAI "
        "LLM fallback is enabled."
    )

    logger.info(
        "============================================================"
    )

    bot = create_bot()

    forecast_reports = run_forecasting(
        bot,
        run_mode,
    )

    bot.log_report_summary(
        forecast_reports,
    )

    print_run_summary_banner(
        forecast_reports,
        will_publish=publish_to_metaculus,
        tournament_url=TOURNAMENT_URLS.get(run_mode),
    )


if __name__ == "__main__":
    main()

