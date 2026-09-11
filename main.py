"""
Entry point for the OpenRouter Nemotron Metaculus bot.

Usage:

    python main.py --mode tournament
    python main.py --mode test_questions
    python main.py --mode metaculus_cup
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

from forecasting_tools import GeneralLlm, MetaculusClient  # noqa: E402
from bot import OpenRouterForecastBot  # noqa: E402

dotenv.load_dotenv()

logger = logging.getLogger(__name__)

OPENROUTER_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Run the Nemotron Metaculus forecasting bot"
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

    configured_model = os.getenv(
        "OPENROUTER_MODEL",
        OPENROUTER_MODEL,
    )

    if configured_model != OPENROUTER_MODEL:
        raise RuntimeError(
            "This bot is intentionally locked to the free OpenRouter "
            f"model '{OPENROUTER_MODEL}'. "
            f"OPENROUTER_MODEL is currently '{configured_model}'."
        )

    publish_to_metaculus = True

    print_startup_banner(
        run_mode,
        will_publish=publish_to_metaculus,
    )

    bot = OpenRouterForecastBot(
        research_reports_per_question=1,
        predictions_per_research_report=3,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=publish_to_metaculus,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
        llms={
            "parser": GeneralLlm(
                model=f"openrouter/{OPENROUTER_MODEL}",
                temperature=0.0,
                timeout=180,
                allowed_tries=3,
            ),
        },
    )

    tournament_urls = {
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

    client = MetaculusClient()

    if run_mode == "tournament":

        seasonal_reports = asyncio.run(
            bot.forecast_on_tournament(
                client.CURRENT_AI_COMPETITION_ID,
                return_exceptions=True,
            )
        )

        minibench_reports = asyncio.run(
            bot.forecast_on_tournament(
                client.CURRENT_MINIBENCH_ID,
                return_exceptions=True,
            )
        )

        forecast_reports = (
            seasonal_reports + minibench_reports
        )

    elif run_mode == "metaculus_cup":

        bot.skip_previously_forecasted_questions = False

        forecast_reports = asyncio.run(
            bot.forecast_on_tournament(
                client.CURRENT_METACULUS_CUP_ID,
                return_exceptions=True,
            )
        )

    else:

        bot.skip_previously_forecasted_questions = False

        forecast_reports = asyncio.run(
            bot.forecast_on_tournament(
                "bot-testing-area",
                return_exceptions=True,
            )
        )

    bot.log_report_summary(forecast_reports)

    print_run_summary_banner(
        forecast_reports,
        will_publish=publish_to_metaculus,
        tournament_url=tournament_urls.get(run_mode),
    )
