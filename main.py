"""
Entry point for the VibeThinker Metaculus bot.

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

from bot import VibeThinkerForecastBot  # noqa: E402


dotenv.load_dotenv()

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Run the VibeThinker Metaculus bot"
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
        help="What to forecast on (default: tournament)",
    )

    args = parser.parse_args()

    run_mode: Literal[
        "tournament",
        "metaculus_cup",
        "test_questions",
    ] = args.mode

    check_environment(strict=True)

    # This bot specifically requires:
    #
    #   HF_TOKEN
    #   OPENROUTER_API_KEY
    #
    # because the main forecasting brain and helper models use
    # separate providers.

    extra_required = [
        "HF_TOKEN",
        "OPENROUTER_API_KEY",
    ]

    missing = [
        var
        for var in extra_required
        if not os.getenv(var)
    ]

    if missing:
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing)}. "
            "Copy .env.template to .env and fill these in."
        )

    publish_to_metaculus = True

    print_startup_banner(
        run_mode,
        will_publish=publish_to_metaculus,
    )

    # OpenRouter parser model.
    #
    # This is deliberately configurable through the environment, but defaults
    # to the same free Nemotron model used by the other OpenRouter helper tasks.

    parser_model_name = os.getenv(
        "OPENROUTER_PARSER_MODEL",
        "nvidia/nemotron-3-ultra-550b-a55b:free",
    )

    bot = VibeThinkerForecastBot(
        research_reports_per_question=1,
        predictions_per_research_report=3,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=publish_to_metaculus,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
        llms={
            # The main forecasting brain is intentionally NOT supplied here.
            #
            # Forecast reasoning is generated directly through:
            #
            # clients/hf_vibethinker.py
            #
            # using VibeThinker-3B through Hugging Face / Featherless.

            "parser": GeneralLlm(
                model=f"openrouter/{parser_model_name}",
                temperature=0.0,
                timeout=60,
                allowed_tries=2,
            ),
        },
    )

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

        forecast_reports = seasonal_reports + minibench_reports

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
        tournament_url=TOURNAMENT_URLS.get(run_mode),
    )
