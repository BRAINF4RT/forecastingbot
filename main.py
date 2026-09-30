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
    PARSER_THIRD_LLM,
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
EXPECTED_PARSER_THIRD_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"


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
        "parser third fallback": _bare_model(PARSER_THIRD_LLM),
    }

    expected = {
        "primary": EXPECTED_PRIMARY_MODEL,
        "fallback": EXPECTED_FALLBACK_MODEL,
        "third fallback": EXPECTED_THIRD_MODEL,
        "parser primary": EXPECTED_PARSER_PRIMARY_MODEL,
        "parser fallback": EXPECTED_PARSER_FALLBACK_MODEL,
        "parser third fallback": EXPECTED_PARSER_THIRD_MODEL,
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
        "Parser chain: %s -> %s -> %s",
        PARSER_PRIMARY_LLM,
        PARSER_FALLBACK_LLM,
        PARSER_THIRD_LLM,
    )

    logger.info(
        "Search query generation: deterministic; original question always included."
    )

    logger.info(
        "Scraper chain: HTTP -> Trafilatura -> BeautifulSoup -> DDGS indexed snippet."
    )

    logger.info("Metaculus URLs are not filtered from research results.")


def create_bot(
    *,
    publish_reports_to_metaculus: bool,
) -> OpenRouterForecastBot:
    return OpenRouterForecastBot(
        research_reports_per_question=1,
        predictions_per_research_report=3,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=publish_reports_to_metaculus,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
    )


# Order in which question types are run in test mode. Types not listed here
# (for example ones added by a newer forecasting-tools) are still tested, after
# these, in the order the API returned them.
_TEST_TYPE_ORDER = (
    "BinaryQuestion",
    "MultipleChoiceQuestion",
    "NumericQuestion",
    "DiscreteQuestion",
    "DateQuestion",
    "ConditionalQuestion",
)


def _pick_questions_per_type(questions: list, per_type: int = 1) -> list:
    """Return up to `per_type` questions of each concrete question type.

    Grouping uses the exact class name, so DiscreteQuestion (a subclass of
    NumericQuestion) counts as its own type. Within a type the API's order is
    kept, so the pick is deterministic for a given tournament state.
    """
    by_type: dict[str, list] = {}
    for question in questions:
        by_type.setdefault(type(question).__name__, []).append(question)

    known = [name for name in _TEST_TYPE_ORDER if name in by_type]
    extra = [name for name in by_type if name not in _TEST_TYPE_ORDER]

    picked: list = []
    for name in known + extra:
        picked.extend(by_type[name][:per_type])
    return picked


async def _run_forecasting_async(
    bot: OpenRouterForecastBot,
    run_mode: Literal[
        "tournament",
        "metaculus_cup",
        "test_questions",
    ],
    limit: int = 1,
) -> list:
    client = MetaculusClient()

    if run_mode == "tournament":
        seasonal = await bot.forecast_on_tournament(
            FALL_FUTUREEVAL_2026_ID,
            return_exceptions=True,
        )

        minibench = await bot.forecast_on_tournament(
            client.CURRENT_MINIBENCH_ID,
            return_exceptions=True,
        )

        return seasonal + minibench

    if run_mode == "metaculus_cup":
        bot.skip_previously_forecasted_questions = False

        return await bot.forecast_on_tournament(
            "metaculus-cup-fall-2026",
            return_exceptions=True,
        )

    bot.skip_previously_forecasted_questions = False

    # The official bot-testing-area contains examples of every supported
    # question type. Test mode forecasts ONE question of each type present
    # (or `limit` per type), without publishing, so CI still exercises the
    # binary, multiple-choice, numeric, discrete, date and conditional paths
    # without forecasting the whole area.
    questions = client.get_all_open_questions_from_tournament("bot-testing-area")
    if not questions:
        logger.warning("No open questions found in bot-testing-area.")
        return []

    chosen = _pick_questions_per_type(questions, per_type=max(1, limit))
    logger.info(
        "Test mode: forecasting %d of %d open question(s) (up to %d per type): %s",
        len(chosen),
        len(questions),
        max(1, limit),
        [
            f"{type(q).__name__}: {getattr(q, 'page_url', None) or q.question_text[:60]}"
            for q in chosen
        ],
    )
    return await bot.forecast_questions(
        chosen,
        return_exceptions=True,
    )


def run_forecasting(
    bot: OpenRouterForecastBot,
    run_mode: Literal[
        "tournament",
        "metaculus_cup",
        "test_questions",
    ],
    limit: int = 1,
) -> list:
    """Run the selected mode inside exactly one asyncio event loop."""
    return asyncio.run(
        _run_forecasting_async(
            bot,
            run_mode,
            limit,
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
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help=(
            "test_questions mode only: how many bot-testing-area questions of "
            "EACH TYPE to forecast (default 1)."
        ),
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

    publish = args.mode != "test_questions"

    logger.info(
        "Selected mode: %s",
        args.mode,
    )

    logger.info(
        "Publishing enabled: %s",
        publish,
    )

    logger.info(
        "Research backend: %s (original question is always query #1)",
        os.getenv("RESEARCH_BACKEND", "free_searcher"),
    )

    logger.info(
        "Metaculus source URLs: ALLOWED"
    )

    print_startup_banner(
        args.mode,
        will_publish=publish,
    )

    bot = create_bot(
        publish_reports_to_metaculus=publish,
    )

    reports = run_forecasting(
        bot,
        args.mode,
        args.limit,
    )

    bot.log_report_summary(reports)

    tournament_urls = {
        "tournament": FALL_FUTUREEVAL_2026_URL,
        "test_questions": (
            "https://www.metaculus.com/tournament/bot-testing-area/"
        ),
    }

    print_run_summary_banner(
        reports,
        will_publish=publish,
        tournament_url=tournament_urls.get(args.mode),
    )


if __name__ == "__main__":
    main()
