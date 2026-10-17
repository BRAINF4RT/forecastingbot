"""Entry point for the OpenRouter Metaculus forecasting bot."""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from typing import Literal

import dotenv

from bot_helpers import (
    check_environment,
    print_run_summary_banner,
    print_startup_banner,
    silence_noisy_dependencies,
)

silence_noisy_dependencies()

import bot as bot_module  # noqa: E402
import model_config  # noqa: E402
from bot import OpenRouterForecastBot  # noqa: E402
from clients import openrouter_helper  # noqa: E402
from forecasting_tools import MetaculusClient  # noqa: E402

dotenv.load_dotenv()

logger = logging.getLogger(__name__)

FALL_FUTUREEVAL_2026_ID = "fall-futureeval-2026"
FALL_FUTUREEVAL_2026_URL = (
    "https://www.metaculus.com/tournament/fall-futureeval-2026/"
)


# --------------------------------------------------------------------------- models
#
# Models are chosen in ONE place. Either leave the defaults in
# clients/openrouter_helper.py (reasoning chain) and bot.py (parser chain), or set
#
#     FORECAST_MODELS=nvidia/nemotron-3-ultra-550b-a55b:free,poolside/laguna-s-2.1:free,openrouter/free
#     PARSER_MODELS=google/gemma-4-31b-it:free,openrouter/free
#
# in .env or the workflow. Use plain OpenRouter ids; "openrouter/free" is allowed.
# model_config.py converts them to the spelling each client needs, so there is no list of
# "expected" models to keep in sync any more.


def configure_models() -> model_config.ModelPlan:
    """Resolve, validate and apply the model plan. Raises only on unusable settings."""
    if not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is not set.")

    default_reasoning, default_parser = model_config.defaults_from_code(openrouter_helper, bot_module)
    plan = model_config.build_plan(os.environ, default_reasoning, default_parser)
    model_config.validate_plan(plan)

    if plan.errors:
        raise RuntimeError(
            "Model configuration problem(s):\n  - " + "\n  - ".join(plan.errors)
        )
    for warning in plan.warnings:
        logger.warning("Model configuration: %s", warning)

    legacy = os.getenv("OPENROUTER_MODEL")
    if legacy and not os.getenv("FORECAST_MODELS"):
        logger.info(
            "OPENROUTER_MODEL is set (%s): it now overrides only the first reasoning model. "
            "Use FORECAST_MODELS to set the whole chain.",
            legacy,
        )

    model_config.apply_plan(plan, helper=openrouter_helper, bot_module=bot_module)

    for line in model_config.describe(plan):
        logger.info(line)
    logger.info(
        "Search query generation: original question is always query #1; "
        "query planner: %s",
        getattr(openrouter_helper, "QUERY_PLANNER_MODEL", "n/a"),
    )
    logger.info(
        "Scraper chain: HTTP -> Trafilatura -> BeautifulSoup -> DDGS indexed snippet."
    )
    logger.info("Metaculus URLs are not filtered from research results.")
    return plan


async def _check_models_async(plan: model_config.ModelPlan) -> int:
    """Send one tiny request per configured model through BOTH clients the bot uses."""
    from forecasting_tools import GeneralLlm

    prompt = "Reply with exactly: OK"

    async def direct(model: str) -> str:
        return await openrouter_helper._generate_with_model(
            prompt, model=model, temperature=0.0, max_tokens=400, timeout=90.0, max_retries=1
        )

    async def via_litellm(model: str) -> str:
        return await GeneralLlm(model=model, temperature=0.0, timeout=90).invoke(prompt)

    print("Checking each configured model (one tiny request per client)...\n")
    rows = await model_config.check_models(plan, direct=direct, via_litellm=via_litellm)
    print(model_config.render_check(plan, rows))
    return 0 if model_config.check_passed(plan, rows) else 1


def create_bot(
    *,
    publish_reports_to_metaculus: bool,
) -> OpenRouterForecastBot:
    bot = OpenRouterForecastBot(
        research_reports_per_question=1,
        predictions_per_research_report=3,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=publish_reports_to_metaculus,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
    )
    problems = model_config.verify_bot_models(bot)
    if problems:
        raise RuntimeError(
            "The bot would send badly formed model names to LiteLLM:\n  - "
            + "\n  - ".join(problems)
        )
    return bot


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


def _all_failed(reports: list) -> bool:
    """True when questions were attempted and every one ended in an exception."""
    return bool(reports) and all(isinstance(report, BaseException) for report in reports)


def build_parser() -> argparse.ArgumentParser:
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
    parser.add_argument(
        "--check-models",
        action="store_true",
        help=(
            "Send one tiny request to every configured model through both clients, "
            "print which work, and exit. Needs only OPENROUTER_API_KEY."
        ),
    )
    parser.add_argument(
        "--no-fail",
        action="store_true",
        help="Exit 0 even if every question failed (default: exit 1 in that case).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    args = build_parser().parse_args(argv)

    if args.check_models:
        plan = configure_models()
        return asyncio.run(_check_models_async(plan))

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

    configure_models()

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

    if _all_failed(reports) and not args.no_fail:
        logger.error(
            "Every one of the %d question(s) failed; exiting with status 1. "
            "Run `python main.py --check-models` to test your models.",
            len(reports),
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
