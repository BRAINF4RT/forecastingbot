"""Runtime helpers for environment checks and concise run banners."""
from __future__ import annotations

import logging
import os
import sys
import warnings
from typing import Any, Sequence

_PLACEHOLDER_ENV_VALUES = {
    "1234567890",
    "REPLACE_ME",
    "your-token-here",
    "your-api-key-here",
}


def _is_real_env(name: str) -> bool:
    value = os.getenv(name)
    return bool(value and value.strip() and value.strip() not in _PLACEHOLDER_ENV_VALUES)


def silence_noisy_dependencies() -> None:
    warnings.filterwarnings("ignore", message=r".*does not support cost tracking.*")
    logging.getLogger("forecasting_tools.ai_models.model_tracker").setLevel(logging.ERROR)
    try:
        from streamlit.logger import set_log_level
        set_log_level("error")
    except ImportError:
        pass
    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False


def check_environment(strict: bool = True) -> None:
    """Require the credentials this bot actually uses."""
    problems: list[str] = []
    if not _is_real_env("METACULUS_TOKEN"):
        problems.append(
            "METACULUS_TOKEN is missing or still a placeholder. "
            "Get one at https://www.metaculus.com/futureeval/participate/"
        )
    if not _is_real_env("OPENROUTER_API_KEY"):
        problems.append("OPENROUTER_API_KEY is missing or still a placeholder.")

    if problems:
        print("❌ Setup problems:")
        for problem in problems:
            print(f"    • {problem}")
        if strict:
            sys.exit(1)


def print_startup_banner(run_mode: str, will_publish: bool) -> None:
    publish = "publish=yes" if will_publish else "publish=no (dry run)"
    print(f"🤖 Running mode={run_mode}, {publish}\n")


def print_run_summary_banner(
    forecast_reports: Sequence[Any],
    will_publish: bool,
    tournament_url: str | None = None,
) -> None:
    from forecasting_tools import ForecastReport

    valid = [report for report in forecast_reports if isinstance(report, ForecastReport)]
    exceptions = [report for report in forecast_reports if isinstance(report, BaseException)]
    banner = "=" * 80

    print()
    print(banner)
    if not forecast_reports:
        print("ℹ️ No new questions to forecast on this run.")
        print(banner)
        print()
        return

    if valid and not exceptions:
        verb = "submitted" if will_publish else "produced (dry run)"
        print(f"🎉 Bot {verb} {len(valid)} forecast(s).")
    elif valid and exceptions:
        print(f"⚠️ Partial — {len(valid)} succeeded, {len(exceptions)} failed.")
    else:
        print(f"❌ All {len(exceptions)} attempt(s) failed.")

    if valid:
        print()
        for report in valid:
            note = f"  (with {len(report.errors)} minor error(s))" if report.errors else ""
            print(f"  ✅ {report.question.page_url}{note}")
        if will_publish and tournament_url:
            print(f"\n  Tournament: {tournament_url}")

    if exceptions:
        print()
        for exc in exceptions:
            message = str(exc)
            if len(message) > 200:
                message = message[:200] + "..."
            print(f"  ❌ {type(exc).__name__}: {message}")
    print(banner)
    print()
