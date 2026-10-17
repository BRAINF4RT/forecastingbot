"""One place to choose the bot's models, with checks that catch bad spellings early.

Why this exists
---------------
Model names appear in two spellings and the bot uses both:

* the *OpenRouter id* used by the direct client (``clients/openrouter_helper.py``),
  e.g. ``nvidia/nemotron-3-ultra-550b-a55b:free`` or ``openrouter/free``;
* the *LiteLLM name* used by the forecasting framework, which is the OpenRouter id
  with one ``openrouter/`` provider prefix in front, e.g.
  ``openrouter/nvidia/nemotron-3-ultra-550b-a55b:free`` or ``openrouter/openrouter/free``.

OpenRouter's own router models (``openrouter/free``, ``openrouter/auto``) already start
with ``openrouter/``, so naive prefix stripping turns ``openrouter/free`` into ``free``.
This module converts between the spellings correctly, so you set each model once.

Choose models without editing code
----------------------------------
    FORECAST_MODELS = a,b,c    reasoning / research chain, tried in order (1 to 3 models)
    PARSER_MODELS   = a,b,c    structured-output parser chain (1 to 3 models)
    OPENROUTER_MODEL = a       legacy: overrides only the first reasoning model

Anything not set keeps the defaults hard-coded in ``clients/openrouter_helper.py`` and
``bot.py``. Use plain OpenRouter ids; ``openrouter/free`` is allowed.

This module has no heavy imports so it can be unit-tested on its own.
"""
from __future__ import annotations

import inspect
import os
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping

ROUTER_IDS = frozenset({"openrouter/free"})
CHAIN_SIZE = 3  # the bot's fallback wrapper has exactly three slots

REASONING_SOURCES = [
    "clients/openrouter_helper.py PRIMARY_MODEL",
    "clients/openrouter_helper.py FALLBACK_MODEL",
    "clients/openrouter_helper.py THIRD_MODEL",
]
PARSER_SOURCES = [
    "bot.py PARSER_PRIMARY_LLM",
    "bot.py PARSER_FALLBACK_LLM",
    "bot.py PARSER_THIRD_LLM",
]

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._+-]*(?::[A-Za-z0-9._-]+)?$")


# -- spelling conversion --------------------------------------------------------


def bare_id(model: str) -> str:
    """OpenRouter id for any spelling. Router ids keep their ``openrouter/`` prefix."""
    m = (model or "").strip()
    while m.startswith("openrouter/openrouter/"):
        m = m[len("openrouter/"):]
    if m in ROUTER_IDS:
        return m
    return m.removeprefix("openrouter/")


def litellm_name(model: str) -> str:
    """LiteLLM name: exactly one provider prefix on top of the OpenRouter id."""
    return "openrouter/" + bare_id(model)


def is_router(model: str) -> bool:
    return bare_id(model) in ROUTER_IDS


def is_free(model: str) -> bool:
    m = bare_id(model)
    return m.endswith(":free") or m == "openrouter/free"


def repair_router_typo(model: str, label: str, source: str, warnings: list[str]) -> str:
    """'free' / 'auto' can only mean OpenRouter's router models: fix them and say where they came from."""
    if model in ("free", "auto"):
        fixed = "openrouter/" + model
        warnings.append(
            f"{label} model {source} is {model!r}, which is not a model id. Using {fixed!r} instead. "
            f"Set it to {fixed!r} in the source to silence this warning."
        )
        return fixed
    return model


# -- plan ---------------------------------------------------------------------------


@dataclass
class ModelPlan:
    reasoning: list[str]  # three OpenRouter ids
    parser: list[str]  # three OpenRouter ids
    notes: list[str] = field(default_factory=list)  # where each chain came from
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def parse_chain(raw: str) -> list[str]:
    return [bare_id(part) for part in re.split(r"[,\s]+", raw or "") if part.strip()]


def pad_chain(chain: list[str], size: int = CHAIN_SIZE) -> list[str]:
    """Repeat the last model to fill every slot (useful with the random free router)."""
    return (chain + [chain[-1]] * size)[:size] if chain else chain


def defaults_from_code(helper: Any, bot_module: Any) -> tuple[list[str], list[str]]:
    """The chains currently hard-coded in the repository."""
    reasoning = [bare_id(helper.PRIMARY_MODEL), bare_id(helper.FALLBACK_MODEL), bare_id(helper.THIRD_MODEL)]
    parser = [bare_id(bot_module.PARSER_PRIMARY_LLM), bare_id(bot_module.PARSER_FALLBACK_LLM),
              bare_id(bot_module.PARSER_THIRD_LLM)]
    return reasoning, parser


def build_plan(env: Mapping[str, str], default_reasoning: list[str], default_parser: list[str]) -> ModelPlan:
    plan = ModelPlan(reasoning=list(default_reasoning), parser=list(default_parser))
    plan.notes.append("reasoning chain: defaults from clients/openrouter_helper.py")
    plan.notes.append("parser chain: defaults from bot.py")
    reasoning_src = list(REASONING_SOURCES)
    parser_src = list(PARSER_SOURCES)

    forecast_raw = (env.get("FORECAST_MODELS") or "").strip()
    legacy = (env.get("OPENROUTER_MODEL") or "").strip()
    parser_raw = (env.get("PARSER_MODELS") or "").strip()

    def from_env(var: str, raw: str) -> tuple[list[str], list[str]]:
        chain = parse_chain(raw)
        if len(chain) > CHAIN_SIZE:
            plan.warnings.append(f"{var} lists {len(chain)} models; only the first {CHAIN_SIZE} are used.")
        if len(chain) < CHAIN_SIZE:
            plan.warnings.append(f"{var} lists {len(chain)} model(s); the last is repeated to fill "
                                 f"{CHAIN_SIZE} slots.")
        chain = chain[:CHAIN_SIZE]
        return chain, [f"{var} entry {i + 1}" for i in range(len(chain))]

    if forecast_raw:
        chain, src = from_env("FORECAST_MODELS", forecast_raw)
        plan.reasoning = chain
        reasoning_src = src
        plan.notes[0] = "reasoning chain: FORECAST_MODELS"
    elif legacy:
        plan.reasoning = [bare_id(legacy)] + plan.reasoning[1:]
        reasoning_src = ["OPENROUTER_MODEL"] + reasoning_src[1:]
        plan.notes[0] = "reasoning chain: defaults, first model replaced by OPENROUTER_MODEL"

    if parser_raw:
        chain, src = from_env("PARSER_MODELS", parser_raw)
        plan.parser = chain
        parser_src = src
        plan.notes[1] = "parser chain: PARSER_MODELS"

    plan.reasoning = pad_chain([repair_router_typo(m, "reasoning", src, plan.warnings)
                                for m, src in zip(plan.reasoning, reasoning_src)])
    plan.parser = pad_chain([repair_router_typo(m, "parser", src, plan.warnings)
                             for m, src in zip(plan.parser, parser_src)])
    return plan


def validate_plan(plan: ModelPlan) -> ModelPlan:
    """Fill plan.errors (fatal) and plan.warnings (informational)."""
    for label, chain in (("reasoning", plan.reasoning), ("parser", plan.parser)):
        if not chain:
            plan.errors.append(f"The {label} chain is empty.")
            continue
        for model in dict.fromkeys(chain):  # padded slots repeat; report each model once
            if not _ID_RE.match(model):
                plan.errors.append(f"{label}: {model!r} does not look like an OpenRouter model id "
                                   "(expected 'vendor/model' or 'vendor/model:variant').")
            elif not is_free(model):
                plan.warnings.append(f"{label}: {model} is not a ':free' model and may be billed to your account.")
        if len(set(chain)) == 1 and not is_router(chain[0]):
            plan.warnings.append(f"{label}: every slot is {chain[0]}, so there is no real fallback.")
    return plan


def describe(plan: ModelPlan) -> list[str]:
    def chain_text(chain: list[str]) -> str:
        return " -> ".join(f"{m}{' (router)' if is_router(m) else ''}" for m in chain)

    return [f"Reasoning/research chain ({plan.notes[0].split(': ', 1)[1]}): {chain_text(plan.reasoning)}",
            f"Parser chain ({plan.notes[1].split(': ', 1)[1]}): {chain_text(plan.parser)}"]


# -- applying the plan --------------------------------------------------------------


def third_model_form(bot_module: Any) -> str:
    """How bot.py passes THIRD_MODEL to the fallback wrapper: 'qualified' or 'bare'.

    bot.py has historically passed it raw (``third_model=THIRD_MODEL``), which would send an
    unprefixed id to LiteLLM. If bot.py already adds the provider prefix itself, we must keep
    the bare id or the prefix would be doubled.
    """
    try:
        source = inspect.getsource(bot_module.OpenRouterForecastBot._llm_config_defaults)
    except (OSError, TypeError, AttributeError):
        return "qualified"
    return "qualified" if re.search(r"third_model\s*=\s*THIRD_MODEL\b", source) else "bare"


def apply_plan(plan: ModelPlan, *, helper: Any, bot_module: Any) -> None:
    """Point both model clients at the plan. Call before the bot is constructed."""
    r, p = plan.reasoning, plan.parser
    helper.PRIMARY_MODEL, helper.FALLBACK_MODEL, helper.THIRD_MODEL = r  # direct client: OpenRouter ids
    bot_module.PRIMARY_LLM = litellm_name(r[0])
    bot_module.FALLBACK_LLM = litellm_name(r[1])
    bot_module.THIRD_MODEL = litellm_name(r[2]) if third_model_form(bot_module) == "qualified" else r[2]
    bot_module.PARSER_PRIMARY_LLM = litellm_name(p[0])
    bot_module.PARSER_FALLBACK_LLM = litellm_name(p[1])
    bot_module.PARSER_THIRD_LLM = litellm_name(p[2])


def verify_bot_models(bot: Any) -> list[str]:
    """Inspect the names the constructed bot will really send to LiteLLM. Returns problems."""
    problems: list[str] = []
    for purpose in ("default", "summarizer", "researcher", "parser"):
        try:
            names = list(getattr(bot.get_llm(purpose, "llm"), "_models", []) or [])
        except Exception:  # noqa: BLE001 - inspection is best effort
            continue
        for name in names:
            if not isinstance(name, str):
                continue
            if name.startswith("openrouter/openrouter/openrouter/"):
                problems.append(f"{purpose}: {name!r} has a doubled provider prefix.")
            elif not name.startswith("openrouter/"):
                problems.append(f"{purpose}: {name!r} has no 'openrouter/' provider prefix, so LiteLLM "
                                "cannot route it.")
    return problems


# -- live check (--check-models) -------------------------------------------------------


@dataclass
class CheckRow:
    model: str
    direct_ok: bool
    direct_detail: str
    litellm_ok: bool
    litellm_detail: str


AsyncTry = Callable[[str], Awaitable[str]]


async def _try(fn: AsyncTry, arg: str) -> tuple[bool, str]:
    try:
        text = (await fn(arg) or "").strip()
    except Exception as exc:  # noqa: BLE001 - report every failure mode
        return False, f"{type(exc).__name__}: {str(exc)[:140]}".replace("\n", " ")
    return (True, text[:30]) if text else (False, "empty reply")


async def check_models(plan: ModelPlan, *, direct: AsyncTry, via_litellm: AsyncTry, pause: float = 1.0,
                       sleep: Callable[[float], Awaitable[None]] | None = None) -> list[CheckRow]:
    import asyncio

    sleeper = sleep or asyncio.sleep
    rows: list[CheckRow] = []
    for model in dict.fromkeys(plan.reasoning + plan.parser):
        d_ok, d_msg = await _try(direct, bare_id(model))
        l_ok, l_msg = await _try(via_litellm, litellm_name(model))
        rows.append(CheckRow(model, d_ok, d_msg, l_ok, l_msg))
        await sleeper(pause)
    return rows


def check_passed(plan: ModelPlan, rows: list[CheckRow]) -> bool:
    by_model = {row.model: row for row in rows}
    reasoning_ok = any(by_model[m].direct_ok and by_model[m].litellm_ok for m in plan.reasoning if m in by_model)
    parser_ok = any(by_model[m].litellm_ok for m in plan.parser if m in by_model)
    return reasoning_ok and parser_ok


def render_check(plan: ModelPlan, rows: list[CheckRow]) -> str:
    lines = [f"{'model':<46} {'direct client':<14} {'LiteLLM (bot)':<14}", "-" * 76]
    for row in rows:
        lines.append(f"{row.model:<46} {'OK' if row.direct_ok else 'FAIL':<14} {'OK' if row.litellm_ok else 'FAIL':<14}")
        for label, ok, detail in (("direct", row.direct_ok, row.direct_detail), ("litellm", row.litellm_ok, row.litellm_detail)):
            if not ok:
                lines.append(f"    {label}: {detail}")
    lines.append("")
    lines.append("RESULT: " + ("usable chains found for both the reasoning and parser steps."
                              if check_passed(plan, rows) else
                              "NOT usable: the reasoning chain needs one model that passes both columns and the "
                              "parser chain needs one that passes the LiteLLM column."))
    return "\n".join(lines)


def env_snapshot() -> Mapping[str, str]:
    return os.environ
