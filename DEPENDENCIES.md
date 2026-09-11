# Adding this bot's dependencies to the template's pyproject.toml

I don't have the exact contents of `Metaculus/metac-bot-template`'s
`pyproject.toml` / `poetry.lock` (they change over time, and `poetry.lock`
is a generated file that must match your installed `forecasting-tools`
version exactly). Rather than guess and hand you a `pyproject.toml` that
might silently downgrade a dependency or drift from the lock file, run this
in your forked repo instead:

```bash
cd metac-bot-template   # your fork, with these new files already added
poetry add httpx trafilatura beautifulsoup4 ddgs
```

That adds the four extra dependencies this bot needs on top of everything
the template already installs (`forecasting-tools`, `python-dotenv`, etc.)
and regenerates `poetry.lock` for you correctly.

Then verify:

```bash
poetry install
poetry run python main.py --mode test_questions
```

Delete `requirements.txt` from this bundle if you're using Poetry (the
template does) -- it was only included in case you preferred pip.
