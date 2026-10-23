# Install the open-stack fork

Use Python 3.11+ and an installed open-source Chromium build. Node.js 18+ is
optional for HTML reports. Start from this fork's source tree or Git bundle,
not a fresh clone of upstream: upstream does not contain these changes.

```bash
uv venv
uv pip install -e '.[test]'
cp .env.example .env
```

Edit `.env` to match the model IDs and URLs exposed by your self-hosted
OpenAI-compatible server. Set `RLENV_BROWSER_EXECUTABLE` to Chromium's absolute
path. Serve a vision-capable model for `local-browser`; the writer/critic can use
a separate text model. The supplied example uses Qwen3-VL-8B-Instruct for both.
Do not ask for Browser Use, OpenRouter or Anthropic API keys for this setup.

Run vLLM in a separate environment with suitable GPU resources, for example:

```bash
vllm serve Qwen/Qwen3-VL-8B-Instruct --host 127.0.0.1 --port 8000
```

Verify the code without calling a model or website:

```bash
RLENV_OPEN_ONLY=0 .venv/bin/python -m pytest -q
RLENV_OPEN_ONLY=1 .venv/bin/python -m pytest -q tests/test_local.py
.venv/bin/rl-env-use --help
```

The first test invocation explicitly exercises upstream compatibility; it does
not enable hosted calls in your subsequent shell commands. Tests use mocks.

Follow [docs/OPEN_STACK.md](docs/OPEN_STACK.md) for usage, panel configuration,
migration, cost accounting and validation limits. The upstream `SKILL.md` is
retained as historical workflow documentation, not the setup contract for this
fork. Do not register it as an installation skill for an open-only setup.
