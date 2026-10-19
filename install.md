---
name: rl-env-use-install
description: Install rl-env-use into the current agent (Claude Code, Codex, Hermes, Openclaw, etc.) and set up the Browser Use and OpenRouter API keys so the user can build an environment right away.
---

# rl-env-use install

Use this file only for first-time install or reconnect. For daily use, read `SKILL.md`.

## What you're doing

You're setting up a tool that turns one line ("predict whether a clinical trial met its primary endpoint from its
registered protocol") into an RL environment built from real cases that Browser Use agents mine from the web, and then
hillclimbs a model on it. After install the user says what skill they want an environment for; you run it by reading
`SKILL.md`.

Four things must exist on this machine:

1. The `rl-env-use` repo cloned somewhere stable, installed into its own virtual environment (Python 3.11+).
2. A Browser Use API key (cloud browsers for the mining agents) and an OpenRouter API key (every LLM call: planner,
   task writer, critic, frontier panel, the models under test) in `.env` at the repo root.
3. Optionally Node.js (>= 18) on `$PATH`: it writes the HTML reports. Without it everything works and the reports are
   skipped.
4. The current agent can discover `SKILL.md` (a skills directory, or an import in its `CLAUDE.md`-equivalent).

## Install prompt contract

- Do everything yourself. Only ask the user for the two API keys.
- Prefer a stable clone path like `~/Developer/rl-env-use` (not `/tmp`, not `~/Downloads`).
- Never spend credits during install. Every check below is free.

## Steps

### 1. Clone and install

```bash
test -d ~/Developer/rl-env-use || git clone https://github.com/browser-use/rl-env-use ~/Developer/rl-env-use
cd ~/Developer/rl-env-use
command -v uv >/dev/null && uv venv && uv pip install -e '.[test]' || (python3 -m venv .venv && .venv/bin/pip install -e '.[test]')
.venv/bin/python -m pytest -q   # offline, no keys needed, ~10 s
```

If the repo is already there, `git pull --ff-only` and re-run the install line.

### 2. Register the skill

- **Claude Code**: `mkdir -p ~/.claude/skills && ln -sfn ~/Developer/rl-env-use ~/.claude/skills/rl-env-use`
- **Codex**: `mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills" && ln -sfn ~/Developer/rl-env-use "${CODEX_HOME:-$HOME/.codex}/skills/rl-env-use"`
- **Another agent**: symlink the repo into its skills directory as `rl-env-use`, or point its system prompt at
  `~/Developer/rl-env-use/SKILL.md`.

Put the CLI on the user's path so it runs from any folder: `ln -sfn ~/Developer/rl-env-use/.venv/bin/rl-env-use ~/.local/bin/rl-env-use`
(any directory on `$PATH` works).

### 3. API keys

Check what is already there and stop at the first hit for each key: an exported `BROWSER_USE_API_KEY` /
`OPENROUTER_API_KEY`, or a non-empty line in `~/Developer/rl-env-use/.env`. For each missing key, ask the user once:

> I need a Browser Use API key for the mining agents' cloud browsers - get one at https://cloud.browser-use.com - and an
> OpenRouter API key for the language models - https://openrouter.ai/keys. Paste them here and I'll write them to
> `~/Developer/rl-env-use/.env`.

```bash
cp -n .env.example .env && chmod 600 .env   # then fill in the two values; never echo them back, never commit .env
```

Free checks:

```bash
.venv/bin/python -c "from rl_env_use.llm import load_dotenv, bu_balance; load_dotenv(); print('Browser Use credits:', bu_balance())"
curl -s -o /dev/null -w '%{http_code}\n' -H "Authorization: Bearer ${OPENROUTER_API_KEY:-$(sed -n 's/^OPENROUTER_API_KEY=//p' .env)}" https://openrouter.ai/api/v1/key
```

`None` or `401` means a wrong key: ask once more.

### 4. Verify end-to-end (free)

```bash
.venv/bin/rl-env-use run envs/trial-forecast --model qwen-35b --flow models --dry-run   # prices a run from the shipped env's measurements; writes nothing
```

### 5. Hand off

Tell the user, in one short message: where it is installed, that a good first message is *"build me an RL environment
for <skill> from <where the data lives>"*, that a build costs real credits (the shipped envs cost $20-75 each; `create`
stops at its `--budget`, $40 by default), and that everything lands in `./envs/<name>/`.

## Keeping it current

`cd ~/Developer/rl-env-use && git pull --ff-only && uv pip install -e .` (or `.venv/bin/pip install -e .`).
