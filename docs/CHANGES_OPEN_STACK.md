# Open-stack changes

Version: `0.1.0+open.1`
Branch: `open-stack`
Upstream: https://github.com/browser-use/rl-env-use
Base commit: `92fe423137c01178692bf0a26c769b20438f9c5f`
License: upstream MIT license retained.

## Resulting behavior

The normal create/build/run/climb workflow now defaults to self-hosted models
and local browser execution. Hosted aliases, provider routes and the Claude
comparison subprocess are rejected in open-only mode. Existing hosted workflows
are available only through explicit `RLENV_OPEN_ONLY=0` compatibility mode.

For example, previously `create` selected `bu` for mining, Claude for task
writing and a Claude/GPT/Gemini panel. It now selects `local-browser`,
`local-text` and the configurable `RLENV_PANEL_MODELS` panel. The minimal panel
is one local model; it must be expanded for independent evaluation.

## Implementation

| Files | Change |
| --- | --- |
| `config.py`, package initialization | Load `.env` before browser-use imports; define mode-dependent defaults; disable Browser Use telemetry, cloud sync and remote cost lookup. |
| `local.py`, `llm.py` | Add a self-hosted compatible transport for both structured text and screenshot-based agent calls; preserve schema validation and token usage; qualify model IDs for local cost accounting; reject hosted routing and skip billing. |
| `browse.py` | Force local sessions in open-only mode and support an explicit Chromium executable. |
| CLI, pipeline, scout, tasks, verify, climb, Env, judges, rollout | Replace implicit hosted role defaults, configure the panel and adjudicators, reject legacy hosted specs, and use the local-call timeout for browser agents. |
| CLI, `vanilla.py` | Block proprietary Claude comparison calls in open-only mode. |
| `.env.example`, README, install guide, this directory | Document local setup, migration, panel limitations, costs and publishing. |
| `tests/test_local.py` | Add 16 offline checks for transport and open-only behavior. |

No weights, browser binary, inference server, credentials or virtual environment
are distributed with the fork. Upstream's dependency pin `browser-use==0.13.10`
is retained. The upstream `SKILL.md` and example results remain historical
material and are explicitly identified as such in the setup guide.

## Validation

- Compatibility suite: **486 passed, 3 live tests deselected** (470 upstream
  tests plus 16 new checks), with `RLENV_OPEN_ONLY=0`.
- Open-only tests: **16 passed**, with `RLENV_OPEN_ONLY=1`.
- Ruff source/new-test lint and formatting checks passed.
- `git diff --check` passed.
- Actual BrowserSession and Agent construction with the local adapter passed;
  no browser was launched and no inference request was made. The construction
  check removed this execution environment's SOCKS proxy variables because its
  optional `socksio` dependency was not installed.

No real model-server/GPU inference or live website mining run was performed.
The transport tests use the actual SDK serializer with mocked HTTP responses.
Model quality, memory requirements, schema support and website access must be
validated on the deployment hardware.

## Portable handoff and GitHub publishing

The delivery ZIP contains this source tree, an incremental patch and a Git
bundle with history. It excludes local test output, temporary agent files,
virtual environments and secrets. To restore the Git checkout:

```bash
git clone --branch open-stack rl-env-use-open-stack.bundle rl-env-use
cd rl-env-use
git remote rename origin delivery
git remote add upstream https://github.com/browser-use/rl-env-use.git
```

The GitHub fork is [versae/rl-env-use](https://github.com/versae/rl-env-use).
The implementation is prepared on the local `open-stack` branch. Publishing is
currently blocked by the connected integration's write permissions. After the
branch is published, clone it directly:

```bash
git clone --branch open-stack https://github.com/versae/rl-env-use.git
cd rl-env-use
git remote add upstream https://github.com/browser-use/rl-env-use.git
```

When restoring the earlier portable bundle, add the fork as the publish remote:

```bash
git remote add origin https://github.com/versae/rl-env-use.git
```

The original delivery bundle records the initial preparation commit. The
published branch also includes these updated fork instructions. The patch can
be applied to a checkout of the recorded base commit using `git apply`.
