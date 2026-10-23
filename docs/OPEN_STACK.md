# Self-hosted open-weight setup

This fork adds an open-only mode to browser-use/rl-env-use. It is enabled by
default, including when no `.env` exists. Python 3.11+ is required. Node.js 18+
is optional for HTML reports. No Claude Code, OpenRouter account, Browser Use
Cloud subscription, or proprietary model API is needed for the normal workflow.

## Install and start a model

From this fork's working tree:

```bash
uv venv
uv pip install -e '.[test]'
cp .env.example .env
```

Run a self-hosted inference server in a **separate environment**. For example,
with a compatible vLLM installation and enough GPU memory:

```bash
vllm serve Qwen/Qwen3-VL-8B-Instruct --host 127.0.0.1 --port 8000
```

The example model has Apache-2.0 weights and vision support. It is a starting
point for integration, not a benchmarked recommendation for reliable mining.
Choose model size, quantization, context limits and concurrency for your hardware.
An 8B vision model can require considerably more memory than its weight size
once context, image tokens and parallel requests are included.

Install an open-source Chromium build using your operating system's package
manager, then set `RLENV_BROWSER_EXECUTABLE` to its absolute path in `.env`.
Otherwise browser-use uses its own browser discovery; that can select a
proprietary Chrome installation already on your machine.

The inference server must accept `/v1/chat/completions`, image content for browser
agents when vision is enabled, and `response_format.type=json_schema`. An OpenAI-compatible API is a
wire protocol: the requests go to your server, not OpenAI. Ollama's compatible
endpoint may work with an appropriate model/version, but this fork does not add
or test a separate native Ollama transport.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `RLENV_OPEN_ONLY` | `1` | Reject hosted model routes and disable the Claude comparison command. Only `0` or `1` is accepted. |
| `RLENV_LOCAL_BASE_URL` | `http://127.0.0.1:8000/v1` | Your OpenAI-compatible server, including its `/v1` prefix. |
| `RLENV_LOCAL_TEXT_MODEL` | Required for `local-text` | Planner, writer, critic, closed-book checks, proposer and case-policy default. |
| `RLENV_LOCAL_BROWSER_MODEL` | Required for `local-browser` | Vision-capable browser-agent default. |
| `RLENV_LOCAL_BROWSER_BASE_URL` | Shared base URL | Optional separate endpoint, used by the `local-browser` alias only. |
| `RLENV_LOCAL_API_KEY` | Dummy local token | Optional authentication token for your own server; shared by both endpoints. |
| `RLENV_LOCAL_MAX_TOKENS` | `8192` | Maximum completion tokens per call. |
| `RLENV_LOCAL_TIMEOUT` | `600` seconds | Per-call server timeout and browser-agent LLM timeout. |
| `RLENV_PANEL_MODELS` | `local-text` | Comma-separated local aliases or `local:MODEL_ID` entries. |
| `RLENV_BROWSER_USE_VISION` | `1` | Set to `0` for a text-only browser model; navigate with DOM text instead of screenshot input. |
| `RLENV_BROWSER_EXECUTABLE` | Browser discovery | Absolute path to your Chromium executable. |

The supplied `.env.example` sets both model aliases to Qwen3-VL-8B-Instruct,
allowing a one-server smoke test. For real dataset generation, use a stronger
writer/critic and multiple independently trained panel models. A single-model
panel is not independent validation, particularly if that model wrote the tasks.

`local:MODEL_ID` sends the exact model ID to the shared base URL. All panel models
must be available there (directly or through your own open-source router).
An explicit `local:` ID does not select the separate browser endpoint: use
`local-browser` for that. The code cannot attest that an endpoint is self-hosted
or that its model has open weights. You control the endpoint and must verify
the selected model's license.

Configuration loads before browser-use is imported: exported variables override
the working directory's `.env`, which overrides this repository's `.env`.
Change configuration before starting Python; restart a running Python process
after edits because workflow defaults are bound at import time.

## Run the workflow

```bash
# Print a proposed environment spec; this calls your local text model.
.venv/bin/rl-env-use create \
  'compute financial ratios from public annual reports' \
  --name financial-ratios --plan-only

# Build a small first environment. The one-model panel is only a smoke-test setup.
.venv/bin/rl-env-use create \
  'compute financial ratios from public annual reports' \
  --name financial-ratios --tasks 12 --missions 2

.venv/bin/rl-env-use run envs/financial-ratios --model local-text --verified
.venv/bin/rl-env-use climb envs/financial-ratios \
  --model local-text --proposer local-text --verified
.venv/bin/rl-env-use export envs/financial-ratios
```

The climb can refuse an underpowered dataset. Twelve tasks are only an initial
integration check; build enough independent cases for meaningful train/test
estimates. `climb` optimizes a system prompt; it does not update model weights.
Actual RL training requires a separate trainer consuming the exported data.

Local usage retains token counts and qualified model IDs (`local:...`). API spend
is recorded as zero/unknown; it excludes GPU rental, hardware and electricity.
Consequently `--budget` is **not a compute cap** for local inference. Control
`--tasks`, `--missions`, climb `--rounds`, `--reps`, `--concurrency`, and the
environment spec's `max_rounds` (default six for creation).

## Separate writers and critics

`create` and `build` accept `--writer` (an alias for the existing `--model`) and
`--critic`. The writer handles planning, synthesis and hardening; the critic
handles evidence support, ambiguity checks and blind answer recomputation.
Mining, closed-book checks, difficulty panels and adjudicators retain their own
model settings.

```bash
# Separate models; both IDs must be served by the shared local endpoint/router.
rl-env-use create 'your skill and sources' --name my-env \
  --writer local:writer-id --critic local:critic-id
rl-env-use build envs/my-env --only verify --critic local:reviewer-id
```

If only one role flag is supplied, that model fills both roles. With neither
flag, commands reuse `writer_model` and `critic_model` saved in `spec.json`,
falling back to the configured text default for new or legacy environments.
Supplying either role flag replaces the saved pair using the same fallback rule.
`--plan-only` prints the resolved roles without saving them. Python callers can
use `build(..., model=writer, critic=critic)` or
`create(..., model=writer, critic=critic)` with the same semantics.

A task records its reviewer in `checks.critic_model`. Verification repeats the
critic review when that identity changes (or is absent in legacy tasks), while
retaining cached live-evidence and closed-book checks. Changing weights behind
the same alias still requires deliberate cache invalidation. This split adds
no dedicated Norwegian language-quality grader.

## What open-only mode enforces

- The supported model factory accepts only `local-text`, `local-browser` and
  `local:MODEL_ID`. Hosted aliases and explicit hosted routes raise an error,
  even when provider keys are present. Retries never switch to a hosted route.
- The browser factory always sets `use_cloud=False`, even with a Browser Use key.
- Browser Use telemetry, cloud synchronization and remote cost lookup are
  disabled before browser-use initializes. Browser agents do not calculate
  hosted token prices. Browser Use billing requests are skipped.
- The CLI's Claude comparison command and its low-level Claude subprocess
  entry point are blocked. Use `run --flow models --model local:MODEL_ID` for
  comparisons between your own policies.

This is application routing, not an OS network sandbox. Web mining still reaches
the sites being researched; model/browser downloads and dependency installation
also need network access. Restrict process egress separately if you need a
network-enforced guarantee. Hosted SDK packages remain transitive dependencies
of the pinned open-source browser-use package; open-only mode does not use their
hosted services.

## Migrate existing environments

Work on a copy of the environment so existing calibration and run history stay
interpretable. In `spec.json`, replace hosted entries in:

```json
{
  "agent_model": "local-browser",
  "writer_model": "local-text",
  "critic_model": "local-text",
  "closed_book": ["local-text"],
  "panel": ["local:your-open-model-a", "local:your-open-model-b"],
  "adjudicators": ["local:your-open-model-b", "local:your-open-model-a"]
}
```

The IDs above are placeholders for models your server actually exposes. Rebuild
the verification checks and panel in the copied environment, and recalibrate any
stored judges that name hosted models. Existing closed-book results may otherwise
remain cached; clear those checks when intentionally repeating them. Use new
run/climb directories after changing the actual model behind an alias: the
resumption cache does not fingerprint model weights or endpoint configuration.

Stored hosted entries are rejected rather than silently remapped. Replacing the
panel changes the benchmark's difficulty definition; its results do not establish
that the original frontier panel would fail the new tasks.

## Compatibility mode and historical documentation

Set `RLENV_OPEN_ONLY=0` **before starting Python** to restore upstream hosted
defaults and comparison commands. This permits paid/proprietary calls and is not
the all-open setup. The upstream README retained below this fork's introduction
and shipped example environments describe the original hosted workflow. Use
`install.md` for setup and [SKILL.md](../SKILL.md) for the agent's open-only daily
workflow. Do not follow the historical README's hosted API-key setup instructions
for an open-only installation; migrate shipped environments before running them.

## Verification and practical limits

Run the existing compatibility suite and the new strict-mode checks separately:

```bash
RLENV_OPEN_ONLY=0 .venv/bin/python -m pytest -q
RLENV_OPEN_ONLY=1 .venv/bin/python -m pytest -q tests/test_local.py
```

The strict tests exercise the actual OpenAI-compatible serializer with an HTTP
mock, including screenshot content, JSON schemas, response parsing and usage.
They also check route rejection, local retries, local browser selection, early
configuration loading and disabled billing/Claude calls. They do not prove model
quality or website accessibility. No real GPU/model-server/browser mining run was
performed while preparing this fork. Local browsers can encounter access blocks
that the upstream cloud's proxies and browser profiles handled.

Primary setup references:

- [Browser Use supported models](https://docs.browser-use.com/open-source/supported-models)
- [vLLM compatible server](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/)
- [Qwen3-VL-8B-Instruct model card and license](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
