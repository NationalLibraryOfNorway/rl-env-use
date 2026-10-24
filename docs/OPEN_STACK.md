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
| `RLENV_LOCAL_MODELS` | `{}` | JSON mapping of local route names to `model`, `base_url`, and optional `api_key_env`. |
| `RLENV_LOCAL_API_KEY` | Dummy local token | Optional authentication token for your own server; shared by both endpoints. |
| `RLENV_LOCAL_MAX_TOKENS` | `262144` | Requested maximum completion tokens per call, shared by all local routes. Servers may enforce lower limits, including input-plus-output context limits. |
| `RLENV_LOCAL_TIMEOUT` | `1200` seconds | Per-call server timeout, structured-call deadline and browser-agent LLM timeout. |
| `RLENV_STEP_TIMEOUT` | `360` seconds | Outer browser-agent step deadline, including model work and actions. |
| `RLENV_MISSION_TIMEOUT` | `3000` seconds | Total wall-clock limit per mining mission (50 minutes). |
| `RLENV_GLOBAL_TIMEOUT` | `0` | Optional global build/create deadline in seconds; zero disables it. |
| `RLENV_PANEL_MODELS` | `local-text` | Comma-separated local aliases or `local:MODEL_ID` entries. |
| `RLENV_BROWSER_USE_VISION` | `1` | Set to `0` for a text-only browser model; navigate with DOM text instead of screenshot input. |
| `RLENV_BROWSER_EXECUTABLE` | Browser discovery | Absolute path to your Chromium executable. |

When a server explicitly rejects input plus requested output as exceeding its context
window and reports both token counts, the transport retries once with output
limited to the remaining context. The configured ceiling stays unchanged; inputs
and evidence are never truncated, and the endpoint does not change. Other errors
are propagated normally.

The supplied `.env.example` sets both model aliases to Qwen3-VL-8B-Instruct,
allowing a one-server smoke test. For real dataset generation, use a stronger
writer/critic and multiple independently trained panel models. A single-model
panel is not independent validation, particularly if that model wrote the tasks.

`local:NAME` first looks up `NAME` in `RLENV_LOCAL_MODELS`. A matching entry can
select a separate endpoint and translate the route name to the exact model ID
served there. Without an entry, the name is sent as the model ID to the shared
base URL, preserving the existing behavior. The code cannot attest that an
endpoint is self-hosted or that its model has open weights; verify both yourself.

### Different endpoints per model

Add a single-line JSON object to `.env`, for example:

```dotenv
RLENV_LOCAL_MODELS={"writer":{"model":"qwen-model-id","base_url":"http://192.168.1.129:8000/v1"},"critic":{"model":"reviewer-model-id","base_url":"http://192.168.1.130:8000/v1","api_key_env":"CRITIC_SERVER_TOKEN"}}
```

Use the actual IDs exposed by each server's `/v1/models` endpoint. If the critic
requires authentication, export `CRITIC_SERVER_TOKEN` or set it in the ignored
`.env`; omit `api_key_env` for servers without endpoint-specific authentication.
A configured `api_key_env` must name a nonempty environment variable. Otherwise
the shared `RLENV_LOCAL_API_KEY` (or a dummy token) is used.

```bash
rl-env-use create 'your skill and sources' --writer local:writer --critic local:critic
# Any named route can also be a panel, adjudicator, closed-book, judge or policy model.
rl-env-use run envs/my-env --model local:critic
```

Each entry may contain `model`, `base_url`, and `api_key_env`; omitted fields use
the existing defaults. Route names are retained as `local:NAME` in accounting and
results, so two servers serving the same raw model ID can be compared separately.
`RLENV_LOCAL_TEXT_MODEL` and `RLENV_LOCAL_BROWSER_MODEL` can also name registry
entries (without `local:`). An entry's `base_url` overrides the separate browser
URL; otherwise `local-browser` keeps its browser endpoint and other models keep
the shared endpoint. Retries stay on the selected route without switching servers.
Restart Python after configuration changes, and use new aliases or run directories
when replacing a server/model behind an existing route to avoid stale caches.

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

## Dataset language and optional diagnostic localization

Use `--language` on `new`, `create` or `build` with a three-letter ISO 639-3
code. Common choices are `eng` (English), `nob` (Norwegian Bokmål), `nno`
(Norwegian Nynorsk), and `spa` (Spanish). `nor` selects Norwegian with Bokmål
as the default writing standard; use `nob` or `nno` to be precise. Two-letter
codes such as `nb` or `nn` are rejected. Other registered ISO 639-3 codes are
accepted using their language names.

```bash
rl-env-use create 'your Norwegian use case and sources' --name norsk-pilot \
  --language nob --writer local:writer --critic local:critic --tasks 12 --missions 2
# Also put newly generated model feedback and mining notes in Bokmål:
rl-env-use build envs/norsk-pilot --localize-diagnostics
# Return future internal diagnostics to English:
rl-env-use build envs/norsk-pilot --no-localize-diagnostics
```

The dataset language applies to generated questions, natural-language answers,
answer choices, derivations and `why_hard`. Each generated task and exported row
records `language`. Policy evaluations request answers in that language too.
Explicit policy reasoning is an internal diagnostic: it stays in English unless
`--localize-diagnostics` is enabled. No option controls a model's hidden reasoning.

The optional flag also covers model-generated critic explanations, mining
observations, mission plans, judge feedback and climb diagnostics. Source
quotations/materials and tool responses remain verbatim. Code-generated logs,
error messages, report UI and machine fields are not translated. Existing mined
notes and cached model outputs are not retroactively translated; a changed critic
language setting does trigger a new critic review during verification.

`rl_env_use/language.py` composes language-specific output templates with shared
workflow instructions. It includes translated dataset templates for English,
Bokmål, Nynorsk, Norwegian and Spanish, and a language-name template for other
ISO codes; the domain rules themselves need not be copied into every language.
Template selection changes neither task counts (`--tasks`) nor grading rules.

Canonical forecast golds (`yes`/`no`), numeric/date formats, JSON keys, enum values,
URLs, identifiers and arithmetic expressions stay unchanged. Writers are asked
to retain original source values as aliases for translated categorical answers;
source evidence stays intact and all normal verification gates still apply.
Source materials may therefore be in a different language from the generated
questions. Language compliance is prompted, not a separate certified language
quality gate; review a pilot before training.

Settings are saved as `language` and `localize_diagnostics` in `spec.json` and
reused by subsequent build/run/climb/report/judge workflows. You can also set
`RLENV_LANGUAGE=eng` and `RLENV_LOCALIZE_DIAGNOSTICS=0` in `.env` as defaults.
Start a new environment for another language: changing the language after tasks
exist is rejected to prevent mixing datasets. Use new run directories when
changing diagnostic settings if you need all traces in the same language.

## Separate writers and critics

`create` and `build` accept `--writer` (an alias for the existing `--model`) and
`--critic`. The writer handles planning, synthesis and hardening; the critic
handles evidence support, ambiguity checks and blind answer recomputation.
Mining, closed-book checks, difficulty panels and adjudicators retain their own
model settings.

```bash
# Separate models; use served IDs or named routes from RLENV_LOCAL_MODELS.
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

## Trial discovery and timeout limits

ClinicalTrials.gov discovery uses `query.locn=Norway`, optional `query.cond` or
`query.term`, `filter.overallStatus=COMPLETED`, and
`filter.advanced=AREA[StudyType]INTERVENTIONAL AND AREA[HasResults]true`.
Do not send `country`, `results=with`, the old `filter.geo=country|NO`, or
invented `fields` names. The trial-source normalizer removes unsupported query
parameters and fetches full records with a small page size. Current API records
are discovery evidence; dated pre-results registration versions are still
required for policy inputs.

Helsenorge's own clinical-trials frontend supplies
`https://www.helsenorge.no/api/v1/kliniskestudier`, a public JSON array.
It exposes titles, summaries, categories, status, hospital names and `lenke`
links. No authentication or Chromium is needed for this listing; filter records
locally, and follow hospital links for registry IDs and published outcomes.
The invented `/api/clinicaltrials` endpoint returns 404. Browser rendering can
still be necessary for dynamic linked pages. Source guidance is injected into
mining plans and known invalid trial-discovery URLs are normalized before use.

Model calls default to 1200 seconds, browser steps to 360 seconds, and mining
missions to 3000 seconds. The optional global build/create deadline defaults
to zero (disabled); a positive value can stop a mission sooner. These are wall-clock limits, not
output-token budgets. Set the four timeout variables in `.env` before starting
Python; restarting is required for the module-level structured/mission limits.

## Optional DeepInfra endpoints

The named-route transport also supports explicitly configured hosted open-weight
endpoints. `local:NAME` identifies the transport route, not physical locality.
DeepInfra uses `https://api.deepinfra.com/v1/openai` and Bearer authentication;
set the raw token in the ignored `.env`:

```dotenv
DEEPINFRA_API_KEY=
```

Merge the four route entries from `.env.example` into your existing single-line
`RLENV_LOCAL_MODELS` JSON object. Each entry uses `api_key_env=DEEPINFRA_API_KEY`.
The prepared names and exact served model IDs are:

| Route | Model |
| --- | --- |
| `local:deepinfra-deepseek-writer` | `deepseek-ai/DeepSeek-V4.1-Flash-1017` |
| `local:deepinfra-deepseek-critic` | `deepseek-ai/DeepSeek-V4.1-Flash` |
| `local:deepinfra-nemotron-writer` | `nvidia/NVIDIA-Nemotron-3.5-Super-VL` |
| `local:deepinfra-nemotron-critic` | `nvidia/NVIDIA-Nemotron-3.5-Lightning` |

```bash
rl-env-use create 'your skill and sources' --name deepinfra-pilot --language nob \
  --writer local:deepinfra-deepseek-writer --critic local:deepinfra-deepseek-critic \
  --tasks 12 --missions 2 --no-localize-diagnostics --plan-only
```

Remove `--plan-only` to build, after checking a small structured-response request.
For Nemotron use the corresponding `deepinfra-nemotron-*` names. `--writer` does
not change mining: to use the same DeepInfra writer for browser agents, set
`agent_model` in the new environment's `spec.json` to that named route, or set
`RLENV_LOCAL_BROWSER_MODEL=deepinfra-deepseek-writer` before creating a new spec.
Do not change the endpoint behind an alias mid-run; use a new environment/run
for comparisons with the existing local/Sigma2 setup.

DeepInfra is paid hosting. Calls on these generic named routes still record
zero/unknown cost in this application's ledger, so `--budget` is not a reliable
spending cap; use provider-side limits and bounded tasks/missions. Provider
output caps may be below the configured 262144 ceiling; verify compatibility
with the selected model. No authenticated DeepInfra inference is implied by
loading these configurations. Closely related writer/critic versions provide
less independent validation than independently trained model families.

References: [DeepInfra chat API](https://docs.deepinfra.com/chat/overview) and
[structured outputs](https://docs.deepinfra.com/chat/structured-outputs).

## Critic recovery and incremental checkpoints

Task critics use independent limits so a stuck review cannot silently consume
the general 1200-second model-call allowance on every retry:

| Variable | Default | Meaning |
| --- | --- | --- |
| `RLENV_CRITIC_CALL_TIMEOUT` | `120` | Seconds per critic API attempt. |
| `RLENV_CRITIC_TASK_TIMEOUT` | `360` | Total seconds for one task's recomputation and critique, including retries. |
| `RLENV_CRITIC_RETRIES` | `2` | Maximum attempts per structured critic call. |
| `RLENV_CRITIC_MAX_TOKENS` | `4096` | Response cap for task-critic calls only. |

These limits do not change writer/browser/panel budgets. Each critic request
logs task ID, model, schema, attempt, deadline, elapsed time and exception type;
prompt text, keys and raw provider error payloads are not included in retry logs.
Successful and failed reviews are atomically checkpointed to `tasks.jsonl` as
each task finishes. Completed evidence gates are saved before the critic starts.
Cancellation preserves completed reviews without stamping pending tasks as done.
Timeout/error verdicts remain unverified and may be retried by a later build.

For recovery, inspect saved evidence before rerunning `build --only verify`.
Reject invalid tasks in `reviews.jsonl` and unusable materials in
`data_reviews.jsonl`; rejected inputs are excluded from future synthesis.
Retain source evidence and pre-audit snapshots. Domain `conventions` now reach
mining agents as collection requirements, not just the writer. For trial forecasts,
a current registration page's first-posted date does not prove that its current
contents were available before results. Require a dated historical version or
pre-results protocol and a documented success rule before readmitting a case.
Run `build --only scout` to collect replacement evidence without drafting tasks;
review that evidence before the next synth/verify pass. Both `build` and `create`
respect `RLENV_GLOBAL_TIMEOUT` when it is positive (zero disables the deadline).

## Complete ClinicalTrials.gov outcomes

On `/api/v2/studies/NCT...` pages, the collector parses the full current page JSON
and locates the unique primary outcome matching the supplied exact title. It saves
the original JSON substring for the complete object under
`resultsSection.outcomeMeasuresModule.outcomeMeasures`, including all reported
analyses, and marks it `capture=complete_primary_outcome_json`. Model-selected
start/end markers cannot shorten this object. Malformed/truncated JSON, missing
or ambiguous titles, and objects exceeding the passage cap fail without saving.
Other sites retain verbatim marker-based extraction and require manual completeness
review. A complete source object does not prove a valid success label: missing
statistics, historical provenance, time points and prespecified rules remain
separate checks. Existing partial records require repair; they are not silently
upgraded by this change.

## Per-route thinking control

A named route may set `"enable_thinking": false` (or `true`). The compatible
transport forwards this as `chat_template_kwargs.enable_thinking` while retaining
the normal message serializer, JSON schema, usage accounting and response parser.
Omit the field for servers that do not support this template option; omission
preserves server defaults. This is separate from the completion-token ceiling.
Reasoning returned in a separate `reasoning_content` field is not imported into
dataset tasks. Disabling thinking prevents its generation on supporting servers;
discarding that response field alone does not reduce generation time.

Clinical-trial discovery should also include studies whose primary results are
published in journals but have no registry `resultsSection`. Requiring
`AREA[HasResults]true` limits discovery to registry-posted results. Use a broader
Norway/interventional query when collecting publication-backed cases and pair
the dated public protocol with the original primary results publication.

For prediction-only drafting (`mix=predict`), the response schema enforces
`kind=predict`, `answer_type=prob`, and a private `yes`/`no` gold label. The
policy still produces a probability: the private resolution is not a forecast.
An insufficient panel is marked unmeasured before any policy calls. Repeated
aliases, including local routes resolving to the same served model ID, do not
count as independent panel members. An endpoint change alone does not establish
a different model.

The task critic is explicitly told that a forecast's private gold is its observed
`yes`/`no` event resolution while the policy output is a probability graded with
a Brier score. A difference between these formats is intentional and must not
itself be reported as a task defect. Evidence and precision checks still apply.
Case-draft schemas place a concise final justification before the private gold
field, so structured decoding can resolve the event before committing its label.
The localized probability instruction is appended only when it is absent.

## Curated SFT explanations

Use `sft` to export **retrospective outcome interpretation**, with published results in the
training input and a short edited explanation in the target. This changes the task from
forecasting an unknown outcome to interpreting evidence. Do not train a registration-only
forecast to output a hindsight explanation or treat its observed yes/no outcome as a calibrated
forecast probability. Rewriting explanations must correct errors against source evidence;
it must not invent support for a preferred target.

```sh
rl-env-use sft envs/<name> --targets envs/<name>/sft-targets.json
```

The targets file is a JSON array, with one object per selected task:

```json
[
  {
    "task_id": "<existing task ID>",
    "answer": "no",
    "label": "Nei",
    "system": "Tolk resultatene ut fra dokumentasjonen. Svar på norsk bokmål.",
    "prompt": "Viste studien overlegenhet for det forhåndsdefinerte primærendepunktet?",
    "explanation": "Den rapporterte p-verdien var 0,78, som overstiger signifikansnivået på 0,05. Statistisk signifikant overlegenhet ble ikke vist."
  }
]
```

Use the saved task's ISO 639-3 language for the system message, new question, localized label
and explanation. Source context and outcome quotations stay verbatim. Do not reuse the
original prospective probability question. The exporter checks the binary label against the
verified task, requires passing live and critic correctness checks, and rejects quarantined
material or evidence. It deliberately ignores RL panel difficulty, closed-book recall and
forecast balance for this different task; the original RL files and gates are unchanged.
It does not automatically establish that a new question or edited explanation is correct:
review both against the supplied evidence before training.

`dataset-sft.jsonl` contains only chat `messages` (system, user, assistant), with protocol
context and published outcome evidence in the user message. Raw traces, old derivations,
critic diagnostics and grading metadata are excluded. `dataset-sft.audit.json` separately
records source IDs/URLs, evidence and target hashes, and original panel verdicts. Edited
explanations are marked as curated final explanations and not human-reviewed; optional
`explanation_model` in each target records the model used to rewrite it. `--out` chooses another
JSONL destination. Keep all examples from the same trial together when splitting data, and
exclude these trials from any later prospective forecasting evaluation after training on results.

For the Norwegian clinical-trial pilot, `envs/norwegian-trial-endpoints-pilot/sft-targets.json`
contains four edited targets. This is a format-review seed set, not a sufficient fine-tuning corpus.

## Automated Norwegian clinical-trial SFT collection

`new`, `build` and `create` accept `--mode sft` (saved in `spec.json`). RL remains
the default. The current automated SFT source collector supports Norwegian clinical
trials: Helsenorge's listing API and hospital study pages, linked ClinicalTrials.gov
IDs, and discovery probes of the configured Norwegian/EU portals. It logs failed
or empty sources; listing a portal does not imply that it contributed an example.
Heavy JavaScript portal search is not automated by this HTTP collector; use the
browser workflow to obtain additional source evidence when needed.

```sh
rl-env-use build envs/norwegian-clinical-trials-sft --mode sft \
  --writer local:writer --critic local:critic \
  --language nob --no-localize-diagnostics
```

A fresh spec can be made with `new --mode sft`; set its sources and limits before
`build`. `create --mode sft --name <existing-name>` runs one bounded pass over the
saved sources and imported document cache. It does not invoke RL panels, hardening,
closed-book recall or forecast label balancing. The automated SFT path has its own
source documents and candidate files; original RL tasks/reviews are not imported
or silently approved. Previously rejected trials are recovered through fresh source
captures and newly checked tasks, not by undoing their forecasting rejections.

The spec supports:

- `recovery_trial_ids`: canonical ClinicalTrials.gov IDs to recapture, including old
  forecasting candidates. Each record must list a Norwegian location.
- `tasks_per_case` (default 2): maximum drafts per captured document, with distinct
  extraction, result interpretation or evidence-sufficiency questions. This is a
  document limit; report independent trial counts separately from example counts.
- `target_tasks`: accepted-example target, including any explicitly imported seeds.
- `sft_discovery_limit` (default 6): linked pages/registry IDs followed per source.
- `sft_model_timeout` (default 120 seconds per attempt), with two attempts per call.
- `sft_question_max_chars` (default 2000), `sft_explanation_max_chars` (2400), and
  `sft_explanation_max_words` (250). Explanations are normally 60–180 words; extra
  length should explain evidence and limitations, not add unsupported detail.

Every registry capture includes the relevant current protocol modules and **complete**
posted primary-outcome objects with all analyses. The protocol modules are a declared
structured projection, not a claimed historical snapshot. Oversized/incomplete outcomes
are rejected rather than cut off. Missing results can support endpoint extraction or
carefully scoped insufficiency questions; they cannot establish trial failure.

The writer drafts a question, final answer, edited explanation and exact supporting
quotations. Structural checks preserve numeric signs in quotations. A separate critic
call checks source support, correctness, question clarity and target-language quality;
only examples passing every check are exported. Writer and critic may use the same
configured model through separate calls. This is model-based validation, not human
review or a guarantee of clinical correctness. Failed calls and rejected drafts stay
in the candidate audit. Completed reviews resume from cache keyed by evidence, model
routes, language and limits; use a fresh output directory after changing weights behind
an unchanged route alias.

Artifacts in the SFT environment:

```text
source-captures/         complete fetched responses, referenced by hash
sft-discovery.jsonl      attempted sources, followed links and explicit exclusions
sft-documents.jsonl      relevant evidence with provenance and transformations
sft-candidates.jsonl     drafts and critic verdicts, including rejected examples
sft-status.json         latest phase and completion status
dataset-sft.jsonl       accepted messages plus separate critic-reasoning attributes
dataset-sft.audit.json  source, model, evidence hash and review provenance
sft-preview.md          readable accepted questions, answers and explanations
```

`build --only scout` collects sources without drafting; `--only synth` or
`--only verify` resumes cached documents and unfinished reviews. `export` re-exports
the accepted SFT pool. To repeat completed criticisms after changing instructions,
retain the original candidate audit in a backup and start a new environment. Split by
trial ID (or study URL when no ID exists); more questions from one trial do not create
independent trial evidence. Do not evaluate prospective forecasting on trials whose
results were included in SFT training.

### Continuous collection, robots policies and portable source snapshots

Set `target_tasks=0` and `sft_continuous=true` to remove the example-count stop.
The collector advances through Helsenorge's listing in `sft_discovery_limit` batches,
writing `sft-frontier.json` after each source. It drafts and checks each batch before
collecting the next. It stops when the source frontier is exhausted or interrupted;
`sft_max_batches=0` imposes no additional batch limit. A large listing can yield
thousands of examples, but source availability and correctness checks determine the
actual count. Restart the same command to resume; cached reviews remain available.

Every download and redirect passes a robots gate using the `rl-env-use` user agent.
The longest applicable Allow/Disallow rule wins, Allow wins ties, and crawl delays
are respected. A missing robots file (404/410) means no published restrictions;
403, server failures, invalid HTML responses and unavailable policies fail closed.
Robots policies and decisions are stored in `robots-captures/` and
`robots-audit.jsonl`. Permission to fetch a page is checked separately from whether
its content provides enough evidence for a task.

**ClinicalTrials.gov's generic robots policy disallows `/api/`, but NLM explicitly
recommends automated use of its public v2 API.** See the [official July 2025
guidance](https://www.nlm.nih.gov/pubs/techbull/ja25/ja25_clinical_trials_screen-scraping.html),
which provides v2 study request examples and instructions for automated collection.
The user authorized restoring this documented API access. The downloader therefore
has a provider-specific exception for HTTPS requests to the exact
`clinicaltrials.gov` host and documented public v2 study/version endpoints. It does
not cover internal APIs, arbitrary v2 paths, lookalike hosts or redirects outside
the permitted scope. All other downloads retain the robots gate.

API provenance preserves `robots_allowed=false` alongside effective `allowed=true`,
`reason=documented_public_api` and the NLM authorization reference. The exception
also requires at least one second between API requests. Throttling and transient
server errors trigger bounded retries with exponential backoff and Retry-After
handling. API study discovery follows `nextPageToken` across batches; complete
JSON snapshots and all primary-outcome analyses remain in the source archive.

`recovery_publications` can also map a trial ID to a known original/related PMID;
its complete indexed abstract is fetched through the permitted Europe PMC service.
These are declared abstracts, not claimed full articles. A related or secondary
publication supports only questions justified by that publication; it cannot
establish an unreported original primary outcome.

The training input includes the complete extracted evidence for the question,
source URL and capture date. Raw permitted responses are saved without truncation
as `.html`, `.json`, `.xml`, `.pdf` or `.txt` according to content type. Full raw
HTML is kept outside the model prompt to avoid filling the context with navigation
and scripts. `dataset-sft.records.jsonl` pairs each messages row with provenance,
including the raw snapshot path, SHA-256 and robots decisions. Keep that file,
`dataset-sft.audit.json`, `source-captures/` and `robots-captures/` together when
moving the dataset. Snapshot checksums are checked again before export; missing or
modified snapshots are excluded. Failed or disallowed earlier captures remain in
the audit but do not appear in the compliant SFT export.

For source/language spot-checks, `sft-reviews.jsonl` accepts per-example rejections
such as `{"id":"sft_...","verdict":"reject","notes":"..."}`. The latest verdict
wins and rejected examples are omitted from every export even if the model critic
passed them. Corrected examples retain their original source snapshot and receive
a new ID and a fresh critic review; the original and its rejection stay in the audit.

### Switching the critic to thinking mode at an accepted-example milestone

Keep writer and critic routes separate. A thinking route can point to the same
weights and endpoint as `local:critic`, with `"enable_thinking": true`; keep
`local:writer` and the original critic route unchanged. Configure a saved SFT spec:

```json
{
  "critic_model": "local:critic",
  "sft_critic_switch_at": 2500,
  "sft_thinking_critic": "local:critic-thinking",
  "sft_thinking_critic_timeout": 1200,
  "sft_thinking_critic_max_tokens": 32768,
  "target_tasks": 10000,
  "sft_continuous": true
}
```

The threshold counts accepted, exported examples, including manual exclusions and
source checks. Selection is checked between examples, even within a two-example
writer batch. After 2,500 acceptances, the first cohort is frozen under
`cohorts/nonthinking-2500/` with its source snapshots. The next review uses the
thinking critic; `sft-critic-transition.json` makes this transition persistent.
Previously completed cases keep their existing reviews. Collection stops at the
total target without overshooting it or when the source frontier is exhausted.

Thinking uses the server's template option, not a request to reveal reasoning in
the final review schema. The compatible transport captures model-provided
`reasoning_content` (or `reasoning`) separately from the structured verdict.
For thinking reviews, missing reasoning is an explicit error and is not silently
accepted. No reasoning is fabricated or reconstructed from the verdict.

Every automatic SFT example in `dataset-sft.jsonl` and
`dataset-sft.records.jsonl` has separate attributes:

```json
{
  "messages": ["...training messages only..."],
  "critic_thinking": true,
  "critic_reasoning": "...model-provided critic reasoning..."
}
```

Non-thinking examples have `critic_thinking=false` and normally
`critic_reasoning=null`. The richer records/audit also identify critic phase,
model, response and capture metadata. Critic reasoning never enters the system,
user or assistant training messages. `dataset-sft.chat.jsonl` contains only
`messages` for trainers that reject extra attributes. The reasoning is diagnostic
model output, not verified explanatory supervision or a human review.
