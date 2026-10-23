---
name: rl-env-use
description: Build and evaluate RL environments from real cases mined from the web, using self-hosted open-weight models and local Browser Use agents. Use when the user knows a domain and the input-to-outcome skill to measure but needs help collecting cases, verifying tasks, benchmarking policies, optimizing prompts, or exporting tasks for an external RL trainer.
---

# rl-env-use

A case pairs the materials a model receives with its true outcome, published elsewhere or later. Local Browser Use
agents mine both; a case policy reads only the materials and answers without a browser. Omitting `--skill` when
creating a spec instead makes a browse-task environment, where the policy finds answers on the live web.

This fork defaults to self-hosted open-weight models. `climb` optimizes a system prompt; it does not update model
weights. Actual RL training requires a separate trainer consuming the exported tasks and rewards.

## Setup and model selection

Run commands from this fork's repository root with its installed `rl-env-use` (or `python -m rl_env_use`) and Python
environment. Relative `envs/` paths use the current working directory. For installation read [install.md](install.md);
for endpoint configuration, migration and troubleshooting read [docs/OPEN_STACK.md](docs/OPEN_STACK.md).

Before model or browser calls:

- Keep `RLENV_OPEN_ONLY=1` (the default even without `.env`). Do not request hosted provider keys or enable
  compatibility mode unless the user explicitly chooses the hosted workflow.
- Configure `RLENV_LOCAL_BASE_URL` (default `http://127.0.0.1:8000/v1`), `RLENV_LOCAL_TEXT_MODEL`, and
  `RLENV_LOCAL_BROWSER_MODEL` to match the user's self-hosted server. It must support `/v1/chat/completions`,
  JSON-schema structured outputs, and image input when browser vision is enabled. Set
  `RLENV_BROWSER_USE_VISION=0` for a text-only browser model using DOM navigation. Verify the chosen weights'
  license and endpoint ownership; the code cannot attest to either.
- Use `local-text` for planning, task writing, checks, case policies, prompt proposals and calibrated judges;
  `local-browser` for mining and browse policies; or `local:MODEL_ID` for an exact model ID. On `create` and `build`,
  `--writer` (alias `--model`) selects the planner/writer and `--critic` selects the task critic. If only one is
  supplied, both roles use it. If neither is supplied, saved `writer_model`/`critic_model` roles are reused, or the
  text default is used for both. Explicit role flags replace the saved pair. Mining uses `spec.json`'s `agent_model` (default `local-browser`).
- `RLENV_LOCAL_MODELS` optionally maps names to per-model settings: for example,
  `{"critic":{"model":"served-id","base_url":"http://host:8000/v1","api_key_env":"CRITIC_TOKEN"}}` routes
  `local:critic` to that server. Named routes work for writers, critics, panels, adjudicators, judges and policies.
  Optional `api_key_env` names an environment variable containing that endpoint's token. Without it, the shared
  `RLENV_LOCAL_API_KEY` or dummy token is used. Text/browser model variables may name registry entries too.
  An entry's `base_url` takes precedence over `RLENV_LOCAL_BROWSER_BASE_URL` for `local-browser`, then the shared
  base URL. Unregistered explicit IDs keep the shared endpoint. See the open-stack guide for full examples.
- Point `RLENV_BROWSER_EXECUTABLE` at an installed open-source Chromium executable. Browsers run locally;
  cloud profile IDs and proxy-country flags do not configure local authentication or proxies. Resolve source access
  with the user before mining if it needs these features.
- Set `RLENV_PANEL_MODELS` to comma-separated local aliases, named routes or exact served IDs.
  The default `local-text` panel is suitable for integration checks only. For meaningful difficulty validation,
  use multiple independently trained models and identify the writer, panel and policy in the results.

Exported variables override the working directory's `.env`, then the repository's `.env`. Set configuration before
starting Python and restart existing processes after changes. Hosted aliases and fallbacks are rejected in open-only
mode. Browser Use telemetry, cloud sync, remote pricing and billing calls are disabled, but web mining still reaches
the source sites; this is routing enforcement, not a network sandbox.

Local API spend is recorded as zero/unknown and excludes GPU time, rental and electricity. **`--budget` is not a local
compute cap**, and `--dry-run` cannot establish a dollar price for local inference. Bound creation with `--tasks`,
`--missions`, and the spec's `max_rounds` (default six), `max_steps` and `concurrency`. Bound runs with `--limit`,
`--reps`, `--concurrency` and `--timeout`; bound climbs with `--rounds`, `--k`, `--reps` and `--concurrency`.
`RLENV_LOCAL_MAX_TOKENS` (default `262144`, shared by all local routes) and `RLENV_LOCAL_TIMEOUT` bound individual
model calls. The token setting is a requested output ceiling; each server still enforces its context/output limits.
Timeout defaults: `RLENV_LOCAL_TIMEOUT=1200` for model calls, `RLENV_STEP_TIMEOUT=360` for browser steps,
`RLENV_MISSION_TIMEOUT=3000` per mining mission, and `RLENV_GLOBAL_TIMEOUT=0` to disable the global deadline (the original behavior).
Creation's `--missions` sets the initial round; later rounds adapt to yield. `--tasks` is a target, not an exact
episode limit. Use the spec's `max_rounds` and per-mission limits when sizing local work.

## Dataset language

Use `--language` on `new`, `create` or `build` with ISO 639-3 codes such as `eng`, `nob` (Bokmål), `nno` (Nynorsk)
or `spa`. Prefer `nob`/`nno` over generic `nor` for Norwegian. Language-specific output templates control generated
questions, textual answers, choices, derivations and `why_hard`. Task counts remain controlled by `--tasks`.
Source materials and quotations remain verbatim, as do grading tokens, numeric/date formats, identifiers and schemas.
Translated categorical answers retain source forms as aliases and still pass normal evidence/grading checks.

Internal model feedback, mining notes, critiques and explicit policy reasoning stay in English by default. Add
`--localize-diagnostics` to generate these in the dataset language too; `--no-localize-diagnostics` turns that off.
Code logs/report UI and existing cached outputs are not translated. Settings persist in `spec.json`; `.env` defaults
are `RLENV_LANGUAGE=eng` and `RLENV_LOCALIZE_DIAGNOSTICS=0`. Start a new environment to change dataset language after
tasks exist. Review pilot language quality; prompt instructions are not an independent language-quality gate.

## Autonomous workflow

Use when the user wants hands-off creation within their chosen scope and resource limits.

1. Start with a small integration run:
   `rl-env-use create "<domain, input -> outcome skill, source URLs, constraints>" --name <name> --tasks 12 --missions 2 --writer local:writer --critic local:critic --language nob --no-localize-diagnostics`.
   This example assumes `writer` and `critic` routes are configured in `.env`; otherwise select available local aliases.
   Pass both role flags when using different models: providing only one sets both roles to that model.
   This is not a statistically meaningful benchmark. `--plan-only` calls the text model and prints the proposed spec
   without building or saving it. For larger runs choose task and mission counts to fit the user's resources; use
   `new` and edit the spec first if custom mining limits or `max_rounds` are needed.
   - A planner defines sources, conventions and an outcome-blind sampling rule. Each round mines, writes, verifies,
     measures panel difficulty, hardens easy tasks, then verifies and measures them again.
   - Derived numbers are computed by code from expressions over the materials. Forecasts must depend on the
     materials and must not be answerable without them; at most 60% of served forecasts resolve one way.
   - Tasks solved by every configured panel model are hardened up to two levels. Difficulty is relative to those
     models; an open-model panel does not establish frontier-model headroom. Use `--panel local:MODEL_A local:MODEL_B`
     to override the panel for a new environment (IDs are placeholders). On resume edit the stored spec instead:
     `--panel` does not replace an existing spec's panel.
   - Naming a registry query or records-listing URL helps mining. Creation stops at the target, limits or low yield.
     Ctrl-C cancels work; resume with `create "<same request>" --name <name>` and the same configuration.
2. Pilot the served pool:
   `rl-env-use run envs/<name> --model local-text --limit 5 --concurrency 1`.
   Check graded results, traces and `errors.jsonl` before scaling. If a smaller policy needs tasks the panel found
   easy, add `--verified` consistently to pilot, baseline and climb. This retains verification gates and includes
   panel-easy tasks; report which pool was used.
3. Start a bounded prompt climb:
   `rl-env-use climb envs/<name> --model local-text --proposer local-text --rounds 2 --reps 3 --concurrency 1`.
   It splits by case, estimates A/A noise from rep halves, screens edits on train and keeps paired case-bootstrap test
   gains, with a final baseline-vs-best evaluation on fresh reps. Respect refusals for saturated or underpowered data;
   collect enough independent cases rather than defaulting to `--force`. Report the final result and generated
   `report.html`, including "within noise, do not merge" outcomes.

Terminal runs show a live board; redirected logs show status lines. Local spend displays are not compute estimates.

## Optional DeepInfra routes

See the open-stack guide and `.env.example` for `deepinfra-deepseek-writer`/`deepinfra-deepseek-critic` and
`deepinfra-nemotron-writer`/`deepinfra-nemotron-critic`. Each uses the configured DeepInfra base URL and reads
`DEEPINFRA_API_KEY` via `api_key_env`; never print the key. Select them as `local:<route-name>`.
The `local:` prefix selects the compatible transport, not physical locality. Generic route accounting does not
measure DeepInfra charges; use provider limits rather than treating `--budget` as a spending cap.
Keep mining's `agent_model` explicit; writer/critic flags do not change it. Use a new environment for provider/model
comparisons and verify structured/image support before a full run.

## Recovering stalled reviews

Task critics have separate bounded defaults: `RLENV_CRITIC_CALL_TIMEOUT=120`, `RLENV_CRITIC_TASK_TIMEOUT=360`,
`RLENV_CRITIC_RETRIES=2` and `RLENV_CRITIC_MAX_TOKENS=4096`. Per-task verdicts are checkpointed immediately;
logs identify task/model/schema, retry count, elapsed time and error type without printing keys or prompts.
These overrides leave general writer/browser/panel budgets unchanged. Resume an interrupted review with
`build envs/<name> --only verify`; already completed reviews keep their cached verdicts.

Audit raw inputs and gold evidence before retrying. Quarantine invalid tasks through `reviews.jsonl` and input
materials through `data_reviews.jsonl`; preserve originals. Domain conventions are passed to miners as collection
requirements. Use `build --only scout` for replacement evidence and inspect it before drafting or verification.
Do not automatically rerun a long full creation loop on known-invalid evidence.

## Forecasting trial endpoints

Use the verified Helsenorge listing API `https://www.helsenorge.no/api/v1/kliniskestudier` for discovery; filter
its JSON array locally and follow `lenke` links. Chromium is needed only when linked pages require rendering.
For ClinicalTrials.gov, use `query.locn=Norway` and `filter.advanced=AREA[StudyType]INTERVENTIONAL AND AREA[HasResults]true`;
avoid invented API routes/fields. See the open-stack guide for verified queries and source provenance rules.

For a clinical-trial endpoint dataset, set `--mix predict`, choose the dataset language explicitly, and name all
requested discovery sources. Use registrations/protocol versions from before results as policy inputs; outcome
pages and publications are private gold evidence, never policy context. Link registry records by trial identifiers
and deduplicate them by trial. Keep the prespecified primary endpoint, time point, and success criterion fixed;
never invent a new threshold to balance yes/no labels. A completed trial is not necessarily a successful trial.
Exclude unresolved endpoints and unclear co-primary success rules rather than inventing gold labels.

When launching a bounded pilot, use one task per trial (`tasks_per_case=1`) and suitable spec limits for the model
server. A spec created with `new` can be edited before `create --name <name>` resumes the autonomous workflow.
`create` does not replace an existing spec's source list or panel: confirm those in `spec.json` before resuming.
If the user wants dataset text localized but internal feedback in English, save `localize_diagnostics=false` and
use `--no-localize-diagnostics`; omitting the positive flag alone does not override a saved or `.env` true value.

Check each configured endpoint before a long run without displaying tokens. Use the returned `/v1/models` ID
exactly and verify structured output support. `api_key_env` contains only the environment variable's name;
the transport sends its value as `Authorization: Bearer <token>`. Keep endpoint keys in ignored `.env`, not in
specs, logs, prompts, commits or dataset exports. A named `local:` route may point to a remote self-hosted gateway;
open-only mode enforces configured transport routes, not physical locality.

## Step by step

Use when the user wants to review cases and tasks before continuing. Ask only for unresolved choices and the
requested review decisions, using the host's available question mechanism.

1. Establish one input -> outcome skill per environment and URLs containing inputs and outcomes, ideally separated
   in place or time. Create the spec:
   `rl-env-use new <name> --topic "<domain>" --skill "<input -> outcome>" --sources URL...`.
   Edit `spec.json` for mining limits and panel models before building. Omit `--skill` only for live browse tasks.
2. `rl-env-use build envs/<name>` mines and stops before task creation on the first fresh build. Show the output
   (`build envs/<name> --only look` repeats it) and open `review envs/<name>` -> Data tab. Let the user review cases
   before the next build. To add cases use `build envs/<name> --only scout`; add `--enrich` to gather table data needed
   for computations on complete cases.
3. `rl-env-use build envs/<name>` again writes and checks tasks: outcome-supported gold, leakage, closed-book recall,
   live evidence and a critic, followed by the configured panel. Graders are programmatic (number/date/set/exact/
   contains/prob) and self-tested against null, foreign and perturbed golds. Show representative tasks in the Tasks
   tab and resolve disputed golds or grading before evaluating policies.
4. Pilot with `rl-env-use run envs/<name> --model local-text --limit 5 --concurrency 1`, then remove `--limit` and add
   `--dry-run` to inspect the full run. Agree on local runtime/concurrency limits if unresolved; zero dollar estimates
   do not mean zero cost.
5. Baseline each policy with `rl-env-use run envs/<name> --model local:MODEL_ID --flow models` (add `--verified` for
   the broader pool). Each model gets its own slot. `report envs/<name>` regrades, audits and writes reports; it can
   call stored trusted judges unless `--no-judges` is supplied. Investigate inverted cases as possible task or grader
   bugs. Do not commit generated environments unless requested.
6. Use `climb` as above for case tasks, or manually `split envs/<name>`, then
   `run envs/<name> --model local-text --variant vN --system prompt.txt`. Label failures in the review app,
   `codify envs/<name> --model local-text`, then `judges envs/<name> --model local-text`; judges become trusted only
   after calibration against the user's labels.

## Export and existing environments

`rl-env-use export envs/<name>` writes `dataset.jsonl` for an external trainer. The CLI exporter uses the default
panel-hard served pool; it does **not** inherit a previous run's `--verified`. To export the broader verified pool:

```python
from rl_env_use import Env

env = Env("envs/<name>", verified=True)
env.export_jsonl("envs/<name>/dataset-verified.jsonl")
observation = env.reset(env.ids[0])  # public input only; gold stays in the reward function
reward = env.reward(env.ids[0], "<policy answer>")
```

For shipped or older environments, follow the migration section of [docs/OPEN_STACK.md](docs/OPEN_STACK.md) before
running them. Work on a copy; replace hosted `agent_model`, `closed_book`, `panel` and `adjudicators` entries,
replace hosted `writer_model`/`critic_model` roles too, rebuild verification and panel results, and recalibrate stored hosted judges. Closed-book checks may be cached;
clear the relevant checks when deliberately repeating verification. Use new run/climb directories after changing
the actual model or endpoint behind an alias: resumption caches do not fingerprint weights or endpoint settings.

`compare` invokes proprietary Claude Code and is blocked in open-only mode. Compare local policies through
`run --flow models --model local:MODEL_ID`. Hosted compatibility requires an explicit user choice and
`RLENV_OPEN_ONLY=0` before starting Python; consult the guide rather than switching automatically on failure.

## Saved artifacts

Each environment is a folder under `envs/<name>/`:

```text
spec.json          skill, sources, sampling, model roles, limits; create adds spend and result
kb.json            mined materials, quote-verified outcome facts, missions
tasks.jsonl        drafted tasks, golds and checks; Env filters the served pool
audit.json         latest audit, also printed by report
reviews.jsonl      case/task verdicts and gold fixes from the review app
annotations.jsonl  trace labels used by codify and judges
models/            comparison slots, model identified in each change.md
climb-<model>/     sanitized model name; adds -verified for the broader pool
                   baseline, v1..vN, _state.json, report.html
hillclimb/         manual runs and split; ablation/ holds --blind runs
<flow>/<slot>/     results.jsonl, errors.jsonl, traces/
dataset.jsonl      default CLI export
```

Slots are `baseline`, `v1`, `v2`, etc. Each climb slot's `change.md` describes the edit and `prompt.txt` stores the
prompt extension. Infrastructure failures are recorded separately from graded answers. Keep model/pool provenance
with results and distinguish a passing integration check from evidence of mining quality or policy improvement.
