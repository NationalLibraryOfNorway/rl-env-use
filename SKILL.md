---
name: rl-env-use
description: Build an RL environment / eval that trains or measures a skill in any domain (finance, medicine, law, math, science, a niche industry) from real cases that Browser Use agents mine from the web, then baseline and hillclimb a model on it. Use when the user knows a domain and what a model should be able to do in it, but not how to get the data or build the environment.
---

# rl-env-use

Anthropic's claude-api `build-eval` -> `hillclimb` workflow, with real inputs. If the claude-api skill is available, its
`shared/evals/build-eval.md` and `shared/evals/eval-hillclimb.md` are the method; this skill changes where the inputs come
from. They are not logs or synthesized prompts: they are real cases that browser agents mine where they live - filings,
registries, case records, contest archives, portals behind search forms or logins. A case = the materials the model is
given + its true outcome, published elsewhere or later. The model under test reads the materials and answers with no
browser: the environment tests the domain skill. Browsing is how the environment gets built.

Commands: `rl-env-use ...` (or `python -m rl_env_use ...` from the repo). Every env is one folder, `./envs/<name>/`.

## Where things are saved

```
envs/<name>/
  spec.json        what the env is: topic, skill, sources, conventions, budget; `create` adds spend so far and its result
  kb.json          what the agents mined: materials, quote-verified outcome facts, every mission
  tasks.jsonl      every drafted task with its gold and its checks (the env serves the ones that pass)
  audit.json       the latest audit (also printed by `report`)
  reviews.jsonl    your verdicts and gold fixes from the review app (annotations.jsonl: trace labels; judges.json)
  models/          model comparisons: one slot per model, named in <slot>/change.md
  climb-<model>/   a hillclimb: baseline, v1..vN (change.md = the edit, prompt.txt = the full prompt), _state.json, report.html
  hillclimb/       runs by hand (`run` without --flow); ablation/: runs with the materials withheld (--blind)
  <flow>/<slot>/   results.jsonl (one graded row per attempt), errors.jsonl (infra failures), traces/ (full transcripts)
```
Slots are `baseline`, `v1`, `v2`, ... because the report reads only these; `change.md`'s first line is the column title.
Traces, trajectory/ and logs are git-ignored (large, regenerable); results, reports and climb state are kept.

## Autonomous (default when the user wants it hands-off)
1. `create "<domain, skill, sources, constraints - one line>" [--tasks 150] [--budget 40]`: a planner writes the spec
   (skill, sources, conventions, an outcome-blind case-selection rule), then mining rounds run until the target number of
   kept tasks, the budget, or two rounds without new tasks. Every round: scout -> synth -> verify -> panel -> harden ->
   verify -> panel. Hard tasks are the design goal: every task the whole panel solved goes back to the drafter with the
   env's calibration (what was too easy, what split the panel and why) and comes back harder on the same case, up to two
   levels; new drafts start from that calibration. `build --only harden` does the same on any env. Derived numbers
   are correct by construction (code-evaluated expression over numbers in the materials, typical mistakes must fail the
   grader); difficulty is measured on a frontier panel from three labs (opus55, gpt-astra, gemini - closed frontier
   models answer only and reason internally): all-pass tasks are dropped, all-fail or converging-wrong ones need an
   adjudicator, tasks whose answers state several values are rejected as multi-value prompts, forecasts (yes/no,
   Brier-scored) must move with the materials and not be answerable without them. When the panel saturates derive
   tasks, later rounds write forecasts - measured on SEC filings, the frontier solves nearly every computation from an
   excerpt; its headroom is in forecasts. Derive tasks remain training data for smaller policies (pass `--panel` with
   the policy family to build that tier). Forecast outcomes are balanced by construction: at most 60% of the served
   forecasts resolve one way, and the drafter is told which side is short.
   Mining is fastest when the line names a source URL that LISTS records (a registry API query, a search results page):
   the planner fetches it and hands every mission its own records by id. Outcomes that are tables (trial results,
   statements) are saved as verbatim passages. Ctrl-C or kill stops a run cleanly (cloud browsers are released);
   `create --name <env>` resumes it with its spend so far.
2. `climb envs/<name> --model M`: the eval-hillclimb loop as code, in `climb-<model>/` (split by case, A/A noise from rep
   halves, refuses saturated or underpowered setups, K edits per round screened on train, kept only on a paired
   case-bootstrap test gain, failure audit on stalls, final baseline-vs-best with fresh reps). Report the final line and
   `climb-<model>/report.html` as they are - "within noise - do not merge" is a result.

On a terminal both commands show a live board (stage bar with ETA, spend vs budget, overall ETA, kept tasks and yes/no
balance for create; baseline, best, per-round marks and score sparkline for climb); in a log they print status lines.

## Step by step (when the user wants to review)
Ask every decision with AskUserQuestion, your pick first as "(Recommended)". Wait for a clear yes at each sign-off.

### Step 0 - what is being evaluated
Ask, in this order:
1. The skill, as input -> outcome: what does the model get, and what must it produce? ("given a trial's registered
   protocol, predict whether it met its primary endpoint"; "given a 10-Q, compute net debt / EBITDA"). One skill per env.
2. Where do inputs and outcomes live? (2-6 URLs). Good skills pair data separated in place or time: an input document and
   an outcome published later or elsewhere. Prefer exact outcomes (a number, a name, a category) over yes/no.
3. Logins (-> Browser Use cloud profile id) or a country (-> proxy country)?
Then `new <name> --topic "<domain>" --skill "<input -> outcome>" --sources URL.. [--profile ID] [--proxy-country CC]`.
Without `--skill` the env is a browse task env instead (the model must find answers on the live web) - only when the
user wants to measure browsing itself.

### Step 1 - inputs: mine the cases, then look at them (sign-off 1)
`build envs/<name>` - browser agents collect cases (~$0.15-0.40 per mission) and it stops before any task exists. Show
the look output (complete cases, how much of the data a plain fetch would have missed; `build envs/<name> --only look` prints it again). Have the user open
`review envs/<name>` -> Data tab: materials and outcome facts, reject wrong, leaky or ambiguous ones.
Too few cases: `build envs/<name> --only scout` runs another round (it first finishes cases that lack an outcome).
Tasks that compute from data (statements, results tables) need that data in the materials, not only in the outcome:
`build envs/<name> --only scout --enrich` sends one agent per complete case to save it.
SIGN-OFF: "Are these the cases the environment should be built from?"

### Step 2 - tasks and grading (sign-off 2)
`build envs/<name>` again: tasks are written per case and checked - gold from the outcome quotes, no leak into prompt or
materials, not answerable from memory (closed-book), outcome evidence still live, critic: unambiguous, gold supported,
not stated in the materials, well posed; a derived number's gold is computed by code from the task's arithmetic
expression, never by the drafting model. Grading is programmatic (number / date / set / exact / contains / prob with
tolerances), self-tested against null, foreign and perturbed golds. Show 5 tasks with materials, gold and outcome
evidence (Tasks tab). SIGN-OFF: "Would you grade any of these differently?" Gold fixes go in through the app.

### Step 3 - pilot and cost (sign-off 3)
`run envs/<name> --model M --limit 5` is the pilot; then `run ... --dry-run` prices the full run from the pilot's
measured $ and seconds per episode. SIGN-OFF on that number before the full run.

### Step 4 - baseline, headroom, hand over
Compare models in their own flow: `run envs/<name> --model M --flow models` per model (each gets its own named slot),
so comparisons stay out of the climb's rounds. Refusals (a model's safeguards declining domain content) are reported
apart from capability misses. `report envs/<name>` regrades every flow, audits and writes each `report.html`. Act on it:
`inverted` cases -> inspect, usually a task or grader bug. Saturated or no gradient: headroom is relative to the policy
being trained. Derive tasks verified by a frontier critic saturate frontier models but leave room for the smaller model
the user will actually train - pick that as the climb's policy. For headroom on frontier models, use forecast tasks
(protocol -> later outcome) with cases mined from lesser-known records, so the closed-book check does not reject them as
memorized. Ask whether to commit the env.

### Step 5 - hillclimb
`climb envs/<name> --model M` (above), or by hand: `split envs/<name>` writes the stratified train/test split into
`hillclimb/_state.json`, then each round is `run envs/<name> --model M --variant vN --system prompt.txt` (the system
prompt is the surface). Open-code failures in the Traces tab, `codify`, then `judges` adds narrow judges, trusted only
once they agree with the user's labels. `export envs/<name>` writes the dataset (prompt + context + verifiable answer
spec) for an RL trainer; `from rl_env_use import Env` serves it in code (`reset`, `reward`).

## Optional - measure the edge
`compare envs/<name> --missions 8` reruns the mining with vanilla Claude Code (WebSearch + WebFetch, same model,
same instructions) and pushes both arms' data through the same synth + verify: cases found, tasks kept, $ per task.
