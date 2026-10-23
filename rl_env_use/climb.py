"""Autonomous hillclimb of the policy's system prompt: the claude-api eval-hillclimb loop as code.

Split by case -> baseline (+ A/A noise from rep halves) -> rounds: a proposer reads only TRAIN failures, buckets them
by root cause and proposes K one-change edits -> each is screened on train -> the best goes to test -> kept only if test
improves with a paired, case-clustered bootstrap (train, on fresh reps, must not fall) -> two rejections in a row trigger
a failure audit that sets aside train tasks the proposer marks as task/grader faults -> final: baseline vs best on test
with fresh reps, never the reps a decision was made on. The proposer never sees test traces or scores, only the
keep/revert bit of its own edits. A rep that errored, timed out or was refused scores 0: a prompt that makes the policy
refuse hard tasks must not look better for it. Resumable: the incumbent, stall count and spend live in _state.json.
"""

import json
import re
from pathlib import Path

from pydantic import BaseModel, Field

from . import ui
from .audit import paired, split
from .config import TEXT_MODEL
from .env import Env
from .llm import LEDGER, call
from .rollout import rollout
from .store import load_jsonl
from .ui import spark


class Bucket(BaseModel):
	name: str = Field(description='Root cause in a few words, e.g. used year-to-date figure instead of the quarter.')
	task_ids: list[str]
	fixable: bool = Field(
		description='True if better instructions to the model could fix it; False if the task or its gold is at fault (ambiguous prompt, wrong gold, missing data).'
	)


class Edit(BaseModel):
	bucket: str = Field(description='The bucket this edit targets.')
	change: str = Field(description='One line: what changed and why.')
	prompt: str = Field(description='The COMPLETE new system prompt: the current one with this single change applied.')


class Proposal(BaseModel):
	buckets: list[Bucket]
	edits: list[Edit]


PROPOSE_SYS = """You improve the system prompt of a model that answers domain tasks from case materials (it gets the materials and one question, no tools).
You see its failures on TRAINING tasks: question, its reasoning, its answer, the gold. First bucket every failure by root cause, and say whether
better instructions could fix it. Then propose {k} DIFFERENT edits, each ONE change aimed at the root cause of a fixable bucket (the largest
first): add, rewrite or delete one instruction - rewrite the part that causes the failure rather than piling on rules. A worked example of a
method (with invented numbers) often beats a rule. Rules: general domain method only - never an entity, number or answer from these cases;
the whole prompt stays under {cap} characters; keep what works; do not resubmit an idea listed as already rejected, not even
reworded or strengthened - it failed on held-out tasks."""

CAP = 3000


def leaked(prompt: str, tasks: list[dict]) -> list[str]:
	"""Case-specific content in a proposed prompt: numbers of 3+ digits from the materials or golds, or case keys."""
	specific = lambda n: len(re.sub(r'\D', '', n).strip('0')) >= 3  # noqa: E731  4,527 or 61.05, not 100 or 1,000
	nums = {n for n in re.findall(r'\d[\d,.]*\d', ' '.join(t.get('context', '') + ' ' + t['answer'] for t in tasks)) if specific(n)}
	hits = [n for n in re.findall(r'\d[\d,.]*\d', prompt) if n in nums]
	keys = {str(t.get('case')) for t in tasks if t.get('case') and len(str(t.get('case'))) > 4}
	return hits + [k for k in keys if k.lower() in prompt.lower()]


def failures(rows: list[dict], flow: Path, variant: str, ids: set, limit: int = 40) -> str:
	out = []
	for r in sorted(
		(r for r in rows if r['prompt_id'] in ids and r.get('status') == 'ok' and r['grade'].get('correct', 0) < 1), key=lambda r: (r['grade']['correct'], r['prompt_id'], r['rep'])
	):
		if any(o.startswith(f'[{r["prompt_id"]}]') for o in out):
			continue  # one rep per task: more tasks beat more reps of the same one
		tp = flow / variant / 'traces' / f'{r["prompt_id"]}_rep{r["rep"]}.json'
		why = next((m.get('thinking') or '' for m in json.loads(tp.read_text()) if m['role'] == 'assistant'), '') if tp.exists() else ''
		out.append(
			f'[{r["prompt_id"]}] Q: {r["prompt"][:600]}\nREASONING: {why[-1500:]}\nANSWER: {r["meta"].get("answer")} | GOLD: {r["meta"].get("gold")} | score {r["grade"]["correct"]:.2f}'
		)
		if len(out) >= limit:
			break
	return '\n\n'.join(out)


def write_variant(flow: Path, v: str, prompt: str, change: str, model: str) -> None:
	d = flow / v
	if (d / 'prompt.txt').exists() and (d / 'prompt.txt').read_text() != prompt and (d / 'results.jsonl').exists():
		raise SystemExit(f'{d} holds runs of a different prompt: resuming would mix them - delete {flow} to start over')
	d.mkdir(parents=True, exist_ok=True)
	(d / 'prompt.txt').write_text(prompt)
	(d / 'change.md').write_text(change + '\n')
	(d / 'summary.json').write_text(json.dumps({'description': change, 'target': 'system_prompt', 'model': model}))


async def climb(
	env: Path,
	model: str,
	rounds: int = 8,
	k: int = 3,
	reps: int = 3,
	budget: float = 20.0,
	proposer: str = TEXT_MODEL,
	system: str = '',
	test_frac: float = 0.5,
	seed: int = 0,
	confidence: float = 0.95,
	mde: float = 0.08,
	force: bool = False,
	concurrency: int = 8,
	verified: bool = False,
	log=print,
) -> dict:
	flow = env / (f'climb-{re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")}' + ('-verified' if verified else ''))
	sp = flow / '_state.json'
	state = json.loads(sp.read_text()) if sp.exists() else {}
	tasks = {t['id']: t for t in Env(env, verified=verified).tasks() if t.get('context')}
	if 'train_ids' not in state:
		state['train_ids'], state['test_ids'] = split(list(tasks.values()), test_frac, seed)
	train, test = set(state['train_ids']) & set(tasks), set(state['test_ids']) & set(tasks)
	case_of = {i: t.get('case') or i for i, t in tasks.items()}
	state |= {
		'policy': model,
		'surface': 'system prompt (CASE_SYS + extension)',
		'proposer': proposer,
		'reps': reps,
		'confidence': confidence,
		'split_seed': seed,
		'approve_each_round': False,
	}
	quarantined, prior = set(state.get('quarantined', [])), state.get('usd', 0.0)
	spent = lambda: prior + LEDGER.usd  # noqa: E731
	done = len(state.get('rounds', []))  # rounds of an earlier run of this climb
	ui.money(spent, budget)
	flow.mkdir(parents=True, exist_ok=True)

	def save():
		state.update(quarantined=sorted(quarantined), usd=round(spent(), 4))
		sp.write_text(json.dumps(state, indent=1))
		show()

	def show():
		"""The board: baseline and incumbent on test, the incumbent's test score per round, one mark per round."""
		b, rl = state.get('baseline'), state.get('rounds', [])
		ui.line('policy', f'{model} · {len(train)} train / {len(test)} test tasks · {reps} reps · {k} edits per round')
		if b:
			ui.line('baseline', f'test {b["test"]:.3f}  train {b["train"]:.3f}  noise ±{b["noise"]}  smallest detectable gain {b["mde"]}')
		if rl:
			best = state.get('best', {})
			ui.line(
				'best',
				f'{best.get("variant")}: test {best.get("test_score", 0):.3f}' + (f'  ({best["test_score"] - b["test"]:+.3f} vs baseline)' if b and 'test_score' in best else ''),
			)
			ui.line(
				'rounds', ' '.join(f'r{r["round"]}' + ('[green]✓[/green]' + f'{r["test"]["diff"]:+.2f}' if r['decision'] == 'keep' else '[red]✗[/red]') for r in rl), markup=True
			)
			hist = [b['test']] + [r['test_now'] for r in rl if 'test_now' in r]
			ui.line('test', spark(hist, min(hist) - 0.02, max(hist) + 0.02) + f'  incumbent test score by round ({min(hist):.2f}-{max(hist):.2f})')
		ui.progress(min(1.0, len(rl) / max(1, done + rounds)) if b else 0.0)

	async def run(v: str, ids: set, prompt: str, hi: int, lo: int = 0) -> dict:
		"""Per-task mean over reps [lo, hi); a missing rep (error, timeout, refusal) scores 0."""
		ui.line('now', f'{v} on {"train" if not ids & test else "test" if not ids & train else "train + test"}, reps {lo}-{hi - 1}')
		await rollout([tasks[i] for i in sorted(ids)], flow, v, model, hi, 25, 1800, concurrency, True, None, prompt or None, lo)  # the ceiling outlasts a call's own retries
		got = reps_of(v)
		return {i: sum(got.get((i, j), 0.0) for j in range(lo, hi)) / (hi - lo) for i in ids}

	def reps_of(v: str) -> dict:
		return {(r['prompt_id'], r['rep']): r['grade'].get('correct', 0) for r in load_jsonl(flow / v / 'results.jsonl') if r.get('status') == 'ok'}

	n_cases = len({case_of[i] for i in test})
	if n_cases < 10 and not force:  # known before any spend
		state['final'] = {'decision': 'not run', 'reason': f'underpowered: {n_cases} test cases - a case bootstrap needs at least 10 (more cases, or --test-frac)'}
		save()
		log(state['final']['reason'])
		return state
	save()
	write_variant(flow, 'baseline', system, 'baseline', model)
	base = await run('baseline', train | test, system, reps)
	got = reps_of('baseline')  # A/A: even vs odd reps, a missing rep scoring 0 as everywhere else
	half = lambda js: {i: sum(got.get((i, j), 0.0) for j in js) / len(js) for i in test}  # noqa: E731
	aa = paired(half(range(0, reps, 2)), half(range(1, reps, 2)), case_of) if reps >= 2 else None
	noise = round(max((aa['hi'] - aa['lo']) / 2 / 2**0.5, 1 / max(1, len(test))), 4) if aa else None  # floor: one test task flipping, even when every rep agrees
	mean = lambda d, ids: round(sum(d[i] for i in ids) / max(1, len(ids)), 4)  # noqa: E731
	mixed = sum(0 < base[i] < 1 for i in train) / max(1, len(train))
	state['baseline'] = {
		'train': mean(base, train),
		'test': mean(base, test),
		'noise': noise,
		'mde': round(1.43 * noise, 4) if noise is not None else None,
		'mixed_train': round(mixed, 3),
	}
	log(
		f'baseline {model}: train {state["baseline"]["train"]:.3f}, test {state["baseline"]["test"]:.3f} on {len(train)}/{len(test)} tasks x {reps} reps; noise ±{noise}; {mixed:.0%} of train tasks mixed across reps; ${spent():.2f}'
	)
	stop = None
	if reps < 2:
		stop = 'need --reps >= 2: without repeated runs there is no noise estimate'
	elif state['baseline']['test'] >= 0.9:
		stop = f'saturated: {model} scores {state["baseline"]["test"]:.0%} on test - climb a weaker policy or build forecast tasks'
	elif state['baseline']['mde'] > mde:
		stop = f'underpowered: the smallest detectable gain is {state["baseline"]["mde"]:.0%} > {mde:.0%} - add test cases or reps (--reps)'
	if stop and not force:
		state['final'] = {'decision': 'not run', 'reason': stop}
		save()
		log(stop)
		return state
	inc_v, stall, s = state.get('incumbent', 'baseline'), state.get('stall', 0), max(1, reps - 1)
	prompt = system if inc_v == 'baseline' else (flow / inc_v / 'prompt.txt').read_text()
	inc = base if inc_v == 'baseline' else (await run(inc_v, train - quarantined, prompt, s + reps, s)) | (await run(inc_v, test, prompt, reps))  # resume: cached
	rounds_log = state.setdefault('rounds', [])
	n_v = max([int(d.name[1:]) for d in flow.glob('v*') if d.name[1:].isdigit()] + [0])
	for rnd in range(len(rounds_log) + 1, len(rounds_log) + rounds + 1):
		if spent() >= budget:
			log(f'budget spent (${spent():.2f} of ${budget:.0f})')
			break
		ids = train - quarantined
		rows = load_jsonl(flow / inc_v / 'results.jsonl')
		rejected = '\n'.join(f'- {r["change"]}' for r in rounds_log if r['decision'] != 'keep') or '-'
		user = f'CURRENT SYSTEM PROMPT EXTENSION (appended to: "You are a domain expert. Read the materials and answer the question."):\n{prompt or "(empty)"}\n\nALREADY REJECTED EDITS:\n{rejected}\n\nTRAIN FAILURES:\n{failures(rows, flow, inc_v, ids)}'
		pr, _, _ = await call(proposer, PROPOSE_SYS.format(k=k, cap=CAP), user, Proposal, temperature=0.7)
		cands = []
		for e in pr.edits[:k]:
			bad = leaked(e.prompt, [tasks[i] for i in train])
			if bad and spent() < budget:  # usually a sound idea whose worked example reuses a case's numbers: one repair, then re-check
				fix, _, _ = await call(
					proposer,
					PROPOSE_SYS.format(k=1, cap=CAP),
					f'This edit copies numbers or names from the training cases ({", ".join(bad[:5])}). Rewrite it with the same idea and invented numbers that appear in no case.\n\nEDIT:\n{e.model_dump_json()}',
					Edit,
					temperature=0.3,
				)
				e, bad = fix, leaked(fix.prompt, [tasks[i] for i in train])
			if len(e.prompt) > CAP or bad or e.prompt.strip() == prompt.strip() or spent() >= budget:
				log(
					f'round {rnd}: edit not run ({"too long" if len(e.prompt) > CAP else f"case content {bad[:3]}" if bad else "no change" if e.prompt.strip() == prompt.strip() else "budget"}): {e.change}'
				)
				continue
			n_v += 1
			v = f'v{n_v}'
			write_variant(flow, v, e.prompt, f'{e.change} [round {rnd}, {e.bucket}]', model)
			cands.append((paired({i: inc[i] for i in ids if i in inc}, await run(v, ids, e.prompt, s), case_of), v, e))
			log(f'round {rnd}: {v} screened on train {cands[-1][0]["diff"]:+.3f} [{cands[-1][0]["lo"]:+.3f}, {cands[-1][0]["hi"]:+.3f}]: {e.change}')
		rec = {'round': rnd, 'candidates': [{'variant': v, 'change': e.change, 'train_screen': sc} for sc, v, e in cands], 'buckets': [b.model_dump() for b in pr.buckets]}
		best = max(cands, key=lambda c: c[0]['diff'], default=None)
		if not best or best[0]['diff'] <= 0 or spent() >= budget:
			rec |= {'change': best[2].change if best else 'no valid edit', 'decision': 'revert (no train gain)' if best is None or best[0]['diff'] <= 0 else 'stop (budget)'}
		else:
			_, v, e = best
			fresh = await run(v, ids, e.prompt, s + reps, s)  # the screen picked the best of K on its own reps: judge train on new ones
			new_test = await run(v, test, e.prompt, reps)
			r_ = paired({i: inc[i] for i in ids if i in inc}, fresh, case_of)
			t_ = paired({i: inc[i] for i in test}, new_test, case_of)
			ok = t_['diff'] > 0 and t_['p_better'] >= confidence and r_['diff'] >= 0
			rec |= {
				'variant': v,
				'change': e.change,
				'train': r_,
				'test': t_,
				'decision': 'keep' if ok else 'revert (test flat or down)' if r_['diff'] >= 0 else 'revert (train fell)',
			}
			if ok:
				inc, inc_v, prompt = fresh | new_test, v, e.prompt
		stall = 0 if rec['decision'] == 'keep' else stall + 1
		rec['test_now'] = mean(inc, test)
		log(
			f'round {rnd}: {rec["decision"]}: {rec["change"]}'
			+ (f' (test {rec["test"]["diff"]:+.3f} [{rec["test"]["lo"]:+.3f}, {rec["test"]["hi"]:+.3f}], P(better) {rec["test"]["p_better"]:.2f})' if 'test' in rec else '')
		)
		if stall == 2:  # failure audit: train tasks the proposer blames on the task, that fail every rep of every variant
			ever = {}
			for d in flow.glob('*/results.jsonl'):
				for r in load_jsonl(d):
					if r.get('status') == 'ok':
						ever[r['prompt_id']] = max(ever.get(r['prompt_id'], 0), r['grade'].get('correct', 0))
			out = {i for b in pr.buckets if not b.fixable for i in b.task_ids if i in train and ever.get(i, 1) == 0} - quarantined
			quarantined |= out
			rec['quarantined'] = sorted(out)
			log(f'round {rnd}: stall - failure audit sets aside {len(out)} train tasks blamed on the task or gold (never solved by any variant)')
		rounds_log.append(rec)
		state |= {'incumbent': inc_v, 'stall': stall, 'best': {'round': 0 if inc_v == 'baseline' else int(inc_v[1:]), 'variant': inc_v, 'test_score': mean(inc, test)}}
		save()
		if stall >= 3 or rec['decision'] == 'stop (budget)':
			log('three rounds without a kept change - stopping' if stall >= 3 else 'budget spent')
			break
	looks = sum('test' in r for r in rounds_log)
	ui.line('final', 'baseline vs best on fresh test reps' if inc_v != 'baseline' else None)
	if inc_v != 'baseline':  # the headline: fresh reps on test for both arms, never the reps a decision was made on
		b2, n2 = await run('baseline', test, system, 2 * reps, reps), await run(inc_v, test, prompt, 2 * reps, reps)
		f = paired(b2, n2, case_of)
		state['final'] = {
			'variant': inc_v,
			'test': f,
			'baseline_test': mean(b2, test),
			'best_test': mean(n2, test),
			'reps': reps,
			'test_looks': looks,
			'decision': 'merge' if f['lo'] > 0 else 'within noise - do not merge',
		}
	else:
		state['final'] = {'variant': 'baseline', 'test_looks': looks, 'decision': 'no change beat the baseline on test'}
	save()
	return state
