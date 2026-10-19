"""Stage 6: executable version of the eval-health checklist. Every finding cites numbers, severity first."""

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from .graders import grade_answer, norm, numbers
from .llm import price_of
from .store import load_jsonl
from .tasks import balanced, keep, load_tasks


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
	if n == 0:
		return (0.0, 1.0)
	p, d = k / n, 1 + z * z / n
	c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
	return (max(0.0, c - h), min(1.0, c + h))


def noise_floor(n: int, reps: int = 1) -> float:
	return 1 / math.sqrt(max(1, n * reps))


def split(tasks: list[dict], test_frac: float = 0.5, seed: int = 0) -> tuple[list[str], list[str]]:
	"""Random split stratified by category - never by score. Tasks of one case (same materials) stay on one side, or the
	test set would share material with what the hillclimber reads. Strata are shuffled then concatenated and assigned
	systematically, so singleton categories still alternate and the test set is never empty."""
	rng, groups = random.Random(seed), defaultdict(list)
	for t in sorted(tasks, key=lambda t: t['id']):
		groups[t.get('case') or t['id']].append(t)
	by = defaultdict(list)  # a case is stratified by its most common category, and by its forecasts' outcomes: a train set of
	for g, ts in sorted(groups.items()):  # all-yes forecasts rewards 'raise every probability', which a balanced test punishes
		outcome = ''.join(sorted({t['answer'] for t in ts if t.get('answer_type') == 'prob'}))
		by[(Counter(t.get('category') or (t.get('tags') or [''])[0] for t in ts).most_common(1)[0][0], outcome)].append(g)
	order = []
	for cat in sorted(by):
		rng.shuffle(by[cat])
		order += by[cat]
	test = {g for k, g in enumerate(order) if math.floor((k + 1) * test_frac) > math.floor(k * test_frac)}
	return sorted(t['id'] for g in order if g not in test for t in groups[g]), sorted(t['id'] for g in test for t in groups[g])


def paired(base: dict[str, float], new: dict[str, float], case_of: dict[str, str] | None = None, B: int = 4000, seed: int = 0) -> dict:
	"""new - base on the tasks both have (per-task means over reps), with a bootstrap that resamples whole cases: tasks of
	one case are correlated, and treating them as independent understates the interval."""
	ids = sorted(set(base) & set(new))
	if not ids:
		return {'n': 0, 'diff': 0.0, 'lo': 0.0, 'hi': 0.0, 'p_better': 0.5}
	by = defaultdict(list)
	for i in ids:
		by[(case_of or {}).get(i) or i].append(new[i] - base[i])
	groups = list(by.values())
	rng, mean = random.Random(seed), sum(new[i] - base[i] for i in ids) / len(ids)
	boot = []
	for _ in range(B):
		pick = [g for _ in groups for g in [groups[rng.randrange(len(groups))]]]
		boot.append(sum(map(sum, pick)) / sum(map(len, pick)))
	boot.sort()
	return {
		'n': len(ids),
		'cases': len(groups),
		'diff': round(mean, 4),
		'lo': round(boot[int(0.025 * B)], 4),
		'hi': round(boot[int(0.975 * B) - 1], 4),
		'p_better': round(sum(b > 0 for b in boot) / B, 4),
	}


def rep_halves(rows: list[dict]) -> tuple[dict, dict]:
	"""A/A check from one run: per-task means of the even and of the odd reps (needs >= 2 reps)."""
	even, odd = case_means([r for r in rows if r['rep'] % 2 == 0]), case_means([r for r in rows if r['rep'] % 2 == 1])
	return even, odd


def perturb(t: dict) -> str | None:
	"""A wrong-but-plausible answer that a sound grader must reject."""
	if t['answer_type'] == 'number':
		g = numbers(t['answer'])[0]
		tol = max(t.get('abs_tol', 0), t.get('rel_tol', 0) * abs(g))
		return str(g + max(1, abs(g) * 0.1, 2 * tol + 1))
	if t['answer_type'] == 'set':
		items = [x.strip() for x in t['answer'].replace(';', ',').split(',') if x.strip()]
		return ', '.join(items[:-1]) if len(items) > 1 else None
	if t['answer_type'] == 'date':
		return '1 January 1900'
	return None


def within_tol(t: dict, other: str) -> bool:
	"""A numeric gold of another task that genuinely lies inside this task's tolerance (two ratios of 1.7) is a coincidence, not a lenient grader."""
	if t['answer_type'] != 'number' or len(numbers(other)) != 1:
		return False
	g, o = numbers(t['answer'])[0], numbers(other)[0]
	return abs(o - g) <= max(t.get('abs_tol', 0), t.get('rel_tol', 0) * abs(g)) + 1e-9


def grader_selftest(tasks: list[dict]) -> dict:
	"""Oracle must pass; null, someone else's gold, and perturbed gold must fail."""
	bad = defaultdict(list)
	golds = [t['answer'] for t in tasks]
	full = lambda t, a: grade_answer(t, a) >= 1 - 1e-9  # noqa: E731  forecasts score in [0, 1]: only a full score counts as accepted
	for t in tasks:
		if not full(t, t['answer']):
			bad['oracle_fails'].append(t['id'])
		for a in t.get('aliases', []) if t['answer_type'] not in ('number', 'date') else []:  # those graders ignore aliases
			if not full(t, a):
				bad['alias_fails'].append(t['id'])
		for null in ('', "I don't know", 'N/A'):
			if full(t, null):
				bad['null_passes'].append(t['id'])
		mine = {norm(g) for g in [t['answer'], *t.get('aliases', [])]}
		foreign = [g for g in golds if t['answer_type'] != 'prob' or norm(g) in ('yes', 'no')]  # a count of 1 or 100 reads as a certain forecast
		if any(norm(g) not in mine and not within_tol(t, g) and full(t, g) for g in foreign):
			bad['other_gold_passes'].append(t['id'])
		p = perturb(t)
		if p and full(t, p):
			bad['perturbed_passes'].append(t['id'])
	return dict(bad)


def strength(model: str) -> float:
	p = price_of(model or '')
	return p[1] if p else 0.0


def load_variants(flow: Path) -> dict[str, list[dict]]:
	return (
		{d.name: load_jsonl(d / 'results.jsonl') for d in sorted(flow.iterdir()) if d.is_dir() and (d.name == 'baseline' or (d.name[0] == 'v' and d.name[1:].isdigit()))}
		if flow.exists()
		else {}
	)


def case_means(rows: list[dict], metric: str = 'correct') -> dict[str, float]:
	acc = defaultdict(list)
	for r in rows:
		if r.get('status') == 'ok' and metric in r['grade']:
			acc[r['prompt_id']].append(float(r['grade'][metric]))
	return {k: sum(v) / len(v) for k, v in acc.items()}


def audit(env_dir: Path) -> list[dict]:
	env, F = Path(env_dir), []

	def find(sev, check, msg, **data):
		F.append({'severity': sev, 'check': check, 'message': msg, **data})

	if not check_tasks(env, find):
		return F
	for flow in sorted(d.name for d in env.iterdir() if d.is_dir() and any(d.glob('*/results.jsonl'))):
		variants_section(env, find, flow)
	judges_p = env / 'judges.json'
	for j in json.loads(judges_p.read_text()) if judges_p.exists() else []:
		s = j.get('stats', {})
		if not s:
			find('medium', 'judge', f'judge {j["mode"]["id"]} is not calibrated against human labels')
		elif not s.get('trusted'):
			find(
				'high',
				'judge',
				f'judge {j["mode"]["id"]} disagrees with humans (TPR {s.get("tpr")}, TNR {s.get("tnr")}, flip {s.get("flip_rate")}, n={s.get("n")}) - do not optimize against it',
			)
		else:
			find('info', 'judge', f'judge {j["mode"]["id"]} calibrated: TPR {s["tpr"]}, TNR {s["tnr"]}, flip {s.get("flip_rate")}, n={s["n"]}')
	order = {'high': 0, 'medium': 1, 'info': 2}
	return sorted(F, key=lambda f: order[f['severity']])


def check_tasks(env: Path, find) -> bool:
	tasks = load_tasks(env)
	every = [t for t in tasks if keep(t)]
	kept = balanced(every)
	if not tasks:
		find('high', 'tasks', 'no tasks.jsonl - run `build` first')
		return False
	c = Counter(t.get('category') for t in kept)
	find('info', 'tasks', f'{len(kept)}/{len(tasks)} tasks pass all filters; categories {dict(c)}')
	if any('needs_browser' in t for t in kept):
		find(
			'info',
			'edge',
			f'{sum(t.get("needs_browser", False) for t in kept)}/{len(kept)} kept tasks rest on evidence a plain HTTP fetch of the same page does not show (browser-only data)',
		)
	if len(every) > len(kept):  # a policy that shifts every probability toward the majority would score without forecasting
		fc = Counter(t['answer'] for t in every if t.get('answer_type') == 'prob')
		find(
			'medium',
			'base_rate',
			f'{fc["yes"]}/{sum(fc.values())} verified forecasts resolve yes: {len(every) - len(kept)} majority-outcome ones are left out so at most 60% resolve one way - more cases, or thresholds on the other side, bring them back',
		)
	if any(t.get('context') for t in kept):
		find(
			'info',
			'cases',
			f'{len({t.get("case") for t in kept})} cases; kinds {dict(Counter(t.get("kind") for t in kept))}; {sum(t["checks"].get("closed_book") == "skipped" for t in kept)} few-choice tasks skip the memorization check (a closed-book hit there is chance)',
		)
	if len(kept) < 15:
		find('medium', 'size', f'{len(kept)} kept tasks: one flaky case swings the score by {100 / max(1, len(kept)):.0f} points (aim for 15-100)')
	for key, label in (('leakage', 'gold leaks into prompt'), ('structural', 'structural problems')):
		ids = [t['id'] for t in tasks if t['checks'].get(key)]
		if ids:
			find('info', key, f'{len(ids)} tasks: {label} (filtered out)', ids=ids)
	mem = [t['id'] for t in tasks if isinstance(t['checks'].get('closed_book'), dict) and any(v is True for v in t['checks']['closed_book'].values())]
	if mem:
		find('info', 'memorizable', f'{len(mem)}/{len(tasks)} tasks answerable closed-book - they test recall, not the skill (filtered out)', ids=mem)
	if not any('closed_book' in t['checks'] for t in tasks):
		find('medium', 'memorizable', 'closed-book check not run: tasks may be answerable from memory')
	dead = [t['id'] for t in tasks if t['checks'].get('live') is False]
	if dead:
		find('info', 'stale', f'{len(dead)} tasks cite evidence no longer on the live page (filtered out); re-run verify periodically', ids=dead)
	amb = [
		t['id']
		for t in tasks
		if 'critic' in t['checks'] and not (t['checks']['critic']['unambiguous'] and t['checks']['critic']['gold_supported'] and t['checks']['critic'].get('agrees', True))
	]
	if amb:
		find('info', 'ambiguous', f'{len(amb)} tasks flagged by critic: ambiguous, unsupported, or its independent recomputation disagrees with gold (filtered out)', ids=amb)
	reviews = {r['id'] for r in load_jsonl(env / 'reviews.jsonl')}
	unreviewed = [t['id'] for t in kept if t['id'] not in reviews]
	if unreviewed:
		auto = json.loads((env / 'spec.json').read_text()).get('auto') if (env / 'spec.json').exists() else False
		find(
			'info' if auto else 'medium',
			'signoff',
			f'{len(unreviewed)}/{len(kept)} kept tasks have no human review' + (' (autonomous env: automated gates only)' if auto else ' yet (open `review`)'),
		)
	st = grader_selftest(kept)
	for k, ids in st.items():
		find('high', 'grader', f'grader self-test: {k} on {len(ids)} tasks', ids=ids)
	if not st:
		find('info', 'grader', f'grader self-test clean on {len(kept)} tasks (oracle passes; null, foreign gold, perturbed gold fail)')
	return True


def variants_section(env: Path, find, flow: str = 'hillclimb') -> None:
	variants = {(v if flow == 'hillclimb' else f'{flow}/{v}'): rows for v, rows in load_variants(env / flow).items()}
	state_p = env / flow / '_state.json'
	state = json.loads(state_p.read_text()) if state_p.exists() else {}
	ids_ = [m['id'] for m in state.get('metrics', []) if m.get('kind', 'binary') == 'binary']
	present = lambda m: any(m in r['grade'] for rows in variants.values() for r in rows)  # noqa: E731
	metric = next((m for m in ids_ + ['supported', 'correct'] if present(m)), 'correct')
	per_model = {}
	for v, rows in variants.items():
		ok = [r for r in rows if r.get('status') == 'ok']
		errs = load_jsonl(env / flow / v.split('/')[-1] / 'errors.jsonl')
		done = {(r['prompt_id'], r['rep']) for r in ok}
		live_errs = [e for e in errs if (e['prompt_id'], e['rep']) not in done]
		refused = [e for e in live_errs if e['failure_class'] == 'refusal']
		live_errs = [e for e in live_errs if e['failure_class'] != 'refusal']
		if not ok:
			continue
		k, n = sum(r['grade'].get(metric, 0) for r in ok), len(ok)
		reps = max(r['rep'] for r in ok) + 1
		cm = case_means(ok, metric)
		ncase = len(cm)
		lo, hi = wilson(sum(cm.values()), ncase)  # reps of one case are correlated: the case is the unit
		models = sorted({r['model'] for r in ok})
		per_model[v] = (models[0] if len(models) == 1 else 'mixed', cm)
		spend = sum(r.get('cost_usd', 0) for r in ok)
		find(
			'info',
			'score',
			f'{v} ({", ".join(models)}): '
			+ (
				f'{k / n:.2f} mean score (1 - Brier where a forecast) [{lo:.2f}, {hi:.2f}]'
				if any(r['grade'].get(metric, 0) not in (0, 1) for r in ok)
				else f'{k / n:.0%} {metric} [{lo:.0%}, {hi:.0%}]'
			)
			+ f' on {ncase} cases x {reps} reps; noise floor ~±{noise_floor(ncase, reps):.0%}; ${spend:.2f}; {len(live_errs)} infra errors',
		)
		if flow == 'ablation':  # materials withheld: a floor is the expected result, not a broken grader
			pass
		elif k / n >= 0.95:
			find('high', 'saturated', f'{v} scores {k / n:.0%} on {metric}: no headroom - add harder tasks before hillclimbing')
		elif k == 0:
			find('high', 'floor', f'{v} scores 0%: suspect a broken task set or grader before concluding the tasks are hard')
		over = [r for r in ok if r['meta'].get('agent_claimed_success') and not r['grade'].get('correct', 1)]
		if over:
			find(
				'info',
				'overclaim',
				f'{v}: agent claimed success on {len(over)}/{n} attempts it got wrong - self-reported success is not a reward signal',
				ids=[r['prompt_id'] for r in over],
			)
		hit = [r for r in ok if r['grade'].get('correct') and 'grounded' in r['grade']]
		short = [r for r in hit if not r['grade']['grounded']]
		if short:
			find(
				'medium' if len(short) / len(hit) > 0.1 else 'info',
				'shortcut',
				f'{v}: {len(short)}/{len(hit)} correct answers were never read at their source (search snippets / third-party copies) - train on `supported`, not `correct`',
				ids=sorted({r['prompt_id'] for r in short}),
			)
		fc = Counter(r['meta'].get('failure_class') for r in ok if not r['grade'].get('correct', 1) and r['meta'].get('failure_class'))
		if fc:
			find('info', 'failures', f'{v}: failure classes {dict(fc)}')
		if refused:
			find(
				'medium',
				'refusal',
				f"{v}: {len(refused)} attempts refused by the model's safeguards - reported apart from capability misses and excluded from the score",
				ids=sorted({e['prompt_id'] for e in refused}),
			)
		if live_errs:
			find(
				'medium',
				'infra',
				f'{v}: {len(live_errs)} attempts failed in the harness (timeouts/crashes) and are excluded from the score',
				classes=dict(Counter(e['failure_class'] for e in live_errs)),
			)
		if v.split('/')[-1] == 'baseline' and state.get('train_ids'):
			tr = [cm[i] for i in state['train_ids'] if i in cm]
			te = [cm[i] for i in state['test_ids'] if i in cm]
			if tr and te and abs(sum(tr) / len(tr) - sum(te) / len(te)) > noise_floor(min(len(tr), len(te)), reps):
				find('medium', 'split', f'baseline train {sum(tr) / len(tr):.0%} vs test {sum(te) / len(te):.0%} differ beyond noise - re-draw the split')

	if len(per_model) >= 2:
		ordered = sorted(per_model.items(), key=lambda kv: strength(kv[1][0]))
		(_, (wm, weak)), (_, (sm, strong)) = ordered[0], ordered[-1]
		common = sorted(set(weak) & set(strong))
		if common and strength(wm) < strength(sm):
			ws, ss = sum(weak[i] for i in common) / len(common), sum(strong[i] for i in common) / len(common)
			inv = [i for i in common if weak[i] - strong[i] > 0.5]  # by more than half the score range: on Brier-scored forecasts small inversions are noise
			sev = 'high' if ss <= ws else 'info'
			find(
				sev,
				'gradient',
				f'capability gradient on {metric}: {wm} {ws:.0%} -> {sm} {ss:.0%} on {len(common)} shared cases'
				+ (' (no gradient: suspect ambiguous tasks or grader)' if ss <= ws else ''),
			)
			if inv:
				find('medium', 'inverted', f'{len(inv)} cases where {wm} beats {sm} - inspect these first, usually a task or grader bug', ids=inv)
		allm = [m for _, m in per_model.values()]
		ids = set.intersection(*(set(m) for m in allm))
		flat = [i for i in ids if len({round(m[i]) for m in allm}) == 1 and all(m[i] in (0, 1) for m in allm)]
		if ids:
			find('info', 'discrimination', f'{len(flat)}/{len(ids)} cases give the same all-pass/all-fail result across every variant - they cannot discriminate between models')
		stuck = [i for i in ids if all(m[i] == 0 for m in allm)]
		if stuck:
			find('medium', 'never_solved', f'{len(stuck)} cases fail for every model - check for broken gold before calling them hard', ids=stuck)
