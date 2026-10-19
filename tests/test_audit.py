import json

import pytest
from conftest import GT, jl, mk_task

import rl_env_use.audit as A


def test_wilson_and_noise_floor():
	lo, hi = A.wilson(7, 10)
	assert 0.39 < lo < 0.40 and 0.89 < hi < 0.90
	assert A.wilson(0, 0) == (0.0, 1.0)
	assert A.noise_floor(25, 2) == pytest.approx(1 / 50**0.5)


def test_split_stratified_deterministic_disjoint():
	tasks = [mk_task(i, category='a' if i < 10 else 'b') for i in range(30)]
	tr, te = A.split(tasks, 0.5, seed=1)
	assert (tr, te) == A.split(list(reversed(tasks)), 0.5, seed=1)
	assert not set(tr) & set(te) and len(tr) + len(te) == 30
	assert sum(t.startswith('t') and int(t[1:]) < 10 for t in te) == 5


def test_selftest_clean_on_ground_truth():
	assert A.grader_selftest(jl(GT / 'web_tasks.jsonl')) == {}


def test_selftest_catches_lenient_grader(monkeypatch):
	monkeypatch.setattr(A, 'grade_answer', lambda t, p: True)
	bad = A.grader_selftest([mk_task(0), mk_task(1, answer='2024-01-01', answer_type='date')])
	assert {'null_passes', 'perturbed_passes', 'other_gold_passes'} <= set(bad)


def test_selftest_catches_strict_grader(monkeypatch):
	monkeypatch.setattr(A, 'grade_answer', lambda t, p: False)
	assert A.grader_selftest([mk_task(0)])['oracle_fails'] == ['t0']


def test_perturb_outside_tolerance():
	assert float(A.perturb(mk_task(0, answer='100', rel_tol=0.5))) == 201
	assert A.perturb(mk_task(0, answer='a, b, c', answer_type='set')) == 'a, b'
	assert A.perturb(mk_task(0, answer='x', answer_type='exact')) is None


def by_check(F):
	out = {}
	for f in F:
		out.setdefault(f['check'], []).append(f)
	return out


def test_audit_on_synthetic_env(env_dir):
	F = by_check(A.audit(env_dir))
	assert '20/23 tasks pass' in F['tasks'][0]['message']
	assert F['leakage'][0]['ids'] == ['t20'] and F['memorizable'][0]['ids'] == ['t21'] and F['stale'][0]['ids'] == ['t22']
	assert F['gradient'][0]['severity'] == 'info' and 'claude-haiku-4-5 30% -> anthropic/claude-sonnet-4-6 70%' in F['gradient'][0]['message']
	assert F['inverted'][0]['ids'] == ['t0']
	assert 'signoff' in F and 'grader' in F
	assert any('agent claimed success' in f['message'] for f in F['overclaim'])
	sev = [f['severity'] for f in A.audit(env_dir)]
	assert sev == sorted(sev, key={'high': 0, 'medium': 1, 'info': 2}.get)


def test_audit_flags_saturation_and_no_gradient(env_dir):
	from conftest import mk_row, write_jl

	flow = env_dir / 'hillclimb'
	write_jl(flow / 'baseline' / 'results.jsonl', [mk_row(f't{i}', 0, 'anthropic/claude-haiku-4-5', True) for i in range(20)])
	F = by_check(A.audit(env_dir))
	assert F['saturated'][0]['severity'] == 'high'
	assert F['gradient'][0]['severity'] == 'high'


def test_audit_split_imbalance(env_dir):
	ids = [f't{i}' for i in range(20)]
	(env_dir / 'hillclimb' / '_state.json').write_text(json.dumps({'train_ids': ids[:6], 'test_ids': ids[6:]}))
	assert 'split' in by_check(A.audit(env_dir))


def test_audit_judge_calibration(env_dir):
	j = lambda i, s: {'mode': {'id': i}, 'stats': s}  # noqa: E731
	(env_dir / 'judges.json').write_text(
		json.dumps([j('a', {}), j('b', {'trusted': False, 'tpr': 0.5, 'tnr': 1.0, 'n': 10}), j('c', {'trusted': True, 'tpr': 1.0, 'tnr': 0.95, 'n': 40})])
	)
	sev = {f['message'].split()[1]: f['severity'] for f in by_check(A.audit(env_dir))['judge']}
	assert sev == {'a': 'medium', 'b': 'high', 'c': 'info'}


def test_audit_without_tasks(tmp_path):
	assert A.audit(tmp_path)[0]['severity'] == 'high'


def test_audit_respects_human_rejections(env_dir):
	from conftest import write_jl

	write_jl(env_dir / 'reviews.jsonl', [{'id': 't3', 'verdict': 'reject'}])
	assert '19/23 tasks pass' in by_check(A.audit(env_dir))['tasks'][0]['message']


def test_split_never_leaves_test_empty():
	tasks = [mk_task(i, category=f'c{i}') for i in range(10)]
	tr, te = A.split(tasks, 0.5)
	assert len(te) == 5 and len(tr) == 5


def test_audit_flags_shortcuts(env_dir):
	from conftest import jl, write_jl

	rows = jl(env_dir / 'hillclimb' / 'v1' / 'results.jsonl')
	for i, r in enumerate(rows):
		r['grade'].update(grounded=float(i % 2), supported=float(r['grade']['correct'] and i % 2), no_loop=1.0)
	write_jl(env_dir / 'hillclimb' / 'v1' / 'results.jsonl', rows)
	f = by_check(A.audit(env_dir))['shortcut'][0]
	assert f['severity'] == 'medium' and 'never read at their source' in f['message']


def test_split_balances_forecast_outcomes_across_train_and_test():
	from rl_env_use.audit import split

	ts = [{'id': f't{i}', 'case': f'c{i}', 'category': 'fc', 'answer_type': 'prob', 'answer': 'yes' if i < 10 else 'no'} for i in range(20)]
	tr, te = split(ts, 0.5, 0)
	yes = lambda ids: sum(int(i[1:]) < 10 for i in ids)  # noqa: E731
	assert yes(tr) == 5 and yes(te) == 5


def test_audit_flags_skewed_forecast_base_rate(tmp_path):
	import json

	from conftest import mk_task, write_jl

	from rl_env_use.audit import audit

	env = tmp_path / 'e'
	ok = {
		'structural': [],
		'leakage': False,
		'closed_book': 'skipped',
		'live': True,
		'critic': {'unambiguous': True, 'gold_supported': True, 'issues': ''},
		'panel': {'verdict': 'keep'},
	}
	write_jl(env / 'tasks.jsonl', [mk_task(i, answer='yes' if i < 9 else 'no', answer_type='prob', context='m', kind='predict', checks=ok) for i in range(10)])
	(env / 'spec.json').write_text(json.dumps({'topic': 'x', 'auto': True}))
	f = next(f for f in audit(env) if f['check'] == 'base_rate')
	assert '9/10' in f['message'] and '8 majority-outcome' in f['message']  # 1 no allows 1 yes (60% cap)
