import asyncio
import hashlib
import inspect
import json
import random
import re
from types import SimpleNamespace as NS

import pytest
from conftest import write_jl

import rl_env_use.audit as A
import rl_env_use.climb as C
import rl_env_use.rollout as R

GOOD = 'WORKED EXAMPLE: margin = (revenue - costs) / revenue, e.g. (50 - 20) / 50 = 0.6.'
OVERFIT = 'TRAIN TRICK: always recompute.'


def mk(c, j):
	tid = f'q{c:02d}{j}'
	ok = {
		'structural': [],
		'leakage': False,
		'closed_book': {'haiku': False},
		'live': True,
		'critic': {'unambiguous': True, 'gold_supported': True, 'realistic': True, 'issues': ''},
	}
	return {
		'id': tid,
		'case': f'acme-case-{c:02d}',
		'kind': 'derive',
		'answer_type': 'number',
		'answer': f'{3000 + c * 37 + j}',
		'aliases': [],
		'category': 'ratio',
		'difficulty': 'hard',
		'prompt': f'For item {tid}, what is revenue minus costs?',
		'context': f'Revenue was {4500 + c * 13:,} thousand and costs were {1200 + c * 11:,} thousand in segment {j}.',
		'checks': ok,
	}


TASKS = [mk(c, j) for c in range(20) for j in range(2)]
GOLD = {t['id']: t['answer'] for t in TASKS}
TRAIN, TEST = A.split(TASKS, 0.5, 0)
H = lambda *a: int(hashlib.md5(repr(a).encode()).hexdigest(), 16)  # noqa: E731


@pytest.fixture
def env(tmp_path):
	write_jl(tmp_path / 'tasks.jsonl', TASKS)
	return tmp_path


def policy(monkeypatch, mode='half'):
	"""Right on every task with the worked example; otherwise per mode. 'noisy' flips a coin per (task, call)."""
	seen = {}

	async def fake(model, system, user, schema, temperature=None):
		tid = re.search(r'item (\w+),', user).group(1)
		if 'REFUSE' in system:
			raise RuntimeError('I cannot help with that')
		n = seen[tid] = seen.get(tid, -1) + 1
		right = GOOD in system or mode == 'all' or (OVERFIT in system and tid in TRAIN) or (H(tid, n) % 2 == 0 if mode == 'noisy' else int(tid[1:3]) % 2 == 0)
		return schema(reasoning=f'thought about {tid}', answer=GOLD[tid] if right else '-1'), {}, 'fake-model'

	monkeypatch.setattr(R, 'call', fake)


def proposer(monkeypatch, script, buckets=(), repair=None):
	"""script: one list of (change, prompt) per round; the last round repeats. repair: the prompt a leak repair returns
	(default: the leaked edit unchanged, so it stays rejected)."""
	users = []

	async def fake(model, system, user, schema, temperature=None):
		if schema is C.Edit:  # one repair attempt for an edit that copies case content
			old = C.Edit.model_validate_json(user.split('EDIT:\n', 1)[1])
			return C.Edit(bucket=old.bucket, change=old.change + ' (repaired)', prompt=repair if repair is not None else old.prompt), {}, 'prop'
		users.append(user)
		edits = script[min(len(users), len(script)) - 1]
		return C.Proposal(buckets=[C.Bucket(**b) for b in buckets], edits=[C.Edit(bucket='b', change=c, prompt=p) for c, p in edits]), {}, 'prop'

	monkeypatch.setattr(C, 'call', fake)
	monkeypatch.setattr(C, 'LEDGER', NS(usd=0.0))
	return users


def climb(env, **kw):
	return asyncio.run(C.climb(env, 'fake-model', log=lambda *_: None, **{'rounds': 1, 'k': 3, 'reps': 2, 'budget': 100, **kw}))


def flow(env):
	return env / 'climb-fake-model'


def test_good_edit_kept_and_files_written(env, monkeypatch):
	policy(monkeypatch)
	proposer(monkeypatch, [[('add worked example', GOOD)]])
	s = climb(env)
	bt = sum(int(i[1:3]) % 2 == 0 for i in TEST) / len(TEST)
	assert s['baseline']['test'] == pytest.approx(bt) and s['baseline']['train'] == pytest.approx(sum(int(i[1:3]) % 2 == 0 for i in TRAIN) / len(TRAIN))
	r = s['rounds'][0]
	assert r['decision'] == 'keep' and r['variant'] == 'v1' and r['test']['diff'] == pytest.approx(1 - bt) and r['test']['p_better'] > 0.99
	assert s['best'] == {'round': 1, 'variant': 'v1', 'test_score': 1.0}
	assert s['final']['variant'] == 'v1' and s['final']['decision'] == 'merge' and s['final']['reps'] == 2 and s['final']['best_test'] == 1.0 and s['final']['test_looks'] == 1
	d = flow(env) / 'v1'
	assert (d / 'prompt.txt').read_text() == GOOD and 'add worked example' in (d / 'change.md').read_text()
	assert json.loads((d / 'summary.json').read_text())['target'] == 'system_prompt'
	assert set(A.load_variants(flow(env))) == {'baseline', 'v1'}
	saved = json.loads((flow(env) / '_state.json').read_text())
	assert saved['final']['decision'] == 'merge' and not set(saved['train_ids']) & set(saved['test_ids'])
	assert max(r['rep'] for r in A.load_jsonl(d / 'results.jsonl')) == 3


def test_neutral_edit_reverted(env, monkeypatch):
	policy(monkeypatch)
	proposer(monkeypatch, [[('be careful', 'Be careful with units.')]])
	s = climb(env)
	r = s['rounds'][0]
	assert r['decision'] == 'revert (no train gain)' and r['candidates'][0]['train_screen']['diff'] == 0 and 'train' not in r['candidates'][0]
	assert s['best']['variant'] == 'baseline' and s['final'] == {'variant': 'baseline', 'test_looks': 0, 'decision': 'no change beat the baseline on test'}


def test_train_only_gain_reverted_on_test(env, monkeypatch):
	policy(monkeypatch)
	proposer(monkeypatch, [[('overfit', OVERFIT)]])
	r = climb(env)['rounds'][0]
	assert r['train']['diff'] > 0 and r['test']['diff'] == 0 and r['decision'] == 'revert (test flat or down)'


def test_best_of_k_goes_to_test(env, monkeypatch):
	policy(monkeypatch)
	proposer(monkeypatch, [[('neutral', 'Be terse.'), ('example', GOOD)]])
	s = climb(env)
	assert [c['variant'] for c in s['rounds'][0]['candidates']] == ['v1', 'v2'] and s['rounds'][0]['variant'] == 'v2' and s['best']['variant'] == 'v2'


TC = next(t for t in TASKS if t['id'] in TRAIN and t['case'] != 'acme-case-00')


@pytest.mark.parametrize(
	'bad', [f'Revenue near {TC["context"].split()[2]} is typical.', f'Like in {TC["case"].upper()}, subtract.', f'Gold is {TC["answer"]}.', 'x' * (C.CAP + 1), '']
)
def test_bad_edits_rejected_before_running(env, monkeypatch, bad):
	policy(monkeypatch)
	proposer(monkeypatch, [[('bad', bad)]])
	s = climb(env)
	assert s['rounds'][0]['candidates'] == [] and s['rounds'][0]['change'] == 'no valid edit'
	assert not (flow(env) / 'v1').exists()


def test_a_leaked_edit_gets_one_repair_with_invented_numbers(env, monkeypatch):
	policy(monkeypatch)
	proposer(monkeypatch, [[('leaky', f'Gold is {TC["answer"]}. ' + GOOD)]], repair=GOOD)
	s = climb(env)
	assert [c['change'] for c in s['rounds'][0]['candidates']] == ['leaky (repaired)'] and (flow(env) / 'v1' / 'prompt.txt').read_text() == GOOD


def test_saturated_refused_unless_forced(env, monkeypatch):
	policy(monkeypatch, 'all')
	users = proposer(monkeypatch, [[('example', GOOD)]])
	s = climb(env)
	assert s['final']['decision'] == 'not run' and s['final']['reason'].startswith('saturated') and users == [] and 'rounds' not in s
	s = climb(env, force=True)
	assert len(users) == 1 and s['rounds'][0]['decision'].startswith('revert')


def test_underpowered_refused(env, monkeypatch):
	policy(monkeypatch, 'noisy')
	users = proposer(monkeypatch, [[('example', GOOD)]])
	s = climb(env, reps=3)
	assert s['baseline']['noise'] > 0 and s['baseline']['mde'] > 0.08
	assert s['final']['decision'] == 'not run' and s['final']['reason'].startswith('underpowered') and users == []


def test_zero_noise_does_not_block(env, monkeypatch):
	policy(monkeypatch)
	proposer(monkeypatch, [[('n', 'Be terse.')]])
	s = climb(env)
	assert s['baseline']['noise'] == 0.05 and s['final']['decision'] != 'not run'  # every rep agrees: floored at one of 20 test tasks flipping


def test_too_few_test_cases_refused(tmp_path, monkeypatch):
	write_jl(tmp_path / 'tasks.jsonl', TASKS[:12])  # 6 cases: 3 in test
	policy(monkeypatch)
	proposer(monkeypatch, [[('n', GOOD)]])
	s = climb(tmp_path)
	assert s['final']['decision'] == 'not run' and '3 test cases' in s['final']['reason'] and 'rounds' not in s


def test_stall_audit_and_stop(env, monkeypatch):
	policy(monkeypatch)
	never = [i for i in TRAIN if int(i[1:3]) % 2]  # odd cases: wrong under every variant
	solved = [i for i in TRAIN if int(i[1:3]) % 2 == 0]
	test_never = [i for i in TEST if int(i[1:3]) % 2]
	buckets = [{'name': 'bad gold', 'task_ids': [never[0], solved[0], test_never[0]], 'fixable': False}, {'name': 'arith', 'task_ids': [never[1]], 'fixable': True}]
	users = proposer(monkeypatch, [[('n1', 'Be terse.')], [('n2', 'Be brief.')], [('n3', 'Be calm.')]], buckets)
	s = climb(env, rounds=6)
	assert len(s['rounds']) == 3 and len(users) == 3
	assert 'quarantined' not in s['rounds'][0] and s['rounds'][1]['quarantined'] == [never[0]]
	assert json.loads((flow(env) / '_state.json').read_text())['quarantined'] == [never[0]]
	assert f'[{never[0]}]' in users[1] and f'[{never[0]}]' not in users[2] and f'[{never[1]}]' in users[2]
	assert 'ALREADY REJECTED EDITS:\n- n1\n- n2' in users[2]


def test_proposer_never_sees_test(env, monkeypatch):
	policy(monkeypatch)
	users = proposer(monkeypatch, [[('n', 'Be terse.')]])
	climb(env, rounds=3)
	assert users and all(f'[{i}]' in users[0] for i in TRAIN if int(i[1:3]) % 2)
	tt = {t['id']: t for t in TASKS if t['id'] in TEST}
	assert not any(i in u or t['prompt'] in u or t['answer'] in u for u in users for i, t in tt.items())


def test_budget_stops_before_proposing(env, monkeypatch):
	policy(monkeypatch)
	users = proposer(monkeypatch, [[('example', GOOD)]])
	monkeypatch.setattr(C, 'LEDGER', NS(usd=1e6))
	s = climb(env, rounds=3)
	assert users == [] and s['rounds'] == [] and s['final']['variant'] == 'baseline'


def test_resume_keeps_split_and_numbers_variants(env, monkeypatch):
	policy(monkeypatch)
	proposer(monkeypatch, [[('n', 'Be terse.')]])
	s1 = climb(env, seed=0)
	s2 = climb(env, seed=5)
	assert s1['train_ids'] == s2['train_ids'] == TRAIN
	assert (flow(env) / 'v2').exists() and [r['round'] for r in s2['rounds']] == [1, 2]


def test_default_confidence():
	assert inspect.signature(C.climb).parameters['confidence'].default == 0.95


def reps_of(env, v, ids):
	return {r['rep'] for r in A.load_jsonl(flow(env) / v / 'results.jsonl') if r['prompt_id'] in ids}


def test_winner_judged_on_fresh_reps_and_headline_on_fresh_reps(env, monkeypatch):
	policy(monkeypatch)
	proposer(monkeypatch, [[('add worked example', GOOD)]])
	r = climb(env)['rounds'][0]  # reps 2: screen on rep 0, train decision on reps 1-2, test on 0-1, headline on 2-3
	assert r['candidates'][0]['train_screen']['n'] == len(TRAIN) and r['train']['n'] == len(TRAIN) and r['test']['n'] == len(TEST)
	assert reps_of(env, 'v1', set(TRAIN)) == {0, 1, 2} and reps_of(env, 'v1', set(TEST)) == {0, 1, 2, 3}
	assert reps_of(env, 'baseline', set(TRAIN)) == {0, 1} and reps_of(env, 'baseline', set(TEST)) == {0, 1, 2, 3}


def test_missing_rep_scores_zero(env, monkeypatch):
	policy(monkeypatch)
	proposer(monkeypatch, [[('refuse', 'REFUSE when unsure.')]])
	s = climb(env)
	r = s['rounds'][0]
	sc = r['candidates'][0]['train_screen']
	assert sc['n'] == len(TRAIN) and sc['diff'] == pytest.approx(-s['baseline']['train']) and r['decision'] == 'revert (no train gain)'
	assert (flow(env) / 'v1' / 'errors.jsonl').exists() and not A.load_jsonl(flow(env) / 'v1' / 'results.jsonl')


def test_single_rep_refused_unless_forced(env, monkeypatch):
	policy(monkeypatch)
	users = proposer(monkeypatch, [[('example', GOOD)]])
	s = climb(env, reps=1)
	assert s['final']['decision'] == 'not run' and s['final']['reason'].startswith('need --reps >= 2') and users == [] and s['baseline']['noise'] is None
	s = climb(env, reps=1, force=True)
	assert len(users) == 1 and s['rounds'][0]['decision'] == 'keep'


def test_resume_restores_incumbent(env, monkeypatch):
	policy(monkeypatch)
	proposer(monkeypatch, [[('example', GOOD)]])
	assert climb(env)['incumbent'] == 'v1'
	users = proposer(monkeypatch, [[('n', 'Be terse.')]])
	s = climb(env)
	assert GOOD in users[0] and s['incumbent'] == 'v1' and s['best']['variant'] == 'v1' and s['final']['variant'] == 'v1'
	assert [r['round'] for r in s['rounds']] == [1, 2] and s['rounds'][1]['decision'] == 'revert (no train gain)' and s['final']['test_looks'] == 1


def test_resume_restores_stall(env, monkeypatch):
	policy(monkeypatch)
	proposer(monkeypatch, [[('n1', 'Be terse.')], [('n2', 'Be brief.')]])
	assert climb(env, rounds=2)['stall'] == 2
	users = proposer(monkeypatch, [[('n3', 'Be calm.')]])
	s = climb(env, rounds=5)
	assert len(users) == 1 and len(s['rounds']) == 3 and s['stall'] == 3


def test_resume_counts_prior_spend(env, monkeypatch):
	policy(monkeypatch)
	proposer(monkeypatch, [[('example', GOOD)]])
	monkeypatch.setattr(C, 'LEDGER', NS(usd=3.0))
	assert climb(env, budget=5)['usd'] == 3.0
	users = proposer(monkeypatch, [[('n', 'Be terse.')]])
	monkeypatch.setattr(C, 'LEDGER', NS(usd=3.0))  # this process spent 3 more: 6 of 5
	s = climb(env, budget=5)
	assert users == [] and len(s['rounds']) == 1 and json.loads((flow(env) / '_state.json').read_text())['usd'] == 6.0


def test_budget_checked_before_each_candidate(env, monkeypatch):
	policy(monkeypatch)
	proposer(monkeypatch, [[('a', 'Be terse.'), ('b', GOOD)]])
	pol = R.call

	async def costly(model, system, user, schema, temperature=None):
		if 'Be terse.' in system:
			C.LEDGER.usd = 50.0
		return await pol(model, system, user, schema, temperature)

	monkeypatch.setattr(R, 'call', costly)
	s = climb(env, budget=10)
	assert [c['variant'] for c in s['rounds'][0]['candidates']] == ['v1'] and not (flow(env) / 'v2').exists()


def test_write_variant_refuses_to_mix_prompts(tmp_path):
	C.write_variant(tmp_path, 'v1', 'A', 'a', 'm')
	C.write_variant(tmp_path, 'v1', 'B', 'b', 'm')  # no runs yet: overwritten
	(tmp_path / 'v1' / 'results.jsonl').write_text('{}\n')
	C.write_variant(tmp_path, 'v1', 'B', 'b', 'm')  # same prompt: resumable
	with pytest.raises(SystemExit):
		C.write_variant(tmp_path, 'v1', 'C', 'c', 'm')
	assert (tmp_path / 'v1' / 'prompt.txt').read_text() == 'B'


def test_paired_null_ci_contains_zero():
	rng = random.Random(7)
	base, new = {f't{i}': float(rng.random() < 0.5) for i in range(60)}, {f't{i}': float(rng.random() < 0.5) for i in range(60)}
	p = A.paired(base, new)
	assert p['lo'] <= 0 <= p['hi'] and p['n'] == 60 and 0.025 < p['p_better'] < 0.975


def test_paired_planted_effect_excludes_zero():
	rng = random.Random(2)
	base = {f't{i}': rng.random() * 0.7 for i in range(60)}
	new = {k: v + 0.2 + rng.gauss(0, 0.1) for k, v in base.items()}
	p = A.paired(base, new)
	assert p['lo'] > 0 and p['diff'] == pytest.approx(0.2, abs=0.05) and p['p_better'] == 1.0


def test_paired_clustering_widens_ci():
	rng = random.Random(3)
	shift = {c: rng.choice([-0.5, 0.5]) for c in range(10)}
	base = {f't{c}_{j}': 0.5 for c in range(10) for j in range(4)}
	new = {k: 0.5 + shift[int(k[1:].split('_')[0])] for k in base}
	case_of = {k: k.split('_')[0] for k in base}
	flat, clus = A.paired(base, new), A.paired(base, new, case_of)
	assert clus['cases'] == 10 and flat['cases'] == 40 and clus['diff'] == flat['diff']
	assert clus['hi'] - clus['lo'] > 1.5 * (flat['hi'] - flat['lo'])


def test_paired_p_better_and_edges():
	b = {f't{i}': 0.0 for i in range(10)}
	assert A.paired(b, {k: 1.0 for k in b})['p_better'] == 1.0
	assert A.paired({k: 1.0 for k in b}, b)['p_better'] == 0.0
	assert A.paired(b, {'zz': 1.0}) == {'n': 0, 'diff': 0.0, 'lo': 0.0, 'hi': 0.0, 'p_better': 0.5}
	assert A.paired(b, {'t0': 1.0, 'zz': 0.0})['n'] == 1


def test_split_groups_cases():
	case = {t['id']: t['case'] for t in TASKS}
	for seed in range(5):
		tr, te = A.split(TASKS, 0.5, seed)
		assert not {case[i] for i in tr} & {case[i] for i in te} and sorted(tr + te) == sorted(case)
		assert (tr, te) == A.split(list(reversed(TASKS)), 0.5, seed)
	assert len({tuple(A.split(TASKS, 0.5, s)[1]) for s in range(5)}) > 1


@pytest.mark.parametrize('frac', [0.2, 0.3, 0.5])
def test_split_test_frac(frac):
	_, te = A.split(TASKS, frac, 0)
	assert abs(len({i[:3] for i in te}) / 20 - frac) <= 0.05


def test_rep_halves():
	rows = [{'prompt_id': t, 'rep': r, 'status': 'ok', 'grade': {'correct': float(v)}} for t, vals in [('a', [1, 0, 1, 0]), ('b', [1, 1, 0, 1])] for r, v in enumerate(vals)]
	rows.append({'prompt_id': 'a', 'rep': 5, 'status': 'error', 'grade': {}})
	even, odd = A.rep_halves(rows)
	assert even == {'a': 1.0, 'b': 0.5} and odd == {'a': 0.0, 'b': 1.0}


def test_leaked():
	ts = [
		{'case': 'acme-case-01', 'context': 'Revenue 4,527 and 61.05 margin, 100 staff, 1,000 units, year 2023.', 'answer': '3,326'},
		{'case': 'abc', 'context': '', 'answer': '7'},
	]
	assert C.leaked('Use 4,527 and 61.05 and 3,326.', ts) == ['4,527', '61.05', '3,326']
	assert C.leaked('Round to 100 or 1,000; compute 9,876.', ts) == []
	assert C.leaked('see Acme-Case-01', ts) == ['acme-case-01'] and C.leaked('the abc rule', ts) == []
	assert C.leaked('the year 2023 matters', ts) == ['2023']
