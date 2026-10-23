"""Autonomous create: planner -> mining rounds with budget, target, dry-round and headroom stops; CLI create/resume."""

import asyncio
import json

import pytest
from conftest import mk_task
from pydantic import ValidationError

import rl_env_use.__main__ as M
import rl_env_use.pipeline as P
from rl_env_use.__main__ import main
from rl_env_use.llm import LEDGER
from rl_env_use.store import load_jsonl, save_jsonl

PLAN = {
	'name': 'SEC Credit!',
	'topic': 'small-cap 10-Qs 2025',
	'skill': '10-Q -> leverage',
	'sources': ['https://www.sec.gov/edgar'],
	'mix': 'derive',
	'conventions': 'leverage = debt / EBITDA',
	'sampling': 'all filers in SIC 7372',
}


@pytest.fixture(autouse=True)
def ledger():
	LEDGER.by_model = {}
	yield LEDGER
	LEDGER.by_model = {}


def spend(usd):
	LEDGER.add('fake', {'reported_cost': usd})


def kept_task(i, kind='derive', scores=None, verdict='keep'):
	t = mk_task(i, kind=kind, case=f'c{i}')
	t['checks']['panel'] = {'verdict': verdict, 'scores': scores or {'haiku': 0.2, 'gpt6': 0.5}}
	return t


class World:
	"""Fake scout + stages: each round mines `missions` cases and keeps rate[rnd] * missions new tasks."""

	def __init__(self, env, rates=(1.0,), cost=0.0, kind='derive', scores=None):
		self.env, self.rates, self.cost, self.kind, self.scores = env, rates, cost, kind, scores
		self.scouts, self.mixes, self.n = [], [], 0

	async def scout(self, topic, env, missions, *a):
		self.scouts.append({'topic': topic, 'missions': missions, 'sources': a[6], 'skill': a[7]})
		p = env / 'kb.json'
		kb = json.loads(p.read_text()) if p.exists() else {'facts': [], 'materials': [], 'missions': []}
		for _ in range(missions):
			k = f'case{len(kb["materials"])}'
			kb['materials'].append({'id': f'm-{k}', 'case': k, 'text': 'x', 'title': 't', 'url': 'https://a.gov'})
			kb['facts'].append({'id': f'f-{k}', 'case': k, 'url': 'https://a.gov', 'quote': 'q'})
		p.write_text(json.dumps(kb))
		spend(self.cost)
		return kb

	async def synth(self, env, spec, model):
		self.mixes.append(spec.get('mix'))
		rate = self.rates[min(len(self.scouts), len(self.rates)) - 1]
		tasks = load_jsonl(env / 'tasks.jsonl')
		new = [kept_task(self.n + j, self.kind, self.scores) for j in range(round(rate * self.scouts[-1]['missions']))]
		self.n += len(new)
		save_jsonl(env / 'tasks.jsonl', tasks + new)
		return f'synth: {len(new)}'

	def install(self, mp):
		async def stage(env, spec, *a):
			return 'ok'

		mp.setattr(P, 'scout', self.scout)
		mp.setattr(P, 'synth_stage', self.synth)
		mp.setattr(P, 'verify_stage', stage)
		mp.setattr(P, 'panel_stage', stage)
		return self


def go(env, spec, logs=None):
	return asyncio.run(P.create(env, spec, log=(logs if logs is not None else []).append))


@pytest.fixture
def env(tmp_path):
	e = tmp_path / 'envs' / 'x'
	e.mkdir(parents=True)
	return e


def spec_(**kw):
	return {'topic': 'T', 'skill': 'S', 'sources': ['https://a.gov'], 'mix': 'derive', 'budget_usd': 1000.0, 'target_tasks': 1000, 'missions': 12} | kw


# plan_spec


def test_plan_spec_calls_model_with_request_and_schema(monkeypatch):
	seen = {}

	async def fake(model, system, user, schema, **kw):
		seen.update(model=model, system=system, user=user, schema=schema)
		return schema(**PLAN), {}, model

	monkeypatch.setattr(P, 'call', fake)
	p = asyncio.run(P.plan_spec('credit of small caps', 'opus55'))
	assert p == PLAN and isinstance(p, dict)
	assert seen['model'] == 'opus55' and seen['user'] == 'REQUEST: credit of small caps' and seen['schema'] is P.SpecPlan and seen['system'] == P.PLAN_SYS


def test_specplan_rejects_unknown_mix():
	with pytest.raises(ValidationError):
		P.SpecPlan(**PLAN | {'mix': 'lookup'})
	assert set(P.SpecPlan.model_fields) == set(PLAN)


# create loop


def test_stops_at_target(env, monkeypatch):
	w = World(env, rates=(2.0,)).install(monkeypatch)
	r = go(env, spec_(target_tasks=30))
	assert [s['missions'] for s in w.scouts] == [12, 2, 1] and r['kept'] == 30 and r['status'] == 'ready'  # 24 -> 28 -> 30


def test_first_round_reaching_target_stops_immediately(env, monkeypatch):
	w = World(env, rates=(2.0,)).install(monkeypatch)
	r = go(env, spec_(target_tasks=5))
	assert len(w.scouts) == 1 and r['kept'] == 24


def test_next_round_missions_sized_from_yield(env, monkeypatch):
	w = World(env, rates=(0.5,)).install(monkeypatch)
	go(env, spec_(target_tasks=30))
	# round 1: 6 kept from 12 cases -> yield 0.5 -> ceil(24 / 0.5 / 3 * 1.3) = 21
	assert [s['missions'] for s in w.scouts[:2]] == [12, 21]


def test_missions_capped_by_remaining_budget(env, monkeypatch):
	w = World(env, rates=(1.0,), cost=0.0).install(monkeypatch)
	go(env, spec_(budget_usd=2.0, max_rounds=1))
	assert w.scouts[0]['missions'] == 2  # half of $2 at $0.35 a mission


def test_stops_when_budget_spent(env, monkeypatch):
	logs = []
	w = World(env, rates=(0.1,), cost=1.6).install(monkeypatch)
	r = go(env, spec_(budget_usd=2.0), logs)
	assert len(w.scouts) == 1 and any('budget spent' in m for m in logs)
	assert r['usd'] == 1.6 and r['status'] == 'incomplete'


def test_no_mining_when_budget_already_gone(env, monkeypatch):
	spend(50)
	w = World(env).install(monkeypatch)
	r = go(env, spec_(budget_usd=40.0))
	assert w.scouts == [] and r['kept'] == 0 and r['status'] == 'incomplete' and r['usd'] == 50
	assert json.loads((env / 'spec.json').read_text())['result'] == r


def test_ledger_spend_shrinks_later_rounds(env, monkeypatch):
	w = World(env, rates=(0.0, 1.0), cost=3.0).install(monkeypatch)
	go(env, spec_(budget_usd=10.0, target_tasks=1000, max_rounds=2))
	# round 1: min(12, 14) = 12; round 2: $7 left -> half of it at $0.35 a mission
	assert w.scouts[0]['missions'] == 12 and w.scouts[1]['missions'] == int(7 * 0.5 / 0.35)


def test_stops_after_two_dry_rounds(env, monkeypatch):
	logs = []
	w = World(env, rates=(0.0,)).install(monkeypatch)
	r = go(env, spec_(), logs)
	assert len(w.scouts) == 2 and 'two rounds without new tasks' in logs[-1] and r['status'] == 'incomplete'


def test_below_ten_percent_gain_counts_as_dry(env, monkeypatch):
	w = World(env, rates=(1 / 12,)).install(monkeypatch)
	go(env, spec_())
	assert len(w.scouts) == 2


def test_productive_round_resets_dry_count(env, monkeypatch):
	w = World(env, rates=(0.0, 1.0, 0.0, 0.0)).install(monkeypatch)
	r = go(env, spec_(target_tasks=100))
	assert len(w.scouts) == 4 and r['kept'] == 87  # 0, +87, 0, 0 -> stop


def test_max_rounds_bounds_the_loop(env, monkeypatch):
	w = World(env, rates=(0.5,)).install(monkeypatch)
	go(env, spec_(max_rounds=3, target_tasks=10**6))
	assert len(w.scouts) == 3


def test_switches_to_predict_when_strongest_saturates_derive(env, monkeypatch):
	logs = []
	w = World(env, rates=(2.0, 0.0, 0.0), scores={'haiku': 0.4, 'gpt6': 0.9}).install(monkeypatch)
	go(env, spec := spec_(target_tasks=100), logs)
	assert w.mixes[:2] == ['derive', 'predict'] and spec['mix'] == 'predict'
	assert sum('next rounds write forecasts' in m for m in logs) == 1
	assert json.loads((env / 'spec.json').read_text())['mix'] == 'predict'


@pytest.mark.parametrize('rates,scores', [((19 / 12, 0.0), {'gpt6': 0.95}), ((2.0, 0.0), {'gpt6': 0.85}), ((2.0, 0.0), {'haiku': 0.5})])
def test_no_switch_without_saturation_or_enough_tasks(env, monkeypatch, rates, scores):
	w = World(env, rates=rates, scores=scores).install(monkeypatch)
	go(env, spec := spec_(target_tasks=100))
	assert set(w.mixes) == {'derive'} and spec['mix'] == 'derive'


def test_predict_scores_do_not_trigger_switch(env, monkeypatch):
	save_jsonl(env / 'tasks.jsonl', [kept_task(1000 + i, 'derive', {'gpt6': 0.5}) for i in range(20)] + [kept_task(2000 + i, 'predict', {'gpt6': 0.99}) for i in range(20)])
	w = World(env, rates=(0.0,), kind='predict', scores={'gpt6': 0.99}).install(monkeypatch)
	go(env, spec := spec_(mix='mixed'))
	assert set(w.mixes) == {'mixed'} and spec['mix'] == 'mixed'


def test_switch_counts_unkept_measured_derive_tasks(env, monkeypatch):
	save_jsonl(env / 'tasks.jsonl', [kept_task(1000 + i, 'derive', {'gpt6': 1.0}, verdict='saturated') for i in range(20)])
	w = World(env, rates=(0.0,)).install(monkeypatch)
	go(env, spec := spec_())
	assert w.mixes[:2] == ['derive', 'predict'] and spec['mix'] == 'predict'


def test_switch_ignores_unmeasured_derive_tasks(env, monkeypatch):
	save_jsonl(env / 'tasks.jsonl', [kept_task(1000 + i, 'derive', {'gpt6': 1.0}, verdict='unmeasured') for i in range(20)])
	w = World(env, rates=(0.0,)).install(monkeypatch)
	go(env, spec := spec_())
	assert set(w.mixes) == {'derive'} and spec['mix'] == 'derive'


def test_resume_subtracts_prior_spend(env, monkeypatch):
	w = World(env, rates=(0.0,), cost=0.5).install(monkeypatch)
	r = go(env, spec := spec_(budget_usd=40.0, spent_usd=38.0, max_rounds=1))
	assert w.scouts[0]['missions'] == 2  # $2 left: half of it at $0.35 a mission
	assert spec['spent_usd'] == 38.5 and r['usd'] == 38.5 and json.loads((env / 'spec.json').read_text())['spent_usd'] == 38.5
	w = World(env, rates=(0.0,)).install(monkeypatch)
	go(env, spec_(budget_usd=40.0, spent_usd=40.0))
	assert w.scouts == []


def test_new_skill_env_gets_a_panel(tmp_path, monkeypatch):
	monkeypatch.chdir(tmp_path)
	main(['new', 'Going Concern', '--topic', 'credit analysis', '--skill', 'predict the auditor opinion', '--sources', 'https://www.sec.gov/'])
	main(['new', 'Browse', '--topic', 'benchmarks'])
	assert json.loads((tmp_path / 'envs/going-concern/spec.json').read_text())['panel'] == P.PANEL
	assert 'panel' not in json.loads((tmp_path / 'envs/browse/spec.json').read_text())


def test_scout_gets_sampling_sources_and_skill(env, monkeypatch):
	w = World(env, rates=(5.0,)).install(monkeypatch)
	go(env, spec_(target_tasks=1, sampling='all filers in Q1'))
	s = w.scouts[0]
	assert s['topic'] == 'T\nCASE SELECTION (never by outcome): all filers in Q1' and s['sources'] == ['https://a.gov'] and s['skill'] == 'S'


def test_result_written_into_spec(env, monkeypatch):
	save_jsonl(env / 'tasks.jsonl', [kept_task(900, verdict='too_easy')])
	World(env, rates=(1.0,), cost=0.5, kind='predict', scores={'haiku': 0.25, 'gpt6': 0.75}).install(monkeypatch)
	r = go(env, spec_(target_tasks=12))
	s = json.loads((env / 'spec.json').read_text())
	assert s['result'] == r and s['topic'] == 'T' and s['target_tasks'] == 12
	assert r['kept'] == 12 and r['cases'] == 12 and r['kinds'] == {'predict': 12} and r['usd'] == 0.5 and r['status'] == 'ready'
	assert r['verdicts'] == {'keep': 12, 'too_easy': 1} and r['panel_means'] == {'haiku': 0.25, 'gpt6': 0.75}


def test_status_ready_at_thirty_when_target_larger(env, monkeypatch):
	World(env, rates=(30 / 12, 0.0, 0.0)).install(monkeypatch)
	assert go(env, spec_(target_tasks=150))['status'] == 'ready'


def test_panel_means_filters_kind_and_unkept():
	ts = [
		kept_task(0, 'derive', {'a': 1.0}),
		kept_task(1, 'derive', {'a': 0.0, 'b': 0.5}),
		kept_task(2, 'predict', {'a': 0.2}),
		kept_task(3, 'derive', {'a': 0.9}, verdict='too_easy'),
	]
	assert P.panel_means(ts, 'derive') == {'a': 0.5, 'b': 0.5}
	assert P.panel_means(ts) == {'a': 0.4, 'b': 0.5}


# CLI


@pytest.fixture
def cli(tmp_path, monkeypatch):
	monkeypatch.chdir(tmp_path)
	exits, calls = [], {}
	monkeypatch.setattr(M.os, '_exit', exits.append)
	monkeypatch.setattr(M, 'report', lambda env, judges=True: calls.setdefault('report', (env, judges)))

	async def plan(line, model='sonnet46'):
		calls['plan'] = (line, model)
		return dict(PLAN)

	async def create(env, spec, model=None, log=print):
		calls['create'] = (env, dict(spec), spec.get('writer_model', model))
		return {'kept': 0, 'status': 'incomplete'}

	monkeypatch.setattr(P, 'plan_spec', plan)
	monkeypatch.setattr(P, 'create', create)
	return calls


def test_cli_plan_only_prints_spec_without_env(cli, tmp_path, capsys):
	main(['create', 'credit analysis of small caps', '--plan-only'])
	out = capsys.readouterr().out
	spec = json.loads(out[out.index('{') : out.rindex('}') + 1])
	assert spec['topic'] == PLAN['topic'] and spec['request'] == 'credit analysis of small caps' and spec['auto'] is True and spec['panel'] == P.PANEL
	assert cli['plan'] == ('credit analysis of small caps', 'sonnet46') and 'create' not in cli and not (tmp_path / 'envs').exists()


def test_cli_create_writes_spec_and_runs(cli, tmp_path):
	main(['create', 'line', '--tasks', '40', '--budget', '7', '--mix', 'predict', '--panel', 'haiku', 'gpt6', '--model', 'opus55'])
	env = tmp_path / 'envs' / 'sec-credit'
	s = json.loads((env / 'spec.json').read_text())
	assert s['target_tasks'] == 40 and s['budget_usd'] == 7.0 and s['mix'] == 'predict' and s['panel'] == ['haiku', 'gpt6'] and s['sampling'] == PLAN['sampling']
	e, spec, model = cli['create']
	assert e.resolve() == env.resolve() and spec == s and model == 'opus55' and cli['report'][1] is False


def test_cli_name_resumes_existing_env_without_planning(cli, tmp_path, capsys):
	env = tmp_path / 'envs' / 'my-env'
	env.mkdir(parents=True)
	(env / 'spec.json').write_text(json.dumps({'topic': 'old', 'mix': 'derive', 'result': {'kept': 17}}))
	main(['create', 'ignored', '--name', 'My Env', '--tasks', '60'])
	assert 'plan' not in cli and 'resuming envs/my-env (17 tasks kept so far)' in capsys.readouterr().out
	_, spec, _ = cli['create']
	assert spec['topic'] == 'old' and spec['target_tasks'] == 60 and spec['budget_usd'] == 40.0 and spec['mix'] == 'derive'


def test_cli_refuses_to_overwrite_planned_env(cli, tmp_path):
	env = tmp_path / 'envs' / 'sec-credit'
	env.mkdir(parents=True)
	(env / 'spec.json').write_text('{"topic": "keep me"}')
	with pytest.raises(SystemExit, match='--name sec-credit'):
		main(['create', 'line'])
	assert 'create' not in cli and json.loads((env / 'spec.json').read_text()) == {'topic': 'keep me'}


def test_verify_can_skip_the_live_recheck(monkeypatch, tmp_path):
	import asyncio
	import json

	from conftest import jl, mk_task, write_jl

	import rl_env_use.pipeline as P

	env = tmp_path / 'e'
	write_jl(env / 'tasks.jsonl', [mk_task(0, checks={'structural': [], 'leakage': False})])
	(env / 'kb.json').write_text(json.dumps({'facts': [{'id': 'f0', 'url': 'https://x.org', 'quote': 'q', 'value': '137'}], 'materials': []}))

	async def nothing(*a, **k):
		return None

	async def boom(*a, **k):
		raise AssertionError('no browser when live_check is off')

	monkeypatch.setattr(P, 'closed_book', nothing)
	monkeypatch.setattr(P, 'critique', nothing)
	monkeypatch.setattr(P, 'check_live', boom)
	asyncio.run(P.verify_stage(env, {'live_check': False}, 'm'))
	assert jl(env / 'tasks.jsonl')[0]['checks']['live'] == 'skipped'


def test_plan_spec_drops_template_and_non_url_sources(monkeypatch):
	async def fake(model, system, user, schema, **kw):
		return (
			schema(**PLAN | {'sources': ['https://clinicaltrials.gov/study/{NCT_ID}', 'clinicaltrials.gov', 'https://clinicaltrials.gov/search?aggFilters=results:with']}),
			{},
			model,
		)

	monkeypatch.setattr(P, 'call', fake)
	assert asyncio.run(P.plan_spec('x'))['sources'] == ['https://clinicaltrials.gov/search?aggFilters=results:with']


@pytest.mark.parametrize('flags, writer, critic', [
	([], 'sonnet46', 'sonnet46'),
	(['--writer', 'local:writer'], 'local:writer', 'local:writer'),
	(['--model', 'local:writer'], 'local:writer', 'local:writer'),
	(['--critic', 'local:critic'], 'local:critic', 'local:critic'),
	(['--writer', 'local:writer', '--critic', 'local:critic'], 'local:writer', 'local:critic'),
])
def test_cli_writer_critic_fallbacks(cli, tmp_path, flags, writer, critic):
	main(['create', 'line', *flags])
	assert cli['plan'][1] == writer
	spec = json.loads((tmp_path / 'envs/sec-credit/spec.json').read_text())
	assert (spec['writer_model'], spec['critic_model']) == (writer, critic)


def test_cli_resume_reuses_saved_roles(cli, tmp_path):
	env = tmp_path / 'envs/saved'
	env.mkdir(parents=True)
	(env / 'spec.json').write_text(json.dumps({'writer_model': 'local:w', 'critic_model': 'local:c'}))
	main(['create', 'line', '--name', 'saved'])
	spec = cli['create'][1]
	assert (spec['writer_model'], spec['critic_model']) == ('local:w', 'local:c')


@pytest.mark.parametrize('command', ['create', 'build'])
def test_pipeline_routes_writing_and_verification_separately(env, monkeypatch, command):
	World(env).install(monkeypatch)
	seen = []
	for name in ('synth_stage', 'harden_stage', 'verify_stage'):
		async def stage(env, spec, model, name=name):
			seen.append((name, model))
			return 'ok'
		monkeypatch.setattr(P, name, stage)
	spec = spec_(max_rounds=1)
	if command == 'create':
		asyncio.run(P.create(env, spec, 'local:w', critic='local:c', log=lambda *_: None))
		assert seen == [('synth_stage', 'local:w'), ('verify_stage', 'local:c'), ('harden_stage', 'local:w'), ('verify_stage', 'local:c')]
	else:
		asyncio.run(P.build(env, spec, only='harden', model='local:w', critic='local:c'))
		assert seen == [('harden_stage', 'local:w'), ('verify_stage', 'local:c')]


def test_changed_critic_reviews_cached_tasks_without_rechecking_evidence(env, monkeypatch):
	task = kept_task(0)
	task['checks']['critic_model'] = 'local:old'
	save_jsonl(env / 'tasks.jsonl', [task])
	(env / 'kb.json').write_text(json.dumps({'facts': []}))
	seen = []
	async def evidence(tasks, *a):
		assert tasks == []
	async def critic(tasks, facts, model):
		seen.append((len(tasks), model))
		for task in tasks:
			task['checks']['critic_model'] = model
	monkeypatch.setattr(P, 'closed_book', evidence)
	monkeypatch.setattr(P, 'check_live', evidence)
	monkeypatch.setattr(P, 'critique', critic)
	asyncio.run(P.verify_stage(env, {}, 'local:new'))
	asyncio.run(P.verify_stage(env, {}, 'local:new'))
	assert seen == [(1, 'local:new'), (0, 'local:new')]


def test_cli_build_persists_roles(env, monkeypatch):
	(env / 'spec.json').write_text(json.dumps({'topic': 'test'}))
	monkeypatch.setattr(M.os, '_exit', lambda *_: None)
	seen = []
	async def build(env, spec, only=None):
		seen.append((spec['writer_model'], spec['critic_model']))
		return 'ok'
	monkeypatch.setattr(P, 'build', build)
	main(['build', str(env), '--critic', 'local:c', '--only', 'verify'])
	main(['build', str(env), '--only', 'verify'])
	assert seen == [('local:c', 'local:c'), ('local:c', 'local:c')]
	assert json.loads((env / 'spec.json').read_text())['critic_model'] == 'local:c'
