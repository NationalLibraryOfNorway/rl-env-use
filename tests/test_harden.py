"""Harden: tasks the whole panel solved go back to the drafter, which writes harder ones on the same case, calibrated by the env's own record."""

import asyncio
import copy
import json

from conftest import jl, write_jl
from test_panel import OK, ct

import rl_env_use.pipeline as P
import rl_env_use.tasks as T

KB = {
	'topic': 'credit',
	'facts': [
		{'id': 'f0', 'case': 'c0', 'claim': 'rev', 'value': '1,500', 'quote': 'total revenue was 1,500', 'url': 'https://sec.gov/a'},
		{'id': 'f1', 'case': 'c1', 'claim': 'rev', 'value': '1,500', 'quote': 'total revenue was 1,500', 'url': 'https://sec.gov/b'},
	],
	'materials': [
		{'id': 'm0', 'case': 'c0', 'title': 'Acme 10-K', 'text': 'Revenue was 1,000 in 2023 and 500 in 2022. Interest 40.', 'url': 'https://sec.gov/a'},
		{'id': 'm1', 'case': 'c1', 'title': 'Beta 10-K', 'text': 'Revenue was 1,000 in 2023 and 500 in 2022. Interest 40.', 'url': 'https://sec.gov/b'},
	],
	'missions': [],
	'ideas': [],
	'sources': [],
}


def sat(i, level=0, **p):
	t = ct(i, checks=copy.deepcopy(OK) | {'panel': {'verdict': 'saturated', 'scores': {'haiku': 1.0, 'gpt6': 1.0}} | p}, level=level)
	return t


def hard_kept(i):
	return ct(i, checks=copy.deepcopy(OK) | {'panel': {'verdict': 'keep', 'scores': {'haiku': 0.0, 'gpt6': 0.5}, 'adjudication': {'panel_error': 'used 2022 revenue'}}})


def drafter(log, n=None):
	async def call(model, system, user, schema, **kw):
		log.append({'system': system, 'user': user})
		k = n if n is not None else user.count('\n- ') or 1
		return (
			T.CaseDrafts(
				tasks=[
					T.CaseTaskDraft(
						prompt=f'Harder task {len(log)}.{j}: interest coverage over two years?',
						kind='derive',
						material_ids=['m0', 'm1'],
						answer='37.5',
						answer_type='number',
						fact_ids=['f0'],
						derivation='(1000+500)/40',
						expression='(1000 + 500) / 40',
						distractors=['1000 / 40'],
						category='coverage',
						difficulty='hard',
						why_hard='two periods',
					)
					for j in range(k)
				]
			),
			{},
			model,
		)

	return call


def test_calibration_lists_too_easy_and_right_level_with_errors():
	txt = T.calibration([sat(0), hard_kept(1), ct(2)])
	assert 'TOO EASY' in txt and 'case 0' in txt and 'RIGHT LEVEL' in txt and 'case 1' in txt and 'used 2022 revenue' in txt and 'case 2' not in txt
	assert T.calibration([ct(2)]) == ''


def test_harden_mode_writes_one_harder_task_per_easy_task_one_level_up(monkeypatch):
	log = []
	monkeypatch.setattr(T, 'call', drafter(log, n=5))
	out = asyncio.run(T.synthesize_cases(KB, model='m', easy={'c0': [sat(0, level=0), sat(5, level=1)]}, calibration='RIGHT LEVEL x'))
	assert len(log) == 1 and len(out) == 2  # capped at one per easy task, only the case with easy tasks
	assert all(t['level'] == 2 and t['hardened_from'] == ['t0', 't5'] and t['case'] == 'c0' for t in out)
	assert 'HARDEN' in log[0]['system'] and 'HARD MODE' in log[0]['system'] and 'TOO EASY on this case' in log[0]['user'] and 'RIGHT LEVEL x' in log[0]['user']
	assert not out[0]['checks']['structural']  # a hardened draft gets the same deterministic checks


def test_normal_synth_carries_calibration_and_level_zero(monkeypatch):
	log = []
	monkeypatch.setattr(T, 'call', drafter(log, n=1))
	out = asyncio.run(T.synthesize_cases(KB, per_case=1, model='m', calibration='TOO EASY:\n- x'))
	assert len(out) == 2 and all(t['level'] == 0 and 'hardened_from' not in t for t in out)
	assert all('DIFFICULTY CALIBRATION' in c['user'] and 'HARDEN' not in c['system'] for c in log)


def env_with(tmp_path, tasks):
	env = tmp_path / 'env'
	env.mkdir()
	(env / 'kb.json').write_text(json.dumps(KB))
	write_jl(env / 'tasks.jsonl', tasks)
	return env


def test_harden_stage_marks_parents_and_respects_levels(monkeypatch, tmp_path):
	log = []
	monkeypatch.setattr(T, 'call', drafter(log))
	env = env_with(tmp_path, [sat(0), sat(1, level=2), sat(2, hardened=True), hard_kept(3)])
	msg = asyncio.run(P.harden_stage(env, {'skill': 's', 'harden_levels': 2}, 'm'))
	ts = {t['id']: t for t in jl(env / 'tasks.jsonl')}
	assert msg.startswith('harden: 1 too-easy tasks on 1 cases') and len(ts) == 5
	assert ts['t0']['checks']['panel']['hardened'] and not ts['t1']['checks']['panel'].get('hardened')  # level cap: t1 is already 2 deep
	new = [t for t in ts.values() if t.get('hardened_from')]
	assert len(new) == 1 and new[0]['hardened_from'] == ['t0'] and new[0]['level'] == 1 and 'panel' not in new[0]['checks']
	assert 'used 2022 revenue' in log[0]['user']  # calibration from the env's own panel record
	assert asyncio.run(P.harden_stage(env, {'skill': 's'}, 'm')) == 'harden: no too-easy tasks left to harden'  # idempotent


def test_hardened_drafts_go_through_verify_and_panel(monkeypatch, tmp_path):
	env = env_with(tmp_path, [sat(0)])
	(env / 'spec.json').write_text(json.dumps({'topic': 'credit', 'skill': 's', 'panel': ['haiku', 'gpt6']}))
	order = []
	for name in ('harden_stage', 'verify_stage'):

		async def f(env, spec, model, name=name):
			order.append(name)
			return name

		monkeypatch.setattr(P, name, f)

	async def panel(env, spec):
		order.append('panel_stage')
		return 'panel'

	monkeypatch.setattr(P, 'panel_stage', panel)
	asyncio.run(P.build(env, json.loads((env / 'spec.json').read_text()), only='harden'))
	assert order == ['harden_stage', 'verify_stage', 'panel_stage']


def test_create_round_hardens_after_the_panel(monkeypatch, tmp_path):
	env = env_with(tmp_path, [])
	order = []

	async def scout(*a, **k):
		return KB

	monkeypatch.setattr(P, 'scout', scout)
	for name in ('synth_stage', 'verify_stage', 'harden_stage'):

		async def f(env, spec, model, name=name):
			order.append(name)
			return name

		monkeypatch.setattr(P, name, f)

	async def panel(env, spec):
		order.append('panel_stage')
		return 'panel'

	monkeypatch.setattr(P, 'panel_stage', panel)
	monkeypatch.setattr(P.LEDGER, 'by_model', {})
	asyncio.run(P.create(env, {'topic': 't', 'skill': 's', 'target_tasks': 0, 'budget_usd': 10, 'missions': 2}, log=lambda *_: None))
	assert order == ['synth_stage', 'verify_stage', 'panel_stage', 'harden_stage', 'verify_stage', 'panel_stage']
