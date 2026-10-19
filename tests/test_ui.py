"""The live board: formatting, stage ticks through ui.gather, the off-terminal status lines, and the climb and create views."""

import asyncio
import io
import json

from rich.console import Console
from test_climb import GOOD, climb, env, policy, proposer  # noqa: F401

import rl_env_use.ui as ui


def test_formatting():
	assert ui.dur(42) == '42s' and ui.dur(190) == '3m10s' and ui.dur(3725) == '1h02m' and ui.dur(None) == '--'
	assert len(ui.bar(0.0)) == 24 and ui.bar(1.0) == '█' * 24 and ui.bar(0.5).startswith('█' * 12) and len(ui.bar(0.37)) == 24 and len(ui.bar(2)) == 24
	assert ui.spark([0, 0.5, 1]) == '▁▄█' and ui.spark([1, 1]) == '██'


def test_gather_without_a_board_is_plain_gather():
	async def f(x):
		return x * 2

	assert asyncio.run(ui.gather((f(i) for i in range(3)), 'x')) == [0, 2, 4] and ui.BOARD is None


def test_gather_ticks_the_stage_even_when_an_item_fails_and_eta_follows(capsys):
	async def f(x):
		if x == 1:
			raise ValueError
		return x

	async def go():
		return await asyncio.gather(ui.gather((f(i) for i in range(4)), 'panel'), return_exceptions=True)

	out = io.StringIO()
	with ui.board('t', console=Console(file=out)) as b:
		asyncio.run(go())
		assert b.stage_[:3] == ['panel', 4, 4] and b.stage_eta() == 0.0
	assert 'panel 0/4' in capsys.readouterr().out


def test_off_terminal_prints_status_lines_and_budget():
	out = io.StringIO()
	import contextlib

	buf = io.StringIO()
	with contextlib.redirect_stdout(buf), ui.board('create', console=Console(file=out)) as b:
		ui.money(lambda: 3.5, 10)
		ui.line('tasks', '12/50 kept')
		ui.progress(0.24)
		ui.stage('mine', 2)
		ui.tick()
		ui.tick()
		ui.line('tasks', None)
	lines = buf.getvalue().splitlines()
	assert lines[0] == '  tasks: 12/50 kept' and any('mine 0/2' in x and '$3.50/$10' in x and 'overall 24%' in x for x in lines) and any('mine 2/2' in x for x in lines)
	assert sum('tasks' in x for x in lines) == 1 and ui.BOARD is None and b.live is None


def test_render_has_overall_stage_budget_and_lines():
	b = ui.Board('rl-env-use create · x', 40, lambda: 10.0)
	b.stage('panel', 10)
	b.tick(5)
	b.set_line('tasks', '30/150 kept [/x]')
	con = Console(record=True, width=120, file=io.StringIO())
	con.print(b)
	txt = con.export_text()
	assert all(s in txt for s in ('rl-env-use create · x', 'overall', 'panel', '5/10', 'budget', '$10.00 / $40.00', '30/150 kept [/x]', ' 25%'))  # literal text, not markup


def test_nested_board_reuses_the_outer_one():
	with ui.board('a', console=Console(file=io.StringIO())) as a, ui.board('b') as b:
		assert a is b and ui.BOARD is a
	assert ui.BOARD is None


def test_climb_fills_the_board(env, monkeypatch):  # noqa: F811
	policy(monkeypatch)
	proposer(monkeypatch, [[('add worked example', GOOD)]])
	with ui.board('climb', console=Console(file=io.StringIO())) as b:
		s = climb(env, rounds=2, force=True)
		lines = {k: v.plain for k, v in b.lines.items()}
	assert s['rounds'][0]['decision'] == 'keep'
	assert 'fake-model' in lines['policy'] and 'test 0.400' in lines['baseline'] and 'r1' in lines['rounds'] and '✓' in lines['rounds'] and lines['best'].startswith('v')
	assert len(lines['test'].split()[0]) == 1 + len(s['rounds']) and b.frac == 1.0


def test_create_fills_the_board(tmp_path, monkeypatch):
	from test_panel import fc

	import rl_env_use.pipeline as P

	(tmp_path / 'kb.json').write_text(json.dumps({'topic': 't', 'facts': [], 'materials': [], 'missions': [], 'ideas': [], 'sources': []}))
	from conftest import write_jl

	write_jl(tmp_path / 'tasks.jsonl', [fc(i, 'yes' if i < 3 else 'no') for i in range(5)])

	async def stage(*a, **k):
		return 'ok'

	async def scout(*a, **k):
		return json.loads((tmp_path / 'kb.json').read_text())

	for n in ('synth_stage', 'verify_stage', 'harden_stage'):
		monkeypatch.setattr(P, n, stage)
	monkeypatch.setattr(P, 'panel_stage', stage)
	monkeypatch.setattr(P, 'scout', scout)
	monkeypatch.setattr(P.LEDGER, 'by_model', {})
	with ui.board('create', console=Console(file=io.StringIO())) as b:
		asyncio.run(P.create(tmp_path, {'topic': 't', 'skill': 's', 'target_tasks': 10, 'budget_usd': 10, 'missions': 2, 'max_rounds': 1}, log=lambda *_: None))
		assert (
			b.lines['tasks'].plain.startswith('5/10 kept (0 derive, 5 forecast: 3 yes / 2 no)') and b.lines['round'].plain == '1 of at most 1' and b.frac == 0.5 and b.budget == 10
		)
