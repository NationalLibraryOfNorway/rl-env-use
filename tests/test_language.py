"""Language routing, grading invariants and persisted CLI settings; no live model calls."""

import asyncio
import json

import httpx
import pytest
from conftest import mk_task, write_jl
from test_create import PLAN

from rl_env_use import language as L
from rl_env_use import llm
from rl_env_use import pipeline as P
from rl_env_use import tasks as T
from rl_env_use.__main__ import main
from rl_env_use.env import Env


@pytest.mark.parametrize('code', ['eng', 'nob', 'nno', 'nor', 'spa', 'deu', 'fra', 'jpn', 'zho'])
def test_iso_codes(code):
	assert L.language_code(code.upper()) == code


@pytest.mark.parametrize('code', ['no', 'nb', 'nn', 'english', 'xyz', 'ger'])
def test_invalid_codes(code):
	with pytest.raises(ValueError, match='ISO 639-3'):
		L.language_code(code)


def test_templates_and_diagnostics_are_independent():
	with L.language_scope({'language': 'nno'}):
		assert 'norsk nynorsk' in L.language_prompt('dataset')
		assert 'diagnostic prose in English' in L.language_prompt()
		assert 'reasoning, memory, evaluation and next-goal explanations in English' in L.language_prompt('policy')
	with L.language_scope({'language': 'nno', 'localize_diagnostics': True}):
		assert 'diagnostic prose in Norwegian Nynorsk' in L.language_prompt()
	with L.language_scope({'language': 'jpn'}):
		assert 'Japanese (ISO 639-3: jpn)' in L.language_prompt('dataset')


def test_parallel_workflows_do_not_leak_language():
	@L.environment_language
	async def stage(env, spec):
		await asyncio.sleep(0)
		return L.current()

	async def both():
		return await asyncio.gather(stage(None, {'language': 'nob'}), stage(None, {'language': 'spa'}))

	with L.language_scope({'language': 'eng'}):
		assert [r['language'] for r in asyncio.run(both())] == ['nob', 'spa']
		assert L.current()['language'] == 'eng'


def test_forecast_language_does_not_translate_gold_or_evidence():
	fact = {'id': 'f0', 'url': 'https://example.org', 'quote': 'The outcome was yes', 'value': 'yes'}
	task = mk_task(0, answer='yes', answer_type='prob', prompt='Vil forsøket nå målet?', kind='predict')
	with L.language_scope({'language': 'nob'}):
		result = T.finish(task, {'f0': fact})
	assert result['language'] == 'nob'
	assert 'Oppgi sannsynligheten' in result['prompt'] and 'Answer with the probability' not in result['prompt']
	assert result['answer'] == 'yes' and result['evidence'][0]['quote'] == fact['quote']


def test_cli_persists_language_and_resumes_without_flags(tmp_path, monkeypatch):
	import rl_env_use.__main__ as M

	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr(M.os, '_exit', lambda *_: None)

	async def build(*a, **kw):
		return 'ok'

	monkeypatch.setattr(P, 'build', build)
	main(['new', 'norsk', '--topic', 'Norwegian cases', '--skill', 'input -> outcome', '--language', 'nno', '--localize-diagnostics'])
	main(['build', 'envs/norsk', '--only', 'look'])
	path = tmp_path / 'envs/norsk/spec.json'
	spec = json.loads(path.read_text())
	assert spec['language'] == 'nno' and spec['localize_diagnostics'] is True
	main(['build', 'envs/norsk', '--only', 'look', '--no-localize-diagnostics'])
	assert json.loads(path.read_text())['localize_diagnostics'] is False
	write_jl(tmp_path / 'envs/norsk/tasks.jsonl', [mk_task(0, language='nno')])
	with pytest.raises(SystemExit, match='mix datasets'):
		main(['build', 'envs/norsk', '--language', 'spa'])
	assert json.loads(path.read_text())['language'] == 'nno'


def test_plan_only_passes_language_without_writing(tmp_path, monkeypatch, capsys):
	import rl_env_use.__main__ as M

	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr(M.os, '_exit', lambda *_: None)

	async def plan(*a):
		assert L.current() == {'language': 'nob', 'localize_diagnostics': True}
		return dict(PLAN)

	monkeypatch.setattr(P, 'plan_spec', plan)
	main(['create', 'Norwegian data', '--language', 'nob', '--localize-diagnostics', '--plan-only'])
	out = capsys.readouterr().out
	assert '"language": "nob"' in out and not (tmp_path / 'envs').exists()


def test_export_keeps_language_and_grading(env_dir, tmp_path):
	tasks = [mk_task(0, language='nob', answer='137', prompt='Hva er verdien?')]
	write_jl(env_dir / 'tasks.jsonl', tasks)
	env = Env(env_dir)
	output = tmp_path / 'dataset.jsonl'
	assert env.export_jsonl(output) == 1
	row = json.loads(output.read_text())
	assert row['language'] == 'nob' and row['answer'] == '137'
	assert env.reward('t0', '137') == 1.0 and env.reward('t0', '138') == 0.0


@pytest.mark.asyncio
async def test_actual_transport_receives_target_language_template(monkeypatch):
	from pydantic import BaseModel

	class Answer(BaseModel):
		answer: str

	seen = []

	def handler(request):
		seen.append(json.loads(request.content))
		return httpx.Response(
			200,
			json={
				'id': 'test',
				'object': 'chat.completion',
				'created': 0,
				'model': 'test',
				'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': '{"answer":"raud"}'}, 'finish_reason': 'stop'}],
				'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
			},
		)

	chat = llm.make_llm('local:test')
	monkeypatch.setattr(llm, 'make_llm', lambda *a, **kw: chat)
	async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
		chat._chat.http_client = client
		with L.language_scope({'language': 'nno'}):
			result, _, _ = await llm.call('local:test', 'Write a task.', 'Example', Answer, audience='dataset')
	assert result.answer == 'raud'
	assert 'norsk nynorsk' in seen[0]['messages'][0]['content']
	assert 'Preserve source quotations' in seen[0]['messages'][0]['content']


def test_environment_defaults_and_spec_override(monkeypatch):
	monkeypatch.setenv('RLENV_LANGUAGE', 'nob')
	monkeypatch.setenv('RLENV_LOCALIZE_DIAGNOSTICS', '1')
	assert L.settings() == {'language': 'nob', 'localize_diagnostics': True}
	assert L.settings({'language': 'spa', 'localize_diagnostics': False}) == {'language': 'spa', 'localize_diagnostics': False}
	monkeypatch.setenv('RLENV_LOCALIZE_DIAGNOSTICS', 'yes')
	with pytest.raises(ValueError, match='0 or 1'):
		L.settings()


def test_translated_answer_keeps_original_evidence_and_alias_grading():
	fact = {'id': 'f0', 'url': 'https://example.org', 'quote': 'The indicator color is red.', 'value': 'red'}
	task = mk_task(0, answer='rød', aliases=['red'], answer_type='exact', prompt='Hvilken farge har indikatoren?')
	with L.language_scope({'language': 'nob'}):
		result = T.finish(task, {'f0': fact})
	assert result['checks']['structural'] == []
	assert result['evidence'][0]['value'] == 'red'
	assert T.grade_answer(result, 'rød') and T.grade_answer(result, 'red')
	assert not T.grade_answer(result, 'blå')
