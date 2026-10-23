"""Self-hosted transport and strict-mode isolation; no GPU, browser or paid APIs."""

import json
import os
import subprocess
import sys

import httpx
import pytest
from browser_use.llm.messages import ContentPartImageParam, ContentPartTextParam, ImageURL, UserMessage
from pydantic import BaseModel

from rl_env_use import llm
from rl_env_use.local import LocalChat


class Answer(BaseModel):
	answer: str


@pytest.fixture
def strict(monkeypatch):
	monkeypatch.setenv('RLENV_OPEN_ONLY', '1')
	monkeypatch.setenv('RLENV_LOCAL_TEXT_MODEL', 'test-text')
	monkeypatch.setenv('RLENV_LOCAL_BROWSER_MODEL', 'test-vision')
	monkeypatch.setenv('RLENV_LOCAL_BASE_URL', 'http://127.0.0.1:8000/v1')
	monkeypatch.delenv('RLENV_LOCAL_BROWSER_BASE_URL', raising=False)


@pytest.mark.parametrize('name', ['haiku', 'bu', 'claude-sonnet', 'openrouter:qwen/test', 'anthropic:test', 'bu:test', 'unknown'])
def test_strict_rejects_hosted_even_with_keys(strict, monkeypatch, name):
	monkeypatch.setenv('BROWSER_USE_API_KEY', 'present')
	monkeypatch.setenv('OPENROUTER_API_KEY', 'present')
	with pytest.raises(ValueError, match='disabled'):
		llm.make_llm(name)


def test_local_aliases_no_key_and_no_fallback(strict, monkeypatch):
	assert llm.routes('local-text') == [('local', 'test-text')]
	assert llm.routes('local:org/model') == [('local', 'org/model')]
	assert llm.make_llm('local-text', agent=False, fallback=True).model == 'local:test-text'
	monkeypatch.delenv('RLENV_LOCAL_TEXT_MODEL')
	with pytest.raises(ValueError, match='RLENV_LOCAL_TEXT_MODEL'):
		llm.make_llm('local-text')
	with pytest.raises(ValueError, match='requires a model ID'):
		llm.make_llm('local:')


def test_separate_browser_endpoint(strict, monkeypatch):
	monkeypatch.setenv('RLENV_LOCAL_BROWSER_BASE_URL', 'http://127.0.0.1:8001/v1')
	assert llm.make_llm('local-browser')._chat.base_url == 'http://127.0.0.1:8001/v1'
	assert llm.make_llm('local-text', agent=False)._chat.base_url == 'http://127.0.0.1:8000/v1'


@pytest.mark.asyncio
async def test_real_serializer_preserves_images_and_validates_schema(strict):
	requests = []

	def handler(request):
		requests.append(request)
		return httpx.Response(
			200,
			json={
				'id': 'local-test',
				'object': 'chat.completion',
				'created': 0,
				'model': 'test-vision',
				'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': '{"answer":"42"}'}, 'finish_reason': 'stop'}],
				'usage': {'prompt_tokens': 12, 'completion_tokens': 4, 'total_tokens': 16},
			},
		)

	chat = llm.make_llm('local-browser')
	async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
		chat._chat.http_client = client
		result = await chat.ainvoke(
			[
				UserMessage(
					content=[
						ContentPartTextParam(text='Read this screenshot'),
						ContentPartImageParam(image_url=ImageURL(url='data:image/png;base64,aGVsbG8=')),
					]
				)
			],
			output_format=Answer,
		)
	assert result.completion.answer == '42'
	assert result.usage.prompt_tokens == 12
	body = json.loads(requests[0].content)
	assert str(requests[0].url) == 'http://127.0.0.1:8000/v1/chat/completions'
	assert body['model'] == 'test-vision'  # accounting prefix never sent to server
	assert body['response_format']['type'] == 'json_schema'
	assert body['messages'][0]['content'][1]['image_url']['url'].startswith('data:image/png;')
	assert llm.cost(chat.model, llm.usage_dict(result.usage)) is None


@pytest.mark.asyncio
async def test_unavailable_local_server_never_falls_back(strict, monkeypatch):
	seen = []

	async def fail(self, *a, **kw):
		seen.append(self.model)
		raise ConnectionError('local endpoint down')

	async def no_sleep(*a):
		pass

	monkeypatch.setattr(LocalChat, 'ainvoke', fail)
	monkeypatch.setattr(llm.asyncio, 'sleep', no_sleep)
	with pytest.raises(ConnectionError):
		await llm.call('local-text', 'sys', 'question', Answer, retries=3)
	assert seen == ['local:test-text'] * 3


def test_browser_stays_local_and_billing_skipped(strict, monkeypatch):
	from rl_env_use import browse

	monkeypatch.setenv('BROWSER_USE_API_KEY', 'present')
	monkeypatch.setattr(browse, 'BrowserSession', lambda **kw: kw)
	monkeypatch.setattr(browse, 'BrowserProfile', lambda **kw: kw)
	assert browse.browser()['use_cloud'] is False
	assert llm.bu_balance() is None


@pytest.mark.parametrize('field', ['panel', 'writer_model', 'critic_model'])
def test_legacy_spec_rejected_before_browser_use(strict, field):
	from rl_env_use.pipeline import use_browser_spec

	with pytest.raises(ValueError, match='disabled'):
		use_browser_spec({'agent_model': 'local-browser', field: ['opus55'] if field == 'panel' else 'opus55'})


@pytest.mark.asyncio
async def test_claude_comparison_blocked(strict):
	from rl_env_use.vanilla import claude_web

	with pytest.raises(ValueError, match='disabled'):
		await claude_web('hello')


@pytest.mark.parametrize('explicit_mode', [False, True])
def test_fresh_import_defaults_and_early_dotenv(tmp_path, explicit_mode):
	(tmp_path / '.env').write_text(('RLENV_OPEN_ONLY=1\n' if explicit_mode else '') + 'RLENV_LOCAL_TEXT_MODEL=from-dotenv\nRLENV_PANEL_MODELS=local:model-a,local:model-b\n')
	env = {k: v for k, v in os.environ.items() if not k.startswith('RLENV_')}
	code = """
import os
from rl_env_use.config import TEXT_MODEL, BROWSER_MODEL
from rl_env_use.llm import routes
from rl_env_use.verify import PANEL, ADJUDICATORS
from rl_env_use.scout import scout
from rl_env_use.climb import climb
from rl_env_use.__main__ import main
import inspect
assert TEXT_MODEL == 'local-text' and BROWSER_MODEL == 'local-browser'
assert routes(TEXT_MODEL) == [('local', 'from-dotenv')]
assert PANEL == ADJUDICATORS == ['local:model-a', 'local:model-b']
assert inspect.signature(scout).parameters['model'].default == 'local-browser'
assert inspect.signature(climb).parameters['proposer'].default == 'local-text'
assert os.environ['ANONYMIZED_TELEMETRY'] == 'false'
assert os.environ['BROWSER_USE_CLOUD_SYNC'] == 'false'
assert os.environ['BROWSER_USE_CALCULATE_COST'] == 'false'
"""
	subprocess.run([sys.executable, '-c', code], cwd=tmp_path, env=env, check=True, capture_output=True, text=True)


@pytest.mark.parametrize('setting, expected', [('0', False), ('1', True)])
def test_browser_vision_mode(monkeypatch, setting, expected):
	from rl_env_use.config import browser_use_vision

	monkeypatch.setenv('RLENV_BROWSER_USE_VISION', setting)
	assert browser_use_vision() is expected
	from rl_env_use.browse import browser
	from browser_use import Agent

	agent = Agent(task='Read the current page', llm=llm.make_llm('local:test'), browser_session=browser(), use_vision=browser_use_vision(), calculate_cost=False)
	assert agent.settings.use_vision is expected


def test_browser_vision_mode_rejects_invalid_setting(monkeypatch):
	from rl_env_use.config import browser_use_vision

	monkeypatch.setenv('RLENV_BROWSER_USE_VISION', 'false')
	with pytest.raises(ValueError, match='RLENV_BROWSER_USE_VISION'):
		browser_use_vision()
