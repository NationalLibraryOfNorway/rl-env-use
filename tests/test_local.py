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
	monkeypatch.setenv('RLENV_LOCAL_MODELS', '{"test-text":{"base_url":"http://unavailable.test:9000/v1","model":"raw-text"}}')
	seen = []

	async def fail(self, *a, **kw):
		seen.append((self.model, self._chat.base_url, self._chat.model))
		raise ConnectionError('local endpoint down')

	async def no_sleep(*a):
		pass

	monkeypatch.setattr(LocalChat, 'ainvoke', fail)
	monkeypatch.setattr(llm.asyncio, 'sleep', no_sleep)
	with pytest.raises(ConnectionError):
		await llm.call('local-text', 'sys', 'question', Answer, retries=3)
	assert seen == [('local:test-text', 'http://unavailable.test:9000/v1', 'raw-text')] * 3


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
	from browser_use import Agent

	from rl_env_use.browse import browser

	agent = Agent(task='Read the current page', llm=llm.make_llm('local:test'), browser_session=browser(), use_vision=browser_use_vision(), calculate_cost=False)
	assert agent.settings.use_vision is expected


def test_browser_vision_mode_rejects_invalid_setting(monkeypatch):
	from rl_env_use.config import browser_use_vision

	monkeypatch.setenv('RLENV_BROWSER_USE_VISION', 'false')
	with pytest.raises(ValueError, match='RLENV_BROWSER_USE_VISION'):
		browser_use_vision()


@pytest.mark.asyncio
async def test_named_local_models_route_requests_and_auth_independently(strict, monkeypatch):
	monkeypatch.setenv('RLENV_LOCAL_MODELS', json.dumps({
		'writer': {'model': 'shared-id', 'base_url': 'http://writer.test:8000/v1', 'api_key_env': 'WRITER_TOKEN'},
		'critic': {'model': 'shared-id', 'base_url': 'http://critic.test:9000/v1', 'api_key_env': 'CRITIC_TOKEN'},
	}))
	monkeypatch.setenv('WRITER_TOKEN', 'writer-token')
	monkeypatch.setenv('CRITIC_TOKEN', 'critic-token')
	requests = []
	def handler(request):
		requests.append(request)
		return httpx.Response(200, json={
			'id': 'test', 'object': 'chat.completion', 'created': 0, 'model': 'shared-id',
			'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': '{"answer":"42"}'}, 'finish_reason': 'stop'}],
			'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
		})
	for alias in ('writer', 'critic'):
		chat = llm.make_llm(f'local:{alias}', agent=False, fallback=True)
		assert chat.model == f'local:{alias}'
		async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
			chat._chat.http_client = client
			result = await chat.ainvoke([UserMessage(content='6 times 7?')], output_format=Answer)
		assert result.completion.answer == '42'
	assert [str(r.url) for r in requests] == ['http://writer.test:8000/v1/chat/completions', 'http://critic.test:9000/v1/chat/completions']
	assert [r.headers['authorization'] for r in requests] == ['Bearer writer-token', 'Bearer critic-token']
	assert all(json.loads(r.content)['model'] == 'shared-id' for r in requests)


def test_named_route_overrides_browser_endpoint_and_preserves_defaults(strict, monkeypatch):
	monkeypatch.setenv('RLENV_LOCAL_BROWSER_MODEL', 'vision')
	monkeypatch.setenv('RLENV_LOCAL_BROWSER_BASE_URL', 'http://browser.test/v1')
	monkeypatch.setenv('RLENV_LOCAL_MODELS', json.dumps({'vision': {'model': 'raw-vision', 'base_url': 'http://vision.test/v1'}}))
	for name in ('local-browser', 'local:vision'):
		chat = llm.make_llm(name)
		assert chat._chat.base_url == 'http://vision.test/v1'
		assert chat._chat.model == 'raw-vision'
	assert llm.make_llm('local-text', agent=False)._chat.base_url == 'http://127.0.0.1:8000/v1'
	assert llm.make_llm('local:unregistered')._chat.model == 'unregistered'


@pytest.mark.parametrize('value', ['{bad', '[]', '{"x":null}', '{"x":{"unknown":"foo"}}', '{"x":{"model":1}}', '{"x":{"base_url":""}}'])
def test_invalid_local_model_registry_rejected(strict, monkeypatch, value):
	monkeypatch.setenv('RLENV_LOCAL_MODELS', value)
	with pytest.raises(ValueError, match='local model|RLENV_LOCAL_MODELS'):
		llm.make_llm('local:x')


def test_named_route_missing_token_fails_without_fallback(strict, monkeypatch):
	monkeypatch.setenv('RLENV_LOCAL_MODELS', '{"x":{"api_key_env":"UNSET_LOCAL_TEST_TOKEN"}}')
	monkeypatch.delenv('UNSET_LOCAL_TEST_TOKEN', raising=False)
	with pytest.raises(ValueError, match='api_key_env'):
		llm.make_llm('local:x', fallback=True)


@pytest.mark.parametrize('url', ['https://user:secret@host/v1', 'file:///tmp/model', 'http://host/v1?key=secret'])
def test_named_route_invalid_endpoint_rejected(strict, monkeypatch, url):
	monkeypatch.setenv('RLENV_LOCAL_MODELS', json.dumps({'x': {'base_url': url}}))
	with pytest.raises(ValueError, match='local base URL'):
		llm.make_llm('local:x')


@pytest.mark.parametrize('alias', ['local-text', 'local-browser', 'local:critic'])
def test_local_completion_default_and_override(strict, monkeypatch, alias):
	monkeypatch.delenv('RLENV_LOCAL_MAX_TOKENS', raising=False)
	assert llm.make_llm(alias)._chat.max_completion_tokens == 262144
	monkeypatch.setenv('RLENV_LOCAL_MAX_TOKENS', '4096')
	assert llm.make_llm(alias)._chat.max_completion_tokens == 4096


@pytest.mark.asyncio
async def test_context_limit_retries_with_remaining_budget_without_mutating_default(strict, monkeypatch):
	monkeypatch.setenv('RLENV_LOCAL_MAX_TOKENS', '262144')
	requests = []
	def handler(request):
		requests.append(request)
		if len(requests) == 1:
			return httpx.Response(400, json={'error': {'message': "Requested token count exceeds the model's maximum context length of 262144 tokens. You requested a total of 262322 tokens: 178 tokens from the input messages and 262144 tokens for the completion."}})
		return httpx.Response(200, json={'id': 'test', 'object': 'chat.completion', 'created': 0, 'model': 'test-text',
			'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': '{"answer":"42"}'}, 'finish_reason': 'stop'}],
			'usage': {'prompt_tokens': 178, 'completion_tokens': 4, 'total_tokens': 182}})
	chat = llm.make_llm('local-text')
	async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
		chat._chat.http_client = client
		result = await chat.ainvoke([UserMessage(content='6 times 7?')], output_format=Answer)
	assert result.completion.answer == '42'
	bodies = [json.loads(r.content) for r in requests]
	assert [b['max_completion_tokens'] for b in bodies] == [262144, 261966]
	assert requests[0].url == requests[1].url
	assert bodies[0]['messages'] == bodies[1]['messages']
	assert chat._chat.max_completion_tokens == 262144


@pytest.mark.asyncio
@pytest.mark.parametrize('message', ['bad schema', 'maximum context length of 100 tokens; 101 tokens from the input messages'])
async def test_other_provider_errors_and_oversized_input_do_not_get_budget_retry(strict, message):
	requests = []
	def handler(request):
		requests.append(request)
		return httpx.Response(400, json={'error': {'message': message}})
	chat = llm.make_llm('local-text')
	async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
		chat._chat.http_client = client
		with pytest.raises(Exception, match='bad schema|maximum context'):
			await chat.ainvoke([UserMessage(content='question')], output_format=Answer)
	assert len(requests) == 1


def test_model_timeout_defaults_and_override(strict, monkeypatch):
	monkeypatch.delenv('RLENV_LOCAL_TIMEOUT', raising=False)
	assert llm.make_llm('local-text')._chat.timeout == 6000
	monkeypatch.setenv('RLENV_LOCAL_TIMEOUT', '123')
	assert llm.make_llm('local-text')._chat.timeout == 123
