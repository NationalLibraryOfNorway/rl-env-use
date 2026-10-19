import asyncio
import json

import pytest
from browser_use.llm.exceptions import ModelProviderError
from browser_use.llm.messages import SystemMessage, UserMessage
from pydantic import BaseModel

import rl_env_use.llm as L
from rl_env_use.audit import strength
from rl_env_use.runner import REFUSAL


class Ans(BaseModel):
	answer: str
	confidence: float


class Resp:
	def __init__(self, status=200, body=None, ctype='application/json', text=''):
		self.status_code, self._body, self.headers, self.text = status, body, {'content-type': ctype}, text

	def json(self):
		return self._body


def ok(content, cost=0.0123, finish='stop', native=None, usage=True):
	u = {'prompt_tokens': 1000, 'completion_tokens': 200, 'prompt_tokens_details': {'cached_tokens': 100}, **({'cost': cost} if cost is not None else {})}
	return Resp(body={'choices': [{'message': {'content': content}, 'finish_reason': finish, 'native_finish_reason': native}], **({'usage': u} if usage else {})})


@pytest.fixture
def fake(monkeypatch):
	"""Patch httpx.AsyncClient; returns a holder with .queue (responses to serve) and .posts (captured requests)."""
	import httpx

	h = type('H', (), {'queue': [], 'posts': []})()

	class Client:  # the sync client OpenRouter.post uses in its worker thread
		def __init__(self, timeout=None, limits=None):
			self.timeout = timeout

		def post(self, url, headers=None, json=None):
			h.posts.append({'url': url, 'headers': headers, 'json': json, 'timeout': self.timeout})
			return h.queue.pop(0)

	monkeypatch.setattr(httpx, 'Client', Client)
	monkeypatch.setattr(L.OpenRouter, 'local', __import__('threading').local())
	monkeypatch.setenv('OPENROUTER_API_KEY', 'sk-or-test')
	monkeypatch.setattr(L, 'LEDGER', L.Ledger())
	return h


def inv(llm, msgs=None, fmt=Ans):
	return asyncio.run(llm.ainvoke(msgs or [SystemMessage(content='sys'), UserMessage(content='q?')], output_format=fmt))


def test_parses_json_inside_prose_and_fence(fake):
	fake.queue.append(ok('Sure, here it is:\n```json\n{"answer": "137", "confidence": 0.9}\n```\nHope that helps!'))
	r = inv(L.OpenRouter('openai/gpt-6.1-sol'))
	assert r.completion == Ans(answer='137', confidence=0.9)
	p = fake.posts[0]
	assert p['url'] == 'https://openrouter.ai/api/v1/chat/completions' and p['headers']['Authorization'] == 'Bearer sk-or-test'
	assert p['json']['model'] == 'openai/gpt-6.1-sol' and p['json']['usage'] == {'include': True} and p['timeout'].read == L.CALL_TIMEOUT and p['timeout'].connect == 30


def test_parses_python_literal_style(fake):
	fake.queue.append(ok("{'answer': 'x', 'confidence': 0.5}"))
	assert inv(L.OpenRouter('m')).completion.answer == 'x'


def test_plain_text_without_schema(fake):
	fake.queue.append(ok('  just text  '))
	r = inv(L.OpenRouter('m'), fmt=None)
	assert r.completion == 'just text'
	assert 'JSON schema' not in fake.posts[0]['json']['messages'][-1]['content']


def test_invalid_json_shape_raises_validation(fake):
	fake.queue.append(ok('{"answer": "x"}'))
	with pytest.raises(Exception, match='validation error'):
		inv(L.OpenRouter('m'))


def test_schema_appended_to_last_message_only(fake):
	fake.queue.append(ok('{"answer": "a", "confidence": 1}'))
	inv(L.OpenRouter('m'), [SystemMessage(content='sys'), UserMessage(content='first'), UserMessage(content='last')])
	msgs = fake.posts[0]['json']['messages']
	assert [m['role'] for m in msgs] == ['system', 'user', 'user']
	assert msgs[0]['content'] == 'sys' and msgs[1]['content'] == 'first'
	assert msgs[2]['content'].startswith('last\n\nReply with ONLY a JSON object')
	assert json.loads(msgs[2]['content'].split('\n')[-1]) == Ans.model_json_schema()


def test_temperature_omitted_when_none(fake):
	fake.queue += [ok('{"answer": "a", "confidence": 1}'), ok('{"answer": "a", "confidence": 1}'), ok('{"answer": "a", "confidence": 1}')]
	inv(L.OpenRouter('m'))
	inv(L.OpenRouter('m', temperature=0.0))
	inv(L.OpenRouter('m', temperature=0.7))
	assert 'temperature' not in fake.posts[0]['json']
	assert fake.posts[1]['json']['temperature'] == 0.0 and fake.posts[2]['json']['temperature'] == 0.7


def test_usage_dict_carries_reported_cost(fake):
	fake.queue.append(ok('{"answer": "a", "confidence": 1}', cost=0.0123))
	u = inv(L.OpenRouter('anthropic/claude-opus-5.5')).usage
	assert u == {'input_tokens': 1000, 'output_tokens': 200, 'cache_read_input_tokens': 100, 'reported_cost': 0.0123}
	assert L.usage_dict(u) is u
	assert L.cost('anthropic/claude-opus-5.5', u) == 0.0123


def test_missing_cost_falls_back_to_prices(fake):
	fake.queue.append(ok('{"answer": "a", "confidence": 1}', cost=None))
	u = inv(L.OpenRouter('anthropic/claude-opus-5.5')).usage
	assert u['reported_cost'] == 0
	assert L.cost('anthropic/claude-opus-5.5', u) == pytest.approx((900 * 4.0 + 100 * 0.4 + 200 * 20.0) / 1e6)


def test_missing_usage_block_is_zeroed(fake):
	fake.queue.append(ok('{"answer": "a", "confidence": 1}', usage=False))
	u = inv(L.OpenRouter('m')).usage
	assert u == {'input_tokens': 0, 'output_tokens': 0, 'cache_read_input_tokens': 0, 'reported_cost': 0}


@pytest.mark.parametrize(
	'resp,code',
	[
		(Resp(429, {'error': {'message': 'rate limited'}}), 429),
		(Resp(502, None, ctype='text/html', text='<html>bad gateway</html>'), 502),
		(Resp(200, {'error': {'message': 'provider down', 'code': 500}}), 200),
		(Resp(200, {'choices': []}), 200),
	],
)
def test_http_errors_raise_provider_error(fake, resp, code):
	fake.queue.append(resp)
	with pytest.raises(ModelProviderError) as e:
		inv(L.OpenRouter('m'))
	assert e.value.status_code == code and f'openrouter {code}' in str(e.value)
	assert not REFUSAL.search(str(e.value))


def test_html_error_body_in_message(fake):
	fake.queue.append(Resp(503, None, ctype='text/plain', text='upstream overloaded'))
	with pytest.raises(ModelProviderError, match='upstream overloaded'):
		inv(L.OpenRouter('m'))


@pytest.mark.parametrize('content,finish,native', [('partial', 'content_filter', None), ('', 'stop', 'refusal'), (None, 'stop', 'refusal')])
def test_refusals_classified_by_runner(fake, content, finish, native):
	fake.queue.append(ok(content, finish=finish, native=native))
	with pytest.raises(ModelProviderError) as e:
		inv(L.OpenRouter('m'))
	assert REFUSAL.search(str(e.value))


def test_native_refusal_with_content_is_not_refusal(fake):
	fake.queue.append(ok('{"answer": "a", "confidence": 1}', native='refusal'))
	assert inv(L.OpenRouter('m')).completion.answer == 'a'


def test_resolve_openrouter_aliases_need_key(monkeypatch):
	monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
	with pytest.raises(RuntimeError, match=r"no API key for model 'gpt6'.*OPENROUTER_API_KEY"):
		L.resolve('gpt6')
	monkeypatch.setenv('OPENROUTER_API_KEY', 'k')
	for alias in ('opus55', 'sonnet55', 'gpt6', 'gemini', 'gemini-flash', 'grok', 'deepseek', 'qwen-27b', 'qwen-35b'):
		route, mid = L.resolve(alias)
		assert route == 'openrouter' and mid == L.MODELS[alias][0][1] and '/' in mid


def test_resolve_explicit_openrouter_and_unknown(monkeypatch):
	monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
	assert L.resolve('openrouter:meta-llama/llama-5-70b') == ('openrouter', 'meta-llama/llama-5-70b')
	with pytest.raises(RuntimeError, match='no API key'):
		L.resolve('no-such-model')


def test_make_llm_openrouter(monkeypatch):
	monkeypatch.setenv('OPENROUTER_API_KEY', 'k')
	llm = L.make_llm('gemini', temperature=0.3, agent=False)
	assert isinstance(llm, L.OpenRouter) and llm.model == 'google/gemini-3.1-pro-preview' and llm.temperature == 0.3
	assert isinstance(L.make_llm('openrouter:a/b', agent=False), L.OpenRouter)
	assert not isinstance(L.make_llm('gemini', agent=True), L.OpenRouter)


def test_call_end_to_end_with_ledger(fake):
	fake.queue += [ok('{"answer": "a", "confidence": 1}', cost=0.01), ok('```\n{"answer": "b", "confidence": 0.2}\n```', cost=0.02)]
	p1, u1, m1 = asyncio.run(L.call('gpt6', 'sys', 'u1', Ans))
	p2, u2, m2 = asyncio.run(L.call('gpt6', 'sys', 'u2', Ans))
	assert (p1.answer, p2.answer) == ('a', 'b') and m1 == m2 == 'openai/gpt-6.1-sol'
	assert fake.posts[0]['json']['temperature'] == 0.0 and fake.posts[0]['json']['messages'][1]['content'].startswith('u1') and 'reasoning' not in fake.posts[0]['json']
	assert L.LEDGER.by_model['openai/gpt-6.1-sol']['input_tokens'] == 2000
	assert L.LEDGER.usd == pytest.approx(0.03)


def test_call_retries_with_feedback_and_resampling(fake, monkeypatch):
	async def nosleep(s):
		pass

	monkeypatch.setattr(L.asyncio, 'sleep', nosleep)
	fake.queue += [Resp(500, {'error': {'message': 'boom'}}), ok('{"answer": "x"}'), ok('{"answer": "c", "confidence": 0.5}', cost=0.005)]
	p, u, m = asyncio.run(L.call('gpt6', 's', 'q', Ans))
	assert p.answer == 'c' and u['reported_cost'] == 0.005
	assert [x['json'].get('temperature') for x in fake.posts] == [0.0, 0.7, 0.7]
	assert 'previous output was invalid' not in fake.posts[1]['json']['messages'][1]['content']
	assert 'previous output was invalid' in fake.posts[2]['json']['messages'][1]['content']
	assert L.LEDGER.usd == pytest.approx(0.005 + 0.0123)  # the 500 is not billed; the unparseable 200 was, and is booked


def test_call_refusal_propagates_after_retries(fake, monkeypatch):
	async def nosleep(s):
		pass

	monkeypatch.setattr(L.asyncio, 'sleep', nosleep)
	fake.queue += [ok('', finish='content_filter')] * 2
	with pytest.raises(ModelProviderError) as e:
		asyncio.run(L.call('grok', 's', 'q', Ans, retries=2))
	assert REFUSAL.search(str(e.value)) and not fake.queue


def test_ledger_mixes_reported_and_priced():
	led = L.Ledger()
	led.add('openai/gpt-6.1-sol', {'input_tokens': 10, 'output_tokens': 10, 'reported_cost': 0.5})
	led.add('openai/gpt-6.1-sol', {'input_tokens': 10, 'output_tokens': 10, 'reported_cost': 0.25})
	led.add('anthropic/claude-haiku-4-5', {'input_tokens': 1_000_000, 'output_tokens': 0})
	assert led.usd == pytest.approx(0.75 + 1.0)
	led.add('google/gemini-3.1-pro-preview', {'input_tokens': 0, 'output_tokens': 1_000_000, 'reported_cost': 0.01})
	led.add('google/gemini-3.1-pro-preview', {'input_tokens': 0, 'output_tokens': 1_000_000, 'reported_cost': 0})  # unbilled reply: priced, not hidden by the billed one
	assert led.usd == pytest.approx(0.75 + 1.0 + 0.01 + 12.0)


def test_strength_orders_models():
	ids = {a: L.MODELS[a][0][1] for a in ('qwen-35b', 'deepseek', 'gemini-flash', 'grok', 'gpt6', 'gemini', 'opus55')}
	s = {a: strength(m) for a, m in ids.items()}
	assert strength('anthropic/claude-haiku-4-5') < s['gpt6'] < s['opus55']
	assert s['deepseek'] < s['qwen-35b'] < s['gemini-flash'] < s['grok'] < s['gpt6'] < s['gemini'] < s['opus55']
	assert strength('unknown/model') == 0.0 and strength(None) == 0.0
	assert all(L.price_of(m) for a, m in ids.items()) and all(L.price_of(L.MODELS[a][0][1]) for a in ('sonnet55', 'qwen-27b'))


def test_last_attempts_fall_back_to_the_same_model_on_another_route(monkeypatch):
	import asyncio

	from pydantic import BaseModel

	import rl_env_use.llm as L

	class A(BaseModel):
		answer: str

	monkeypatch.setenv('BROWSER_USE_API_KEY', 'k')
	monkeypatch.setenv('OPENROUTER_API_KEY', 'k')
	monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
	assert L.routes('sonnet46') == [('openrouter', 'anthropic/claude-sonnet-4.6'), ('bu', 'anthropic/claude-sonnet-4-6')]
	seen = []

	class Fake:
		def __init__(self, route, mid):
			self.model = mid
			self.route = route

		async def ainvoke(self, messages, output_format=None, **kw):
			seen.append(self.route)
			if self.route == 'openrouter':
				raise RuntimeError('openrouter 502: upstream error')
			return L.SimpleNamespace(completion=A(answer='ok'), usage={'input_tokens': 1, 'output_tokens': 1, 'reported_cost': 0.01})

	monkeypatch.setattr(L, 'make_llm', lambda name, t=None, agent=True, fallback=False: Fake(*L.routes(name)[int(fallback)]))
	nap = asyncio.sleep
	monkeypatch.setattr(L.asyncio, 'sleep', lambda *_: nap(0))
	r, u, mid = asyncio.run(L.call('sonnet46', 's', 'u', A))
	assert r.answer == 'ok' and mid == 'anthropic/claude-sonnet-4-6' and seen == ['openrouter', 'openrouter', 'bu']


def test_frontier_models_reason_internally_and_answer_only(fake):
	fake.queue += [ok('{"answer": "a", "confidence": 1}', cost=0.01)]
	p, _, mid = asyncio.run(L.call('opus55', 'sys', 'u', Ans))
	body = fake.posts[0]['json']
	assert p.answer == 'a' and mid == 'anthropic/claude-opus-5.5' and body['reasoning'] == {'effort': 'high', 'exclude': True} and 'temperature' not in body
	assert {'opus55', 'sonnet55', 'gpt-astra'} <= L.ANSWER_ONLY and L.price_of('openai/gpt-6-astra') == (10.0, 50.0)


def test_reply_with_json_then_a_repeated_final_answer_parses(fake):
	fake.queue += [ok('{"answer": "6.6%", "confidence": 1}\n```\n\nFinal answer:\n\n{"answer": "6.6%", "confidence": 1}', cost=0.01)]
	p, _, _ = asyncio.run(L.call('gpt6', 'sys', 'u', Ans, retries=1))
	assert p.answer == '6.6%'


def test_gemini_gets_its_schema_enforced(fake):
	fake.queue += [ok('{"answer": "a", "confidence": 1}', cost=0.0), ok('{"answer": "a", "confidence": 1}', cost=0.0)]
	asyncio.run(L.call('gemini', 'sys', 'u', Ans))
	asyncio.run(L.call('gpt6', 'sys', 'u', Ans))
	assert fake.posts[0]['json']['response_format']['json_schema']['name'] == 'Ans' and 'response_format' not in fake.posts[1]['json']


def test_a_hung_request_never_blocks_its_timeout(monkeypatch):
	import threading
	import time

	import httpx

	gate = threading.Event()

	class Hung:
		def __init__(self, timeout=None, limits=None):
			pass

		def post(self, *a, **k):
			gate.wait(5)  # the server never answers (and never finishes closing)
			raise RuntimeError('late')

	monkeypatch.setattr(httpx, 'Client', Hung)
	monkeypatch.setattr(L.OpenRouter, 'local', threading.local())
	monkeypatch.setenv('OPENROUTER_API_KEY', 'k')
	t0 = time.monotonic()
	with pytest.raises(asyncio.TimeoutError):
		asyncio.run(asyncio.wait_for(L.OpenRouter('openai/gpt-6.1-sol').ainvoke([L.UserMessage(content='q')]), 0.2))
	gate.set()
	assert time.monotonic() - t0 < 2


def test_book_bu_charges_what_the_balance_lost_beyond_agent_tokens(monkeypatch):
	monkeypatch.setattr(L, 'LEDGER', L.Ledger())
	L.LEDGER.add('bu-2-0', {'reported_cost': 1.0})
	monkeypatch.setattr(L, 'bu_balance', lambda: 96.5)
	L.book_bu(100.0)
	assert L.LEDGER.usd == pytest.approx(3.5) and L.LEDGER.by_model[L.BU_CLOUD]['reported_cost'] == pytest.approx(2.5)
	L.book_bu(None)  # no Browser Use key: nothing measured, nothing booked twice
	assert L.LEDGER.usd == pytest.approx(3.5)
