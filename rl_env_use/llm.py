"""Model registry (routes: Browser Use gateway, your Claude account via `claude -p`, Anthropic API, OpenRouter), pricing, structured calls."""

import asyncio
import json
import os
import random
import shutil
import threading
import time
from dataclasses import dataclass, field
from types import SimpleNamespace

from browser_use import ChatAnthropic
from browser_use.llm.exceptions import ModelProviderError
from browser_use.llm.messages import SystemMessage, UserMessage
from browser_use.llm.views import ChatInvokeCompletion, ChatInvokeUsage
from pydantic import BaseModel, Field, ValidationError

from .config import load_dotenv as load_dotenv
from .config import open_only
from .language import language_prompt

KEYS = {
	'bu': 'BROWSER_USE_API_KEY',
	'anthropic': 'ANTHROPIC_API_KEY',
	'openrouter': 'OPENROUTER_API_KEY',
	'claude': None,
	'local': None,
}  # claude = local `claude -p` (your Claude account)
# alias -> candidate (route, model id), first route with a key wins. LLM calls go to OpenRouter first, so Browser Use
# credits pay only for browsers (and the bu models, which live only there); the gateway is the fallback route.
MODELS = {
	'bu': [('bu', 'bu-2-0')],
	'bu-mini': [('bu', 'bu-2-0-mini-preview')],
	'haiku': [('openrouter', 'anthropic/claude-haiku-4.5'), ('anthropic', 'claude-haiku-4-5'), ('bu', 'anthropic/claude-haiku-4-5')],
	'sonnet46': [('openrouter', 'anthropic/claude-sonnet-4.6'), ('anthropic', 'claude-sonnet-4-6'), ('bu', 'anthropic/claude-sonnet-4-6')],
	'sonnet5': [('openrouter', 'anthropic/claude-sonnet-5'), ('anthropic', 'claude-sonnet-5'), ('bu', 'anthropic/claude-sonnet-5')],
	'opus5': [('openrouter', 'anthropic/claude-opus-5'), ('anthropic', 'claude-opus-5'), ('bu', 'anthropic/claude-opus-5')],
	'gpt': [('openrouter', 'openai/gpt-5.5'), ('bu', 'openai/gpt-5.5')],
	# text-only structured calls through the local Claude Code CLI (`claude -p`, billed to your Claude account)
	'claude-sonnet': [('claude', 'sonnet')],
	'claude-opus': [('claude', 'opus')],
	'claude-haiku': [('claude', 'haiku')],
	# OpenRouter: other labs' frontier models (difficulty panel, adjudication) and open-weight policies to train
	'opus55': [('openrouter', 'anthropic/claude-opus-5.5')],
	'sonnet55': [('openrouter', 'anthropic/claude-sonnet-5.5')],
	'gpt-astra': [('openrouter', 'openai/gpt-6-astra')],
	'gpt6': [('openrouter', 'openai/gpt-6.1-sol')],
	'gemini': [('openrouter', 'google/gemini-3.1-pro-preview')],
	'gemini-flash': [('openrouter', 'google/gemini-3.8-flash')],
	'grok': [('openrouter', 'x-ai/grok-4.7')],
	'deepseek': [('openrouter', 'deepseek/deepseek-v4-pro')],
	'qwen-27b': [('openrouter', 'qwen/qwen3.8-27b')],
	'qwen-35b': [('openrouter', 'qwen/qwen3.6-35b-a3b')],
}
# Frontier closed models: they reason internally at high effort and return only the answer. They decline requests to
# write their reasoning out (a safeguard against copying their outputs), and the grade needs only the answer.
ANSWER_ONLY = {'opus55', 'sonnet55', 'gpt-astra'}
# $/MTok (input, output), matched by substring in order. Cache reads 0.1x input, writes 1.25x input.
# bu-2-0: fitted exactly to the gateway's reported cost over 9 mining runs (Oct 2026); bu-2-0-mini: unknown.
# OpenRouter calls carry their billed cost; these rows only rank models by strength (audit) and price old rows.
PRICES = [
	('bu-2-0-mini', None),
	('bu-2-0', (0.6, 3.5)),
	('sonnet-4', (3.0, 15.0)),
	('sonnet', (2.0, 10.0)),
	('haiku-4', (1.0, 5.0)),
	('haiku', (0.1, 0.5)),
	('opus-5.5', (4.0, 20.0)),
	('opus-5-5', (4.0, 20.0)),
	('opus', (5.0, 25.0)),
	('fable', (10.0, 50.0)),
	('gpt-5.5', (5.0, 30.0)),
	('gpt-6-astra', (10.0, 50.0)),
	('gpt-6', (2.0, 10.0)),
	('gemini-3.1-pro', (2.0, 12.0)),
	('gemini-3.8-flash', (0.75, 3.75)),
	('grok-4', (2.0, 6.0)),
	('deepseek-v4-pro', (0.21, 0.42)),
	('qwen3.8-27b', (0.43, 2.55)),
	('qwen3.6-35b', (0.15, 1.0)),
]


def routes(name: str) -> list[tuple[str, str]]:
	"""Every route of an alias that has a key, in preference order; an explicit 'route:model_id' is its only route."""
	if name in ('local-text', 'local-browser'):
		var = 'RLENV_LOCAL_BROWSER_MODEL' if name == 'local-browser' else 'RLENV_LOCAL_TEXT_MODEL'
		mid = os.getenv(var)
		if not mid:
			raise ValueError(f'{var} must name a model served by your local endpoint')
		return [('local', mid)]
	if open_only() and not name.startswith('local:'):
		raise ValueError(f'hosted model {name!r} is disabled by RLENV_OPEN_ONLY=1; use local-text, local-browser or local:MODEL_ID')
	if name == 'local:':
		raise ValueError('local: requires a model ID')
	if ':' in name and name.split(':', 1)[0] in KEYS:
		return [tuple(name.split(':', 1))]
	return [(r, m) for r, m in MODELS.get(name, []) if (shutil.which('claude') if r == 'claude' else os.getenv(KEYS[r]))]


def resolve(name: str) -> tuple[str, str]:
	"""alias ('haiku') or explicit 'route:model_id' ('bu:bu-2-0', 'anthropic:claude-opus-5-5')."""
	if rs := routes(name):
		return rs[0]
	raise RuntimeError(f'no API key for model {name!r} (need one of {[KEYS[r] for r, _ in MODELS.get(name, [])] or list(KEYS.values())})')


def first_json(txt: str):
	"""The first complete JSON object in a reply that may wrap it in prose, code fences or a repeated 'Final answer: {...}'."""
	dec = json.JSONDecoder()
	for i, ch in enumerate(txt):
		if ch == '{':
			try:
				return dec.raw_decode(txt, i)[0]
			except ValueError:
				continue
	raise ValueError(f'no JSON object in: {txt[:120]}')


def loads_lenient(txt: str):
	"""JSON as models and gateways actually send it: plain, double-escaped (\\" inside), or Python-literal style."""
	import ast

	for attempt in (lambda t: json.loads(t), lambda t: json.loads(json.loads(f'"{t}"')), ast.literal_eval):
		try:
			return attempt(txt)
		except (ValueError, SyntaxError):
			continue
	raise ValueError(f'unparseable structured field: {txt[:120]}')


class ClaudeCLI:
	"""Plain structured calls through `claude -p` (no tools, our own system prompt). For text-only stages (case
	building, critic, judges) - not for browser agents, which send screenshots every step."""

	sem = asyncio.Semaphore(int(os.getenv('RLENV_CLAUDE_CONCURRENCY', '4')))

	def __init__(self, model: str):
		self.model = f'claude-cli:{model}'
		self.cli_model = model

	async def ainvoke(self, messages, output_format=None, **kwargs):
		system = '\n'.join(m.content for m in messages if getattr(m, 'role', '') == 'system') or 'You are a precise assistant.'
		user = '\n\n'.join(str(m.content) for m in messages if getattr(m, 'role', '') != 'system')
		if output_format is not None:
			user += f'\n\nReply with ONLY a JSON object matching this JSON schema (no prose, no code fence):\n{json.dumps(output_format.model_json_schema())}'
		args = [
			'claude',
			'-p',
			'--model',
			self.cli_model,
			'--output-format',
			'json',
			'--system-prompt',
			system,
			'--disallowedTools',
			'Bash,Edit,Write,Read,WebFetch,WebSearch,Task,Glob,Grep',
		]
		async with self.sem:  # prompt on stdin: --disallowedTools is variadic and would swallow a positional prompt
			proc = await asyncio.create_subprocess_exec(*args, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, cwd='/tmp')
			try:
				out, err = await asyncio.wait_for(proc.communicate(user.encode()), 300)
			except (TimeoutError, asyncio.CancelledError):
				proc.kill()
				raise
		w = json.loads(out or b'{}')
		if w.get('is_error') or 'result' not in w:
			raise ModelProviderError(message=f'claude -p failed: {(w.get("result") or err.decode())[:200]}', status_code=500, model=self.model)
		u = w.get('usage', {})
		pt = u.get('input_tokens', 0) + u.get('cache_read_input_tokens', 0) + u.get('cache_creation_input_tokens', 0)
		usage = ChatInvokeUsage(
			prompt_tokens=pt,
			prompt_cached_tokens=u.get('cache_read_input_tokens', 0),
			prompt_cache_creation_tokens=u.get('cache_creation_input_tokens'),
			prompt_image_tokens=None,
			completion_tokens=u.get('output_tokens', 0),
			total_tokens=pt + u.get('output_tokens', 0),
		)
		if output_format is None:
			return ChatInvokeCompletion(completion=w['result'], usage=usage)
		txt = w['result'].strip()
		txt = txt[txt.find('{') : txt.rfind('}') + 1] if '{' in txt else txt
		return ChatInvokeCompletion(completion=output_format.model_validate(loads_lenient(txt)), usage=usage)


class GatewayStructured:
	"""Plain structured calls on the Browser Use gateway (billed to the Browser Use account). The gateway serves
	agent-step outputs only, so the schema travels as the single custom action of a browser-use agent step and is
	unwrapped again; no browser is involved."""

	_types: dict = {}

	def __init__(self, model: str):
		from browser_use import ChatBrowserUse

		self.llm, self.model = ChatBrowserUse(model=model), model

	@staticmethod
	def nested_fields(schema: type[BaseModel]) -> dict:
		"""Fields holding models (directly or in lists/dicts): the gateway flattens nested objects, so they travel as JSON strings."""
		import typing

		out = {}
		for name, f in schema.model_fields.items():
			ann = f.annotation
			args = [a for a in typing.get_args(ann) if a is not type(None)]
			base = args[0] if typing.get_origin(ann) in (typing.Union, type(int | None)) and args else ann
			inner = (typing.get_args(base) or [base])[-1]  # list[X] / dict[str, X] -> X
			if isinstance(inner, type) and issubclass(inner, BaseModel):  # primitive lists travel fine; nested objects get flattened
				out[name] = ann
		return out

	@classmethod
	def wire_schema(cls, schema: type[BaseModel]) -> type[BaseModel]:
		from pydantic import TypeAdapter, create_model

		nested = cls.nested_fields(schema)
		if not nested:
			return schema
		fields = {}
		for name, f in schema.model_fields.items():
			if name in nested:
				js = json.dumps(TypeAdapter(nested[name]).json_schema(), separators=(',', ':'))
				fields[name] = (str, Field(description=f'{f.description or name} - a JSON value matching this JSON schema: {js}'))
			else:
				fields[name] = (f.annotation, f)
		return create_model(schema.__name__ + 'Wire', **fields)

	@classmethod
	def step_type(cls, schema: type[BaseModel]):
		schema = cls.wire_schema(schema)
		if schema not in cls._types:
			from browser_use import Tools
			from browser_use.agent.views import ActionResult, AgentOutput

			tools = Tools(exclude_actions=list(Tools().registry.registry.actions))

			@tools.registry.action('Submit your complete answer in this structure (call exactly once).', param_model=schema)
			async def submit(params):  # never executed: only its schema is used
				return ActionResult(is_done=True)

			cls._types[schema] = AgentOutput.type_with_custom_actions(tools.registry.create_action_model())
		return cls._types[schema]

	async def ainvoke(self, messages, output_format=None, **kwargs):
		if output_format is None:
			return await self.llm.ainvoke(messages, None, **kwargs)
		msgs = (
			[*messages[:-1], messages[-1].model_copy(update={'content': f'{messages[-1].content}\n\nRespond by calling the submit action exactly once.'})] if messages else messages
		)
		r = await self.llm.ainvoke(msgs, output_format=self.step_type(output_format), **kwargs)
		payload = next((a.model_dump(exclude_none=True).get('submit') for a in r.completion.action if a.model_dump(exclude_none=True).get('submit')), None)
		if payload is None:
			raise ModelProviderError(message='gateway response had no submit action', status_code=500, model=self.model)
		for name in self.nested_fields(output_format):
			if isinstance(payload.get(name), str):
				txt = payload[name].strip()
				txt = txt.split('\n', 1)[1].rsplit('```', 1)[0] if txt.startswith('```') else txt
				payload[name] = loads_lenient(txt)
		return ChatInvokeCompletion(completion=output_format.model_validate(payload), usage=r.usage)


class OpenRouter:
	"""Plain structured calls on OpenRouter: the schema travels in the prompt (not every provider enforces
	response_format), the reply is parsed leniently, and the billed cost comes back with the usage."""

	sem = asyncio.Semaphore(int(os.getenv('RLENV_OPENROUTER_CONCURRENCY', '16')))
	pool = None  # requests run in threads: cancelling an awaited httpx request closes its TLS stream in a shielded section,
	local = threading.local()  # and a peer that never finishes the close froze whole climbs (every slot, every timeout)

	@classmethod
	def post(cls, body: dict):
		import httpx

		c = getattr(cls.local, 'client', None)
		if c is None or getattr(c, 'is_closed', False) or type(c) is not httpx.Client:
			c = cls.local.client = httpx.Client(timeout=httpx.Timeout(CALL_TIMEOUT, connect=30))
		return c.post('https://openrouter.ai/api/v1/chat/completions', headers={'Authorization': f'Bearer {os.environ[KEYS["openrouter"]]}'}, json=body)

	def __init__(self, model: str, temperature: float | None = None, effort: str | None = None):
		self.model, self.temperature, self.effort = model, temperature, effort

	async def ainvoke(self, messages, output_format=None, **kwargs):
		msgs = [{'role': getattr(m, 'role', 'user'), 'content': str(m.content)} for m in messages]
		if output_format is not None:
			msgs[-1]['content'] += f'\n\nReply with ONLY a JSON object matching this JSON schema (no prose, no code fence):\n{json.dumps(output_format.model_json_schema())}'
		body = {'model': self.model, 'messages': msgs, 'usage': {'include': True}, 'max_tokens': 32000} | (
			{'temperature': self.temperature} if self.temperature is not None and not self.effort else {}
		)
		body |= {'reasoning': {'effort': self.effort, 'exclude': True}} if self.effort else {}
		if output_format is not None and self.model.startswith('google/'):  # Gemini sometimes echoes the schema instead of filling it
			body['response_format'] = {'type': 'json_schema', 'json_schema': {'name': output_format.__name__, 'strict': False, 'schema': output_format.model_json_schema()}}
		if OpenRouter.pool is None:
			from concurrent.futures import ThreadPoolExecutor

			OpenRouter.pool = ThreadPoolExecutor(64, thread_name_prefix='openrouter')
		async with self.sem:  # a timeout or cancellation returns at once; the thread finishes or times out on its own
			r = await asyncio.get_running_loop().run_in_executor(OpenRouter.pool, OpenRouter.post, body)
		w = r.json() if r.headers.get('content-type', '').startswith('application/json') else {'error': {'message': r.text[:300]}}
		if r.status_code != 200 or w.get('error') or not w.get('choices'):
			raise ModelProviderError(message=f'openrouter {r.status_code}: {str(w.get("error") or w)[:300]}', status_code=r.status_code, model=self.model)
		u = w.get('usage') or {}
		usage = {
			'input_tokens': u.get('prompt_tokens', 0),
			'output_tokens': u.get('completion_tokens', 0),
			'cache_read_input_tokens': (u.get('prompt_tokens_details') or {}).get('cached_tokens', 0) or 0,
			'reported_cost': u.get('cost') or 0,
		}
		txt = (w['choices'][0]['message'].get('content') or '').strip()
		if w['choices'][0].get('finish_reason') == 'content_filter' or (not txt and w['choices'][0].get('native_finish_reason') == 'refusal'):
			raise ModelProviderError(message='stop_reason refusal: declined to respond', status_code=400, model=self.model)
		if output_format is not None:
			try:
				try:
					obj = loads_lenient(txt[txt.find('{') : txt.rfind('}') + 1] if '{' in txt else txt)
				except ValueError:
					obj = first_json(txt)  # JSON followed by more text, e.g. a repeated final answer
				txt = output_format.model_validate(obj)
			except Exception:
				LEDGER.add(self.model, usage)  # an unparseable reply is still billed; call() books only the one it returns
				raise
		return SimpleNamespace(completion=txt, usage=usage)  # usage: a plain dict, so the billed cost survives


def make_llm(name: str, temperature: float | None = None, agent: bool = True, fallback: bool = False):
	"""agent=True: an LLM for browser-use agents. agent=False: plain structured calls (Browser Use gateway via
	GatewayStructured when a Browser Use key is set, or `claude -p` for claude-* models). fallback: the alias's next route."""
	rs = routes(name) or [resolve(name)]
	route, mid = rs[min(int(fallback), len(rs) - 1)]
	if route == 'local':
		from .local import LocalChat

		return LocalChat(mid, temperature, browser=agent and name == 'local-browser')
	if route == 'claude':
		if agent:
			raise ValueError('claude -p models serve text-only structured calls, not browser agents (screenshots every step)')
		return ClaudeCLI(mid)
	if route == 'bu':
		if not agent:
			return GatewayStructured(mid)
		from browser_use import ChatBrowserUse

		return ChatBrowserUse(model=mid)
	if route == 'openrouter':
		if not agent:
			return OpenRouter(mid, temperature, 'high' if name in ANSWER_ONLY else None)
		from browser_use.llm.openrouter.chat import ChatOpenRouter

		return ChatOpenRouter(model=mid, api_key=os.environ[KEYS['openrouter']])
	return ChatAnthropic(model=mid, temperature=temperature)


def price_of(model: str) -> tuple[float, float] | None:
	m = (model or '').lower()
	return next((p for k, p in PRICES if k in m), None) if m else None


def cost(model: str, usage: dict) -> float | None:
	if (model or '').startswith('local:'):
		return None  # hardware/electricity costs are not API token prices
	if (usage or {}).get('reported_cost'):
		return usage['reported_cost']
	p = price_of(model)
	if not p or not usage:
		return None
	i, o = p
	read = usage.get('cache_read_input_tokens', 0) or 0
	write = usage.get('cache_creation_input_tokens', 0) or 0
	fresh = (usage.get('input_tokens', 0) or 0) - read - write
	return (fresh * i + read * 0.1 * i + write * 1.25 * i + (usage.get('output_tokens', 0) or 0) * o) / 1e6


def usage_dict(u) -> dict:
	if u is None:
		return {}
	if isinstance(u, dict):  # OpenRouter: already ours, with the billed cost
		return u
	if hasattr(u, 'total_prompt_tokens'):  # browser_use UsageSummary (cost filled by the gateway for bu-* models)
		d = {'input_tokens': u.total_prompt_tokens, 'output_tokens': u.total_completion_tokens, 'cache_read_input_tokens': u.total_prompt_cached_tokens}
		return d | ({'reported_cost': u.total_cost} if u.total_cost else {})
	return {'input_tokens': u.prompt_tokens, 'output_tokens': u.completion_tokens, 'cache_read_input_tokens': u.prompt_cached_tokens or 0}


def add_usage(a: dict, b: dict) -> dict:
	return {k: (a.get(k, 0) or 0) + (b.get(k, 0) or 0) for k in set(a) | set(b)}


@dataclass
class Ledger:
	"""Running spend across every call in a command."""

	by_model: dict = field(default_factory=dict)

	def add(self, model: str, usage: dict):
		if usage and not usage.get('reported_cost'):  # priced per call: a few billed replies must not hide the unbilled ones' tokens in the sum
			usage = usage | {'reported_cost': cost(model, usage) or 0}
		self.by_model[model] = add_usage(self.by_model.get(model, {}), usage)

	@property
	def usd(self) -> float:
		return sum(cost(m, u) or 0 for m, u in list(self.by_model.items()))  # read from the board's thread while calls add to it


LEDGER = Ledger()
BU_CLOUD = 'browser-use cloud (balance)'  # what the account was charged beyond the agents' tokens: cloud browsers, proxies


def bu_balance() -> float | None:
	"""The Browser Use account's credit balance: the ground truth for what mining cost."""
	if open_only() or not os.getenv(KEYS['bu']):
		return None
	try:
		import httpx

		r = httpx.get('https://api.browser-use.com/api/v2/billing/account', headers={'X-Browser-Use-API-Key': os.environ[KEYS['bu']]}, timeout=10)
		return float(r.json()['totalCreditsBalanceUsd'])
	except Exception:
		return None


def book_bu(start: float | None) -> None:
	"""Charge LEDGER with the Browser Use spend its token counts miss (cloud browser time), measured on the balance."""
	now = bu_balance() if start is not None else None
	if now is not None:
		booked = sum(cost(m, u) or 0 for m, u in list(LEDGER.by_model.items()) if m.startswith('bu-'))
		LEDGER.by_model[BU_CLOUD] = {'reported_cost': max(0.0, start - now - booked)}


CALL_TIMEOUT = 600  # a hung gateway connection otherwise stalls a whole stage (long structured outputs take minutes)


async def call(name: str, system: str, user: str, schema: type[BaseModel], retries: int = 4, temperature: float | None = 0.0, audience: str = 'diagnostics'):
	"""One structured LLM call. Returns (parsed, usage, model). Jittered backoff on transient errors; an invalid output
	is fed back to the model on the next attempt (e.g. an enum value outside the allowed set)."""
	system += language_prompt(audience)
	feedback = ''
	for attempt in range(retries):
		# a deterministic (temperature 0) failure would repeat identically: later attempts resample
		# the last attempts go to the alias's next route (the same model elsewhere): one gateway's outage is not the model's
		llm = make_llm(name, temperature if attempt == 0 or temperature is None else 0.7, agent=False, fallback=attempt >= retries - 2)  # None: the model's own sampling, kept
		try:
			r = await asyncio.wait_for(llm.ainvoke([SystemMessage(content=system), UserMessage(content=user + feedback)], output_format=schema), CALL_TIMEOUT)
			u = usage_dict(r.usage)
			LEDGER.add(llm.model, u)
			return r.completion, u, llm.model
		except Exception as e:
			if attempt == retries - 1:
				raise
			if isinstance(e, ValidationError) or 'validation error' in str(e):
				feedback = f'\n\nYour previous output was invalid - fix this and follow the schema exactly:\n{str(e)[:600]}'
			in_flight = 'in_flight' in str(e) or 'Retry-After' in str(e)  # OpenRouter credit-based concurrency cap: wait it out
			await asyncio.sleep((60 if in_flight else min(30, 2**attempt)) * (0.5 + random.random()))


def now() -> str:
	return time.strftime('%Y-%m-%dT%H:%M:%S')
