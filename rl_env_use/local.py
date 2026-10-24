"""Self-hosted OpenAI-compatible transport shared by browser agents and structured calls."""

import json
import os
import re
from copy import copy
from dataclasses import dataclass
from urllib.parse import urlparse

from browser_use import ChatOpenAI
from browser_use.llm.exceptions import ModelProviderError

from .reasoning import REASONING_RECORDS, record_reasoning


@dataclass
class TemplateChat(ChatOpenAI):
	"""Keep browser-use serialization while forwarding an explicit server template option."""

	enable_thinking: bool | None = None

	def get_client(self):
		client = super().get_client()
		if self.enable_thinking is not None or REASONING_RECORDS.get() is not None:
			create = client.chat.completions.create

			async def configured_create(**kwargs):
				extra = dict(kwargs.pop('extra_body', None) or {})
				if self.enable_thinking is not None:
					extra['chat_template_kwargs'] = dict(extra.get('chat_template_kwargs') or {}) | {'enable_thinking': self.enable_thinking}
				response = await create(**kwargs, extra_body=extra)
				record_reasoning(response)
				return response

			client.chat.completions.create = configured_create
		return client


def local_model_config(name: str) -> dict:
	"""Optional named local routes; keep aliases distinct even if servers share model IDs."""
	try:
		models = json.loads(os.getenv('RLENV_LOCAL_MODELS', '{}'))
	except json.JSONDecodeError:
		raise ValueError('RLENV_LOCAL_MODELS must be a JSON object of local model configurations') from None
	if not isinstance(models, dict):
		raise ValueError('RLENV_LOCAL_MODELS must be a JSON object of local model configurations')
	config = models.get(name, {})
	if not isinstance(config, dict) or set(config) - {'model', 'base_url', 'api_key_env', 'enable_thinking'}:
		raise ValueError('local model configuration must contain only model, base_url, api_key_env and enable_thinking')
	if 'enable_thinking' in config and not isinstance(config['enable_thinking'], bool):
		raise ValueError('local model enable_thinking must be a boolean')
	if any(not isinstance(value, str) or not value.strip() for key, value in config.items() if key != 'enable_thinking'):
		raise ValueError('local model configuration values must be nonempty strings')
	return config


class LocalChat:
	"""Expose a qualified model ID for accounting; send the raw ID to the server.

	Both paths use browser-use's serializer, retaining screenshot/image content and
	Pydantic schema validation. There is no fallback to a hosted provider.
	"""

	def __init__(self, model: str, temperature: float | None = None, browser: bool = False):
		config = local_model_config(model)
		base = config.get('base_url') or (os.getenv('RLENV_LOCAL_BROWSER_BASE_URL') if browser else None)
		base = base or os.getenv('RLENV_LOCAL_BASE_URL', 'http://127.0.0.1:8000/v1')
		url = urlparse(base)
		if url.scheme not in ('http', 'https') or not url.hostname or url.username or url.password or url.query or url.fragment:
			raise ValueError('local base URL must be an HTTP(S) endpoint without embedded credentials, query or fragment')
		self.model = f'local:{model}'
		key = os.getenv('RLENV_LOCAL_API_KEY') or 'local-no-key'
		if key_env := config.get('api_key_env'):
			key = os.getenv(key_env)
			if not key:
				raise ValueError('local model api_key_env names an unset or empty environment variable')
		self._chat = TemplateChat(
			model=config.get('model', model),
			base_url=base.rstrip('/'),
			api_key=key,
			temperature=temperature,
			frequency_penalty=None,
			max_completion_tokens=int(os.getenv('RLENV_LOCAL_MAX_TOKENS', '262144')),
			timeout=float(os.getenv('RLENV_LOCAL_TIMEOUT', '1200')),
			max_retries=0,  # call()/the browser agent own retries; never change providers
			enable_thinking=config.get('enable_thinking'),
		)

	@property
	def model_name(self) -> str:
		return self.model

	@property
	def provider(self) -> str:
		return 'local'

	@property
	def name(self) -> str:
		return self.model

	async def ainvoke(self, messages, output_format=None, **kwargs):
		try:
			return await self._chat.ainvoke(messages, output_format=output_format, **kwargs)
		except ModelProviderError as error:
			# Some servers enforce input + requested output <= context, even for short answers.
			# Use their exact input count; never truncate evidence or switch endpoints.
			limit = re.search(r'maximum context length of (\d+) tokens', str(error), re.I)
			inputs = re.search(r'(\d+) tokens from the input messages', str(error), re.I)
			if error.status_code != 400 or not limit or not inputs:
				raise
			remaining = int(limit[1]) - int(inputs[1])
			if not 0 < remaining < self._chat.max_completion_tokens:
				raise
			# A request-local copy avoids changing the configured ceiling or concurrent calls.
			chat = copy(self._chat)
			chat.max_completion_tokens = remaining
			return await chat.ainvoke(messages, output_format=output_format, **kwargs)
