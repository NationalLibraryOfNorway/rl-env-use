"""Request-scoped capture of model-provided reasoning, separate from final structured output."""

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime

REASONING_RECORDS: ContextVar[list | None] = ContextVar('local_reasoning_records', default=None)


@contextmanager
def capture_reasoning():
	records = []
	token = REASONING_RECORDS.set(records)
	try:
		yield records
	finally:
		REASONING_RECORDS.reset(token)


def record_reasoning(response):
	records = REASONING_RECORDS.get()
	if records is None:
		return
	for choice in response.choices:
		message = choice.message
		text, field = None, None
		for name in ('reasoning_content', 'reasoning'):
			value = getattr(message, name, None)
			if isinstance(value, str) and value.strip():
				text, field = value, name
				break
		records.append(
			{
				'reasoning': text,
				'field': field,
				'model': response.model,
				'response_id': response.id,
				'finish_reason': choice.finish_reason,
				'captured_at': datetime.now(UTC).isoformat(),
			}
		)
