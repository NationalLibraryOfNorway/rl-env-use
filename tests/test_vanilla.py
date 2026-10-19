import asyncio
import json

import pytest

import rl_env_use.vanilla as V
from rl_env_use.graders import grounded


def test_claude_code_stream_becomes_a_gradeable_trace(monkeypatch):
	events = [
		{'type': 'assistant', 'message': {'content': [{'type': 'tool_use', 'id': 'a', 'name': 'WebSearch', 'input': {'query': 'PCSK9 uniprot'}}]}},
		{'type': 'user', 'message': {'content': [{'type': 'tool_result', 'tool_use_id': 'a', 'content': 'results...'}]}},
		{
			'type': 'assistant',
			'message': {'content': [{'type': 'tool_use', 'id': 'b', 'name': 'WebFetch', 'input': {'url': 'https://www.uniprot.org/uniprotkb/Q8NBP7/entry', 'prompt': 'domains'}}]},
		},
		{'type': 'user', 'message': {'content': [{'type': 'tool_result', 'tool_use_id': 'b', 'content': 'Domain 155-461 Peptidase S8'}]}},
		{'type': 'result', 'result': 'It is 155-461.\nANSWER: 155-461\nSOURCE: https://www.uniprot.org/uniprotkb/Q8NBP7/entry', 'total_cost_usd': 0.12},
	]

	class P:
		returncode = 0

		async def communicate(self, _):
			return '\n'.join(map(json.dumps, events)).encode(), b''

	async def fake(*a, **k):
		return P()

	monkeypatch.setattr(V.asyncio, 'create_subprocess_exec', fake)
	text, trace, usd = asyncio.run(V.claude_web('q'))
	assert usd == 0.12 and text.endswith('Q8NBP7/entry')
	assert [t['role'] for t in trace] == ['system', 'assistant', 'tool_call', 'tool_result', 'assistant', 'tool_call', 'tool_result']
	assert grounded([{'url': 'https://www.uniprot.org/uniprotkb/Q8NBP7/entry', 'value': '155-461'}], trace)
	assert not grounded([{'url': 'https://gtexportal.org/home/gene/SCN5A', 'value': '697'}], trace)
	events[-1] |= {'is_error': True, 'result': 'Claude usage limit reached'}
	with pytest.raises(RuntimeError, match='usage limit'):  # an infra stop goes to errors.jsonl, never scored as a wrong answer
		asyncio.run(V.claude_web('q'))


def test_last_json_takes_the_final_object():
	assert V.last_json('notes {"a": 1} more\n{"facts": [{"value": "5"}]}') == {'facts': [{'value': '5'}]}
	assert V.last_json('no json here') == {}


def test_reading_the_sources_own_api_counts_as_grounded():
	tr = [
		{'role': 'assistant', 'content': '[step 0 @ https://rest.uniprot.org/uniprotkb/Q8NBP7.txt]'},
		{'role': 'tool_call', 'name': 'extract', 'content': '{}'},
		{'role': 'tool_result', 'content': 'FT DOMAIN 155..461 /note="Peptidase S8"'},
	]
	assert grounded([{'url': 'https://www.uniprot.org/uniprotkb/Q8NBP7/entry', 'value': '461'}], tr)
	tr[0]['content'] = '[step 0 @ https://www.genecards.org/x]'
	assert not grounded([{'url': 'https://www.uniprot.org/uniprotkb/Q8NBP7/entry', 'value': '461'}], tr)
