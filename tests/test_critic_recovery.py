"""Partial critic persistence and bounded failures must survive interruption."""

import asyncio

import pytest
from conftest import mk_task

from rl_env_use import verify as V
from rl_env_use.pipeline import unverified
from rl_env_use.store import load_jsonl, save_jsonl


def test_completed_review_survives_cancellation_of_another(tmp_path, monkeypatch):
	tasks = [mk_task(0), mk_task(1)]
	path = tmp_path / 'tasks.jsonl'
	options = []

	async def fake(model, system, user, schema, **kw):
		options.append(kw)
		if 'Question 1' in user:
			await asyncio.sleep(10)
		if schema is V.Recompute:
			return schema(calculation='source states 137', recomputed_answer='137'), {}, model
		return schema(unambiguous=True, gold_supported=True, realistic=True, issues=''), {}, model

	monkeypatch.setattr(V, 'call', fake)

	async def run():
		worker = asyncio.create_task(V.critique(tasks, {}, checkpoint=lambda: save_jsonl(path, tasks)))
		for _ in range(100):
			await asyncio.sleep(0.001)
			if path.exists():
				break
		assert path.exists()
		worker.cancel()
		with pytest.raises(asyncio.CancelledError):
			await worker

	asyncio.run(run())
	rows = load_jsonl(path)
	assert rows[0]['checks']['critic']['agrees'] is True
	assert rows[0]['checks']['critic_model']
	# The second task retains its original verdict; its model stamp was not changed.
	assert 'critic_model' not in rows[1]['checks']
	assert all(k['timeout'] == 120 and k['retries'] == 2 and k['max_completion_tokens'] == 4096 for k in options)


def test_total_review_deadline_is_saved_as_retryable_error(tmp_path, monkeypatch):
	task = mk_task(0)
	path = tmp_path / 'tasks.jsonl'
	monkeypatch.setenv('RLENV_CRITIC_TASK_TIMEOUT', '0.01')

	async def slow(*args, **kw):
		await asyncio.sleep(10)

	monkeypatch.setattr(V, 'call', slow)
	asyncio.run(V.critique([task], {}, checkpoint=lambda: save_jsonl(path, [task])))
	row = load_jsonl(path)[0]
	assert 'total review deadline' in row['checks']['critic']['issues']
	assert unverified(row)
