"""Stage 3: run the policy over tasks and write results in the claude-api hillclimb contract. Case tasks: the model
reads the materials and answers (one structured call, no tools). Browse tasks: a browser agent on the live web.

<flow>/<variant>/results.jsonl     one row per (case, rep), appended as each completes (resume-safe)
<flow>/<variant>/traces/<id>_rep<k>.json
<flow>/<variant>/errors.jsonl      infra failures (timeouts, crashes) - never scored as model failures
"""

import asyncio
import json
import shutil
import time
from pathlib import Path
from urllib.parse import urlparse

from browser_use import Agent
from pydantic import BaseModel, Field

from .browse import browser
from .graders import grade_answer, grounded, max_repeats
from .llm import LEDGER, add_usage, call, cost, make_llm, now, usage_dict
from .runner import run_eval

LOOP = 5  # identical tool calls at which a trajectory counts as looping

POLICY_SUFFIX = '\n\nReturn just the answer (value, name, date or comma-separated list) in `answer` - no explanation - and the page you found it on in `source_url`.'


CASE_SYS = 'You are a domain expert. Read the materials and answer the question. Reason it through, then commit to one answer.'
CASE_SUFFIX = '\n\nPut your work in `reasoning` and just the final answer (value, name, date or comma-separated list) in `answer`, no explanation.'
ANSWER_SUFFIX = '\n\nThink it through, then give just the final answer (value, name, date or comma-separated list) in `answer`, no explanation.'


class Reasoned(BaseModel):
	reasoning: str = Field(description='Your work: what in the materials you used and how you reached the answer.')
	answer: str = Field(description='Just the final answer, no explanation.')


class Answer(BaseModel):
	answer: str = Field(description='Just the final answer, no explanation.')
	source_url: str = ''


def trace_grade(t: dict, correct: bool, trace: list[dict]) -> dict:
	"""Reward + process metrics. `supported` = right answer AND the agent opened a source page: the reward to train on,
	since correctness alone rewards copying search snippets (found in error analysis)."""
	g = grounded(t.get('evidence') or [{'url': u} for u in t.get('source_urls', [])], trace)
	return {'correct': float(correct), 'supported': float(correct and g), 'grounded': float(g), 'no_loop': float(max_repeats(trace) < LOOP)}


def public(t: dict) -> dict:
	"""The only task fields a policy ever sees. Gold, evidence and checks stay out of reach."""
	if t.get('context'):
		return {'id': t['id'], 'prompt': t['prompt'], 'context': t['context']}
	return {'id': t['id'], 'prompt': t['prompt'], 'start_url': t['start_url'], 'allowed_domains': t.get('allowed_domains')}


def case_messages(pub: dict, extend_system: str | None = None, answer_only: bool = False) -> tuple[str, str]:
	return CASE_SYS + (f'\n\n{extend_system}' if extend_system else ''), f'MATERIALS:\n{pub["context"]}\n\nQUESTION: {pub["prompt"]}{ANSWER_SUFFIX if answer_only else CASE_SUFFIX}'


async def solve_case(pub: dict, model: str, extend_system: str | None = None):
	"""One policy answer on a case task: (system, user, answer, written reasoning, usage, model id). Frontier closed models
	answer only (their reasoning stays internal); the rest write their work, which the hillclimb's error analysis reads."""
	from .llm import ANSWER_ONLY

	quiet = model in ANSWER_ONLY
	system, user = case_messages(pub, extend_system, quiet)
	r, usage, mid = await call(model, system, user, Answer if quiet else Reasoned, temperature=None)
	return system, user, r.answer, getattr(r, 'reasoning', ''), usage, mid


async def run_case_task(t: dict, rep: int, model: str, judges=None, extend_system: str | None = None) -> tuple[list, dict]:
	t0 = time.monotonic()
	system, user, answer, reasoning, usage, mid = await solve_case(public(t), model, extend_system)
	r = Reasoned(reasoning=reasoning, answer=answer)
	correct = grade_answer(t, r.answer)
	trace = [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}, {'role': 'assistant', 'content': f'FINAL ANSWER: {r.answer}', 'thinking': r.reasoning}]
	grade, expl, judge_usage, judge_model = {'correct': float(correct)}, {}, {}, None
	for j in judges or []:
		v = await j.run(t, trace, r.answer)
		grade[j.id], expl[j.id], judge_model = v['score'], v['reasoning'], v['model']
		judge_usage = add_usage(judge_usage, v['usage'])
	return trace, {
		'prompt_id': t['id'],
		'rep': rep,
		'prompt': t['prompt'],
		'tags': [t.get('category', 'task'), t.get('kind', '?'), t.get('difficulty', '?')],
		'grade': grade,
		**({'explanation': expl} if expl else {}),
		'status': 'ok',
		'stop_reason': 'done',
		'model': mid,
		'latency_s': round(time.monotonic() - t0, 2),
		'tool_calls': 0,
		'steps': 1,
		'usage': usage,
		'cost_usd': round(cost(mid, usage) or 0, 5),
		**({'judge_model': judge_model, 'judge_usage': judge_usage} if judge_usage else {}),
		'meta': {'answer': r.answer, 'gold': t['answer'], 'failure_class': None if correct else 'wrong_answer', 'case': t.get('case'), 'at': now()},
	}


def history_to_trace(h, system: str) -> list[dict]:
	"""Text trace of an episode (no screenshots kept: they cost ~1 MB per episode and the text says what happened)."""
	turns = [{'role': 'system', 'content': system}]
	for i, step in enumerate(h.history):
		mo = step.model_output
		if mo:
			text = '\n'.join(f'{k}: {v}' for k, v in [('eval', mo.evaluation_previous_goal), ('memory', mo.memory), ('next_goal', mo.next_goal)] if v)
			turns.append({'role': 'assistant', 'content': f'[step {i} @ {step.state.url}]\n{text}', 'thinking': mo.thinking or None})
			for a in mo.action:
				for name, params in a.model_dump(exclude_none=True).items():
					turns.append({'role': 'tool_call', 'name': name, 'content': json.dumps(params, indent=1, default=str)})
		for r in step.result:
			turns.append({'role': 'tool_result', 'content': r.error or r.extracted_content or ('done' if r.is_done else '')})
	return turns


async def run_case(t: dict, rep: int, model: str, vdir: Path, max_steps: int, headless: bool, judges=None, extend_system: str | None = None) -> tuple[list, dict]:
	pub = public(t)
	llm = make_llm(model)
	task = pub['prompt'] + POLICY_SUFFIX
	agent = Agent(
		task=task,
		llm=llm,
		browser_session=browser(pub['allowed_domains'], headless),
		output_model_schema=Answer,
		use_judge=False,
		enable_signal_handler=False,
		calculate_cost=True,
		extend_system_message=extend_system,
		initial_actions=[{'navigate': {'url': pub['start_url']}}],
	)
	t0 = time.monotonic()
	try:
		h = await agent.run(max_steps=max_steps)
	finally:  # a cancelled episode (wall-clock ceiling) must still release its cloud browser
		try:
			await asyncio.wait_for(agent.close(), 30)
		except (Exception, asyncio.CancelledError):
			pass
	latency = time.monotonic() - t0
	errs = [e for e in h.errors() if e]
	if not h.is_done() and h.number_of_steps() < max_steps and errs and h.errors()[-1]:  # browser-use stops after repeated LLM errors without raising: infra, not a miss
		shutil.rmtree(agent.agent_directory, ignore_errors=True)
		raise RuntimeError(f'agent stopped on errors: {errs[-1][:300]}')
	try:
		out = h.structured_output if h.is_done() else None
	except ValueError:  # done with free text instead of the schema: no scorable answer
		out = None
	answer = out.answer if out else None
	correct = grade_answer(t, answer)
	trace = history_to_trace(h, task)
	shutil.rmtree(agent.agent_directory, ignore_errors=True)  # browser-use's per-agent temp dir (~10 MB of screenshots)
	trace.append({'role': 'assistant', 'content': f'FINAL ANSWER: {answer}' + (f'\nsource: {out.source_url}' if out and out.source_url else '')})
	usage = usage_dict(h.usage)
	LEDGER.add(llm.model, usage)
	grade, expl, judge_usage, judge_model = trace_grade(t, correct, trace), {}, {}, None
	for j in judges or []:
		v = await j.run(t, trace, answer)
		grade[j.id], expl[j.id], judge_model = v['score'], v['reasoning'], v['model']
		judge_usage = add_usage(judge_usage, v['usage'])
	fc = None if correct else ('no_answer' if answer is None else ('gave_up' if h.is_successful() is False else 'wrong_answer'))
	domains = list(dict.fromkeys(urlparse(u).netloc for u in h.urls() if u))
	return trace, {
		'prompt_id': t['id'],
		'rep': rep,
		'prompt': t['prompt'],
		'tags': [t.get('category', 'task'), t.get('difficulty', '?')],
		'grade': grade,
		**({'explanation': expl} if expl else {}),
		'status': 'ok',
		'stop_reason': 'done' if h.is_done() else 'max_steps',
		'model': llm.model,
		'latency_s': round(latency, 2),
		'tool_calls': sum(len(s.model_output.action) for s in h.history if s.model_output),
		'steps': h.number_of_steps(),
		'usage': usage,
		'cost_usd': round(cost(llm.model, usage) or 0, 5),
		**({'judge_model': judge_model, 'judge_usage': judge_usage} if judge_usage else {}),
		'meta': {
			'answer': answer,
			'gold': t['answer'],
			'failure_class': fc,
			'agent_claimed_success': h.is_successful(),
			'domains': domains,
			'errors': [e for e in h.errors() if e][:5],
			'at': now(),
		},
	}


async def rollout(
	tasks: list[dict],
	flow: Path,
	variant: str = 'baseline',
	model: str = 'bu',
	reps: int = 1,
	max_steps: int = 25,
	timeout_s: float = 600,
	concurrency: int = 4,
	headless: bool = True,
	judges=None,
	extend_system: str | None = None,
	first: int = 0,
) -> Path:
	"""Reps [first, reps) of every task; done (task, rep) pairs are skipped, so a rerun resumes."""

	async def one(t, rep, vdir):
		if t.get('context'):
			return await run_case_task(t, rep, model, judges, extend_system)
		return await run_case(t, rep, model, vdir, max_steps, headless, judges, extend_system)

	return await run_eval(tasks, one, flow, variant, reps, timeout_s, concurrency, first)
