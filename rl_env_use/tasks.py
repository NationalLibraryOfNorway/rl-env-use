"""Stage 2: writing tasks from the mined cases, and which of them an env serves.

Two kinds of task. Case tasks (spec has a `skill`): the model under test is handed a case's materials as `context` and
must derive or predict its true outcome - no browser; the browser agents' job was finding and pairing the data.
Browse tasks: the model under test must find the answer on the live web from `start_url`.
Checks that need no model live in checks.py; checks that ask models (critic, closed book, panel) in verify.py.
"""

import hashlib
import math
import re
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from . import ui
from .checks import code_gold, derived, gold_issues, grade_answer, leaks, rounding_tol, structural_issues
from .config import TEXT_MODEL
from .graders import norm
from .language import PROBABILITY, current
from .llm import call, now
from .store import load_jsonl

AnswerType = Literal['exact', 'number', 'date', 'set', 'contains', 'prob']


class TaskDraft(BaseModel):
	prompt: str = Field(description='What a real user would type. Self-contained, unambiguous, never contains the answer.')
	start_url: str = Field(description='Homepage or section root of the source site - NOT the page containing the answer.')
	answer: str
	answer_type: AnswerType
	aliases: list[str] = Field(default_factory=list, description='Other acceptable surface forms of the answer.')
	choices: list[str] = Field(default_factory=list, description='Which-of questions only: every candidate named in the prompt (the gold is one of them).')
	rel_tol: float = 0.0
	fact_ids: list[str]
	derivation: str = Field('', description='Empty if the answer is literally in a cited quote, else how it is computed from the cited facts.')
	category: str = Field(description='Short snake_case task family, e.g. lookup, multi_hop, comparison, list, temporal.')
	difficulty: Literal['easy', 'medium', 'hard']
	why_hard: str = Field(description='Articulate in one sentence what makes this hard for a browser agent BEFORE anyone runs it.')


class Drafts(BaseModel):
	tasks: list[TaskDraft]


class CaseTaskDraft(BaseModel):
	prompt: str = Field(description='The question the model answers after reading the materials. Self-contained given them; never contains the answer.')
	kind: Literal['derive', 'predict'] = Field(
		description='derive: an expert can reach the answer from the materials alone; predict: a yes/no question about a later outcome the materials make forecastable but do not state (answer_type prob).'
	)
	material_ids: list[str] = Field(description='The materials the model is given (ids from MATERIALS).')
	answer: str
	answer_type: AnswerType
	aliases: list[str] = Field(default_factory=list, description='Other acceptable surface forms of the answer.')
	choices: list[str] = Field(default_factory=list, description='Which-of questions only: every candidate named in the prompt (the gold is one of them).')
	rel_tol: float = 0.0
	fact_ids: list[str] = Field(description='The outcome facts the gold comes from.')
	derivation: str = Field('', description='Empty if the answer is literally in a cited outcome quote, else how it follows from them.')
	expression: str = Field(
		'',
		description='derive tasks with a numeric answer: the whole computation as ONE arithmetic expression over the numbers exactly as written in the materials (digits, + - * / ** and parentheses only), e.g. 10875 / (10875 + 6939) * 100. Code evaluates it: the gold must be its value rounded as the prompt asks.',
	)
	distractors: list[str] = Field(
		default_factory=list,
		description='derive tasks with a numeric answer: 2-3 expressions for the typical expert mistakes (wrong period, year-to-date not de-cumulated, wrong sign, wrong scale, wrong denominator), same syntax as expression. The grader must reject each.',
	)
	category: str = Field(description='Short snake_case task family in the domain, e.g. ratio_analysis, endpoint_outcome, final_diagnosis.')
	difficulty: Literal['easy', 'medium', 'hard']
	why_hard: str = Field(description='One sentence: the domain expertise or reasoning the task demands.')


class CaseDrafts(BaseModel):
	tasks: list[CaseTaskDraft]


class ForecastDraft(CaseTaskDraft):
	"""The policy predicts a probability; the private gold is a resolved binary event."""
	kind: Literal['predict'] = 'predict'
	answer_type: Literal['prob'] = 'prob'
	answer: Literal['yes', 'no'] = Field(description='Private outcome label from evidence; never a forecast probability.')


class ForecastDrafts(BaseModel):
	tasks: list[ForecastDraft]


SYNTH_SYS = """You turn scouted web evidence into tasks for a browser-agent RL environment.
Rules:
- Every task's answer MUST come from the cited facts' quotes (copy the value exactly), or be a simple derivation from them (state it in `derivation`).
- The prompt must pin down exactly one correct answer (name the entity, version, year, unit). Never put the answer in the prompt.
- start_url is where a user would naturally begin (site root/section), never the answer page. The agent must navigate.
- State the problem plainly - difficulty must come from the web task, not obscure wording.
- Exactly ONE question with ONE answer per task - never 'X, and also Y?'. Combine facts only through a single derived value (a difference, a sum, which-is-larger).
- Which-of questions: list every candidate in `choices` and use answer_type exact.
- Any numeric answer (counts, percentages, sizes, '1.18 million') MUST use answer_type number; set rel_tol only if the source rounds.
- Famous headline numbers (a paper's abstract) are usually memorized by models: prefer long-tail facts, and combine facts across pages (multi_hop, comparison, temporal arithmetic) - aim for at least half multi-step tasks.
- Prefer facts unlikely to be memorized and unlikely to change soon.
- Do not hand over the investigation: no answer-page URLs, no table names, no hints about where the value sits.
- Name entities the way a person would (gene/product/place names), never by the source's internal record IDs or accessions (no 'entry Q8NBP7', no 'VCV000017661'): finding the right record is part of the task. Name the source only when values differ between sources or versions."""


def task_id(prompt: str) -> str:
	return 't_' + hashlib.sha1(norm(prompt).encode()).hexdigest()[:8]


def apply_reviews(tasks: list[dict], reviews: list[dict]) -> list[dict]:
	"""Attach the latest human review; a gold fix replaces answer AND aliases and re-runs the gold checks."""
	latest = {r['id']: r for r in reviews}
	for t in tasks:
		r = latest.get(t['id'])
		if not r:
			continue
		t['review'] = r
		if r.get('gold_fix'):
			before = set(gold_issues(t))
			t.update(answer=norm(r['gold_fix']) if t['answer_type'] == 'prob' else r['gold_fix'], aliases=[], gold_source='human-corrected')
			for k in ('abs_tol', 'rel_tol'):  # tolerance follows the new gold's precision
				t.pop(k, None)
			t.update(rounding_tol(t))
			c = t['checks']
			c['leakage'] = leaks(t)
			c['structural'] = [i for i in c['structural'] if i not in before and 'gold not literally' not in i] + gold_issues(t)
			if isinstance(c.get('critic'), dict) and 'recomputed_answer' in c['critic']:
				c['critic']['agrees'] = grade_answer(t, c['critic']['recomputed_answer']) >= 0.5
			if c.get('closed_book_answers'):
				c['closed_book'] = {m: grade_answer(t, a) for m, a in c['closed_book_answers'].items()}
	return tasks


MIX = '\n- Mix families: direct lookups deep in a site, multi-hop (combine two pages), comparisons, lists (answer_type set), and dates.'


HARD = '\nHARD MODE: no single lookups - single lookups saturate (any agent searches the name and reads one page). Each task needs the same attribute from 3+ different records/pages, reduced to ONE answer: a superlative (which of these genes has the lowest X?), a ranking-derived pick, a count over records (how many of these have X above Y?), or a difference between records. Name the candidate entities in the prompt; never put any of their values in it.'


async def synthesize(kb: dict, n: int, model: str = TEXT_MODEL, existing: list[str] = (), hard: bool = False) -> list[dict]:
	facts = {f['id']: f for f in kb['facts']}
	ev = '\n'.join(f'[{f["id"]}] ({f.get("angle", "")}) {f["claim"]} = {f["value"]} | quote: "{f["quote"]}" | url: {f["url"]}' for f in kb['facts'])
	ideas = '\n'.join(f'- {i["question"]} (hard because: {i["why_hard"]})' for i in kb['ideas'])
	obs = '\n'.join(o for m in kb['missions'] for o in m.get('observations', []))
	user = f'Topic: {kb["topic"]}\n\nALREADY WRITTEN (do not duplicate):\n{chr(10).join(existing) or "-"}\n\nEVIDENCE (only source of answers):\n{ev}\n\nREAL USER QUESTIONS seen while scouting (match their style and intent):\n{ideas or "-"}\n\nSCOUT OBSERVATIONS about what is hard here:\n{obs or "-"}\n\nWrite {n} tasks.'
	d, _, _ = await call(model, SYNTH_SYS + (HARD if hard else MIX), user, Drafts, temperature=0.7, audience='dataset')
	return [finish(td.model_dump(), facts) for td in d.tasks]


CASE_SYS = """You turn real cases into tasks for an RL environment that trains a model in a skill. Each case has MATERIALS (what the model will be given) and OUTCOME facts (the truth, published elsewhere or later).
Rules:
- The model sees only the chosen materials and the prompt. The gold MUST come from the outcome facts' quotes (copy the value exactly) or a simple stated derivation from them.
- The task must test the skill in the domain, not reading: never ask for something the materials state outright. Good tasks make an expert reason, compute, diagnose or forecast from the materials.
- Forecast only outcomes the materials make forecastable: an outcome driven by events the materials cannot foresee (a financing, a split, an acquisition, a trial halted for outside reasons) is not a task.
- derive: prefer specific answers (a number, a name, a category). Any numeric answer MUST use answer_type number, exact (rel_tol 0 unless the outcome is rounded), with `expression` over numbers exactly as written in the MATERIALS - never a number from the OUTCOME FACTS, which the model never sees - and 2-3 `distractors` (the typical mistakes). Which-of questions: list every candidate in `choices`.
- derive: name the quantity and its definition (the convention an expert would use), never the steps or which lines to use - choosing the inputs, period and basis is the skill. The hardest good tasks combine figures from different statements or periods (de-cumulate year-to-date figures, annualize, net items across tables).
- predict: answer_type prob, gold `yes` or `no`, phrased as a precise yes/no question about the outcome with a threshold set where an expert would genuinely hesitate (near the value the materials imply, not far from it); across a case's forecasts, put thresholds on both sides so about half resolve no. The prompt describes only what the materials show: never the outcome, the posted results or a hint at them. The policy answers with a probability and is scored by Brier score, so a well-chosen binary question is a strong reward.
- Exactly ONE question with ONE answer per task: ask for the final quantity only, never 'X and Y' or 'X, then its change' - intermediate values are the model's work, not its answer. Pin units, periods and definitions in the prompt.
- Never restate figures or findings from the materials in the prompt: locating and selecting the relevant evidence is part of the skill.
- Vary the analysis: across cases, prefer the less obvious analyses an expert would run over repeating the first textbook metric; do not duplicate a computation listed under ALREADY WRITTEN for another case.
- Refer to the materials as 'the materials'/'the excerpt', never by URL."""


CASE_HARD = """
HARD MODE: one-step lookups and textbook ratios saturate (every model gets them). Each derive task must chain at least 3 steps across at least 2 parts of the materials, and hinge on a judgment an expert gets right and a novice gets wrong - which items belong in the quantity, which period or basis applies, which disclosure overrides the headline figure. Still exactly one numeric (or named) answer, fully pinned by the prompt. Each predict task must need the expert's reading of the materials (design, size, trends, base rates), not a guess from the question alone."""


MIXES = {'derive': '\nWrite derive tasks only.', 'predict': '\nWrite predict tasks only (answer_type prob).', 'mixed': ''}


def cases(kb: dict) -> dict[str, dict]:
	"""Complete cases only: at least one material and one outcome fact under the same key."""
	out = {}
	for m in kb.get('materials', []):
		out.setdefault(m['case'], {'materials': [], 'facts': []})['materials'].append(m)
	for f in kb['facts']:
		if f.get('case') in out:
			out[f['case']]['facts'].append(f)
	return {k: c for k, c in out.items() if c['facts']}


def context_of(materials: list[dict]) -> str:
	"""The materials a task hands the model, minus passages already contained in another one (re-recorded tables)."""
	keep = [m for i, m in enumerate(materials) if not any(m['text'] in o['text'] and (m['text'] != o['text'] or j < i) for j, o in enumerate(materials) if j != i)]
	return '\n\n'.join(f'### {m["title"]}\n{m["text"]}' for m in keep)


HARDEN = """
HARDEN: the tasks listed as TOO EASY were solved by every model on the panel, weak and strong - they teach nothing. Replace each with a
HARDER task on this same case. Make it hard the way the domain is hard, never by obscurity or ambiguity: a longer chain across statements,
tables or periods; a convention the expert must apply unprompted (de-cumulate year-to-date figures, annualize, exclude a non-recurring item,
use the right denominator); a disclosure in the notes that overrides the headline figure; or a forecast of the later outcome instead of a
computation. A derive task's inputs come only from the MATERIALS (the outcome facts are hidden from the model: a later share count or
balance belongs in a forecast, not in a computation). Do NOT spell out the formula line by line ("defined as (A + B) / (C - D)"): that hands over the method. Name the metric
and its standard definition in words, as an analyst would ("net debt including finance leases", "quarterly burn excluding non-cash items");
choosing the lines, period and basis is the expert's job. Learn from the RIGHT LEVEL examples what actually trips the models. Still exactly
one answer, fully pinned by the prompt."""


def calibration(tasks: list[dict], k: int = 6) -> str:
	"""The env's own difficulty record for the drafter: what every model solved, and what split the panel and why."""
	verdict = lambda t: t['checks'].get('panel', {}).get('verdict')  # noqa: E731
	easy = [t for t in tasks if verdict(t) == 'saturated'][-k:]
	right = [t for t in tasks if keep(t) and t['checks'].get('panel', {}).get('scores')][-k:]
	flat = [t for t in tasks if verdict(t) in ('no_shift', 'memorized')][-k:]
	if not easy and not right and not flat:
		return ''
	why = lambda p: f" | the models' error: {p['adjudication']['panel_error'][:200]}" if p.get('adjudication') else ''  # noqa: E731
	out = (
		'TOO EASY (every model solved it):\n'
		+ ('\n'.join(f'- {t["prompt"][:300]}' for t in easy) or '-')
		+ '\nRIGHT LEVEL (some models failed it):\n'
		+ ('\n'.join(f'- {t["prompt"][:300]} | scores {t["checks"]["panel"]["scores"]}{why(t["checks"]["panel"])}' for t in right) or '-')
	)
	if flat:  # forecasts the panel answered the same with or without the materials, or knew blind: write ones the materials decide
		out += '\nREJECTED FORECASTS (the materials did not move the forecast, or it was answerable without them):\n' + '\n'.join(
			f'- {t["prompt"][:300]} [{verdict(t)}]' for t in flat
		)
	fc = Counter(t['answer'] for t in tasks if t.get('answer_type') == 'prob' and keep(t, verified=True))
	if sum(fc.values()) >= 4 and max(fc.values()) > 0.55 * sum(fc.values()):
		minor = min(('yes', 'no'), key=lambda a: fc[a])
		out += f"\nOUTCOME BALANCE: {fc['yes']} of this environment's {sum(fc.values())} forecasts resolve yes. Majority-outcome forecasts beyond 60% are dropped, so set new thresholds (on the side an expert would genuinely hesitate about) so that they resolve {minor.upper()}."
	return out


async def synthesize_cases(
	kb: dict,
	per_case: int = 2,
	model: str = TEXT_MODEL,
	existing: list[str] = (),
	concurrency: int = 4,
	have: dict | None = None,
	hard: bool = False,
	mix: str = 'mixed',
	conventions: str = '',
	calibration: str = '',
	easy: dict | None = None,
) -> list[dict]:
	"""Cases in waves of `concurrency`: each wave sees the prompts written so far, so the task set does not collapse into
	one computation repeated per case. `have`: tasks a case already has - it is topped up to `per_case`. `calibration`:
	the panel's record so far, so new drafts start at the right difficulty. `easy` (case -> tasks every model solved):
	harden mode - each case gets one harder replacement per too-easy task, one level up."""
	facts = {f['id']: f for f in kb['facts']}
	written = list(existing)
	system = (
		CASE_SYS
		+ (CASE_HARD if hard or easy else '')
		+ (HARDEN if easy else '')
		+ MIXES[mix]
		+ (f'\nCONVENTIONS of this domain (state the one a task relies on in its prompt):\n{conventions}' if conventions else '')
	)

	async def one(key, c):
		n = len(easy[key]) if easy else per_case - (have or {}).get(key, 0)
		if n <= 0:
			return []
		mats = {m['id']: m for m in c['materials']}
		flag = lambda t: ' [it asked for several values: ask for ONE final quantity]' if t['checks'].get('panel', {}).get('verdict') == 'multi_value' else ''  # noqa: E731
		ask = (
			(
				'TOO EASY on this case (every model solved them):\n'
				+ '\n'.join(f'- {t["prompt"]} => {t["answer"]}{flag(t)}' for t in easy[key])
				+ f'\n\nWrite {n} HARDER tasks for this case, one per too-easy task.'
			)
			if easy
			else f'Write up to {n} new tasks for this case (fewer if the case supports fewer good ones).'
		)
		user = (
			f'Topic: {kb["topic"]}\nSKILL: {kb.get("skill", "")}\nCASE: {key}\n\nMATERIALS:\n'
			+ '\n\n'.join(f'[{m["id"]}] {m["title"]}\n{m["text"]}' for m in c['materials'])
			+ '\n\nOUTCOME FACTS:\n'
			+ '\n'.join(f'[{f["id"]}] {f["claim"]} = {f["value"]} | quote: "{f["quote"]}"' for f in c['facts'])
			+ (f"\n\nDIFFICULTY CALIBRATION from this environment's panel (aim for the right level):\n{calibration}" if calibration else '')
			+ f'\n\nALREADY WRITTEN (do not duplicate):\n{chr(10).join(written) or "-"}\n\n{ask}'
		)
		try:
			d, _, _ = await call(model, system, user, ForecastDrafts if mix == 'predict' else CaseDrafts, temperature=0.7, audience='dataset')
		except Exception as e:  # one case must not sink the round: it is topped up on the next synth
			print(f'synth: case {key} skipped ({type(e).__name__}: {str(e)[:120]})')
			return []
		level = 1 + max(t.get('level', 0) for t in easy[key]) if easy else 0
		out = []
		for td in d.tasks[:n]:
			chosen = [mats[i] for i in td.material_ids if i in mats] or c['materials']
			out.append(
				finish(
					td.model_dump()
					| {'material_ids': [m['id'] for m in chosen], 'context': context_of(chosen), 'case': key, 'start_url': '', 'level': level}
					| ({'hardened_from': [t['id'] for t in easy[key]]} if easy else {}),
					facts,
				)
			)
		return out

	out, todo = [], [(k, c) for k, c in cases(kb).items() if not easy or k in easy]
	ui.stage('harden' if easy else 'write', len(todo))
	for i in range(0, len(todo), concurrency):
		for ts in await ui.gather(one(k, c) for k, c in todo[i : i + concurrency]):
			out += ts
			written += [t['prompt'] for t in ts]
	return out


def finish(t: dict, facts: dict) -> dict:
	"""Shared post-processing of a drafted task: evidence, id, numeric typing, tolerance, deterministic checks."""
	t['language'] = current()['language']
	t |= {'source_urls': list(dict.fromkeys(facts[f]['url'] for f in t['fact_ids'] if f in facts)), 'created': now()}
	t['evidence'] = [{'url': facts[f]['url'], 'value': facts[f]['value'], 'quote': facts[f]['quote']} for f in t['fact_ids'] if f in facts]
	t['id'] = task_id(t['prompt'] + t.get('case', ''))
	t |= code_gold(t)
	version_like = re.fullmatch(r'\d+(\.\d+){2,}|\d+\.\d*0', t['answer'].strip()) or 'version' in t['prompt'].lower()
	if not version_like and t['answer_type'] in ('exact', 'contains') and re.fullmatch(r'[\d.,]+\s*(%|percent|thousand|million|billion|k|m)?', t['answer'].strip(), re.I):
		t['answer_type'] = 'number'  # numeric gold graded as text is brittle (59.1 vs 59.1%)
	t.update(rounding_tol(t))
	if t['answer_type'] == 'prob':
		t['answer'] = norm(t['answer'])
		if 'probab' not in t['prompt'].lower():
			t['prompt'] += ' ' + PROBABILITY.get(t['language'], '(0–1)')
	t['checks'] = {'structural': structural_issues(t, facts), 'leakage': leaks(t)}
	return t


def balanced(tasks: list[dict], hi: float = 0.6) -> list[dict]:
	"""Forecasts that mostly resolve one way reward 'move every probability toward the majority' (measured: a climbed
	prompt gained +0.34 on an 8-yes/2-no test while ranking outcomes worse). Majority-outcome forecasts beyond `hi` of
	the forecast set are left out, from the cases holding the most forecasts first; they return as the minority grows."""
	fc = [t for t in tasks if t.get('answer_type') == 'prob']
	n = Counter(t['answer'] for t in fc)
	if not fc or max(n.values()) <= hi * len(fc):
		return tasks
	major = max(n, key=n.get)
	per_case = Counter(t.get('case') for t in fc)
	allowed = math.floor(hi / (1 - hi) * (len(fc) - n[major]) + 1e-9)
	stable = lambda t: hashlib.sha1(t['id'].encode()).hexdigest()  # noqa: E731  - one task's fate does not depend on which others are in the pool
	drop = {t['id'] for t in sorted((t for t in fc if t['answer'] == major), key=lambda t: (-min(per_case[t.get('case')], 3), stable(t)))[: n[major] - allowed]}
	return [t for t in tasks if t['id'] not in drop]


def keep(t: dict, allow_memorizable: bool = False, verified: bool = False) -> bool:
	"""Kept = correct by every gate AND hard for the panel. verified=True: correct by every gate, however easy - the pool
	model comparisons run on (scores must scale with capability) and a small policy trains on."""
	c = t['checks']
	if c['structural'] or c['leakage'] or c.get('live', None) not in (True, 'skipped'):
		return False  # never verified = not kept; `verify --no-live` records 'skipped' explicitly
	if 'critic' not in c or 'closed_book' not in c:
		return False
	# by construction (code-evaluated expression over numbers in the materials) a failed recompute marks a hard task, not a
	# wrong gold - but only once the panel ran, so a disputed gold meets an adjudicator
	by_code = derived(t) and bool(t.get('expression')) and c.get('panel', {}).get('verdict') in ('keep', 'saturated', 'capped')
	if t['answer_type'] == 'prob' and 'panel' not in c:
		return False  # a forecast's memorization and forecastability checks are the panel's
	if c['critic'] != 'skipped' and not (c['critic']['unambiguous'] and c['critic']['gold_supported'] and (by_code or c['critic'].get('agrees', True))):
		return False
	if c.get('panel', {}).get('verdict', 'keep') not in (('keep', 'saturated', 'capped') if verified else ('keep',)):
		return False
	if c['critic'] != 'skipped' and (c['critic'].get('answer_in_context') or not c['critic'].get('well_posed', True)):
		return False
	cb = [] if c['closed_book'] == 'skipped' else c['closed_book'].values()
	if any(not isinstance(v, bool) for v in cb):  # errored check = unverified: fail closed
		return False
	if not allow_memorizable and any(cb):
		return False
	return t.get('review', {}).get('verdict') != 'reject'


def rejected_facts(env: Path) -> set[str]:
	"""Facts rejected in the review app's Data tab (latest label wins)."""
	return {r['id'] for r in {r['id']: r for r in load_jsonl(env / 'data_reviews.jsonl')}.values() if r['verdict'] == 'reject'}


def load_tasks(env_dir) -> list[dict]:
	"""All tasks with the latest human reviews applied - the single source every stage reads."""
	env = Path(env_dir)
	tasks = apply_reviews(load_jsonl(env / 'tasks.jsonl'), load_jsonl(env / 'reviews.jsonl'))
	bad = rejected_facts(env)
	for t in tasks:  # a fact rejected in the Data tab after tasks were written takes its tasks with it
		if bad & set(t.get('fact_ids', [])) and t.get('review', {}).get('verdict') != 'accept':
			t['review'] = {'verdict': 'reject', 'notes': 'cites a fact rejected in the Data tab'}
	return tasks
