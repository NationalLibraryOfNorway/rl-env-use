"""Stage 1: browser agents mine the sources and record evidence-backed facts, case materials, sources and task ideas.

Facts are only accepted if their quote is literally present in the live page text (checked via CDP at record time),
and materials are sliced from the live page text itself, so the knowledge base cannot contain hallucinated evidence.
A case = materials (what the model under test is given) + facts with the same case key (its true outcome).
"""

import asyncio
import json
import os
import re
import shutil
from pathlib import Path
from urllib.parse import urlparse

from browser_use import Agent, Tools
from browser_use.agent.views import ActionResult
from browser_use.browser import BrowserSession
from pydantic import BaseModel, Field

from . import ui
from .browse import SEARCH_ENGINES, UNUSED, browser, http_capture, page_text, quote_in
from .config import BROWSER_MODEL, TEXT_MODEL, browser_use_vision, open_only
from .graders import norm
from .language import language_prompt
from .llm import LEDGER, call, make_llm, now, usage_dict
from .sources import discovery_guidance, normalize_mission
from .store import write_atomic
from .tasks import rejected_facts

ANGLES = {
	'database': "Mine a small table: pick ONE record type and 2-4 fixed fields (e.g. a score, a count, a position, a classification), then use the site's search, filters and pagination to open 5+ different records and record the SAME fields from each record page (prefer fields pinned to a versioned release over live counts that grow) - values only reachable through search or several clicks, never from the landing page.",
	'reference': 'Find the authoritative primary sources for this topic (official sites, registries, databases, documentation). Record facts that require navigating INTO these sites (detail pages, tables, filters), not facts from the landing page.',
	'longtail': 'Find specific, obscure, verifiable values (numbers, dates, names, lists) buried deep in pages or tables, that a person would only know by actually looking them up. Prefer stable facts that will not change week to week.',
	'users': 'Find what real people ask or struggle with on this topic (forums, Q&A sites, issue trackers, FAQ pages). Record the concrete questions people actually ask and, where the page answers them, the answer as a fact.',
	'deep': 'Find facts buried deep inside long documents: specific rows/cells of results tables and appendices in papers (prefer arXiv HTML versions), per-version entries in changelogs/release notes, specific comments inside closed issues. Record the table/section context in the quote. Skip anything stated in an abstract or README headline.',
	'cases': 'Collect complete CASES for the skill below, each about a DIFFERENT entity: open the page holding the input (a filing, a problem, a protocol, a case record) and save it with record_material; then find where the TRUE OUTCOME for that same entity is published (often a later or separate page: results, verdict, official solution, next report) and record it under the same case key: record_fact for one value, record_outcome for a results section or table (verbatim). A case without its outcome is worthless. The materials must not already state the outcome. If a document opens inside a heavy viewer, open its plain document version instead.',
	'interaction': 'Find tasks on this topic that need real site interaction: search forms, filters, sorting, pagination, comparing multiple pages, or reading a specific version/date of something. Record facts that can only be reached through such interaction.',
}


class Mission(BaseModel):
	angle: str
	instructions: str
	start_url: str = Field(description='Concrete https URL to start from (no search engines).')


class Plan(BaseModel):
	missions: list[Mission]


class ScoutSummary(BaseModel):
	observations: list[str] = Field(description='What makes this domain hard for a browser agent: navigation traps, blockers, ambiguity, data that changes.')


_RELATIVE = re.compile(r'\b(ago|today|yesterday|tomorrow|this (week|month|year)|last (week|month|year)|currently|now)\b', re.I)


def fact_issue(value: str, quote: str) -> str | None:
	"""The evidence bar every miner's facts must pass, whoever found them."""
	if len(norm(quote).split()) < len(norm(value).split()) + 2:  # table labels are terse: 'Enrollment (Actual) 28'
		return 'Quote must include the surrounding words that say what the value means (at least 2 words besides the value).'
	if len(value) > 80 or _RELATIVE.search(value):
		return 'Record a short, atomic, time-stable value (a number, date, name or short list) - not a sentence or a relative time like "3 days ago".'
	if not quote_in(value, quote):
		return 'The quote must contain the value.'
	if re.match(r'\s*[-\u2212\u2013(]', value) and not re.search(r'[-\u2212\u2013(]', quote):  # norm() drops signs: a loss must be one on the page
		return 'The value is negative but the quote shows no minus sign or parentheses: copy the value with the sign the page uses.'
	if re.match(r'\s*\$?\d', value) and re.search(r'(?<![\w\d])[-\u2212]\s?\$?\s?' + re.escape(value.strip().lstrip('$')), quote):
		return 'The quote shows this value with a minus sign: record the value with its sign.'
	return None


MATERIAL_CHARS = (300, 20000)
MISSION_TIMEOUT = int(os.getenv('RLENV_MISSION_TIMEOUT', '3000'))  # a cloud browser stuck reconnecting held a mission for 80 minutes; what it saved stays saved
MAX_MATERIALS = 3  # per case before its outcome is recorded: input is cheap, the outcome is the hard part


def marker(words: str) -> re.Pattern | None:
	"""A marker as page text renders it: any case (CSS uppercases headings), any punctuation or whitespace between words (table cells, quote styles)."""
	toks = re.findall(r'\w+', words)  # words only: table cells render with tabs, colons, pipes or nothing between them
	return re.compile(r'[\W_]*'.join(map(re.escape, toks)) + r'[^\w\s]*', re.I) if toks else None  # a closing '.' or ')' stays in the passage


def near(text: str, words: str) -> str:
	"""The page text where the first two words of a failed marker occur: lets the agent copy the real wording."""
	m = marker(' '.join(words.split()[:2]))
	hits = [h.start() for h in m.finditer(text)][:3] if m else []
	return (' Text on the page there: ' + ' | '.join(repr(' '.join(text[max(0, h - 80) : h + 100].split())) for h in hits)) if hits else ''


def slice_passage(text: str, starts_with: str, ends_with: str) -> str:
	"""The page text from the first `starts_with` through the next `ends_with`, verbatim; '!reason' when it cannot be cut."""
	s, e = marker(starts_with), marker(ends_with)
	i = s.search(text) if s else None
	if not i:
		return '!starts_with not found on the current page: copy the first words of the passage exactly as shown.' + near(text, starts_with)
	j = e.search(text, i.end()) if e else None
	if not j:
		return '!ends_with not found after starts_with: copy the last words of the passage exactly as shown.' + near(text[i.end() :], ends_with)
	passage = text[i.start() : j.end()]
	if not MATERIAL_CHARS[0] <= len(passage) <= MATERIAL_CHARS[1]:
		return f'!passage is {len(passage)} chars; it must be {MATERIAL_CHARS[0]}-{MATERIAL_CHARS[1]}: pick the part a solver needs.'
	return passage


def clinical_outcome(text: str, title: str) -> str:
	"""One complete primary-outcome object, verbatim from a full ClinicalTrials.gov JSON record.

	Never fall back to marker slicing: a partial analysis can look like a valid quotation.
	"""
	try:
		document = json.loads(text)
		measures = document['resultsSection']['outcomeMeasuresModule']['outcomeMeasures']
		matches = [m for m in measures if m.get('type') == 'PRIMARY' and norm(m.get('title', '')) == norm(title)]
	except (ValueError, KeyError, TypeError, AttributeError):
		return '!Open the complete ClinicalTrials.gov API record: valid JSON with resultsSection.outcomeMeasuresModule.outcomeMeasures is required.'
	if len(matches) != 1:
		return '!Use the exact title of one primary outcome. Available titles: ' + ', '.join(m.get('title', '') for m in measures if m.get('type') == 'PRIMARY')
	decoder = json.JSONDecoder()
	for array in re.finditer(r'"outcomeMeasures"\s*:\s*\[', text):
		at = array.end()
		while at < len(text):
			while at < len(text) and text[at] in ' \t\r\n,':
				at += 1
			if at == len(text) or text[at] == ']':
				break
			try:
				item, end = decoder.raw_decode(text, at)
			except ValueError:
				break
			if item == matches[0]:
				passage = text[at:end]
				if len(passage) > MATERIAL_CHARS[1]:
					return '!Complete outcome exceeds the passage limit; use a focused primary source. It has not been truncated or saved.'
				return passage
			at = end
	return '!Could not locate the complete outcome object verbatim; nothing saved.'


def make_tools(kb: dict, angle: str, save=lambda: None, tally: dict | None = None) -> Tools:
	tally = {} if tally is None else tally  # what this mission saved: the source-health record
	own = set()  # case keys this mission opened
	tools = Tools(exclude_actions=[x for x in UNUSED if x not in ('search_page', 'find_elements')])  # zero-LLM grep/CSS queries over big tables

	@tools.action(
		'Record one verifiable fact you can see on the CURRENT page. `quote` must be copied verbatim from the page and contain the value. `case`: the case key when this fact is the outcome of a case, else empty.'
	)
	async def record_fact(claim: str, quote: str, browser_session: BrowserSession, value: str = '', case: str = '') -> ActionResult:
		keys = {m['case'] for m in kb.get('materials', [])}
		if case and keys and case not in keys:  # an outcome must land on the case its material was saved under
			return ActionResult(error=f'Unknown case key {case!r}. Use the exact key you saved its material under: {sorted(keys)[-12:]}')
		if not value.strip():  # agent models often drop this field: a quote holding exactly one number names it
			nums = re.findall(r'[-+]?\$?\d[\d,]*(?:\.\d+)?%?', quote)
			if len(nums) != 1:
				return ActionResult(error='Give `value`: the short atomic value (number, date, name) inside the quote that the claim is about.')
			value = nums[0]
		if issue := fact_issue(value, quote):
			return ActionResult(error=issue)
		url, text = await browser_session.get_current_page_url(), await page_text(browser_session)
		if SEARCH_ENGINES.match(url):
			return ActionResult(error='Search result snippets are not evidence. Open the primary source page and record the fact there.')
		if not quote_in(quote, text):
			return ActionResult(error='Quote not found verbatim on the current page. Copy the exact text that contains the value.' + near(text, value))
		if any(f['url'] == url and norm(f['value']) == norm(value) and f.get('case') == (case or None) for f in kb['facts']):  # another case may share the value
			return ActionResult(extracted_content=f'Already recorded: {value}')
		kb['facts'].append(
			{'id': f'f{len(kb["facts"])}', 'claim': claim, 'value': value, 'quote': quote, 'url': url, 'angle': angle, 'at': now(), **({'case': case} if case else {})}
		)
		tally['outcomes' if case else 'facts'] = tally.get('outcomes' if case else 'facts', 0) + 1
		save()
		return ActionResult(extracted_content=f'Recorded fact: {claim} = {value}', long_term_memory=f'Recorded fact #{len(kb["facts"])}: {claim}')

	@tools.action(
		'Save the OUTCOME of a case from the CURRENT page. For ClinicalTrials.gov API JSON, give the exact primary-outcome title: the complete outcome object including all analyses is saved automatically and text markers are ignored. Else give the first and last few words of the COMPLETE results passage, including its statistical analysis. `case`: the key its material was saved under. Never compute or summarize the source.'
	)
	async def record_outcome(case: str, title: str, starts_with: str, ends_with: str, browser_session: BrowserSession) -> ActionResult:
		keys = {m['case'] for m in kb.get('materials', [])}
		if case not in keys:
			return ActionResult(error=f"Unknown case key {case!r}: save the case's material first, or use the exact key it was saved under: {sorted(keys)[-12:]}")
		url, text = await browser_session.get_current_page_url(), await page_text(browser_session)
		if SEARCH_ENGINES.match(url):
			return ActionResult(error='Search result snippets are not evidence. Open the primary source page.')
		clinical = urlparse(url).hostname in ('clinicaltrials.gov', 'www.clinicaltrials.gov') and bool(re.fullmatch(r'/api/v2/studies/NCT\d+', urlparse(url).path))
		passage = clinical_outcome(text, title) if clinical else slice_passage(text, starts_with, ends_with)
		if passage.startswith('!'):
			return ActionResult(error=passage[1:])
		if any(f.get('case') == case and passage in f['quote'] for f in kb['facts']):
			return ActionResult(extracted_content=f'Already saved for case {case}.')
		kb['facts'].append(
			{'id': f'f{len(kb["facts"])}', 'claim': title, 'value': '(results passage)', 'quote': passage, 'url': url, 'angle': angle, 'at': now(), 'case': case, 'passage': True,
			 **({'capture': 'complete_primary_outcome_json'} if clinical else {})}
		)
		tally['outcomes'] = tally.get('outcomes', 0) + 1
		save()
		return ActionResult(extracted_content=f'Saved outcome for case {case}: {title} ({len(passage)} chars)', long_term_memory=f'Outcome for case {case}: {title}')

	@tools.action(
		'Save a passage of the CURRENT page as input material for a case (what the model under test will read). Give the first and last few words of the passage exactly as on the page; everything between them is saved verbatim. `case`: a short key naming the entity, reused for its outcome fact.'
	)
	async def record_material(case: str, title: str, starts_with: str, ends_with: str, browser_session: BrowserSession) -> ActionResult:
		url, text = await browser_session.get_current_page_url(), await page_text(browser_session)
		passage = slice_passage(text, starts_with, ends_with)
		if passage.startswith('!'):
			return ActionResult(error=passage[1:])
		mine = [m for m in kb.get('materials', []) if m['case'] == case]
		if mine and angle not in ('outcome', 'enrich') and case not in own:  # parallel missions pick keys independently
			return ActionResult(error=f'Case key {case!r} is already taken by another case. Use a distinct key that includes the record id (e.g. the registry or filing number).')
		own.add(case)
		if len(mine) >= MAX_MATERIALS and not any(f.get('case') == case for f in kb['facts']):  # complete cases may be enriched
			return ActionResult(
				error=f'Case {case} already has {len(mine)} materials and no outcome: record its outcome with record_fact or record_outcome (case="{case}") now (cut larger passages instead of many small ones).'
			)
		if any(passage in m['text'] for m in mine):
			return ActionResult(extracted_content=f'Already saved for case {case}: this passage is part of a saved material. Move on to its outcome.')
		kb.setdefault('materials', []).append({'id': f'm{len(kb["materials"])}', 'case': case, 'title': title, 'text': passage, 'url': url, 'at': now()})
		tally['materials'] = tally.get('materials', 0) + 1
		save()
		return ActionResult(extracted_content=f'Saved material for case {case}: {title} ({len(passage)} chars)', long_term_memory=f'Material for case {case}: {title}')

	@tools.action('Record a realistic question a user might ask on this topic, why it is hard for a browser agent, and the page where its answer lives (if known).')
	async def record_task_idea(question: str, why_hard: str, answer_url: str = '', browser_session: BrowserSession = None) -> ActionResult:
		kb['ideas'].append({'question': question, 'why_hard': why_hard, 'answer_url': answer_url or await browser_session.get_current_page_url(), 'angle': angle})
		save()
		return ActionResult(extracted_content=f'Recorded task idea: {question}', long_term_memory=f'Recorded idea: {question}')

	@tools.action('Record a useful source website for this topic (the CURRENT page) with a short note on what it contains and how to navigate it.')
	async def record_source(note: str, browser_session: BrowserSession) -> ActionResult:
		url, title = await browser_session.get_current_page_url(), await browser_session.get_current_page_title()
		kb['sources'].append({'url': url, 'title': title, 'note': note, 'angle': angle})
		save()
		return ActionResult(extracted_content=f'Recorded source {url}', long_term_memory=f'Recorded source {url}')

	return tools


def source_health(kb: dict) -> str:
	"""Per start site: missions run, crashed, what they saved, and the miners' first note - so the next plan moves away
	from sources that yield nothing (an unstable search UI, a login wall) instead of repeating them."""
	by = {}
	for m in kb.get('missions', []):
		h = by.setdefault(urlparse(m['start_url']).netloc, {'n': 0, 'crashed': 0, 'materials': 0, 'outcomes': 0, 'note': ''})
		h['n'] += 1
		h['crashed'] += bool(m.get('error'))
		for k in ('materials', 'outcomes'):
			h[k] += m.get('recorded', {}).get(k, 0)
		h['note'] = h['note'] or next(iter(m.get('observations') or []), '')[:200]
	return '\n'.join(
		f'- {host}: {h["n"]} missions ({h["crashed"]} crashed) saved {h["materials"]} materials and {h["outcomes"]} outcomes' + (f' - miner note: {h["note"]}' if h['note'] else '')
		for host, h in by.items()
	)


async def plan(topic: str, n: int, model: str, angles: list[str] | None = None, sources: list[str] | None = None, health: str = '') -> list[Mission]:
	angles = angles or (['database', 'deep', 'interaction', 'longtail'] if sources else None)
	pool = [(k, ANGLES[k]) for k in angles] if angles else list(ANGLES.items())
	angles = [pool[i % len(pool)] for i in range(max(1, n))]
	sys = 'You plan web research missions for browser agents. Each mission gets a concrete, real https start URL (never a search engine) that is likely to work without login or captcha. Missions must cover DIFFERENT entities/sub-areas of the topic (name at least 3 distinct entities per mission, no entity in two missions) so the resulting benchmark is diverse.'
	if sources:  # the user knows where the data lives: every source gets missions, start URLs come from them
		sys += (
			' The user has named the sources where this data lives: start every mission on (a page of) one of these sources, spread missions evenly over them, and plan missions that go deep into each one.'
			' SOURCE PAGES shows what each source returns to a plain fetch: when it lists records (ids, names, links), give every mission its OWN records by id or name with their direct URLs - two missions must never get the same record.'
		)
		pages = await asyncio.gather(*(http_capture(u) for u in sources))  # free, and it lets the plan name concrete records instead of guessing filters
		topic += (
			'\nSOURCES:\n'
			+ '\n'.join(sources)
			+ '\n\nSOURCE PAGES (plain fetch, truncated):\n'
			+ '\n\n'.join(f'[{u}]\n{p["text"][:6000]}' for u, p in zip(sources, pages) if p.get('text'))
		)
	if guidance := discovery_guidance(sources):
		sys += '\nVERIFIED SOURCE ROUTES:\n' + guidance
	if health:
		sys += ' SOURCE HEALTH below reports what earlier missions saved per site: move missions away from sites that crash or save no complete cases, to other official pages with the same data (record pages, report or docket pages, bulk downloads, APIs).'
	user = (
		(f'SOURCE HEALTH from earlier missions:\n{health}\n\n' if health else '')
		+ f'Topic: {topic}\n\nWrite exactly one mission per line below (angles may repeat - then pick different entities). Make instructions specific to the topic.\n'
		+ '\n'.join(f'- angle `{k}`: {v}' for k, v in angles)
	)
	missions = []
	for _ in range(3):  # planners sometimes stop early: ask for the rest
		p, _, _ = await call(
			model,
			sys,
			user + (f'\n\nALREADY PLANNED (write only the {n - len(missions)} remaining, different ones):\n' + '\n'.join(m.instructions for m in missions) if missions else ''),
			Plan,
		)
		missions += [normalize_mission(m) for m in p.missions]
		if len(missions) >= n:
			break
	return missions[:n]


def outcome_missions(kb: dict, skill: str) -> list[Mission]:
	"""One mission per case that has materials but no outcome yet, starting where its material was cut."""
	done = {f.get('case') for f in kb['facts']} | {
		x.group(1) for m in kb.get('missions', []) if m.get('angle') == 'outcome' and not m.get('error') for x in [re.search(r'Case `([^`]+)`', m.get('instructions', ''))] if x
	}  # one try per case: an outcome not yet published must not starve new mining
	first = {}
	for m in kb.get('materials', []):
		if m['case'] not in done:
			first.setdefault(m['case'], m)
	return [
		Mission(
			angle='outcome',
			start_url=m['url'],
			instructions=f'Case `{k}` already has its input material ("{m["title"]}", on the start page). Do not collect new cases: find where the TRUE OUTCOME of this same case is published (for a forecast: the later document that reports it; otherwise the figures that settle the answer) and record it verbatim: record_fact(case="{k}") for one value, record_outcome(case="{k}") for a results section or table. If no outcome exists yet, say so and stop.',
		)
		for k, m in first.items()
	]


def enrich_missions(kb: dict) -> list[Mission]:
	"""One mission per complete case: save the data its computations need that the saved materials lack."""
	facts = {}
	for f in kb['facts']:
		facts.setdefault(f.get('case'), f)
	return [
		Mission(
			angle='enrich',
			start_url=f['url'],
			instructions=f'Case `{k}` is complete, but its saved materials may lack the data a solver needs to compute answers (results tables, financial statements, appendices). Save those sections from this page (or the case\'s other pages) with record_material(case="{k}"): whole tables, not single values. Do not collect new cases.',
		)
		for k, f in facts.items()
		if k and any(m['case'] == k for m in kb.get('materials', []))
	]


def case_task(topic: str, skill: str, m: Mission) -> str:
	if m.angle in ('outcome', 'enrich'):
		return (
			f'You are completing a case for an environment that trains a model in "{topic}". The model is given a case\'s input material and must: {skill}\n'
			f'{m.instructions}\nStart at {m.start_url}. Record the outcome verbatim under the case key as given: record_fact for one value, record_outcome for a results section or table. When done, report what made the outcome hard to find.'
		)
	return (
		f'You are collecting real cases for an environment that trains a model in "{topic}".\nThe model will be given the input materials of a case and must: {skill}\n'
		f'Mission: {m.instructions}\nStart at {m.start_url}. For each case pick a short key (e.g. the entity name and year), save its input with record_material '
		'(the passage a solver needs, cut from the page itself), then record its true outcome under the same key, verbatim from the page that states it: record_fact for one value, record_outcome for a results section or table (do not compute or summarize - save what the page shows). '
		'Finish one case completely (material AND outcome) before starting the next: 2 complete cases are worth more than 6 inputs without outcomes. '
		'Aim for 3-6 complete cases about DIFFERENT entities. Skip a case if its materials already state the outcome, or if no outcome is published. '
		'When done, report your observations about where inputs and outcomes live and what made them hard to reach.'
	)


async def run_mission(m: Mission, topic: str, kb: dict, model: str, max_steps: int, headless: bool, save=lambda: None, skill: str | None = None) -> dict:
	tally = {}
	tools = make_tools(kb, m.angle, save, tally)
	task = (
		case_task(topic, skill, m)
		if skill
		else (
			f'You are scouting the topic "{topic}" to build a benchmark of realistic web tasks.\nMission ({m.angle}): {m.instructions}\n'
			f'Start at {m.start_url}. Use record_source for good sources, record_fact for verifiable facts (quote verbatim from the current page), '
			'and record_task_idea for realistic hard questions (at least 3). Aim for 8-12 facts about several DIFFERENT entities from DIFFERENT pages, including some reachable only after several clicks. '
			'Facts must be short atomic values that will not change soon (no view counts, stars, "N days ago"). '
			'When done, report your observations about what makes this domain hard for a browser agent.'
		)
	)
	agent = Agent(
		task=task,
		llm=make_llm(model),
		tools=tools,
		browser_session=browser(headless=headless),
		output_model_schema=ScoutSummary,
		extend_system_message=language_prompt('diagnostics'),
		use_judge=False,
		use_vision=browser_use_vision(),
		enable_signal_handler=False,
		calculate_cost=not open_only(),
		llm_timeout=int(os.getenv('RLENV_LOCAL_TIMEOUT', '1200')) if open_only() else None,
		step_timeout=int(os.getenv('RLENV_STEP_TIMEOUT', '360')),
	)  # its handler exits on SIGTERM without closing cloud browsers; ours cancels and cleans up
	try:
		h = await agent.run(max_steps=max_steps)
	finally:
		try:  # a cancelled or crashed mission must still release its cloud browser (billed until it times out)
			await asyncio.wait_for(agent.close(), 30)
		except (Exception, asyncio.CancelledError):
			pass
		shutil.rmtree(agent.agent_directory, ignore_errors=True)  # browser-use's per-agent temp dir (screenshots)
		try:  # a mission that crashed or timed out was still billed for its steps
			LEDGER.add(agent.llm.model, usage_dict(await agent.token_cost_service.get_usage_summary()))
		except Exception:
			pass
	try:
		out = h.structured_output
	except ValueError:  # agent finished with free text instead of the schema
		out = ScoutSummary(observations=[h.final_result() or ''])
	return {
		'angle': m.angle,
		'start_url': m.start_url,
		'instructions': m.instructions,
		'steps': h.number_of_steps(),
		'done': h.is_done(),
		'observations': out.observations if out else [],
		'urls': list(dict.fromkeys(u for u in h.urls() if u)),
		'recorded': tally,
	}


async def scout(
	topic: str,
	out_dir: Path,
	n_missions: int = 4,
	model: str = BROWSER_MODEL,
	max_steps: int = 30,
	headless: bool = True,
	concurrency: int = 4,
	plan_model: str = TEXT_MODEL,
	angles: list[str] | None = None,
	sources: list[str] | None = None,
	skill: str | None = None,
	enrich: bool = False,
) -> dict:
	out_dir.mkdir(parents=True, exist_ok=True)
	kb_path = out_dir / 'kb.json'
	kb = json.loads(kb_path.read_text()) if kb_path.exists() else {'topic': topic, 'created': now(), 'facts': [], 'ideas': [], 'sources': [], 'missions': []}
	seen = '\n'.join(dict.fromkeys([m['case'] for m in kb.get('materials', [])] + [f['claim'] for f in kb['facts']]))[:3000]  # another round: go wide, not over the same records
	goal = topic + (f'\nSKILL the environment trains (input -> outcome): {skill}' if skill else '')
	bad = rejected_facts(out_dir)
	eligible = kb | {key: [item for item in kb.get(key, []) if item['id'] not in bad] for key in ('materials', 'facts')}
	todo = (enrich_missions(eligible) if enrich else outcome_missions(eligible, skill))[:n_missions] if skill else []  # finish accepted cases before opening new ones
	n_new = max(n_missions - len(todo), len(sources or ()) if not todo else 0)
	missions = todo + (
		await plan(
			goal + (f'\nALREADY MINED (choose DIFFERENT entities/records):\n{seen}' if seen else ''),
			n_new,
			plan_model,
			angles or (['cases'] if skill else None),
			sources,
			source_health(kb),
		)
		if n_new
		else []
	)
	sem = asyncio.Semaphore(concurrency)
	save = lambda: write_atomic(out_dir / 'kb.json', json.dumps(kb, indent=1))  # noqa: E731

	async def one(m):
		async with sem:
			for attempt in range(3):
				try:
					kb['missions'].append(await asyncio.wait_for(run_mission(m, topic, kb, model, max_steps, headless, save, skill), MISSION_TIMEOUT))
					break
				except (Exception, asyncio.CancelledError) as e:  # browser-use timeouts surface as CancelledError; one mission must not sink the rest
					if attempt < 2 and re.search(r'Failed to create cloud browser|HTTP 503|Too many concurrent|HTTP 429', str(e)):
						await asyncio.sleep(60)  # the browser never started: wait for the pool, then retry
						continue
					kb['missions'].append({'angle': m.angle, 'start_url': m.start_url, 'instructions': m.instructions, 'error': f'{type(e).__name__}: {e}'})
					break
			save()

	await ui.gather((one(m) for m in missions), 'mine')
	kb['spend_usd'] = round(LEDGER.usd, 4)
	save()
	return kb
