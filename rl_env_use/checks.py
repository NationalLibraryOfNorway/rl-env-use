"""Checks on a task that need no model, recorded under `checks`:

  structural   gold parses for its answer_type, cited facts exist, gold is supported by the cited quotes
  leakage      gold answer (or alias) does not appear in the prompt (nor, as a phrase, in the context)

Derived numbers are correct by construction, not by a frontier model re-solving them (that gate keeps only what
frontier models already solve): code evaluates the expression, every operand is in the materials, the gold is not,
and the drafter's typical wrong answers must fail the grader.
"""

import ast
import operator as op
import re
from urllib.parse import urlparse

from .browse import SEARCH_ENGINES, quote_in
from .graders import dates, norm, numbers, parse_date
from .graders import grade_answer as _grade


def grade_answer(t: dict, a) -> bool | float:
	"""A gold the grader cannot read (no number, two dates) fails the answer instead of crashing a whole stage; such a
	gold is already a structural issue."""
	try:
		return _grade(t, a)
	except ValueError:
		return 0.0 if t.get('answer_type') == 'prob' else False


PREDICT_TOL = (0.10, 0.15)  # forecasts: default and maximum relative band


def rounding_tol(t: dict) -> dict:
	"""Derived decimals: accept anything that rounds to the gold at the gold's own precision (97.63 for 97.6),
	never a different value (98.3). A relative tolerance would accept neighbours."""
	if t['answer_type'] == 'prob':
		return {}
	if t.get('kind') == 'predict':
		return {'rel_tol': t.get('rel_tol') or PREDICT_TOL[0]} if t['answer_type'] == 'number' else {}
	m = re.search(r'\d\.(\d+)', t['answer'])
	g, raw = numbers(t['answer']), numbers(t['answer'], scale=False)
	if t['answer_type'] == 'number' and not t.get('abs_tol') and len(g) == 1 and raw[0] and (t.get('expression') or (t.get('derivation') and m)):
		return {'abs_tol': 0.5 * 10 ** -(len(m.group(1)) if m else 0) * abs(g[0] / raw[0]), 'rel_tol': 0.0}  # '4.5%' -> 0.05; '4.5 million' -> 50,000; an integer rounds too
	return {}


ACCESSION = re.compile(r'\w*\d{5,}\w*|[A-Za-z]+\d{4,}\w*|[A-Za-z]*\d+[A-Za-z]+\d+\w*')  # Q8NBP7, P38398, ENSG00000139618, 17661 - not gene symbols like BRCA1


def evaluate(expr: str) -> float | None:
	"""Value of a plain arithmetic expression, None if it is anything else."""
	ops = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.Pow: op.pow, ast.USub: op.neg, ast.UAdd: op.pos}

	def ev(n):
		if isinstance(n, ast.Constant) and type(n.value) in (int, float):
			return n.value
		if isinstance(n, ast.BinOp) and type(n.op) in ops:
			return ops[type(n.op)](ev(n.left), ev(n.right))
		if isinstance(n, ast.UnaryOp) and type(n.op) in ops:
			return ops[type(n.op)](ev(n.operand))
		raise ValueError

	if len(expr) > 400:
		return None
	ops[ast.Pow] = lambda a, b: float(a) ** b if abs(b) <= 1000 else (_ for _ in ()).throw(ValueError)  # float pow overflows fast; int 9**9**9 never finishes
	try:
		return float(ev(ast.parse(re.sub(r'(?<=\d),(?=\d{3})', '', expr), mode='eval').body))
	except (SyntaxError, ValueError, ZeroDivisionError, OverflowError, TypeError, RecursionError, MemoryError):
		return None


CONSTANTS = {0, 1, 2, 3, 4, 6, 7, 10, 12, 24, 30, 52, 60, 90, 91, 100, 360, 365, 366, 1000, 1e6, 1e9}  # unit and period conversions


SIGNLESS = re.compile(r'\b(as a positive (number|value|figure)|absolute value|in absolute terms|magnitude of)\b', re.I)


def derived(t: dict) -> bool:
	return t.get('kind') == 'derive' and t['answer_type'] == 'number' and bool(t.get('context'))


def printed(text: str) -> set[float]:
	"""Every number as printed, unsigned (accounting negatives, ranges and ISO dates use dashes), currency codes allowed (USD1,234)."""
	return {round(float(x.replace(',', '')), 9) for x in re.findall(r'\d[\d,]*(?:\.\d+)?|\.\d+', text) if re.search(r'\d', x)}


def operands(expr: str) -> list[float]:
	try:
		tree = ast.parse(re.sub(r'(?<=\d),(?=\d{3})', '', expr), mode='eval')
	except SyntaxError:
		return []
	return [abs(n.value) for n in ast.walk(tree) if isinstance(n, ast.Constant) and type(n.value) in (int, float)]


def expression_issue(t: dict) -> str | None:
	"""A derived numeric gold is computed by code, not by the drafting model: it must round from its expression, and the
	expression may only use numbers written in the materials (plus unit and period constants)."""
	if not derived(t) or t.get('gold_source') == 'human-corrected':  # a reviewer's gold overrides the drafted computation
		return None
	expr = t.get('expression') or ''
	v = evaluate(expr)
	if v is None:
		return 'derived number without an evaluable expression'
	g = numbers(t['answer'], scale=False)
	m = re.search(r'\.(\d+)', t['answer'])
	half = 0.5 * 10 ** -len(m.group(1)) if m else 0.5
	signed = len(g) == 1 and abs(v - g[0]) <= half + 1e-9
	if not signed and not (len(g) == 1 and abs(abs(v) - abs(g[0])) <= half + 1e-9 and SIGNLESS.search(t['prompt'])):
		return f'gold {t["answer"]} does not match its expression ({v:.6g})'
	have = printed(t['context'])  # a figure from an 'in thousands' table may enter scaled (4,564 -> 4564000)
	alien = [x for x in operands(expr) if x not in CONSTANTS and not any(round(x / f, 9) in have for f in (1, 1e3, 1e6, 1e9, 1e-3, 1e-6))]
	return f'expression uses numbers not in the materials: {alien[:4]}' if alien else None


def code_gold(t: dict) -> dict:
	"""The expression is the computation of record: a derived gold the drafter rounded or added up wrong is replaced by the
	code's value at the drafted precision (the drafter's number is kept as `drafted_answer`). Operands are still checked."""
	v = evaluate(t.get('expression') or '') if derived(t) else None
	g = numbers(t['answer'], scale=False)
	if v is None or len(g) != 1:
		return t
	m = re.search(r'\.(\d+)', t['answer'])
	d = len(m.group(1)) if m else 0
	if SIGNLESS.search(t['prompt']) and abs(abs(v) - abs(g[0])) <= 0.5 * 10**-d + 1e-9:
		return t  # the prompt fixed the sign convention and the magnitude is right
	new = f'{v:.{d}f}'
	slip = v * g[0] > 0 and 0.8 <= v / g[0] <= 1.25  # an arithmetic or rounding slip; a percent-vs-fraction or sign error is a wrong task, rejected
	if float(new) == g[0] or not slip:
		return t
	tok = re.search(r'\d[\d,]*(?:\.\d+)?', t['answer'])  # replace the number, keep its unit and scale words ('4.5 million')
	return t | {'answer': t['answer'][: tok.start()] + new + t['answer'][tok.end() :] if tok else new, 'drafted_answer': t['answer']}


def gold_stated(t: dict) -> bool:
	"""A derived gold that is itself printed in the materials (a reported subtotal or ratio): reading, not computing."""
	if not derived(t):
		return False
	g = numbers(t['answer'], scale=False)
	sig = re.sub(r'\D', '', t['answer']).strip('0')  # short values (10.0, 1,538) recur by coincidence in a statement
	return len(g) == 1 and (len(sig) >= 5 or (len(sig) >= 4 and '.' in t['answer'])) and round(abs(g[0]), 9) in printed(t['context'])


def distractor_issue(t: dict) -> str | None:
	"""The drafter's typical mistakes must fail the grader: a tolerance that accepts one cannot tell skill from error."""
	if not derived(t) or len(numbers(t['answer'])) != 1:
		return None  # the gold itself is malformed: reported by its own check
	gold = evaluate(t.get('expression') or '')
	vals = [v for d in t.get('distractors') or [] if (v := evaluate(d)) is not None and (gold is None or abs(v - gold) > 1e-9 * max(1, abs(gold)))]  # same value: not a mistake
	bad = [v for v in vals if grade_answer(t, f'{v:.6f}')]
	return f'grader accepts a typical mistake ({bad[0]:.6g})' if bad else None


def gold_issues(t: dict) -> list[str]:
	"""Every structural check that depends on the gold (re-run when a review fixes it)."""
	issues = [x for x in [expression_issue(t), distractor_issue(t)] if x]
	if gold_stated(t):
		issues.append('derived gold is printed in the materials')
	if not t['answer'].strip():
		issues.append('empty answer')
	if t['answer_type'] == 'prob' and norm(t['answer']) not in ('yes', 'no'):
		issues.append('forecast gold must be yes or no')
	if t['answer_type'] == 'number' and len(numbers(t['answer'])) != 1:
		issues.append('number gold must contain exactly one number')
	cap = PREDICT_TOL[1] if t.get('kind') == 'predict' else 0.05
	if t.get('rel_tol', 0) > cap or t.get('abs_tol', 0) < 0:
		issues.append(f'tolerance too loose (rel_tol > {cap:.0%})')
	if t['answer_type'] == 'date' and not parse_date(t['answer']):
		issues.append('date gold does not parse')
	return issues


def structural_issues(t: dict, facts: dict) -> list[str]:
	issues = gold_issues(t)
	missing = [f for f in t['fact_ids'] if f not in facts]
	if missing:
		issues.append(f'unknown facts {missing}')
	quotes = ' '.join(facts[f]['quote'] for f in t['fact_ids'] if f in facts)
	if not t.get('derivation') and t['answer_type'] == 'set' and not all(quote_in(x, quotes) for x in re.split(r'[;,]', t['answer']) if x.strip()):
		issues.append('set gold items not literally in cited quotes and no derivation given')
	if not t.get('derivation') and t['answer_type'] not in ('set', 'prob') and not quote_in(t['answer'], quotes) and not any(quote_in(a, quotes) for a in t.get('aliases', [])):
		issues.append('gold not literally in cited quotes and no derivation given')
	if any(SEARCH_ENGINES.match(facts[f]['url']) for f in t['fact_ids'] if f in facts):
		issues.append('evidence comes from a search-engine results page')
	ids = {
		x for f in t['fact_ids'] if f in facts for x in re.split(r'[/?=&#._-]+', urlparse(facts[f]['url']).path + '?' + urlparse(facts[f]['url']).query) if ACCESSION.fullmatch(x)
	}
	if t.get('context'):
		return issues + ([] if t.get('material_ids') else ['case task without materials'])
	if any(re.search(rf'\b{re.escape(x)}\b', t['prompt'], re.I) for x in ids):
		issues.append('prompt names the record id of the answer page: finding the record is part of the task')
	if urlparse(t['start_url']).scheme not in ('http', 'https'):
		issues.append('start_url not http(s)')
	return issues


def leaks(t: dict) -> bool:
	"""Does the prompt reveal the gold? Plain token match (no hedge/negation logic: 'Paris or Lyon?' leaks Paris),
	plus value-level matches for numbers and dates written differently. Exception: an exact-graded which-of question."""
	if t['answer_type'] == 'prob':
		return False  # yes/no: the gold is the resolution, never a string to leak; the materials-vs-blind panel catches tells
	golds = [t['answer'], *t.get('aliases', [])]
	ctx = f' {norm(t.get("context", ""))} '
	if any(len(norm(g).split()) >= 3 and f' {norm(g)} ' in ctx for g in golds):
		return True  # a multi-word gold quoted in the materials; shorter golds are the critic's call (answer_in_context)
	p = f' {norm(t["prompt"])} '
	ch = [norm(c) for c in t.get('choices', [])]
	if t['answer_type'] == 'exact' and len(ch) >= 2 and norm(t['answer']) in ch and all(f' {c} ' in p for c in ch):
		return False  # a which-of question names its candidates by design; exact grading rejects answers naming several
	if t['answer_type'] == 'set':
		golds += re.split(r'[;,]', t['answer'])
	if any(norm(g) and f' {norm(g)} ' in p for g in golds):
		return True
	if t['answer_type'] == 'number':
		g = numbers(t['answer'])
		return len(g) == 1 and any(abs(n - g[0]) < 1e-9 for n in numbers(t['prompt']))
	if t['answer_type'] == 'date':
		return parse_date(t['answer']) in dates(t['prompt'])
	return False
