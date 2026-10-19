"""Task quality gates: code-evaluated derive golds, forecast (prob) tasks, keep() and the difficulty panel."""

import asyncio

import pytest
from conftest import mk_task

import rl_env_use.checks as C
import rl_env_use.rollout as R
import rl_env_use.tasks as T
import rl_env_use.verify as V
from rl_env_use.audit import grader_selftest
from rl_env_use.graders import grade_answer, prob_score

CTX = 'Revenue was $10,875 thousand in 2024 and $6,939 thousand in 2023. Net loss was (1,234) and cash was 4.25 million.'
FACTS = {'f0': {'quote': 'the outcome', 'url': 'https://sec.gov/outcome', 'value': 'v'}}
OK = {
	'structural': [],
	'leakage': False,
	'live': True,
	'closed_book': {'haiku': False},
	'critic': {'unambiguous': True, 'gold_supported': True, 'agrees': True, 'answer_in_context': False, 'well_posed': True},
}


def drv(answer, expression='', prompt='What share of the two years of revenue fell in 2024, in percent?', **kw):
	return mk_task(0, answer=answer, kind='derive', context=CTX, expression=expression, material_ids=['m0'], start_url='', prompt=prompt, derivation='computed', case='acme') | kw


def fc(answer='yes', **kw):
	return (
		mk_task(
			0, answer=answer, answer_type='prob', kind='predict', context=CTX, material_ids=['m0'], start_url='', prompt='Will revenue exceed $12 million in 2025?', case='acme'
		)
		| kw
	)


def checks(**kw):
	return OK | {'critic': OK['critic'] | kw.pop('critic', {})} | kw


def test_expression_must_match_gold_at_its_precision():
	assert C.expression_issue(drv('61.0', '10875 / (10875 + 6939) * 100')) is None
	assert C.expression_issue(drv('61.05', '10875 / (10875 + 6939) * 100')) is None
	assert 'does not match' in C.expression_issue(drv('61.1', '10875 / (10875 + 6939) * 100'))
	assert 'does not match' in C.expression_issue(drv('61', '10875 / 6939 * 100'))
	assert C.expression_issue(drv('61.0', '')) == 'derived number without an evaluable expression'
	assert C.expression_issue(drv('61.0', 'max(10875, 6939)')) == 'derived number without an evaluable expression'
	assert C.expression_issue(drv('61.0', '10875 / 0')) == 'derived number without an evaluable expression'


def test_expression_sign_only_relaxed_when_prompt_asks_for_magnitude():
	e = '6939 - 10875'  # -3936
	assert C.expression_issue(drv('-3936', e)) is None
	assert 'does not match' in C.expression_issue(drv('3936', e))
	for ask in ('as a positive number', 'as an absolute value', 'in absolute terms', 'the magnitude of'):
		assert C.expression_issue(drv('3936', e, prompt=f'What is the change in revenue ({ask}), in thousands?')) is None
	for domain in ('false positive', 'absolute risk reduction', 'positive predictive value'):  # domain terms, not a sign convention
		assert 'does not match' in C.expression_issue(drv('3936', e, prompt=f'What is the change in the {domain} count?'))
	assert 'does not match' in C.expression_issue(drv('3937', e, prompt='What is the change, in absolute terms?'))


def test_expression_operands_from_materials_and_constants():
	assert C.expression_issue(drv('15.4', '10,875 / (10,875 + 6,939) * 100 / 4 + 0.1')) == 'expression uses numbers not in the materials: [0.1]'  # 0.1 is alien
	assert C.expression_issue(drv('15.3', '10,875 / (10,875 + 6,939) * 100 / 4')) is None  # thousands separators in the expression
	assert C.expression_issue(drv('-1234', '-(1234)')) is None  # accounting parentheses in the materials
	assert C.expression_issue(drv('4250000', '4.25 * 1e6')) is None
	assert C.expression_issue(drv('354.2', '4.25 * 1000 / 12 + 0')) is None  # 0 is a constant (a missing line item)
	assert C.expression_issue(drv('354.2', '4.25 * 1000 / 12 + 0.01')) == 'expression uses numbers not in the materials: [0.01]'
	assert C.expression_issue(drv('29.8', '10875 / 365')) is None
	iss = C.expression_issue(drv('17814', '10875 + 6939 + 5000 - 5000'))
	assert iss and '5000' in iss
	assert '0.37' in C.expression_issue(drv('4024', '10875 * .37'))  # no leading digit: still an operand
	assert C.expression_issue(drv('4350', '10875 * 4e-1'))  # scientific notation: 0.4 is not in the materials
	assert C.expression_issue(drv('3', '2024 - 2021', context='Sales over 2021-2024 rose.')) is None  # ranges are not negatives
	assert C.expression_issue(drv('3976', '5210 - 1234', context='Revenue 5,210; investing cash flow -1,234.')) is None
	assert C.expression_issue(drv('1290', '1234 + 56', context='Revenue USD1,234 and fees EUR56.')) is None  # currency-prefixed figures
	assert C.expression_issue(drv('3936', '10875 - 6939', kind='predict')) is None  # not a derive task: not checked
	assert C.expression_issue(drv('3936', '1 + 1', context='')) is None


def test_printed_numbers_are_unsigned_and_allow_currency_codes():
	assert C.printed('USD1,234 and EUR56; loss -5.5, (7), .25 over 2021-2024') == {1234, 56, 5.5, 7, 0.25, 2021, 2024}
	assert C.expression_issue(drv('-5.5', '-5.5 * 1', context='Net loss was -5.5.')) is None
	assert C.expression_issue(drv('-1.5', '-(4 - 2.5)', context='Values 4 and -2.5.')) is None  # material signs never block an operand


def test_rounding_tol_for_every_task_with_an_expression():
	assert C.rounding_tol(drv('3936', '10875 - 6939')) == {'abs_tol': 0.5, 'rel_tol': 0.0}  # integer gold: rounds too
	assert C.rounding_tol(drv('61.05', '10875 / (10875 + 6939) * 100')) == {'abs_tol': pytest.approx(0.005), 'rel_tol': 0.0}
	assert C.rounding_tol(drv('3936', '10875 - 6939', derivation='')) == {'abs_tol': 0.5, 'rel_tol': 0.0}
	assert C.rounding_tol(drv('3936')) == {}  # derivation only, integer: exact
	assert C.rounding_tol(drv('3936', '10875 - 6939', abs_tol=2)) == {}  # an explicit tolerance stays


def test_distractor_issue_skips_malformed_gold():
	for gold in ('61 or 62', 'about sixty', ''):
		assert C.distractor_issue(drv(gold, '10875 / (10875 + 6939) * 100', distractors=['10875 / 6939 * 100'])) is None


def test_gold_fix_reruns_gold_checks():
	t = drv('61.0', '10875 / (10875 + 6939) * 100', checks=checks(structural=["unknown facts ['f9']"]))
	(t,) = T.apply_reviews([t], [{'id': 't0', 'verdict': 'accept', 'gold_fix': '62.0'}])
	assert t['answer'] == '62.0' and t['checks']['structural'] == ["unknown facts ['f9']"] and t['abs_tol'] == 0.05  # the reviewer overrides the drafted expression
	(t,) = T.apply_reviews([t], [{'id': 't0', 'verdict': 'accept', 'gold_fix': '61'}])
	assert t['checks']['structural'] == ["unknown facts ['f9']"] and t['abs_tol'] == 0.5
	p = fc('yes', checks=checks(closed_book='skipped'))
	(p,) = T.apply_reviews([p], [{'id': 't0', 'verdict': 'accept', 'gold_fix': 'probably'}])
	assert p['checks']['structural'] == ['forecast gold must be yes or no']


def test_gold_stated_needs_enough_significant_digits():
	assert C.gold_stated(drv('10875'))  # 5 significant digits, printed in the materials
	assert C.gold_stated(drv('-10,875'))  # sign and separators ignored
	assert C.gold_stated(drv('4.25', context=CTX + ' Ratio 42.75.')) is False  # 3 sig digits: coincidence, not reading
	assert C.gold_stated(drv('42.75', context=CTX + ' Ratio 42.75.'))  # 4 sig digits with a decimal point
	assert not C.gold_stated(drv('1,234'))  # 4 digits, no decimal: recurs by coincidence
	assert not C.gold_stated(drv('10.0', context='Ten is 10.0.'))
	assert not C.gold_stated(drv('17814'))  # not in the materials
	assert not C.gold_stated(drv('10875', kind='predict'))


def test_distractor_inside_tolerance_is_flagged():
	t = drv('61.0', '10875 / (10875 + 6939) * 100') | C.rounding_tol(drv('61.0'))
	assert t['abs_tol'] == pytest.approx(0.05)
	assert C.distractor_issue(t | {'distractors': ['10875 / 6939 * 100', '6939 / (10875 + 6939) * 100']}) is None
	iss = C.distractor_issue(t | {'distractors': ['10875 / 6939 * 100', '6100 / 100 + 0.04']})
	assert iss and '61.04' in iss
	assert C.distractor_issue(t | {'distractors': ['not an expression']}) is None


def test_structural_issues_on_derive_tasks():
	good = drv('61.0', '10875 / (10875 + 6939) * 100', distractors=['10875 / 6939 * 100'])
	assert C.structural_issues(good, FACTS) == []
	bad = C.structural_issues(drv('10875', '10875 * 1', distractors=['10875 + 0.1']), FACTS)
	assert 'derived gold is printed in the materials' in bad and not any('typical mistake' in i for i in bad)
	assert any('typical mistake' in i for i in C.structural_issues(drv('10875', '10875 * 1', distractors=['10875.3 * 1'], abs_tol=0.5), FACTS))
	assert not any('typical mistake' in i for i in C.structural_issues(drv('10875', '10875 * 1', distractors=['10875 * 1']), FACTS))  # same value: not a mistake


def test_structural_prob_gold_must_be_yes_or_no():
	assert C.structural_issues(fc('yes'), FACTS) == []
	assert C.structural_issues(fc('No'), FACTS) == []
	assert 'forecast gold must be yes or no' in C.structural_issues(fc('0.7'), FACTS)
	assert 'forecast gold must be yes or no' in C.structural_issues(fc('maybe'), FACTS)
	assert not any('literally' in i for i in C.structural_issues(fc('yes', derivation=''), FACTS))


def test_finish_normalizes_prob_and_appends_probability_once():
	d = {
		'prompt': 'Will revenue exceed $12 million in 2025?',
		'kind': 'predict',
		'material_ids': ['m0'],
		'answer': 'Yes.',
		'answer_type': 'prob',
		'aliases': [],
		'choices': [],
		'rel_tol': 0.0,
		'fact_ids': ['f0'],
		'derivation': '',
		'expression': '',
		'distractors': [],
		'category': 'growth',
		'difficulty': 'hard',
		'why_hard': 'x',
		'context': CTX,
		'case': 'acme',
		'start_url': '',
	}
	t = T.finish(dict(d), FACTS)
	assert t['answer'] == 'yes' and t['answer_type'] == 'prob' and t['checks'] == {'structural': [], 'leakage': False}
	assert t['prompt'].endswith(' Answer with the probability (0 to 1) that it did.') and t['prompt'].count('probability') == 1
	assert 'abs_tol' not in t and t['rel_tol'] == 0.0
	assert T.finish(dict(t), FACTS)['prompt'] == t['prompt']  # re-finishing does not append again
	asked = T.finish(dict(d, prompt='What is the probability that revenue exceeds $12 million?'), FACTS)
	assert asked['prompt'] == 'What is the probability that revenue exceeds $12 million?'
	assert T.finish(dict(d, answer='perhaps'), FACTS)['checks']['structural'] == ['forecast gold must be yes or no']


def test_leaks_guessable_and_rounding_tol_for_prob():
	assert not C.leaks(fc('yes', prompt='Yes or no: will revenue exceed $12 million?'))
	assert not C.leaks(fc('no', context=CTX + ' The answer is no, no and no.'))
	assert V.guessable(fc()) and not V.guessable(fc(context=''))
	assert C.rounding_tol(fc(rel_tol=0.1)) == {}
	assert C.rounding_tol(mk_task(0, answer='12.5', kind='predict')) == {'rel_tol': C.PREDICT_TOL[0]}
	assert C.rounding_tol(drv('58.60')) == {'abs_tol': pytest.approx(0.005), 'rel_tol': 0.0}
	assert C.rounding_tol(drv('58')) == {}


def test_keep_by_code_derive_survives_critic_disagreement_only_with_panel_keep():
	t = drv('61.0', '10875 / (10875 + 6939) * 100')
	pk = {'panel': {'verdict': 'keep'}}
	assert T.keep(t | {'checks': checks(critic={'agrees': False}, **pk)})
	assert not T.keep(t | {'checks': checks(critic={'agrees': False})})  # no panel yet: needs the critic's agreement like any task
	assert T.keep(t | {'checks': checks()})
	for v in ('saturated', 'gold_disputed', 'unmeasured', 'capped'):
		assert not T.keep(t | {'checks': checks(critic={'agrees': False}, panel={'verdict': v})})
	assert not T.keep(t | {'checks': checks(critic={'agrees': False, 'unambiguous': False}, **pk)})
	assert not T.keep(t | {'checks': checks(critic={'agrees': False, 'gold_supported': False}, **pk)})
	assert not T.keep(t | {'checks': checks(critic={'well_posed': False}, **pk)})
	assert not T.keep(t | {'checks': checks(critic={'answer_in_context': True}, **pk)})


def test_keep_without_expression_still_needs_agreement():
	assert T.keep(drv('61.0') | {'checks': checks()})
	assert not T.keep(drv('61.0') | {'checks': checks(critic={'agrees': False})})
	assert not T.keep(mk_task(0, kind='predict', context=CTX, expression='1 + 1', checks=checks(critic={'agrees': False})))  # predict is never by-code
	assert not T.keep(mk_task(0, expression='1 + 1', checks=checks(critic={'agrees': False})))  # browse task: no context


@pytest.mark.parametrize(
	'verdict,kept', [('keep', True), ('saturated', False), ('gold_disputed', False), ('unmeasured', False), ('no_shift', False), ('memorized', False), ('capped', False)]
)
def test_keep_requires_panel_keep(verdict, kept):
	assert T.keep(drv('61.0', '10875 / (10875 + 6939) * 100') | {'checks': checks(panel={'verdict': verdict})}) is kept


def test_keep_prob_task_with_skipped_closed_book():
	assert T.keep(fc() | {'checks': checks(closed_book='skipped', panel={'verdict': 'keep'})})
	assert not T.keep(fc() | {'checks': checks(closed_book='skipped')})  # no panel: memorization and forecastability unchecked
	assert not T.keep(fc() | {'checks': checks(closed_book={'haiku': 'error: boom'})})


def test_cap_all_fail_caps_by_id():
	ts = [drv('61.0', '10875 / (10875 + 6939) * 100', id=f't{i}', checks=checks(panel={'verdict': 'keep', 'all_fail': i < 4})) for i in range(10)]
	assert V.cap_all_fail(ts) == 3  # 6 kept non-all-fail, cap 20%: floor(0.25 * 6) = 1 all-fail task stays
	assert [t['checks']['panel']['verdict'] for t in ts[:4]] == ['keep', 'capped', 'capped', 'capped']
	assert sum(T.keep(t) for t in ts) == 7
	assert V.cap_all_fail(ts) == 3 and sum(T.keep(t) for t in ts) == 7


@pytest.mark.parametrize(
	'pred,gold,score',
	[
		('0.7', 'yes', 0.91),
		('70%', 'yes', 0.91),
		('70 %', 'no', 0.51),
		('70', 'yes', 0.91),
		('1', 'yes', 1.0),
		('0', 'no', 1.0),
		('100%', 'no', 0.0),
		('Yes.', 'yes', 1.0),
		('no', 'yes', 0.0),
		('TRUE', 'no', 0.0),
		('False', 'no', 1.0),
		('0.5', 'Yes', 0.75),
		('Probability: 0.2', 'no', 0.96),
		('0.6 or 0.7', 'yes', 0.0),
		('between 60% and 70%', 'yes', 0.0),
		('yes or no', 'yes', 0.0),
		('probably', 'yes', 0.0),
		('', 'yes', 0.0),
		('150%', 'yes', 0.0),
		('-0.2', 'no', 0.0),
		('0.7, i.e. 0.70', 'yes', 0.91),
	],
)
def test_prob_score(pred, gold, score):
	assert prob_score(pred, gold) == pytest.approx(score)


def test_grade_answer_prob_returns_float():
	t = fc()
	assert grade_answer(t, None) == 0.0 and isinstance(grade_answer(t, None), float)
	assert grade_answer(t, '   ') == 0.0 and grade_answer(t, '0.9') == pytest.approx(0.99)
	assert grade_answer(drv('61.0'), None) is False


def test_grader_selftest_clean_on_prob_tasks():
	ts = [fc('yes', id='p0'), fc('no', id='p1'), fc('yes', id='p2', aliases=['true'])]
	assert grader_selftest(ts) == {}


def test_grader_selftest_clean_on_mixed_env():
	ts = [fc('yes', id='p0'), drv('100', '10875 / 10875 * 100', id='d0'), drv('61.0', '10875 / (10875 + 6939) * 100', id='d1', abs_tol=0.05)]
	assert grader_selftest(ts) == {}
	assert grader_selftest([fc('yes', id='p0'), drv('1', '10875 / 10875', id='d0'), drv('0', '10875 - 10875', id='d1')]) == {}


def test_grader_selftest_still_checks_foreign_golds_for_other_types():
	ts = [mk_task(0, answer='Acme', answer_type='contains'), mk_task(1, answer='Acme Corp', answer_type='exact'), fc('yes', id='p0'), fc('no', id='p1')]
	assert grader_selftest(ts) == {'other_gold_passes': ['t0']}  # a lenient contains grader is still caught; yes/no forecasts are not


def test_critique_passes_expression_and_scores_prob_agreement(monkeypatch):
	seen = []

	async def call(model, system, user, schema, **kw):
		seen.append(user)
		if schema is V.Recompute:
			return V.Recompute(calculation='c', recomputed_answer='no' if 'exceed' in user else '60.9'), {}, model
		return schema(unambiguous=True, gold_supported=True, realistic=True, issues='', answer_in_context=False, well_posed=True), {}, model

	monkeypatch.setattr(V, 'call', call)
	d, p = drv('61.0', '10875 / (10875 + 6939) * 100', abs_tol=0.05), fc('yes')
	asyncio.run(V.critique([d, p], FACTS))
	assert any('expression evaluated by code: 10875 / (10875 + 6939) * 100' in u for u in seen)
	assert d['checks']['critic']['agrees'] is False and not T.keep(d)  # 60.9 disagrees: kept only once the panel keeps it
	d['checks']['panel'] = {'verdict': 'keep'}
	assert T.keep(d)  # the gold is correct by construction
	assert p['checks']['critic']['agrees'] is False  # a 'no' recompute scores 0 against a 'yes' gold


def test_critique_agrees_is_a_bool_threshold(monkeypatch):
	async def call(model, system, user, schema, **kw):
		if schema is V.Recompute:
			return V.Recompute(calculation='c', recomputed_answer='0.8' if 'exceed' in user else '61.0'), {}, model
		return schema(unambiguous=True, gold_supported=True, realistic=True, issues='', answer_in_context=False, well_posed=True), {}, model

	monkeypatch.setattr(V, 'call', call)
	d, y, n = drv('61.0', '10875 / (10875 + 6939) * 100', abs_tol=0.05), fc('yes'), fc('no')
	asyncio.run(V.critique([d, y, n], FACTS))
	assert d['checks']['critic']['agrees'] is True and y['checks']['critic']['agrees'] is True and n['checks']['critic']['agrees'] is False  # 0.96 vs 0.36


def _panel(monkeypatch, answer, adjudication=None):
	async def call(model, system, user, schema, **kw):
		if schema is V.Adjudication:
			return V.Adjudication(**adjudication, panel_error='x'), {}, model
		return R.Reasoned(reasoning='r', answer=answer(model, '(materials withheld)' in user)), {}, model

	monkeypatch.setattr(V, 'call', call)
	monkeypatch.setattr(R, 'call', call)


def test_panel_verdicts_for_derive(monkeypatch):
	t = drv('61.0', '10875 / (10875 + 6939) * 100', abs_tol=0.05, checks=checks())
	_panel(monkeypatch, lambda m, b: '61.0')
	asyncio.run(V.panel([t], FACTS, models=['haiku', 'sonnet46', 'gpt6', 'gemini'], adjudicators=['gpt6', 'gemini', 'sonnet46']))
	assert t['checks']['panel']['verdict'] == 'saturated'
	_panel(monkeypatch, lambda m, b: '64.1', {'gold_correct': True, 'ambiguous': False})
	asyncio.run(V.panel([t], FACTS, models=['haiku', 'sonnet46', 'gpt6', 'gemini'], adjudicators=['gpt6', 'gemini', 'sonnet46']))
	p = t['checks']['panel']
	assert p['verdict'] == 'keep' and p['all_fail'] and p['adjudication']['model'] == 'gpt6' and T.keep(t)
	_panel(monkeypatch, lambda m, b: '64.1', {'gold_correct': True, 'ambiguous': True})
	asyncio.run(V.panel([t], FACTS, models=['haiku', 'sonnet46', 'gpt6', 'gemini'], adjudicators=['gpt6', 'gemini', 'sonnet46']))
	assert t['checks']['panel']['verdict'] == 'gold_disputed' and not T.keep(t)
	_panel(monkeypatch, lambda m, b: '61.0' if m in ('haiku', 'gpt6') else f'{len(m)}')  # different wrong answers: no adjudication needed
	asyncio.run(V.panel([t], FACTS, models=['haiku', 'sonnet46', 'gpt6', 'gemini'], adjudicators=['gpt6', 'gemini', 'sonnet46']))
	assert t['checks']['panel']['verdict'] == 'keep' and not t['checks']['panel']['all_fail'] and 'adjudication' not in t['checks']['panel']


def test_panel_prob_shift(monkeypatch):
	t = fc(checks=checks(closed_book='skipped'))
	_panel(monkeypatch, lambda m, b: '0.5' if b else '0.7')
	asyncio.run(V.panel([t], FACTS, models=['haiku', 'sonnet46', 'gpt6', 'gemini'], adjudicators=['gpt6', 'gemini', 'sonnet46']))
	assert t['checks']['panel']['shift'] == pytest.approx(0.2) and t['checks']['panel']['verdict'] == 'keep' and T.keep(t)
	_panel(monkeypatch, lambda m, b: '0.6')
	asyncio.run(V.panel([t], FACTS, models=['haiku', 'sonnet46', 'gpt6', 'gemini'], adjudicators=['gpt6', 'gemini', 'sonnet46']))
	assert t['checks']['panel']['verdict'] == 'no_shift' and not T.keep(t)
	_panel(monkeypatch, lambda m, b: '0.7')  # a 0.7 lean without the materials is a prior, not memory
	asyncio.run(V.panel([t], FACTS, models=['haiku', 'sonnet46', 'gpt6', 'gemini'], adjudicators=['gpt6', 'gemini', 'sonnet46']))
	assert t['checks']['panel']['blind_score'] == pytest.approx(0.91) and t['checks']['panel']['verdict'] == 'no_shift'
	_panel(monkeypatch, lambda m, b: '0.9')
	asyncio.run(V.panel([t], FACTS, models=['haiku', 'sonnet46', 'gpt6', 'gemini'], adjudicators=['gpt6', 'gemini', 'sonnet46']))
	assert t['checks']['panel']['blind_score'] == pytest.approx(0.99) and t['checks']['panel']['verdict'] == 'memorized'
	_panel(monkeypatch, lambda m, b: '0.5' if b else 'yes')
	asyncio.run(V.panel([t], FACTS, models=['haiku', 'sonnet46', 'gpt6', 'gemini'], adjudicators=['gpt6', 'gemini', 'sonnet46']))
	assert t['checks']['panel']['verdict'] == 'saturated'


def test_scaled_operands_and_same_value_distractors():
	ctx = 'Net cash used in operating activities (in thousands) 4,564. Notes of $120.0 million.'
	assert C.expression_issue(drv('-0.49', '-4564000 / 9343771', context=ctx + ' Shares 9,343,771.')) is None  # 4,564 thousand
	assert C.expression_issue(drv('1920000', '120000 * 16', context=ctx + ' 16 shares per note.')) is None  # $120.0 million in thousands
	assert 'not in the materials' in C.expression_issue(drv('3373273', '30023814 - 26650541', context=ctx + ' 26,650,541.'))
	t = drv('1920000', '120000 * 16', context=ctx) | {'abs_tol': 0.5}
	assert C.distractor_issue(t | {'distractors': ['120000000 / 62.50']}) is None  # another route to the gold is not a mistake
	assert C.distractor_issue(t | {'distractors': ['120000 * 16.0000001']})  # a real mistake inside tolerance is flagged


def test_code_gold_replaces_a_drafted_arithmetic_slip():
	t = C.code_gold(drv('61.1', '10875 / (10875 + 6939) * 100'))
	assert t['answer'] == '61.0' and t['drafted_answer'] == '61.1' and C.expression_issue(t) is None
	assert 'drafted_answer' not in C.code_gold(drv('61.0', '10875 / (10875 + 6939) * 100'))  # right already: untouched
	assert C.code_gold(drv('150', '10875 / 6939 * 100'))['answer'] == '157'  # integer precision kept
	assert C.code_gold(drv('62', '10875 / 6939 * 100'))['answer'] == '62' and C.expression_issue(drv('62', '10875 / 6939 * 100'))  # not a slip: rejected, never rewritten
	assert C.code_gold(drv('12.6%', '(10875 - 9500) / 10875'))['answer'] == '12.6%'  # percent vs fraction
	assert C.code_gold(drv('3936', '6939 - 10875', prompt='Change (as a positive number)?'))['answer'] == '3936'  # stated sign convention
	assert C.code_gold(drv('5', '1 / 0'))['answer'] == '5' and C.code_gold(drv('5', '', kind='predict'))['answer'] == '5'  # nothing to compute from


def test_finish_uses_the_code_gold_and_its_tolerance():
	kb_facts = {'f0': {'url': 'https://sec.gov/a', 'value': '1', 'quote': 'q'}}
	t = T.finish(
		{
			'prompt': 'Share of arm A, in percent?',
			'kind': 'derive',
			'answer': '61.1',
			'answer_type': 'number',
			'fact_ids': ['f0'],
			'derivation': 'x',
			'expression': '10875 / (10875 + 6939) * 100',
			'context': 'Arm A 10,875; arm B 6,939.',
			'material_ids': ['m0'],
			'start_url': '',
			'case': 'c',
		},
		kb_facts,
	)
	assert t['answer'] == '61.0' and t['abs_tol'] == 0.05 and not t['checks']['structural']


def test_verified_pool_includes_tasks_the_panel_found_easy():
	t = drv('61.0', '10875 / (10875 + 6939) * 100') | {'checks': checks(panel={'verdict': 'saturated'})}
	assert not T.keep(t) and T.keep(t, verified=True)
	for v in ('multi_value', 'gold_disputed', 'memorized', 'no_shift', 'unmeasured'):
		assert not T.keep(t | {'checks': checks(panel={'verdict': v})}, verified=True)  # correctness or design failures stay out


def test_rounding_tol_follows_the_golds_own_precision_and_scale():
	assert C.rounding_tol(drv('12.5%', 'x'))['abs_tol'] == pytest.approx(0.05)
	assert C.rounding_tol(drv('4.5 million', 'x'))['abs_tol'] == pytest.approx(50000)
	assert C.rounding_tol(drv('$4.50', 'x'))['abs_tol'] == pytest.approx(0.005) and C.rounding_tol(drv('45', 'x'))['abs_tol'] == 0.5


def test_evaluate_refuses_runaway_expressions():
	assert C.evaluate('9**9**9') is None and C.evaluate('-' * 3000 + '1') is None and C.evaluate('1' + '+1' * 5000) is None and C.evaluate('2**10') == 1024


def test_task_grade_never_raises_on_a_malformed_gold():
	assert C.grade_answer({'answer_type': 'number', 'answer': '12 or 13'}, '12') is False and C.grade_answer({'answer_type': 'prob', 'answer': 'maybe'}, '0.5') == 0.0


def test_rescore_moves_verdicts_back_and_rejudges_forecasts():
	t = drv('6', '79 - 73', checks=checks(panel={'verdict': 'multi_value', 'answers': {'a': ['6 percentage points (from 73% to 79%)'], 'b': ['5']}, 'scores': {}}))
	V.rescore_panel(t)
	assert t['checks']['panel']['verdict'] == 'keep'  # the grader now reads the answer: one value with its working
	t = drv('6', '79 - 73', checks=checks(panel={'verdict': 'saturated', 'answers': {'a': ['5'], 'b': ['5']}, 'scores': {}}))
	V.rescore_panel(t)
	assert t['checks']['panel']['verdict'] == 'unmeasured'  # shared wrong answer: needs the adjudicator, so it is measured again
	f = fc(
		'yes', checks=checks(closed_book='skipped', panel={'verdict': 'memorized', 'answers': {'a': ['0.85'], 'b': ['0.6']}, 'blind': {'a': ['0.7'], 'b': ['0.7']}, 'scores': {}})
	)
	V.rescore_panel(f)
	assert f['checks']['panel']['verdict'] == 'keep' and f['checks']['panel']['blind_score'] == pytest.approx(0.91)  # a 0.7 prior is not memory


def test_second_review_task_cases():
	assert C.evaluate('1.05**30') == pytest.approx(4.3219, rel=1e-4) and C.evaluate('(1+0.005)**360') > 6
	t = C.code_gold(drv('4.4 million', '4.55 * 1.0'))
	assert t['answer'] == '4.5 million' or t['answer'] == '4.6 million'  # the number is replaced, its scale word kept
