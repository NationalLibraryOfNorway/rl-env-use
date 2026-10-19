import pytest
from conftest import mk_task

from rl_env_use.checks import leaks, structural_issues
from rl_env_use.tasks import apply_reviews, keep, task_id

OK = {'structural': [], 'leakage': False, 'live': True, 'closed_book': {}, 'critic': {'unambiguous': True, 'gold_supported': True, 'agrees': True}}


def ck(**kw):
	return {**OK, **kw}


FACTS = {'f0': {'quote': 'collected from 137 websites spanning 31 domains', 'url': 'u'}, 'f1': {'quote': 'task success rate of 14.41%', 'url': 'u'}}


def test_structural_clean():
	assert structural_issues(mk_task(0, fact_ids=['f0']), FACTS) == []


def test_structural_catches_unsupported_gold_and_missing_facts():
	t = mk_task(0, answer='138', fact_ids=['f0', 'nope'])
	issues = structural_issues(t, FACTS)
	assert any('unknown facts' in i for i in issues) and any('not literally' in i for i in issues)


def test_structural_allows_derivations_and_checks_types():
	assert structural_issues(mk_task(0, answer='63.83', fact_ids=['f1'], derivation='78.24 - 14.41'), FACTS) == []
	assert 'date gold does not parse' in structural_issues(mk_task(0, answer='soon', answer_type='date', fact_ids=['f0'], derivation='x'), FACTS)
	assert 'start_url not http(s)' in structural_issues(mk_task(0, fact_ids=['f0'], start_url='arxiv.org'), FACTS)


def test_search_engine_evidence_is_structural_issue():
	facts = {'f0': {'quote': 'collected from 137 websites', 'url': 'https://duckduckgo.com/?q=mind2web'}}
	assert 'evidence comes from a search-engine results page' in structural_issues(mk_task(0, fact_ids=['f0']), facts)


def test_leakage():
	assert leaks(mk_task(0, prompt='Mind2Web uses 137 websites; how many?'))
	assert not leaks(mk_task(0, prompt='How many websites does Mind2Web use?'))
	assert not leaks(mk_task(0, prompt='Of the 1370 tasks, how many websites?'))
	assert leaks(mk_task(0, answer='Shuyan Zhou', answer_type='exact', prompt='Is Shuyan Zhou the first author?'))
	assert leaks(mk_task(0, answer='Reddit, GitLab', answer_type='set', prompt='Which sites besides GitLab?'))


def test_keep_filters():
	assert keep(mk_task(0))
	assert not keep(mk_task(0, checks=ck(structural=['x'])))
	assert not keep(mk_task(0, checks=ck(live=False)))
	assert not keep(mk_task(0, checks=ck(closed_book={'gpt': True})))
	assert keep(mk_task(0, checks=ck(closed_book={'gpt': True})), allow_memorizable=True)
	assert not keep(mk_task(0, checks=ck(critic={'unambiguous': False, 'gold_supported': True})))
	assert not keep(mk_task(0, review={'verdict': 'reject'}))
	assert not keep(mk_task(0, checks=ck(closed_book={'gpt': 'error: x'})))  # errored check fails closed


def test_task_id_stable_under_formatting():
	assert task_id('How many  websites?') == task_id('how many websites') != task_id('how many domains')


def test_leakage_review_findings():
	assert leaks(mk_task(0, answer='Paris', answer_type='exact', prompt='Is the capital Paris or Lyon?'))  # multiple choice
	assert leaks(mk_task(0, answer='Paris', answer_type='exact', prompt='It is not Paris, so which city?'))
	assert leaks(mk_task(0, answer='2021-05-05', answer_type='date', prompt='What happened on May 5, 2021?'))
	assert not leaks(mk_task(0, answer='2021-05-05', answer_type='date', prompt='When was v1 submitted?'))
	assert not leaks(mk_task(0, answer='N/A', answer_type='number', prompt='In 2023 how many?'))  # no crash on bad gold


def test_number_gold_must_be_single_number():
	assert 'number gold must contain exactly one number' in structural_issues(mk_task(0, answer='Q3 2023: 4.5 billion', fact_ids=[], derivation='x'), {})


def test_gold_fix_clears_aliases_and_rechecks():
	t = mk_task(0, answer='wrong', answer_type='exact', aliases=['wrong2'], prompt='Who is it?')
	(t2,) = apply_reviews([t], [{'id': 't0', 'verdict': 'accept', 'gold_fix': 'right'}])
	assert t2['answer'] == 'right' and t2['aliases'] == [] and t2['gold_source'] == 'human-corrected'
	(t3,) = apply_reviews([mk_task(0, prompt='Who is right?', answer_type='exact')], [{'id': 't0', 'verdict': 'accept', 'gold_fix': 'right'}])
	assert t3['checks']['leakage'] and not keep(t3)
	(t4,) = apply_reviews([mk_task(0)], [{'id': 't0', 'verdict': 'accept', 'gold_fix': 'many'}])
	assert t4['checks']['structural'] and not keep(t4)


def test_keep_requires_independent_recomputation_to_agree():
	crit = {'unambiguous': True, 'gold_supported': True, 'realistic': True, 'issues': ''}
	assert keep(mk_task(0, checks=ck(critic=crit | {'agrees': True})))
	assert not keep(mk_task(0, checks=ck(critic=crit | {'agrees': False})))


def test_browse_review_is_triage_not_filter():
	br = {'answer': '20000', 'other_defensible_answers': ['10000'], 'notes': '', 'agrees': False, 'alternatives': ['20000']}
	assert keep(mk_task(0, checks=ck(browse_review=br)))


def test_keep_fails_closed_when_unverified():
	assert not keep(mk_task(0, checks={'structural': [], 'leakage': False}))  # synthesized but never verified
	assert keep(mk_task(0, checks={'structural': [], 'leakage': False, 'live': 'skipped', 'critic': 'skipped', 'closed_book': 'skipped'}))  # explicit skips


def test_gold_fix_keeps_real_issues_and_rescores():
	t = mk_task(
		0,
		checks=ck(
			structural=['evidence comes from a search-engine results page'],
			closed_book={'haiku': False},
			closed_book_answers={'haiku': '138'},
			critic={'unambiguous': True, 'gold_supported': True, 'recomputed_answer': '138', 'agrees': False},
		),
	)
	(t2,) = apply_reviews([t], [{'id': 't0', 'verdict': 'accept', 'gold_fix': '138'}])
	assert t2['checks']['structural'] == ['evidence comes from a search-engine results page']
	assert t2['checks']['closed_book'] == {'haiku': True} and t2['checks']['critic']['agrees'] is True


def test_version_like_golds_and_loose_tolerance():
	assert 'tolerance too loose (rel_tol > 5%)' in structural_issues(mk_task(0, rel_tol=0.2, fact_ids=[], derivation='x'), {})


def test_rounding_tolerance_accepts_precision_not_neighbours():
	from rl_env_use.checks import rounding_tol
	from rl_env_use.graders import grade_answer

	t = mk_task(0, answer='97.6', derivation='1180000 / 12087')
	t.update(rounding_tol(t))
	assert t['abs_tol'] == pytest.approx(0.05) and t['rel_tol'] == 0.0
	assert grade_answer(t, '97.63') and grade_answer(t, '97.6') and not grade_answer(t, '98.3') and not grade_answer(t, '98')
	assert rounding_tol(mk_task(0, answer='603', derivation='x')) == {} and rounding_tol(mk_task(0, answer='5.43')) == {}


def test_prompt_naming_the_answer_record_id_is_structural():
	facts = {'f1': {'quote': 'Domain 155-461 Peptidase S8', 'url': 'https://www.uniprot.org/uniprotkb/Q8NBP7/entry#family_and_domains'}}
	t = {
		'prompt': 'In the UniProt entry for PCSK9 (Q8NBP7), where is the Peptidase S8 domain?',
		'answer': '155-461',
		'answer_type': 'exact',
		'fact_ids': ['f1'],
		'start_url': 'https://www.uniprot.org/',
	}
	assert any('record id' in i for i in structural_issues(t, facts))
	assert not any('record id' in i for i in structural_issues(t | {'prompt': 'Where is the Peptidase S8 domain of human PCSK9?'}, facts))


def test_which_of_question_is_not_a_leak_but_needs_exact_grading():
	t = mk_task(0, answer='TP53', answer_type='exact', prompt='In gnomAD v4, which of VHL, TP53 or NFE2L2 has the lowest pLoF LOEUF?', choices=['VHL', 'TP53', 'NFE2L2'])
	assert not leaks(t)
	assert leaks(t | {'answer_type': 'contains'}) and leaks(t | {'choices': []}) and leaks(t | {'choices': ['TP53', 'KRAS']})
	from rl_env_use.graders import grade_answer

	assert grade_answer(t, 'TP53') and not grade_answer(t, 'TP53 or VHL')


def test_gene_symbols_are_not_record_ids():
	facts = {'f1': {'quote': 'x y z 0.42', 'url': 'https://gtexportal.org/home/gene/BRCA1'}}
	t = {'prompt': 'What is X for BRCA1 in GTEx?', 'answer': '0.42', 'answer_type': 'number', 'fact_ids': ['f1'], 'start_url': 'https://gtexportal.org/'}
	assert not any('record id' in i for i in structural_issues(t, facts))
	facts['f1']['url'] = 'https://www.ncbi.nlm.nih.gov/clinvar/variation/17661/'
	assert any('record id' in i for i in structural_issues(t | {'prompt': 'What is X for variation 17661?'}, facts))


def test_numbers_hyphen_after_word_is_a_separator():
	from rl_env_use.graders import numbers

	assert numbers('Top-5') == [5.0] and numbers('GPT-4') == [4.0] and numbers('-3 degrees') == [-3.0]


def test_years_are_not_record_ids_but_accessions_are():
	from rl_env_use.checks import ACCESSION

	assert not ACCESSION.fullmatch('2024') and not ACCESSION.fullmatch('BRCA1') and not ACCESSION.fullmatch('TP53')
	assert all(ACCESSION.fullmatch(x) for x in ('Q8NBP7', 'P38398', '17661', 'ENSG00000139618', 'rs2017481'))


def test_two_char_values_can_be_evidence():
	from rl_env_use.browse import quote_in

	assert quote_in('US', 'The US has 50 states') and quote_in('5', 'has 5 exons') and not quote_in('a', 'a b c')


def test_errored_checks_are_retried_real_failures_are_not():
	from rl_env_use.pipeline import unverified

	ok = mk_task(0)
	assert not unverified(ok)
	assert unverified(mk_task(1, checks={**ok['checks'], 'live': None}))
	assert unverified(mk_task(2, checks={**ok['checks'], 'closed_book': {'haiku': 'error: 529'}}))
	assert unverified(mk_task(3, checks={**ok['checks'], 'critic': {'unambiguous': False, 'gold_supported': False, 'issues': 'critic error: timeout'}}))
	assert not unverified(mk_task(4, checks={**ok['checks'], 'live': False}))
	assert not unverified(mk_task(5, checks={**ok['checks'], 'closed_book': 'skipped', 'live': 'skipped'}))
