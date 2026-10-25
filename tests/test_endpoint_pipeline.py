import json
from unittest.mock import AsyncMock

import httpx
import pytest

from rl_env_use import endpoint_pipeline as ep
from rl_env_use import pipeline
from rl_env_use.store import load_jsonl, save_jsonl


def source(tmp_path, role, text, verified=False, **extra):
	path = role + '.txt'
	(tmp_path / path).write_text(text)
	return dict(
		id=role,
		case='NCT00000001',
		role=role,
		text=text,
		url='https://example.org/' + role,
		capture=path,
		response_sha256=ep.digest(text),
		captured_at='2026-01-01',
		robots=[{'allowed': True}],
		prespecification_verified=verified,
		**extra,
	)


def fixture(tmp_path):
	registration = source(
		tmp_path,
		'registration',
		'NCT00000001 Primary pain at 6 weeks. ITT A minus B; superiority; adjusted alpha 0.025. One primary endpoint.',
		True,
		registered_endpoints=['Primary pain'],
	)
	results = source(tmp_path, 'results', 'NCT00000001 ITT A minus B at 6 weeks: pain difference -2, p=0.01.')
	bundle = {'trial': 'NCT00000001', 'id': 'b1', 'sources': [registration, results]}
	draft = ep.Assessment(
		question='Oppfylte studien sitt forhåndsspesifiserte primære endepunkt?',
		answer='Ja.',
		explanation='Det primære endepunktet ble oppfylt etter seks uker. Den rapporterte p-verdien er lavere enn den forhåndsspesifiserte grensen.',
		endpoints=[
			ep.Endpoint(
				registered_name='Primary pain',
				timepoint='6 uker',
				population='ITT',
				comparison='A minus B',
				metric='forskjell',
				hypothesis='superiority',
				success_rule='p<0.025',
				reported_result='-2; p=0.01',
				label='met',
				explanation='Grensen ble oppfylt.',
				registration_evidence=[ep.Citation(source_id='registration', quote='Primary pain at 6 weeks.')],
				rule_evidence=[ep.Citation(source_id='registration', quote='adjusted alpha 0.025.')],
				result_evidence=[ep.Citation(source_id='results', quote='pain difference -2, p=0.01.')],
			)
		],
		trial_success='met',
		overall_success_rule='Ett primært endepunkt.',
		overall_rule_evidence=[ep.Citation(source_id='registration', quote='One primary endpoint.')],
		limitations=[],
	)
	return bundle, draft


def review(**updates):
	return ep.EndpointReview(
		**(
			dict(
				supported=True,
				correct=True,
				language_ok=True,
				all_primary_endpoints_covered=True,
				prespecification_supported=True,
				endpoint_result_matching=True,
				statistical_rules_respected=True,
				blocking_issues=[],
				notes=[],
			)
			| updates
		)
	)


def test_prespecification_requires_review_and_strict_chronology():
	entry = dict(
		version_date='2010-01-01', first_results_date='2012-01-01', reviewed_by='curator', verification_note='Verified archived version against earliest results availability.'
	)
	assert ep.prespecified(entry)
	assert not ep.prespecified(entry | {'reviewed_by': ''})
	assert not ep.prespecified(entry | {'version_date': '2012-01-01'})
	assert not ep.prespecified(entry | {'version_date': 'invalid'})


def test_current_registry_cannot_establish_prespecification(tmp_path):
	b, d = fixture(tmp_path)
	b['sources'][0]['prespecification_verified'] = False
	assert 'missing reviewed pre-results registration version' in ep.eligibility(b)
	assert 'missing/invalid registration evidence' in ep.structural_assessment(d, b)
	assert 'missing verified success rule' in ep.structural_assessment(d, b)


def test_missing_results_not_trial_failure(tmp_path):
	b, d = fixture(tmp_path)
	b['sources'] = b['sources'][:1]
	assert ep.eligibility(b) == ['missing paired results']
	assert 'missing matching results citation' in ep.structural_assessment(d, b)


def test_exact_quotes_primary_coverage_and_overall_rule(tmp_path):
	b, d = fixture(tmp_path)
	assert ep.eligibility(b) == []
	assert ep.structural_assessment(d, b) == []
	d.endpoints[0].result_evidence[0].quote = 'pain difference 2, p=0.01.'
	assert 'unsupported quotation' in ep.structural_assessment(d, b)
	b['sources'][0]['registered_endpoints'].append('Co-primary survival')
	assert 'primary endpoint coverage mismatch' in ep.structural_assessment(d, b)
	d.overall_rule_evidence = []
	assert 'overall conclusion requires prespecified rule' in ep.structural_assessment(d, b)


def test_noninferiority_rule_cannot_come_from_results_paper(tmp_path):
	b, d = fixture(tmp_path)
	d.endpoints[0].hypothesis = 'noninferiority'
	d.endpoints[0].rule_evidence = [ep.Citation(source_id='results', quote='p=0.01.')]
	assert 'missing verified success rule' in ep.structural_assessment(d, b)


def test_mixed_summary_is_separate_from_overall_success(tmp_path):
	_, d = fixture(tmp_path)
	row = d.model_dump()
	row['endpoints'].append(row['endpoints'][0] | {'label': 'not_met'})
	assert ep.endpoint_summary(row) == 'mixed'
	row['endpoints'][1]['label'] = 'insufficient_evidence'
	assert ep.endpoint_summary(row) == 'insufficient_evidence'


@pytest.mark.asyncio
@pytest.mark.parametrize('blocking,accepted', [([], 1), (['Fix Norwegian agreement'], 0)])
async def test_critic_corrections_block_export_and_reasoning_stays_separate(tmp_path, monkeypatch, blocking, accepted):
	b, d = fixture(tmp_path)
	save_jsonl(tmp_path / 'endpoint-bundles.jsonl', [b])

	async def fake(model, system, user, schema, **kwargs):
		if schema is ep.Assessment:
			return d, {}, 'writer'
		from rl_env_use.reasoning import REASONING_RECORDS

		REASONING_RECORDS.get().append({'reasoning': 'Independent raw critic reasoning.', 'field': 'reasoning_content'})
		return review(blocking_issues=blocking), {}, 'critic'

	monkeypatch.setattr(ep, 'call', fake)
	spec = dict(language='nob', localize_diagnostics=False, target_tasks=10)
	assert await ep.synthesize_endpoints(tmp_path, spec, 'writer', 'critic') == accepted
	assert not (tmp_path / 'dataset-sft.jsonl').exists()
	if accepted:
		row = load_jsonl(tmp_path / 'dataset-endpoints.jsonl')[0]
		assert row['critic_reasoning'] == 'Independent raw critic reasoning.'
		assert 'Independent raw critic reasoning.' not in json.dumps(row['messages'])
		chat = load_jsonl(tmp_path / 'dataset-endpoints.chat.jsonl')[0]
		assert list(chat) == ['messages']
		assert 'Primary pain' in chat['messages'][1]['content']
		# Export rechecks the raw evidence, not just the old acceptance flag.
		(tmp_path / 'results.txt').write_text('tampered')
		assert ep.export_endpoint(tmp_path) == 0
	# Reviewed candidates are checkpointed; no repeated model calls on resume.
	monkeypatch.setattr(ep, 'call', AsyncMock(side_effect=AssertionError('must not rerun')))
	await ep.synthesize_endpoints(tmp_path, spec, 'writer', 'critic')


@pytest.mark.asyncio
async def test_missing_evidence_is_queued_without_model_calls(tmp_path, monkeypatch):
	b, _ = fixture(tmp_path)
	b['sources'] = b['sources'][:1]
	save_jsonl(tmp_path / 'endpoint-bundles.jsonl', [b])
	mock = AsyncMock(side_effect=AssertionError('ineligible trial must not be drafted'))
	monkeypatch.setattr(ep, 'call', mock)
	assert await ep.synthesize_endpoints(tmp_path, {'language': 'nob'}, 'w', 'c') == 0
	assert load_jsonl(tmp_path / 'endpoint-needs-evidence.jsonl')[0]['issues'] == ['missing paired results']
	mock.assert_not_called()


@pytest.mark.asyncio
async def test_mode_dispatch_does_not_enter_rl_or_general_sft(tmp_path, monkeypatch):
	mock = AsyncMock(return_value={'mode': 'endpoint-assessment', 'kept': 0})
	monkeypatch.setattr(ep, 'build_endpoints', mock)
	monkeypatch.setattr(pipeline, 'use_browser_spec', lambda s: None)
	spec = {'mode': 'endpoint-assessment', 'language': 'nob'}
	assert (await pipeline.create(tmp_path, spec))['mode'] == 'endpoint-assessment'
	assert json.loads(await pipeline.build(tmp_path, spec))['mode'] == 'endpoint-assessment'
	assert mock.await_count == 2


@pytest.mark.asyncio
async def test_manifest_download_uses_robots_and_retains_snapshot(tmp_path):
	from rl_env_use.robots import RobotsGuard

	async def handler(req):
		if req.url.path == '/robots.txt':
			return httpx.Response(200, text='User-agent: *\nDisallow: /blocked\n')
		return httpx.Response(200, text='NCT00000001 Historical registration: Primary pain.', headers={'Content-Type': 'text/plain'})

	async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
		guard = RobotsGuard(tmp_path, client, 0)
		entry = dict(
			url='https://example.org/protocol',
			role='registration',
			version_date='2010-01-01',
			first_results_date='2012-01-01',
			reviewed_by='curator',
			verification_note='Archived registration confirmed.',
		)
		s = await ep.fetch_source(guard, tmp_path, entry, 'NCT00000001')
		assert ep.allowed(tmp_path, s)
		assert s['prespecification_verified']
		with pytest.raises(PermissionError):
			await ep.fetch_source(guard, tmp_path, entry | {'url': 'https://example.org/blocked'}, 'NCT00000001')


@pytest.mark.asyncio
async def test_pairing_full_registry_and_manifest_is_resumable(tmp_path, monkeypatch):
	registry = {
		'protocolSection': {'identificationModule': {'nctId': 'NCT00000001'}, 'outcomesModule': {'primaryOutcomes': [{'measure': 'Pain'}]}},
		'resultsSection': {'outcomeMeasuresModule': {'outcomeMeasures': [{'type': 'PRIMARY', 'analyses': [{'pValue': '0.01'}]}]}},
	}
	text = json.dumps(registry)
	raw = source(tmp_path, 'current_registry', text)
	raw['source_kind'] = 'registry'
	save_jsonl(tmp_path / 'sft-documents.jsonl', [raw])

	async def handler(req):
		if req.url.path == '/robots.txt':
			return httpx.Response(200, text='User-agent: *\nAllow: /')
		return httpx.Response(200, text=text, headers={'Content-Type': 'application/json'})

	factory = httpx.AsyncClient
	client = factory(transport=httpx.MockTransport(handler))
	monkeypatch.setattr(ep.httpx, 'AsyncClient', lambda **kwargs: client)
	spec = {
		'endpoint_cases': {
			'NCT00000001': {
				'sources': [
					dict(
						url='https://example.org/history',
						role='registration',
						version_date='2010-01-01',
						first_results_date='2012-01-01',
						reviewed_by='curator',
						verification_note='Verified historical version',
					)
				]
			}
		}
	}
	b = (await ep.pair_sources(tmp_path, spec))[0]
	assert not ep.eligibility(b)
	assert 'analyses' in b['sources'][0]['text']
	assert any(s.get('registered_endpoints') == ['Pain'] and s['prespecification_verified'] for s in b['sources'])

	# New client, same cached historical source: no redownload, stable bundle identity.
	async def no_network(req):
		raise AssertionError('cached source should not be downloaded again')

	client = factory(transport=httpx.MockTransport(no_network))
	again = (await ep.pair_sources(tmp_path, spec))[0]
	assert b['id'] == again['id']


@pytest.mark.asyncio
async def test_current_api_cannot_be_relabeled_as_historical_registration(tmp_path):
	class Downloader:
		async def get(self, url):
			return httpx.Response(200, request=httpx.Request('GET', url), text='NCT00000001 Current registration.', headers={'Content-Type': 'text/plain'}), [{'allowed': True}]

	entry = dict(
		url='https://clinicaltrials.gov/api/v2/studies/NCT00000001',
		role='registration',
		version_date='2010-01-01',
		first_results_date='2012-01-01',
		reviewed_by='curator',
		verification_note='A claimed old date cannot turn the current API into an archive.',
	)
	result = await ep.fetch_source(Downloader(), tmp_path, entry, 'NCT00000001')
	assert not result['prespecification_verified']


@pytest.mark.asyncio
async def test_pdf_text_and_raw_bytes_retained(tmp_path, monkeypatch):
	from io import BytesIO

	from pypdf import PdfWriter

	writer = PdfWriter()
	writer.add_blank_page(width=100, height=100)
	buffer = BytesIO()
	writer.write(buffer)

	class Downloader:
		async def get(self, url):
			return httpx.Response(200, request=httpx.Request('GET', url), content=buffer.getvalue(), headers={'Content-Type': 'application/pdf'}), [{'allowed': True}]

	from pypdf._page import PageObject

	monkeypatch.setattr(PageObject, 'extract_text', lambda self: 'NCT00000001 Registered primary endpoint: pain.')
	result = await ep.fetch_source(Downloader(), tmp_path, {'url': 'https://example.org/paper.pdf', 'role': 'registration'}, 'NCT00000001')
	assert result['text'] == 'NCT00000001 Registered primary endpoint: pain.'
	assert (tmp_path / result['capture']).read_bytes() == buffer.getvalue()
	assert ep.allowed(tmp_path, result)


@pytest.mark.asyncio
async def test_scout_populates_evidence_queue_without_model_calls(tmp_path, monkeypatch):
	b, _ = fixture(tmp_path)
	b['sources'] = b['sources'][:1]
	monkeypatch.setattr(ep, 'collect', AsyncMock())
	monkeypatch.setattr(ep, 'pair_sources', AsyncMock(return_value=[b]))
	call = AsyncMock(side_effect=AssertionError('scout must not call models'))
	monkeypatch.setattr(ep, 'call', call)
	result = await ep.build_endpoints(tmp_path, {'language': 'nob', 'target_tasks': 100}, 'w', 'c', 'scout')
	assert result['status'] == 'needs_evidence'
	assert load_jsonl(tmp_path / 'endpoint-needs-evidence.jsonl')[0]['issues'] == ['missing paired results']
	call.assert_not_called()


def test_cli_new_and_export_route_to_separate_endpoint_artifacts(tmp_path, monkeypatch):
	from rl_env_use.__main__ import main

	monkeypatch.chdir(tmp_path)
	main(['new', 'Endpoints', '--topic', 'primary endpoints', '--mode', 'endpoint-assessment', '--language', 'nob'])
	env = tmp_path / 'envs' / 'endpoints'
	assert json.loads((env / 'spec.json').read_text())['mode'] == 'endpoint-assessment'
	main(['export', str(env)])
	assert (env / 'dataset-endpoints.jsonl').exists()
	assert not (env / 'dataset-sft.jsonl').exists()
	assert not (env / 'dataset.jsonl').exists()


def test_cli_refuses_repurposing_running_general_dataset(tmp_path):
	from argparse import Namespace

	from rl_env_use.__main__ import cmd_build

	spec = {'mode': 'sft', 'language': 'nob'}
	(tmp_path / 'spec.json').write_text(json.dumps(spec))
	save_jsonl(tmp_path / 'sft-candidates.jsonl', [{'id': 'existing'}])
	with pytest.raises(SystemExit, match='new environment'):
		cmd_build(Namespace(env=str(tmp_path), only=None, mode='endpoint-assessment', language=None, localize_diagnostics=None))
	assert json.loads((tmp_path / 'spec.json').read_text()) == spec
