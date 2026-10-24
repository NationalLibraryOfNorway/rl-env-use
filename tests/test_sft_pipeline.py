import json
from unittest.mock import AsyncMock

import httpx
import pytest

from rl_env_use import pipeline
from rl_env_use.robots import RobotsGuard
from rl_env_use.sft_pipeline import Critic, Draft, Drafts, collect, export_accepted, registry_document, structural, synthesize_sft
from rl_env_use.store import load_jsonl, save_jsonl


def record(country='Norway'):
	return {
		'protocolSection': {'identificationModule': {'nctId': 'NCT00000001'}, 'contactsLocationsModule': {'locations': [{'country': country}]}},
		'resultsSection': {'outcomeMeasuresModule': {'outcomeMeasures': [{'type': 'PRIMARY', 'title': 'Pain', 'timeFrame': '6 weeks', 'analyses': [{'pValue': '0.78'}]}]}},
	}


@pytest.mark.asyncio
async def test_registry_capture_preserves_complete_outcome_and_rejects_wrong_country(tmp_path):
	raw = json.dumps(record())
	transport = httpx.MockTransport(lambda r: httpx.Response(200, text='User-agent: *\nAllow: /' if r.url.path == '/robots.txt' else raw))
	async with httpx.AsyncClient(transport=transport) as client:
		d = await registry_document(RobotsGuard(tmp_path, client, 0), tmp_path, 'NCT00000001', ['cached'])
		assert '"pValue": "0.78"' in d['text'] and d['primary_outcomes'] == 1
		assert (tmp_path / d['capture']).read_text() == raw
	transport = httpx.MockTransport(lambda r: httpx.Response(200, text='User-agent: *\nAllow: /') if r.url.path == '/robots.txt' else httpx.Response(200, json=record('Sweden')))
	async with httpx.AsyncClient(transport=transport) as client:
		with pytest.raises(ValueError, match='Norwegian'):
			await registry_document(RobotsGuard(tmp_path, client, 0), tmp_path, 'NCT00000001', [])


def test_structural_preserves_negative_sign_and_enforces_longer_explanation_limit():
	d = Draft(
		task_type='result_interpretation',
		question='Hva forteller konfidensintervallet om resultatet?',
		answer='Ikke dokumentert.',
		explanation='Resultatet viser ikke en statistisk sikker effekt. ' * 3,
		quotes=['-0.22'],
	)
	assert not structural(d, {'text': 'The confidence limit is -0.22.'}, {})
	assert 'unsupported quotation' in structural(d, {'text': 'The confidence limit is 0.22.'}, {})
	assert 'explanation word limit' in structural(d, {'text': '-0.22'}, {'sft_explanation_max_words': 3})


@pytest.mark.asyncio
async def test_sft_is_separate_from_rl_and_checkpoints_critic_rejections(tmp_path, monkeypatch):
	import rl_env_use.sft_pipeline as s

	save_jsonl(
		tmp_path / 'sft-documents.jsonl',
		[{'id': 'd1', 'case': 'trial1', 'text': 'Primary pain at 6 weeks.', 'robots_allowed': True, 'capture': 'raw.html', 'url': 'https://example.org/trial'}],
	)
	draft = Draft(
		task_type='endpoint_extraction',
		question='Hva er det primære endepunktet og tidspunktet for målingen?',
		answer='Smerte etter seks uker.',
		explanation='Kilden oppgir smerte som primært endepunkt. Målingen gjelder seks uker, ikke et annet oppfølgingstidspunkt.',
		quotes=['Primary pain at 6 weeks.'],
	)
	calls = []

	async def fake(model, system, user, schema, **kwargs):
		calls.append(schema)
		if schema is Drafts:
			return Drafts(examples=[draft]), {}, 'writer-id'
		return Critic(supported=False, correct=False, well_posed=True, language_ok=True, issues=['Unsupported']), {}, 'critic-id'

	monkeypatch.setattr(s, 'call', fake)
	spec = {'language': 'nob', 'localize_diagnostics': False, 'tasks_per_case': 1}
	assert await synthesize_sft(tmp_path, spec, 'writer', 'critic') == 0
	assert not (tmp_path / 'tasks.jsonl').exists()
	assert load_jsonl(tmp_path / 'sft-candidates.jsonl')[0]['review']['supported'] is False
	assert await synthesize_sft(tmp_path, spec, 'writer', 'critic') == 0
	assert len(calls) == 2  # completed review is cached


@pytest.mark.asyncio
async def test_sft_mode_routes_around_rl_panel(tmp_path, monkeypatch):
	import rl_env_use.sft_pipeline as s

	fake = AsyncMock(return_value={'kept': 2})
	monkeypatch.setattr(s, 'build_sft', fake)
	monkeypatch.setattr(pipeline, 'use_browser_spec', lambda spec: None)
	monkeypatch.setattr(pipeline, 'panel_stage', AsyncMock(side_effect=AssertionError('must not run panel')))
	spec = {'mode': 'sft', 'language': 'nob', 'writer_model': 'local:writer', 'critic_model': 'local:critic'}
	assert json.loads(await pipeline.build(tmp_path, spec))['kept'] == 2
	assert (await pipeline.create(tmp_path, spec))['kept'] == 2
	assert fake.await_count == 2


@pytest.mark.asyncio
async def test_all_sources_are_attempted_and_failed_sources_are_logged(tmp_path):
	# Empty recovery IDs and invalid schemes exercise discovery audit without external requests.
	spec = {'sources': ['bad://helsenorge', 'bad://nortrials', 'bad://norcrin']}
	assert await collect(tmp_path, spec) == []
	assert len(load_jsonl(tmp_path / 'sft-discovery.jsonl')) == 3
	assert all(r['status'] == 'error' for r in load_jsonl(tmp_path / 'sft-discovery.jsonl'))


@pytest.mark.asyncio
async def test_continuous_sft_has_no_twenty_four_example_stop(tmp_path, monkeypatch):
	import rl_env_use.sft_pipeline as s

	count = 0

	async def collect_fake(env, spec):
		nonlocal count
		count += 1
		(env / 'sft-frontier.json').write_text(json.dumps({'https://a.org': {'exhausted': count == 2}}))

	async def synth_fake(*args):
		return count * 30

	monkeypatch.setattr(s, 'collect', collect_fake)
	monkeypatch.setattr(s, 'synthesize_sft', synth_fake)
	monkeypatch.setattr(s, 'audit_sources', AsyncMock())
	result = await s.build_sft(tmp_path, {'language': 'nob', 'mode': 'sft', 'sft_continuous': True, 'target_tasks': 0, 'sources': ['https://a.org']}, 'w', 'c')
	assert result['kept'] == 60 and result['batches_this_run'] == 2


def test_export_bundles_evidence_and_raw_snapshot_provenance_and_rejects_corruption(tmp_path):
	import hashlib

	raw = b'<html>Complete original study page.</html>'
	(tmp_path / 'source.html').write_bytes(raw)
	proof = {'capture': 'source.html', 'response_sha256': hashlib.sha256(raw).hexdigest(), 'captured_at': '2026-10-24', 'robots': [{'allowed': True}]}
	row = {
		'id': 's1',
		'case': 't1',
		'document_id': 'd1',
		'source_url': 'https://a.org/study',
		'language': 'nob',
		'writer': 'w',
		'critic': 'c',
		'review': {'supported': True},
		'evidence_sha256': 'e',
		'source_capture': proof,
		'robots_allowed': True,
		'accepted': True,
		'system': 'Svar på norsk.',
		'evidence': 'Complete supplied evidence.',
		'draft': {'question': 'Hva viser studien?', 'answer': 'Svar.', 'explanation': 'Forklaring.', 'task_type': 'result_interpretation'},
	}
	save_jsonl(tmp_path / 'sft-candidates.jsonl', [row])
	assert export_accepted(tmp_path) == 1
	record = load_jsonl(tmp_path / 'dataset-sft.records.jsonl')[0]
	assert record['provenance']['source_capture'] == proof
	assert 'Complete supplied evidence.' in record['messages'][1]['content']
	(tmp_path / 'source.html').write_bytes(b'truncated')
	assert export_accepted(tmp_path) == 0


@pytest.mark.asyncio
async def test_ctg_discovery_paginates_and_reopens_only_after_last_page(tmp_path, monkeypatch):
	from urllib.parse import parse_qs, urlsplit

	import rl_env_use.sft_pipeline as module

	seen = []

	async def fake_capture(client, env, url):
		seen.append(url)
		token = parse_qs(urlsplit(url).query).get('pageToken')
		payload = {'studies': [{'protocolSection': {'identificationModule': {'nctId': 'NCT00000001' if not token else 'NCT00000002'}}}]}
		if not token:
			payload['nextPageToken'] = 'page2'
		return httpx.Response(200, json=payload), {'url': url}

	async def fake_registry(*args):
		nct = args[2]
		return {'id': nct, 'case': nct, 'url': 'https://clinicaltrials.gov/api/v2/studies/' + nct, 'text': 'evidence'}

	monkeypatch.setattr(module, 'capture', fake_capture)
	monkeypatch.setattr(module, 'registry_document', fake_registry)
	source = 'https://clinicaltrials.gov/api/v2/studies?query.locn=Norway&pageSize=3'
	spec = {'sources': [source], 'sft_discovery_limit': 25}
	await module.collect(tmp_path, spec)
	assert not json.loads((tmp_path / 'sft-frontier.json').read_text())[source]['exhausted']
	await module.collect(tmp_path, spec)
	assert json.loads((tmp_path / 'sft-frontier.json').read_text())[source]['exhausted']
	assert parse_qs(urlsplit(seen[1]).query)['pageToken'] == ['page2']
	assert parse_qs(urlsplit(seen[0]).query)['pageSize'] == ['25']
