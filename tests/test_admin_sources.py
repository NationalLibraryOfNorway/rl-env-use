import json

import httpx
import pytest

from rl_env_use import admin_sources as admin
from rl_env_use.robots import rules_allow
from rl_env_use.store import load_jsonl


def spec():
	return {
		'administrative_sources': {'example.org': {'paths': ['/guidance'], 'seeds': ['https://example.org/guidance']}},
		'source_max_pages': 10,
		'source_max_pages_per_agency': 10,
		'source_max_depth': 2,
		'source_min_chars': 10,
		'source_min_delay': 0,
	}


@pytest.mark.parametrize(
	'url',
	[
		'http://example.org/guidance',
		'https://evil.org/guidance',
		'https://example.org/login',
		'https://example.org/guidance?token=secret',
		'https://user:pass@example.org/guidance',
		'https://example.org/guidance/../login',
		'https://example.org/guidance/%2e%2e/login',
		'https://example.org/guidance/search',
		'https://example.org/guidance/image.png',
		'https://example.org/guidance2',
	],
)
def test_scope_rejects_nonpublic_and_unconfigured_urls(url):
	assert admin.scoped_url(url, spec()) is None


def test_scope_normalizes_fragments_without_widening_prefix():
	assert admin.scoped_url('https://www.example.org/guidance/rules#section', spec()) == 'https://www.example.org/guidance/rules'


def test_robots_bom_does_not_erase_disallow():
	assert not rules_allow('\ufeffUser-agent: *\r\nDisallow: /globalassets\r\n', 'https://www.skatteetaten.no/globalassets/doc.pdf')[0]


def test_extract_preserves_numbers_tables_and_dates():
	r = httpx.Response(
		200,
		request=httpx.Request('GET', 'https://example.org/guidance'),
		headers={'Content-Type': 'text/html'},
		text='<html lang="nb"><head><meta property="article:modified_time" content="2026-10-01"></head><body><nav>Noise</nav><main><h1>Rules</h1><table><tr><td>2026</td><td>-2.5%</td></tr></table><p hidden>hidden</p><a href="/guidance/next">Next</a></main></body></html>',
	)
	result = admin.extract(r)
	assert '-2.5%' in result['text'] and '2026' in result['text']
	assert 'hidden' not in result['text'] and 'Noise' not in result['text']
	assert result['language_declared'] == 'nb'
	assert result['dates_declared'] == [{'field': 'article:modified_time', 'value': '2026-10-01'}]
	assert result['links'] == ['https://example.org/guidance/next']


@pytest.mark.asyncio
async def test_collection_checkpoints_blocks_redirects_and_resumes_without_redownload(tmp_path, monkeypatch):
	requested = []

	async def handler(req):
		requested.append(str(req.url))
		if req.url.path == '/robots.txt':
			return httpx.Response(200, text='User-agent: *\nDisallow: /guidance/blocked\n')
		if req.url.path == '/guidance/redirect':
			return httpx.Response(302, headers={'Location': 'https://evil.org/guidance/private'})
		return httpx.Response(
			200,
			headers={'Content-Type': 'text/html'},
			text='<main><h1>Public rules</h1><p>Complete evidence -2.5%.</p><a href="/guidance/blocked">Blocked</a><a href="/guidance/redirect">Redirect</a></main>',
		)

	factory = httpx.AsyncClient
	monkeypatch.setattr(admin.httpx, 'AsyncClient', lambda **kwargs: factory(transport=httpx.MockTransport(handler), **kwargs))
	result = await admin.collect_admin(tmp_path, spec())
	assert result['documents'] == 1 and result['attempted'] == 3
	assert 'https://example.org/guidance/blocked' not in requested
	assert not any('evil.org' in u for u in requested)
	docs = load_jsonl(tmp_path / 'admin-documents.jsonl')
	assert docs[0]['reuse_status'] == 'not_assessed'
	assert (tmp_path / docs[0]['capture']).read_text().startswith('<main>')
	assert json.loads((tmp_path / 'admin-status.json').read_text())['counts']['blocked'] == 2
	before = requested.copy()
	await admin.collect_admin(tmp_path, spec())
	assert requested == before
	assert not (tmp_path / 'dataset-sft.jsonl').exists()


@pytest.mark.asyncio
async def test_limit_keeps_frontier_for_resume(tmp_path, monkeypatch):
	async def handler(req):
		if req.url.path == '/robots.txt':
			return httpx.Response(200, text='User-agent: *\nAllow: /')
		return httpx.Response(200, headers={'Content-Type': 'text/html'}, text='<main>Enough guidance text.<a href="/guidance/next">Next</a></main>')

	factory = httpx.AsyncClient
	monkeypatch.setattr(admin.httpx, 'AsyncClient', lambda **kwargs: factory(transport=httpx.MockTransport(handler), **kwargs))
	s = spec() | {'source_max_pages': 1}
	result = await admin.collect_admin(tmp_path, s)
	assert result['queued'] == 1
	assert json.loads((tmp_path / 'admin-status.json').read_text())['phase'] == 'limit_reached'
	s['source_max_pages'] = 2
	assert (await admin.collect_admin(tmp_path, s))['documents'] == 2


def test_split_is_lossless_bounded_and_retains_legal_context():
	text = (
		'Document title\nKapittel 1. Virkeområde\n§ 1-1. Formål\n'
		+ ('Alle vilkår og unntak gjelder.\n' * 30)
		+ 'Kapittel 2. Plikter\n§ 2-1. Arbeidsgiver\n'
		+ 'Rettigheter gjelder.\n' * 30
	)
	parts = admin.split_source(text, 180)
	assert ''.join(p['text'] for p in parts) == text
	assert all(0 < len(p['text']) <= 180 for p in parts)
	assert all(text[p['source_text_start'] : p['source_text_end']] == p['text'] for p in parts)
	assert any(p['section_path'] == ['Kapittel 1. Virkeområde', '§ 1-1. Formål'] and p['continues_section'] for p in parts)
	assert any(p['section_path'] == ['Kapittel 2. Plikter', '§ 2-1. Arbeidsgiver'] for p in parts)


def test_long_unbroken_section_splits_without_dropping_characters():
	text = 'x' * 301
	parts = admin.split_source(text, 100)
	assert ''.join(p['text'] for p in parts) == text
	assert [len(p['text']) for p in parts] == [100, 100, 100, 1]


def test_parts_share_full_source_provenance_and_hash():
	text = 'Kapittel 1. Test\n§ 1-1. Vilkår\n' + 'Komplette vilkår -2.5% med unntak.\n' * 40
	proof = {'capture': 'raw.html', 'response_sha256': 'rawhash', 'robots': [{'allowed': True}]}
	parts = admin.source_documents(
		'https://example.org/guidance', text, proof, {'title': 'Lov'}, {'depth': 0, 'discovered_from': 'seed'}, spec() | {'source_max_chars': 250, 'source_chunk_chars': 120}
	)
	assert len(parts) > 1
	assert len({p['parent_document_id'] for p in parts}) == 1
	assert all(p['source_text_sha256'] == admin.digest(text) and p['context_required'] for p in parts)
	assert all(p['capture'] == 'raw.html' and p['response_sha256'] == 'rawhash' and p['title'] == 'Lov' for p in parts)
	assert [p['part_index'] for p in parts] == list(range(1, len(parts) + 1))
	assert ''.join(p['text'] for p in parts) == text


@pytest.mark.asyncio
async def test_retry_recovers_oversized_page_and_preserves_audit_and_other_frontier(tmp_path, monkeypatch):
	from rl_env_use.store import save_jsonl

	url = 'https://example.org/guidance/long-law'
	old = {'url_requested': url, 'agency': 'example.org', 'status': 'error', 'error': 'ValueError', 'detail': 'oversized'}
	save_jsonl(tmp_path / 'admin-collection.jsonl', [old])
	(tmp_path / 'admin-frontier.json').write_text(json.dumps({'queue': [{'url': 'https://example.org/guidance/untouched', 'depth': 0}]}))
	text = '§ 1-1. Vilkår\n' + 'Alle vilkår gjelder.\n' * 50

	async def handler(req):
		if req.url.path == '/robots.txt':
			return httpx.Response(200, text='User-agent: *\nAllow: /')
		assert req.url.path == '/guidance/long-law'
		return httpx.Response(200, headers={'Content-Type': 'text/html'}, text='<main>' + text + '</main>')

	factory = httpx.AsyncClient
	monkeypatch.setattr(admin.httpx, 'AsyncClient', lambda **kwargs: factory(transport=httpx.MockTransport(handler), **kwargs))
	s = spec() | {'source_max_chars': 250, 'source_chunk_chars': 120}
	result = await admin.collect_admin(tmp_path, s, [url], retry_only=True)
	assert result['documents'] > 1
	assert load_jsonl(tmp_path / 'admin-retries.jsonl')[0]['previous_result'] == old
	row = load_jsonl(tmp_path / 'admin-collection.jsonl')[0]
	assert row['status'] == 'collected' and row['part_count'] == result['documents']
	assert 'document_id' not in row
	parts = load_jsonl(tmp_path / 'admin-documents.jsonl')
	assert ''.join(p['text'] for p in parts) == text.strip()
	assert json.loads((tmp_path / 'admin-frontier.json').read_text())['queue'][0]['url'].endswith('/untouched')
	# Retrying the same command after success does not duplicate the recovered parts.
	assert (await admin.collect_admin(tmp_path, s, [url], retry_only=True))['documents'] == result['documents']
