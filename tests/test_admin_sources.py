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
