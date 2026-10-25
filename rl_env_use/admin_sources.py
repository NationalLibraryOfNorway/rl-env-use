"""Resumable, model-free collection of Norwegian public administrative guidance."""

import json
import re
from collections import Counter
from io import BytesIO
from urllib.parse import unquote, urljoin, urlsplit, urlunsplit

import httpx
from bs4 import BeautifulSoup
from pypdf import PdfReader

from .robots import RobotsGuard
from .sft_pipeline import capture, digest, stamp
from .store import load_jsonl, save_jsonl, write_atomic

BLOCKED_PATH = re.compile(r'/(?:login|logg-?inn|logout|auth|oauth|sok|search|søk|minside|min-side|mine-sider|dine-sider|ditt-nav|oss|api)(?:/|$)', re.I)
BINARY = re.compile(r'\.(?:png|jpe?g|gif|svg|webp|mp[34]|zip|xlsx?|docx?|pptx?|woff2?|css|js|ico)$', re.I)


def scoped_url(url, spec):
	"""An explicit domain/path scope; no login, search, query traps or youth Q&A."""
	try:
		p = urlsplit(url)
		host = (p.hostname or '').lower()
		if p.scheme != 'https' or p.port not in (None, 443) or p.username or p.password or p.query:
			return None
		scopes = spec['administrative_sources'].get(host.removeprefix('www.'))
		if not scopes:
			return None
		path = unquote(p.path) or '/'
		if any(x in path for x in ['\\', '..', '//']) or BLOCKED_PATH.search(path) or BINARY.search(path):
			return None
		if not any(prefix == '/' or path == prefix.rstrip('/') or path.startswith(prefix.rstrip('/') + '/') for prefix in scopes['paths']):
			return None
		return urlunsplit(('https', host, p.path or '/', '', ''))
	except ValueError:
		return None


class ScopedGuard(RobotsGuard):
	def __init__(self, env, client, spec):
		super().__init__(env, client, spec.get('source_min_delay', 1.0))
		self.spec = spec

	async def check(self, url, pace=False):
		if not scoped_url(url, self.spec):
			raise PermissionError('source or redirect outside public administrative scope')
		return await super().check(url, pace)


def extract(response):
	mime = response.headers.get('content-type', '').lower()
	if 'pdf' in mime or response.content.startswith(b'%PDF-'):
		reader = PdfReader(BytesIO(response.content))
		return {
			'title': str(reader.metadata.title or '') if reader.metadata else '',
			'text': '\n\n'.join(p.extract_text() or '' for p in reader.pages),
			'language_declared': None,
			'dates_declared': [],
			'links': [],
			'transformation': 'all PDF pages; extracted text requires table/sign review',
		}
	if 'html' not in mime:
		raise ValueError('not an HTML or PDF guidance document')
	soup = BeautifulSoup(response.text, 'html.parser')
	dates = []
	for tag in soup.select('meta[property],meta[name],time[datetime]'):
		name = tag.get('property', tag.get('name', 'datetime'))
		if tag.name == 'time' or re.search('date|modified|published', name, re.I):
			dates.append({'field': name, 'value': tag.get('content', tag.get('datetime', ''))})
	for tag in soup.select('script[type="application/ld+json"]'):
		try:
			obj = json.loads(tag.get_text())

			def visit(item):
				if isinstance(item, dict):
					for k, v in item.items():
						if k in ('datePublished', 'dateModified') and isinstance(v, str):
							dates.append({'field': k, 'value': v})
						elif isinstance(v, (dict, list)):
							visit(v)
				elif isinstance(item, list):
					for v in item:
						visit(v)

			visit(obj)
		except (ValueError, TypeError):
			pass
	main = soup.find('main') or soup.find('article') or soup.body
	if main is None:
		raise ValueError('no article/main/body content')
	links = [urljoin(str(response.url), a['href']) for a in main.select('a[href]')]
	title = soup.find('h1') or soup.title
	title = title.get_text(' ', strip=True) if title else ''
	lang = soup.html.get('lang') if soup.html else None
	for tag in main.select('script,style,nav,footer,header,form,noscript,[hidden],[aria-hidden="true"]'):
		tag.decompose()
	return {
		'title': title,
		'text': main.get_text('\n', strip=True),
		'language_declared': lang,
		'dates_declared': dates,
		'links': links,
		'transformation': 'complete main/article/body DOM text; navigation/scripts/forms/explicitly hidden elements removed; no browser rendering',
	}


def report(env, phase, rows, queue, **extra):
	counts = Counter(r['status'] for r in rows.values())
	domains = Counter(r['agency'] for r in rows.values() if r['status'] == 'collected')
	value = {'phase': phase, 'at': stamp(), 'counts': dict(counts), 'collected_by_agency': dict(domains), 'queued': len(queue), **extra}
	write_atomic(env / 'admin-status.json', json.dumps(value, ensure_ascii=False, indent=2))
	print(json.dumps(value, ensure_ascii=False), flush=True)


async def collect_admin(env, spec):
	"""Only source collection: never calls writer, critic, browser agent or training export."""
	rows = {r['url_requested']: r for r in load_jsonl(env / 'admin-collection.jsonl')}
	documents = {d['id']: d for d in load_jsonl(env / 'admin-documents.jsonl')}
	state_file = env / 'admin-frontier.json'
	queue = json.loads(state_file.read_text())['queue'] if state_file.exists() else []
	queued = {x['url'] for x in queue}
	for config in spec['administrative_sources'].values():
		for url in config['seeds']:
			url = scoped_url(url, spec)
			if url and url not in rows and url not in queued:
				queue.append({'url': url, 'depth': 0, 'discovered_from': 'configured_seed'})
				queued.add(url)
	host_counts = Counter(urlsplit(u).hostname.removeprefix('www.') for u in rows)
	max_pages = spec.get('source_max_pages', 2000)
	max_host = spec.get('source_max_pages_per_agency', 300)

	def checkpoint():
		save_jsonl(env / 'admin-collection.jsonl', list(rows.values()))
		save_jsonl(env / 'admin-documents.jsonl', list(documents.values()))
		write_atomic(state_file, json.dumps({'queue': queue}, ensure_ascii=False))

	checkpoint()
	try:
		async with httpx.AsyncClient(timeout=45, follow_redirects=False) as client:
			guard = ScopedGuard(env, client, spec)
			while queue and len(rows) < max_pages:
				item = queue[0]
				url = item['url']
				host = urlsplit(url).hostname.removeprefix('www.')
				if host_counts[host] >= max_host:
					# Preserve capped work so raising the cap and resuming can collect it.
					next_item = next((i for i, x in enumerate(queue) if host_counts[urlsplit(x['url']).hostname.removeprefix('www.')] < max_host), None)
					if next_item is None:
						break
					queue.insert(0, queue.pop(next_item))
					continue
				row = dict(item, url_requested=url, agency=host, at=stamp())
				report(env, 'downloading', rows, queue, current_url=url)
				try:
					response, proof = await capture(guard, env, url)
					page = extract(response)
					text = page.pop('text')
					links = page.pop('links')
					if not spec.get('source_min_chars', 300) <= len(text) <= spec.get('source_max_chars', 150000):
						raise ValueError('empty/short/scanned or oversized evidence; no text was truncated')
					key = 'admin_' + digest(str(response.url) + digest(text))[:24]
					documents[key] = dict(
						id=key,
						agency=host,
						**proof,
						**page,
						text=text,
						text_sha256=digest(text),
						discovered_from=item['discovered_from'],
						depth=item['depth'],
						reuse_status='not_assessed',
						source_kind='public_administrative_guidance',
						evidence_only=True,
					)
					row.update(status='collected', document_id=key, final_url=str(response.url), capture=proof['capture'])
					if item['depth'] < spec.get('source_max_depth', 2):
						for link in links:
							normalized = scoped_url(link, spec)
							if normalized and normalized not in rows and normalized not in queued:
								queue.append({'url': normalized, 'depth': item['depth'] + 1, 'discovered_from': str(response.url)})
								queued.add(normalized)
				except PermissionError as exc:
					row.update(status='blocked', error=str(exc))
				except Exception as exc:
					row.update(status='error', error=type(exc).__name__, detail=str(exc)[:300])
				rows[url] = row
				host_counts[host] += 1
				queue.pop(0)
				checkpoint()
			report(env, 'complete' if not queue else 'limit_reached', rows, queue)
	except BaseException:
		checkpoint()
		report(env, 'interrupted', rows, queue)
		raise
	return {'documents': len(documents), 'attempted': len(rows), 'queued': len(queue)}
