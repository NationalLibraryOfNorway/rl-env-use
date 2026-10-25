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


LEGAL_HEADING = re.compile(r'(?mi)^(?P<heading>(?:Kapittel|Kap\.|Del)\s+[^\n]+|§\s*\d[^\n]*|[IVX]+\.\s+[^\n]+)$')


def split_source(text, limit):
	"""Lossless contiguous slices; prefer legal headings, then lines/words. Offsets are characters."""
	if not isinstance(limit, int) or limit < 1:
		raise ValueError('source_chunk_chars must be a positive integer')
	headings = []
	path = []
	for match in LEGAL_HEADING.finditer(text):
		heading = match['heading']
		level = 1 if heading.lower().startswith('del ') else 3 if heading.startswith('§') else 2
		path = [(lev, value) for lev, value in path if lev < level] + [(level, heading)]
		headings.append((match.start(), [value for _, value in path]))
	parts = []
	start = 0
	while start < len(text):
		end = min(start + limit, len(text))
		boundary = 'end'
		if end < len(text):
			choices = [pos for pos, _ in headings if start < pos <= end]
			if choices:
				end, boundary = choices[-1], 'legal_heading'
			else:
				cut = text.rfind('\n', start + 1, end + 1)
				if cut >= 0:
					end, boundary = cut + 1, 'line'
				else:
					cut = text.rfind(' ', start + 1, end + 1)
					if cut >= 0:
						end, boundary = cut + 1, 'word'
					else:
						boundary = 'character'
				# A separator at the window end must not exceed the ceiling.
				end = min(end, start + limit)
		active = [(pos, context) for pos, context in headings if pos <= start]
		context = active[-1][1] if active else []
		parts.append(
			{
				'text': text[start:end],
				'source_text_start': start,
				'source_text_end': end,
				'section_path': context,
				'continues_section': bool(active and active[-1][0] < start),
				'boundary_after': boundary,
			}
		)
		start = end
	return parts


def source_documents(url, text, proof, page, item, spec):
	minimum, maximum = spec.get('source_min_chars', 300), spec.get('source_max_chars', 150000)
	if len(text) < minimum:
		raise ValueError('empty/short/scanned evidence; no text was truncated')
	parent = 'admin_' + digest(url + digest(text))[:24]
	parts = split_source(text, min(spec.get('source_chunk_chars', 24000), maximum)) if len(text) > maximum else [{'text': text}]
	documents = []
	for i, part in enumerate(parts, 1):
		chunk = part.pop('text')
		key = parent if len(parts) == 1 else parent + '_' + str(i).zfill(4)
		metadata = (
			{}
			if len(text) <= maximum
			else dict(parent_document_id=parent, part_index=i, part_count=len(parts), source_text_sha256=digest(text), source_text_length=len(text), context_required=True, **part)
		)
		documents.append(
			dict(
				id=key,
				**proof,
				**page,
				text=chunk,
				text_sha256=digest(chunk),
				discovered_from=item['discovered_from'],
				depth=item['depth'],
				reuse_status='not_assessed',
				source_kind='public_administrative_guidance',
				evidence_only=True,
				**metadata,
			)
		)
	return documents


def report(env, phase, rows, queue, **extra):
	counts = Counter(r['status'] for r in rows.values())
	domains = Counter(r['agency'] for r in rows.values() if r['status'] == 'collected')
	value = {'phase': phase, 'at': stamp(), 'counts': dict(counts), 'collected_by_agency': dict(domains), 'queued': len(queue), **extra}
	write_atomic(env / 'admin-status.json', json.dumps(value, ensure_ascii=False, indent=2))
	print(json.dumps(value, ensure_ascii=False), flush=True)


async def collect_admin(env, spec, retry_urls=(), retry_only=False):
	"""Only source collection: never calls writer, critic, browser agent or training export."""
	rows = {r['url_requested']: r for r in load_jsonl(env / 'admin-collection.jsonl')}
	documents = {d['id']: d for d in load_jsonl(env / 'admin-documents.jsonl')}
	state_file = env / 'admin-frontier.json'
	queue = json.loads(state_file.read_text())['queue'] if state_file.exists() else []
	if retry_only and not retry_urls:
		raise ValueError('retry-only requires explicit retry URLs')
	retry_pending = set()
	retries = load_jsonl(env / 'admin-retries.jsonl')
	for url in dict.fromkeys(retry_urls):
		url = scoped_url(url, spec)
		if url is None:
			raise ValueError('retry URL outside configured public scope')
		if url in rows:
			if rows[url]['status'] == 'collected':
				continue
			if rows[url]['status'] != 'error':
				raise ValueError('only failed collection rows can be retried explicitly')
			retries.append({'at': stamp(), 'previous_result': rows.pop(url)})
		retry_pending.add(url)
		queue = [item for item in queue if item['url'] != url]
		queue.insert(0, {'url': url, 'depth': 0, 'discovered_from': 'explicit_retry'})
	if retry_urls:
		save_jsonl(env / 'admin-retries.jsonl', retries)
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
				if retry_only:
					position = next(
						(i for i, item in enumerate(queue) if item['url'] in retry_pending and host_counts[urlsplit(item['url']).hostname.removeprefix('www.')] < max_host), None
					)
					if position is None:
						break
					queue.insert(0, queue.pop(position))
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
					parts = source_documents(str(response.url), text, proof, page, item, spec)
					for part in parts:
						documents[part['id']] = dict(part, agency=host)
					ids = [part['id'] for part in parts]
					row.update(status='collected', document_ids=ids, part_count=len(parts), final_url=str(response.url), capture=proof['capture'])
					if len(ids) == 1:
						row['document_id'] = ids[0]
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
				retry_pending.discard(url)
				rows[url] = row
				host_counts[host] += 1
				queue.pop(0)
				checkpoint()
			report(env, 'retry_complete' if retry_only and not retry_pending else 'complete' if not queue else 'limit_reached', rows, queue)
	except BaseException:
		checkpoint()
		report(env, 'interrupted', rows, queue)
		raise
	return {'documents': len(documents), 'attempted': len(rows), 'queued': len(queue)}
