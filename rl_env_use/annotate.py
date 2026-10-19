"""Stage 4: local annotation app. Review tasks (input sign-off) and traces (open coding) before any evaluator exists."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .llm import now
from .store import load_jsonl

HTML = Path(__file__).with_name('annotate.html')


def latest(rows: list[dict]) -> dict:
	return {r['id']: r for r in rows}


class Store:
	def __init__(self, env_dir: Path, variant: str = 'baseline'):
		self.env, self.variant = Path(env_dir), variant
		self.flow = self.env / 'hillclimb'

	@property
	def tasks(self) -> list[dict]:
		return load_jsonl(self.env / 'tasks.jsonl')

	def kb(self) -> dict:
		p = self.env / 'kb.json'
		kb = json.loads(p.read_text()) if p.exists() else {'facts': []}
		mats = [
			{
				'id': m['id'],
				'claim': f'[{m["case"]}] material: {m["title"]}',
				'value': f'{len(m["text"])} chars',
				'quote': m['text'],
				'url': m['url'],
				'angle': 'material',
				'http': m.get('http', ''),
			}
			for m in kb.get('materials', [])
		]
		return kb | {'facts': [f | ({'angle': f'outcome of {f["case"]}'} if f.get('case') else {}) for f in kb['facts']] + mats}  # case materials are reviewed like facts

	FILES = {'tasks': 'reviews.jsonl', 'traces': 'annotations.jsonl', 'data': 'data_reviews.jsonl'}

	def rows(self) -> list[dict]:
		return load_jsonl(self.flow / self.variant / 'results.jsonl')

	def labels(self, mode: str) -> dict:
		return latest(load_jsonl(self.env / self.FILES[mode]))

	def modes(self) -> list[dict]:
		p = self.env / 'failure_modes.json'
		return json.loads(p.read_text())['modes'] if p.exists() else []

	def items(self, mode: str) -> list[dict]:
		lab = self.labels(mode)
		if mode == 'data':  # every mined fact, with what plain HTTP saw of it
			return [
				{
					'id': f['id'],
					'title': f'{f["claim"]} = {f["value"]}',
					'tags': [f.get('angle', ''), urlparse(f['url']).netloc, f.get('http', '')],
					'label': lab.get(f['id'], {}).get('verdict'),
				}
				for f in self.kb()['facts']
			]
		if mode == 'tasks':
			return [
				{'id': t['id'], 'title': t['prompt'], 'tags': [t.get('category', ''), t.get('difficulty', '')], 'label': lab.get(t['id'], {}).get('verdict')} for t in self.tasks
			]
		return [
			{
				'id': f'{self.variant}/{r["prompt_id"]}_rep{r["rep"]}',
				'title': r['prompt'],
				'tags': r.get('tags', []),
				'label': lab.get(f'{self.variant}/{r["prompt_id"]}_rep{r["rep"]}', {}).get('verdict'),
			}
			for r in self.rows()
		]

	def detail(self, mode: str, iid: str) -> dict:
		lab = self.labels(mode).get(iid)
		if mode == 'data':
			return {'fact': next(f for f in self.kb()['facts'] if f['id'] == iid), 'label': lab}
		if mode == 'tasks':
			t = next(t for t in self.tasks if t['id'] == iid)
			facts = {f['id']: f for f in self.kb()['facts']}
			return {'task': t, 'evidence': [facts[f] for f in t.get('fact_ids', []) if f in facts], 'label': lab}
		variant, rest = iid.split('/', 1)
		pid, rep = rest.rsplit('_rep', 1)
		row = next(r for r in load_jsonl(self.flow / variant / 'results.jsonl') if r['prompt_id'] == pid and r['rep'] == int(rep))
		tp = self.flow / variant / 'traces' / f'{pid}_rep{rep}.json'
		return {'row': row, 'trace': json.loads(tp.read_text()) if tp.exists() else [], 'label': lab, 'modes': self.modes()}

	def save(self, mode: str, body: dict) -> dict:
		rec = {'id': body['id'], 'verdict': body['verdict'], 'notes': body.get('notes', ''), 'annotator': body.get('annotator') or 'human', 'at': now()}
		if mode == 'tasks':
			rec['gold_fix'] = body.get('gold_fix') or None
		elif mode == 'traces':
			rec['modes'] = body.get('modes', [])
		with (self.env / self.FILES[mode]).open('a') as f:
			f.write(json.dumps(rec) + '\n')
		return rec


def make_handler(store: Store):
	class H(BaseHTTPRequestHandler):
		def log_message(self, *a):
			pass

		def _send(self, code: int, body: bytes, ctype: str):
			self.send_response(code)
			self.send_header('Content-Type', ctype)
			self.send_header('Content-Security-Policy', "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'")
			self.end_headers()
			self.wfile.write(body)

		def _json(self, obj, code=200):
			self._send(code, json.dumps(obj).encode(), 'application/json')

		def do_GET(self):
			if self.headers.get('Host', '').split(':')[0] not in ('127.0.0.1', 'localhost'):  # DNS rebinding could read gold answers
				return self._json({'error': 'forbidden'}, 403)
			u = urlparse(self.path)
			q = {k: v[0] for k, v in parse_qs(u.query).items()}
			try:
				if u.path == '/':
					return self._send(200, HTML.read_bytes(), 'text/html; charset=utf-8')
				if u.path == '/api/items':
					return self._json(store.items(q['mode']))
				if u.path == '/api/item':
					return self._json(store.detail(q['mode'], q['id']))
				self._json({'error': 'not found'}, 404)
			except (KeyError, StopIteration) as e:
				self._json({'error': f'missing {e}'}, 404)

		def _same_origin(self) -> bool:
			"""Block cross-site writes and DNS rebinding: labels can overwrite gold answers."""
			host = self.headers.get('Host', '')
			origin = self.headers.get('Origin')
			local = host.split(':')[0] in ('127.0.0.1', 'localhost')
			return local and self.headers.get('Content-Type', '').startswith('application/json') and (origin is None or urlparse(origin).netloc == host)

		def do_POST(self):
			if urlparse(self.path).path != '/api/label':
				return self._json({'error': 'not found'}, 404)
			if not self._same_origin():
				return self._json({'error': 'forbidden'}, 403)
			body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))) or b'{}')
			if body.get('mode') not in ('tasks', 'traces', 'data') or not body.get('id') or body.get('verdict') not in ('pass', 'fail', 'accept', 'reject'):
				return self._json({'error': 'bad label'}, 400)
			self._json(store.save(body['mode'], body))

	return H


def serve(env_dir: Path, variant: str = 'baseline', port: int = 8765) -> ThreadingHTTPServer:
	srv = ThreadingHTTPServer(('127.0.0.1', port), make_handler(Store(env_dir, variant)))
	return srv
