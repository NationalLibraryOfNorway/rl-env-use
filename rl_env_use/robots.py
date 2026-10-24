"""Robots-aware download gate and per-origin crawl pacing for SFT source collection."""

import asyncio
import hashlib
import re
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from urllib.parse import urljoin, urlsplit

from .store import load_jsonl, save_jsonl, write_atomic

AGENT = 'rl-env-use'
CTG_API_AUTHORIZATION = 'https://www.nlm.nih.gov/pubs/techbull/ja25/ja25_clinical_trials_screen-scraping.html'


def documented_public_api(url):
	"""Provider-documented study API only; never private/internal paths or lookalike hosts."""
	p = urlsplit(url)
	if p.scheme != 'https' or p.hostname != 'clinicaltrials.gov' or p.port not in (None, 443) or p.username or p.password:
		return False
	return bool(re.fullmatch(r'/api/v2/(?:version|studies(?:/(?:NCT\d{8}|metadata|search-areas|enums))?)', p.path))


def rules_allow(text, url, agent=AGENT):
	"""Longest matching rule wins; Allow wins ties. Supports * and terminal $."""
	groups, names, rules, delay = [], [], [], 0.0

	def flush():
		if names:
			groups.append((names.copy(), rules.copy(), delay))

	started = False
	for raw in text.splitlines():
		line = raw.split('#', 1)[0].strip()
		if ':' not in line:
			continue
		key, value = (p.strip() for p in line.split(':', 1))
		key = key.lower()
		if key == 'user-agent':
			if started:
				flush()
				names, rules, delay, started = [], [], 0.0, False
			names.append(value.lower())
		elif names:
			started = True
			if key in ('allow', 'disallow') and value:
				rules.append((key == 'allow', value))
			elif key == 'crawl-delay':
				try:
					delay = max(delay, float(value))
				except ValueError:
					pass
	flush()
	matching = [(max((len(n) if n != '*' else 0 for n in ns if n == '*' or n in agent.lower()), default=-1), rs, d) for ns, rs, d in groups]
	best = max((score for score, _, _ in matching), default=-1)
	selected = [(rs, d) for score, rs, d in matching if score == best and score >= 0]
	parts = urlsplit(url)
	path = parts.path or '/'
	if parts.query:
		path += '?' + parts.query
	# Decode only unreserved characters; reserved escapes such as %2F stay distinct.
	canonical = lambda s: re.sub(
		r'%([0-9a-fA-F]{2})', lambda m: chr(int(m[1], 16)) if chr(int(m[1], 16)) in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~' else '%' + m[1].upper(), s
	)
	path = canonical(path)
	hits = []
	for rs, _ in selected:
		for allow, pattern in rs:
			pattern = canonical(pattern)
			end = pattern.endswith('$')
			pattern = pattern[:-1] if end else pattern
			expr = '^' + re.escape(pattern).replace(r'\*', '.*') + ('$' if end else '')
			if re.search(expr, path):
				hits.append((len(pattern.replace('*', '').encode()), allow))
	allowed = max(hits)[1] if hits else True
	return allowed, max((d for _, d in selected), default=0.0)


class RobotsGuard:
	def __init__(self, env, client, min_delay=0.5):
		self.env, self.client, self.min_delay = env, client, min_delay
		self.policies, self.last = {}, {}
		self.audit = load_jsonl(env / 'robots-audit.jsonl')

	async def check(self, url, pace=False):
		parts = urlsplit(url)
		if parts.scheme not in ('http', 'https') or not parts.netloc:
			raise PermissionError('unsupported source URL')
		origin = parts.scheme + '://' + parts.netloc
		if origin not in self.policies:
			robots_url = origin + '/robots.txt'
			folder = self.env / 'robots-captures'
			folder.mkdir(exist_ok=True)
			try:
				r = await self.client.get(robots_url, follow_redirects=True, headers={'User-Agent': AGENT})
				policy = {'robots_url': robots_url, 'http_status': r.status_code, 'checked_at': datetime.now(UTC).isoformat(), 'text': r.text}
				policy['capture'] = 'robots-captures/' + hashlib.sha256((robots_url + r.text).encode()).hexdigest() + '.txt'
				write_atomic(self.env / policy['capture'], r.text)
				policy['valid'] = r.status_code == 200 and '<html' not in r.text.lower()
				policy['missing'] = r.status_code in (404, 410)
			except Exception as exc:
				policy = {'robots_url': robots_url, 'valid': False, 'missing': False, 'error': type(exc).__name__, 'text': ''}
			self.policies[origin] = policy
		p = self.policies[origin]
		allowed, delay = rules_allow(p['text'], url) if p['valid'] else (bool(p['missing']), 0)
		decision = {k: v for k, v in p.items() if k not in ('text', 'valid', 'missing')}
		decision.update(url=url, allowed=allowed, reason='no_robots_file' if p['missing'] else 'robots_rules' if p['valid'] else 'robots_unavailable', crawl_delay=delay)
		if documented_public_api(url):
			decision.update(robots_allowed=allowed if p['valid'] or p['missing'] else None, allowed=True, reason='documented_public_api', authorization_url=CTG_API_AUTHORIZATION)
			allowed, delay = True, max(delay, 1.0)
			decision['crawl_delay'] = delay
		self.audit.append(decision)
		save_jsonl(self.env / 'robots-audit.jsonl', self.audit)
		if not allowed:
			raise PermissionError('robots.txt does not allow download: ' + url)
		if pace:
			wait = max(delay, self.min_delay) - (time.monotonic() - self.last.get(origin, 0))
			if wait > 0:
				await asyncio.sleep(wait)
			self.last[origin] = time.monotonic()
		return decision

	async def get(self, url):
		decisions = []
		for _ in range(10):
			decisions.append(await self.check(url, pace=True))
			for attempt in range(3):
				if attempt:
					decisions.append(await self.check(url, pace=True))
				r = await self.client.get(url, follow_redirects=False, headers={'User-Agent': AGENT})
				if r.status_code not in (429, 500, 502, 503, 504) or attempt == 2:
					break
				try:
					retry_after = float(r.headers.get('Retry-After', '0'))
				except ValueError:
					try:
						retry_after = (parsedate_to_datetime(r.headers['Retry-After']) - datetime.now(UTC)).total_seconds()
					except (KeyError, TypeError, ValueError):
						retry_after = 0
				await asyncio.sleep(max(2 ** (attempt + 1), retry_after))
			if r.is_redirect:
				url = urljoin(url, r.headers['location'])
				continue
			return r, decisions
		raise ValueError('too many source redirects')
