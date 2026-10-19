"""Browser layer: cloud browsers (stealth, residential proxies, persistent login profiles), page capture, evidence matching.
This is what lets environments be built from data that plain HTTP and fetch tools cannot reach."""

import asyncio
import html as htmllib
import os
import re
from urllib.parse import urlparse

import httpx

os.environ.setdefault('TIMEOUT_BrowserStateRequestEvent', '90')  # long records (court opinions, filings, accident dockets) outgrow the 30s DOM snapshot
from browser_use import BrowserProfile, BrowserSession  # noqa: E402

from .graders import norm

SEARCH_ENGINES = re.compile(r'^https?://([a-z]+\.)?(google|bing|duckduckgo|yahoo|baidu|yandex|search\.brave)\.[a-z.]+/', re.I)


def browser(allowed_domains: list[str] | None = None, headless: bool = True, profile_id: str | None = None, proxy_country: str | None = None) -> BrowserSession:
	"""Cloud browser when a Browser Use key is set: stealth fingerprint, residential proxy (US by default, or
	`proxy_country`), parallel-safe. `profile_id` attaches a synced cloud login profile, so harvesting can reach
	logged-in communities and helpdesk UIs. Falls back to local headless Chrome without a key."""
	kw = {'allowed_domains': allowed_domains} if allowed_domains else {}
	if os.getenv('BROWSER_USE_API_KEY') and os.getenv('RLENV_LOCAL_BROWSER') != '1':
		profile_id = profile_id or os.getenv('RLENV_CLOUD_PROFILE_ID')
		proxy_country = proxy_country or os.getenv('RLENV_PROXY_COUNTRY')
		cloud = ({'cloud_profile_id': profile_id} if profile_id else {}) | ({'cloud_proxy_country_code': proxy_country} if proxy_country else {})
		return BrowserSession(use_cloud=True, **cloud, **kw)
	return BrowserSession(browser_profile=BrowserProfile(headless=headless, **kw))


# the page as a person reads it, including same-origin frames (document viewers such as EDGAR's inline XBRL render the document in an iframe)
PAGE_TEXT_JS = """(() => { const t = d => { let s = d.body ? d.body.innerText : '';
  for (const f of d.querySelectorAll('iframe, frame')) { try { if (f.contentDocument) s += '\\n' + t(f.contentDocument); } catch (e) {} }
  return s; }; return t(document); })()"""


async def page_text(browser_session: BrowserSession) -> str:
	cdp = await browser_session.get_or_create_cdp_session()
	r = await cdp.cdp_client.send.Runtime.evaluate(params={'expression': PAGE_TEXT_JS, 'returnByValue': True}, session_id=cdp.session_id)
	return r.get('result', {}).get('value', '') or ''


def quote_in(quote: str, text: str) -> bool:
	"""Whole-token containment after normalization; one-character values allowed only if numeric. A quote may skip table
	cells with '...': each piece must be verbatim, in order, and within 300 characters of the previous one."""
	parts = [p for p in re.split(r'\s*(?:\.\.\.|\u2026)\s*', quote) if p.strip()]
	t = f' {norm(text)} '
	if len(parts) > 1:
		at = 0
		for i, p in enumerate(parts):
			q = norm(p)
			j = t.find(f' {q} ', at) if q else -1
			if j < 0 or (i and j - at > 300):
				return False
			at = j + len(q) + 1
		return True
	q = norm(quote)
	return (len(q) >= 2 or q.isdigit()) and f' {q} ' in t


# Anthropic's strict-schema grammar has a size cap; drop actions scouts/policies never need.
UNUSED = ['upload_file', 'write_file', 'read_file', 'replace_file', 'save_as_pdf', 'screenshot', 'evaluate', 'close', 'switch', 'search_page', 'find_elements']


# Generic "show me everything" pass, run repeatedly until the page stops growing: cookie walls, collapsed <details>,
# "Load more" / "Show 12 more replies" / "View hidden items" buttons, and lazy-loaded content below the fold.
EXPAND_JS = r"""(() => {
  const vis = e => { const r = e.getBoundingClientRect(); return r.width > 0 && r.height > 0 && getComputedStyle(e).visibility !== 'hidden'; };
  let n = 0;
  for (const b of document.querySelectorAll('button, [role=button], a')) {
    const t = (b.innerText || b.getAttribute('aria-label') || '').trim().toLowerCase();
    if (t.length > 40 || !vis(b) || b.dataset.rlenvClicked) continue;
    const href = b.tagName === 'A' ? (b.getAttribute('href') || '') : '';
    const inPage = !href || href.startsWith('#') || href.startsWith('javascript');
    if (/^(accept( all)?( cookies)?|agree|i agree|got it|allow all|ok, got it)$/.test(t) && inPage) { b.click(); b.dataset.rlenvClicked = 1; n++; continue; }
    if (inPage && /(load|show|view|see|read|expand)( \d+)? (more|all|hidden|earlier|older|previous|replies|comments)|^\d+ (hidden|more) (items|replies|comments)|^more replies|^expand( all)?$|^show more$/.test(t)) { b.click(); b.dataset.rlenvClicked = 1; n++; }
  }
  for (const d of document.querySelectorAll('details:not([open])')) { d.open = true; n++; }
  window.scrollTo(0, document.body.scrollHeight);
  return n;
})()"""


async def _eval(session: BrowserSession, expr: str):
	cdp = await session.get_or_create_cdp_session()
	r = await cdp.cdp_client.send.Runtime.evaluate(params={'expression': expr, 'returnByValue': True}, session_id=cdp.session_id)
	return r.get('result', {}).get('value')


async def capture(session: BrowserSession, url: str, expand: bool = True, rounds: int = 8, settle: float = 1.5) -> dict:
	"""Render `url` in a real browser and return its full text after expanding hidden content (no LLM involved)."""
	await session.navigate_to(url)
	await asyncio.sleep(settle)
	text, clicks = await page_text(session), 0
	for _ in range(rounds if expand else 0):
		n = await _eval(session, EXPAND_JS) or 0
		clicks += n
		await asyncio.sleep(settle)
		new = await page_text(session)
		if not n and len(new) <= len(text):
			break
		text = max(text, new, key=len)  # virtualized lists can shrink after clicks: keep the fullest view
	return {'url': url, 'final_url': await session.get_current_page_url(), 'title': await session.get_current_page_title(), 'text': text, 'expand_clicks': clicks, 'via': 'browser'}


def _visible_text(raw: str) -> str:
	"""What a fetch-to-markdown tool shows: rendered-markup text, <noscript> kept, scripts/styles dropped."""
	raw = re.sub(r'(?is)<(script|style|template)[^>]*>.*?</\1>', ' ', raw)
	raw = re.sub(r'(?i)<br\s*/?>|</(p|div|li|h[1-6]|tr|section|article)>', '\n', raw)
	return re.sub(r'[ \t]+', ' ', htmllib.unescape(re.sub(r'<[^>]+>', ' ', raw))).strip()


def _raw_strings(raw: str) -> str:
	"""Most charitable HTTP view: every string in the HTML incl. embedded <script> JSON (e.g. SSR payloads), unescaped."""
	raw = htmllib.unescape(raw)
	raw = re.sub(r'\\u([0-9a-fA-F]{4})', lambda m: chr(int(m.group(1), 16)), raw).replace('\\n', '\n').replace('\\"', '"').replace('\\/', '/')
	return re.sub(r'<[^>]+>', ' ', raw)


BLOCK_MARKERS = re.compile(
	r'(captcha|are you a robot|verify you are human|access denied|enable javascript|checking your browser|just a moment|cf-chl|unusual traffic|sign in to (view|continue)|log in to (view|continue))',
	re.I,
)
TRANSIENT = {None, 408, 425, 429, 500, 502, 503, 504}


async def http_capture(url: str, timeout: float = 20, retries: int = 3) -> dict:
	"""The vanilla baseline: what a fetch tool / plain HTTP crawler sees for the same URL. `raw` is the upper bound."""
	for attempt in range(retries):
		try:
			async with httpx.AsyncClient(
				follow_redirects=True,
				timeout=timeout,
				headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36'},
			) as c:
				r = await c.get(url)
		except Exception as e:
			r, err = None, f'{type(e).__name__}: {e}'[:200]
		status = r.status_code if r is not None else None
		if status not in TRANSIENT or attempt == retries - 1:
			break
		await asyncio.sleep(2 * (attempt + 1))
	if r is None:
		return {'url': url, 'status': None, 'text': '', 'raw': '', 'blocked': None, 'error': err, 'via': 'http'}
	html = 'html' in r.headers.get('content-type', '')
	text = _visible_text(r.text) if html else r.text
	# a block marker only counts on a short page (a "CAPTCHA" docs link in a long sidebar is not a wall)
	blocked = status in (401, 403, 429, 503) or len(text.strip()) < 200 or (bool(BLOCK_MARKERS.search(text[:3000])) and len(text) < 2000)
	return {'url': url, 'status': status, 'text': text, 'raw': _raw_strings(r.text) if html else r.text, 'blocked': blocked, 'transient': status in TRANSIENT, 'via': 'http'}


def host(url: str) -> str:
	return urlparse(url).netloc.lower().removeprefix('www.')
