import json
import threading
import urllib.error
import urllib.request

import pytest
from conftest import jl

from rl_env_use.annotate import HTML, serve


@pytest.fixture
def server(env_dir):
	srv = serve(env_dir, port=0)
	threading.Thread(target=srv.serve_forever, daemon=True).start()
	yield f'http://127.0.0.1:{srv.server_address[1]}'
	srv.shutdown()


def get(url):
	with urllib.request.urlopen(url) as r:
		return r.status, r.read()


def post(url, body):
	req = urllib.request.Request(url, json.dumps(body).encode(), {'Content-Type': 'application/json'})
	with urllib.request.urlopen(req) as r:
		return json.loads(r.read())


def test_items_and_details(server):
	tasks = json.loads(get(server + '/api/items?mode=tasks')[1])
	traces = json.loads(get(server + '/api/items?mode=traces')[1])
	assert len(tasks) == 23 and len(traces) == 20
	assert traces[0]['label'] == 'pass' and traces[19]['label'] is None
	d = json.loads(get(server + '/api/item?mode=traces&id=baseline/t7_rep0')[1])
	assert d['row']['prompt_id'] == 't7' and d['trace'][-1]['content'].startswith('FINAL ANSWER') and d['label']['modes'] == ['premature_stop']
	d = json.loads(get(server + '/api/item?mode=tasks&id=t3')[1])
	assert d['evidence'][0]['quote'] == 'from 137 websites'


def test_label_roundtrip_latest_wins(server, env_dir):
	post(server + '/api/label', {'mode': 'traces', 'id': 'baseline/t19_rep0', 'verdict': 'fail', 'notes': 'n', 'modes': ['x']})
	post(server + '/api/label', {'mode': 'traces', 'id': 'baseline/t19_rep0', 'verdict': 'pass', 'notes': 'fixed'})
	post(server + '/api/label', {'mode': 'tasks', 'id': 't1', 'verdict': 'reject', 'notes': 'ambiguous', 'gold_fix': ''})
	items = {i['id']: i['label'] for i in json.loads(get(server + '/api/items?mode=traces')[1])}
	assert items['baseline/t19_rep0'] == 'pass'
	assert jl(env_dir / 'reviews.jsonl')[-1]['verdict'] == 'reject'


def test_rejects_bad_input_and_traversal(server):
	with pytest.raises(urllib.error.HTTPError) as e:
		post(server + '/api/label', {'mode': 'traces', 'id': 'x', 'verdict': 'maybe'})
	assert e.value.code == 400
	for p in ['/file?p=../tasks.jsonl', '/../../etc/passwd', '/tasks.jsonl']:  # nothing but the app and its API is served
		with pytest.raises(urllib.error.HTTPError) as e:
			get(server + p)
		assert e.value.code == 404


def test_ui_never_injects_untrusted_html():
	src = HTML.read_text()
	assert 'innerHTML' not in src and 'insertAdjacentHTML' not in src and 'document.write' not in src


def test_blocks_cross_site_writes(server):
	body = json.dumps({'mode': 'tasks', 'id': 't1', 'verdict': 'accept', 'gold_fix': 'poison'}).encode()
	for headers in ({'Content-Type': 'text/plain'}, {'Content-Type': 'application/json', 'Origin': 'http://evil.com'}, {'Content-Type': 'application/json', 'Host': 'evil.com'}):
		with pytest.raises(urllib.error.HTTPError) as e:
			urllib.request.urlopen(urllib.request.Request(server + '/api/label', body, headers))
		assert e.value.code == 403


def test_data_tab_shows_mined_facts_and_rejects_feed_synth(server, env_dir):
	from rl_env_use.pipeline import kb_for_synth

	facts = json.loads(get(server + '/api/items?mode=data')[1])
	assert len(facts) == 23 and facts[0]['title'] == 'c = 137' and 'arxiv.org' in facts[0]['tags']
	assert json.loads(get(server + '/api/item?mode=data&id=f2')[1])['fact']['quote'] == 'from 137 websites'
	post(server + '/api/label', {'mode': 'data', 'id': 'f2', 'verdict': 'reject', 'notes': 'wrong entity'})
	ids = [f['id'] for f in kb_for_synth(env_dir)['facts']]
	assert 'f2' not in ids and len(ids) == 22
