import json

import pytest

from rl_env_use.sft import export_sft


def fixture(tmp_path):
	task = {'id': 't1', 'case': 'trial1', 'language': 'nob', 'answer': 'no', 'context': 'Pre-results protocol',
		'fact_ids': ['f1'], 'material_ids': ['m1'], 'derivation': 'DO NOT COPY RAW REASONING',
		'checks': {'structural': [], 'leakage': False, 'live': True, 'closed_book': {'model': True},
			'critic': dict.fromkeys(('unambiguous', 'gold_supported', 'agrees', 'well_posed'), True),
			'panel': {'verdict': 'no_shift'}}}
	target = {'task_id': 't1', 'answer': 'no', 'label': 'Nei', 'prompt': 'Ble endepunktet oppfylt?',
		'explanation': 'Resultatene viste ikke overlegenhet.', 'system': 'Svar på norsk bokmål.'}
	(tmp_path / 'tasks.jsonl').write_text(json.dumps(task) + '\n')
	(tmp_path / 'kb.json').write_text(json.dumps({'facts': [{'id': 'f1', 'case': 'trial1', 'quote': 'Published result: p=0.78', 'url': 'https://example.org/result'}]}))
	targets = tmp_path / 'targets.json'
	targets.write_text(json.dumps([target]))
	return task, target, targets, tmp_path / 'dataset-sft.jsonl'


def test_sft_includes_results_and_edited_explanation_without_rl_filters(tmp_path):
	_, _, targets, out = fixture(tmp_path)
	original = (tmp_path / 'tasks.jsonl').read_bytes()
	assert export_sft(tmp_path, targets, out) == 1
	row = json.loads(out.read_text())
	assert list(row) == ['messages']
	assert 'Published result: p=0.78' in row['messages'][1]['content']
	assert row['messages'][2]['content'] == 'Nei. Resultatene viste ikke overlegenhet.'
	assert 'DO NOT COPY' not in out.read_text()
	assert (tmp_path / 'tasks.jsonl').read_bytes() == original
	assert json.loads(out.with_suffix('.audit.json').read_text())[0]['rl_panel_verdict'] == 'no_shift'


@pytest.mark.parametrize('failure', ['wrong_answer', 'wrong_label', 'critic', 'live', 'rejected_fact', 'rejected_material', 'review', 'overwrite'])
def test_sft_rejects_unsupported_or_contradictory_targets(tmp_path, failure):
	task, target, targets, out = fixture(tmp_path)
	if failure == 'wrong_answer':
		target['answer'] = 'yes'
	if failure == 'wrong_label':
		target['label'] = 'Ja'
	if failure == 'critic':
		task['checks']['critic']['gold_supported'] = False
	if failure == 'live':
		task['checks']['live'] = 'skipped'
	if failure in ('rejected_fact', 'rejected_material'):
		(tmp_path / 'data_reviews.jsonl').write_text(json.dumps({'id': 'f1' if failure == 'rejected_fact' else 'm1', 'verdict': 'reject'}) + '\n')
	if failure == 'review':
		(tmp_path / 'reviews.jsonl').write_text(json.dumps({'id': 't1', 'verdict': 'reject'}) + '\n')
	if failure == 'overwrite':
		out = tmp_path / 'tasks.jsonl'
	(tmp_path / 'tasks.jsonl').write_text(json.dumps(task) + '\n')
	targets.write_text(json.dumps([target]))
	with pytest.raises(ValueError):
		export_sft(tmp_path, targets, out)
