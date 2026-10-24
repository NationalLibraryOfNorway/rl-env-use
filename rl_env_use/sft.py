"""Export curated outcome-interpretation targets, never hindsight forecasts or raw traces."""

import hashlib
import json
from pathlib import Path

from .store import write_atomic
from .tasks import load_tasks, rejected_facts


def export_sft(env_dir, targets_path, out_path) -> int:
	"""Targets are explicitly edited prompts/explanations; evidence is always included in the input.

	Correctness checks remain required. RL difficulty, memorization and forecast balance do not
	apply to this different, evidence-interpretation task. Original RL tasks are never modified.
	"""
	env = Path(env_dir)
	tasks = {t['id']: t for t in load_tasks(env)}
	kb = json.loads((env / 'kb.json').read_text())
	facts = {f['id']: f for f in kb['facts']}
	bad = rejected_facts(env)
	targets = json.loads(Path(targets_path).read_text())
	rows, audit, seen = [], [], set()
	for target in targets:
		tid = target['task_id']
		if tid in seen:
			raise ValueError(f'{tid}: duplicate target')
		seen.add(tid)
		task = tasks[tid]
		c = task['checks']
		critic = c.get('critic')
		if (c.get('structural') != [] or c.get('leakage') is not False or c.get('live') is not True
			or not isinstance(critic, dict) or not all(critic.get(k) is True for k in ('unambiguous', 'gold_supported', 'agrees', 'well_posed'))
			or task.get('review', {}).get('verdict') == 'reject'):
			raise ValueError(f'{tid}: requires passing source and correctness checks')
		if task['answer'] not in ('yes', 'no') or target['answer'] != task['answer']:
			raise ValueError(f'{tid}: target must match the verified binary outcome, not a forecast probability')
		for key in ('prompt', 'explanation', 'label', 'system'):
			if not isinstance(target.get(key), str) or not target[key].strip():
				raise ValueError(f'{tid}: missing curated {key}')
		labels = {'eng': {'yes': 'yes', 'no': 'no'}, 'nob': {'yes': 'ja', 'no': 'nei'},
			'nno': {'yes': 'ja', 'no': 'nei'}, 'nor': {'yes': 'ja', 'no': 'nei'}, 'spa': {'yes': 'sí', 'no': 'no'}}
		if task['language'] in labels and target['label'].casefold() != labels[task['language']][target['answer']]:
			raise ValueError(f'{tid}: localized label contradicts the verified answer')
		linked = [facts[fid] for fid in task['fact_ids']]
		if bad.intersection(task.get('material_ids', [])) or not linked or any(f['id'] in bad or f['case'] != task['case'] or not f.get('quote') or not f.get('url') for f in linked):
			raise ValueError(f'{tid}: missing, rejected or mismatched outcome evidence')
		# Source quotations stay verbatim, including original language. Generated text uses target language.
		evidence = '\n\n'.join(f"[{f['id']}] {f['url']}\n{f['quote']}" for f in linked)
		messages = [
			{'role': 'system', 'content': target['system']},
			{'role': 'user', 'content': target['prompt'] + '\n\n---\n\n' + task['context'] + '\n\n---\n\n' + evidence},
			{'role': 'assistant', 'content': target['label'] + '. ' + target['explanation']},
		]
		rows.append({'messages': messages})
		audit.append({'row': len(rows), 'task_id': tid, 'case': task['case'], 'language': task['language'],
			'outcome': target['answer'], 'task_type': 'outcome_interpretation',
			'explanation_origin': 'curated_final_explanation', 'explanation_model': target.get('explanation_model', 'unspecified'), 'human_reviewed': False,
			'source_urls': [f.get('primary_publication', f['url']) for f in linked],
			'evidence_sha256': hashlib.sha256(evidence.encode()).hexdigest(),
			'target_sha256': hashlib.sha256(json.dumps(target, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
			'rl_panel_verdict': c.get('panel', {}).get('verdict')})
	out = Path(out_path)
	if out.suffix != '.jsonl':
		raise ValueError('SFT output must use the .jsonl extension')
	if out.resolve() in {Path(targets_path).resolve()} | {(env / name).resolve() for name in ('tasks.jsonl', 'dataset.jsonl', 'kb.json', 'spec.json', 'reviews.jsonl', 'data_reviews.jsonl')}:
		raise ValueError('SFT output must be separate from RL tasks, exports and reviews')
	write_atomic(out, ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
	write_atomic(out.with_suffix('.audit.json'), json.dumps(audit, indent=2, ensure_ascii=False) + '\n')
	return len(rows)
