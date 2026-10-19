"""JSONL persistence shared by every stage."""

import json
import os
from pathlib import Path


def load_jsonl(p: Path) -> list[dict]:
	out = []
	for line in p.read_text().splitlines() if p.exists() else []:
		try:
			out.append(json.loads(line)) if line.strip() else None
		except json.JSONDecodeError:  # the last line of a killed run's append: that row is redone, never a crash
			pass
	return out


def write_atomic(p: Path, text: str) -> None:
	"""A kill mid-write leaves the old file, never a truncated one (kb.json holds every mined case)."""
	tmp = p.with_name(p.name + '.tmp')
	tmp.write_text(text)
	os.replace(tmp, p)


def save_jsonl(p: Path, rows: list[dict]) -> None:
	write_atomic(p, ''.join(json.dumps(r) + '\n' for r in rows))
