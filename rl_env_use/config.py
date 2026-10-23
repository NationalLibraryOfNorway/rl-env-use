"""Open-stack defaults. Load configuration before importing browser-use."""

import os
from pathlib import Path


def load_dotenv() -> None:
	"""Working-directory .env wins over repository .env; exported variables win over both."""
	for path in (Path('.env'), Path(__file__).resolve().parents[1] / '.env'):
		for line in path.read_text().splitlines() if path.is_file() else []:
			k, _, v = line.strip().partition('=')
			if k and not k.startswith('#') and v:
				os.environ.setdefault(k, v.strip('"\''))


load_dotenv()


def open_only() -> bool:
	value = os.getenv('RLENV_OPEN_ONLY', '1')
	if value not in ('0', '1'):
		raise ValueError('RLENV_OPEN_ONLY must be 1 (self-hosted only) or 0 (hosted compatibility)')
	return value == '1'


if open_only():
	# Set before browser-use initializes telemetry and cloud synchronization.
	os.environ['ANONYMIZED_TELEMETRY'] = 'false'
	os.environ['BROWSER_USE_CLOUD_SYNC'] = 'false'
	os.environ['BROWSER_USE_CALCULATE_COST'] = 'false'

TEXT_MODEL = 'local-text' if open_only() else 'sonnet46'
BROWSER_MODEL = 'local-browser' if open_only() else 'bu'
CLOSED_BOOK = ['local-text'] if open_only() else ['haiku', 'sonnet46']
PANEL_MODELS = [m.strip() for m in os.getenv('RLENV_PANEL_MODELS', 'local-text').split(',') if m.strip()]
if open_only() and not PANEL_MODELS:
	raise ValueError('RLENV_PANEL_MODELS must contain at least one local model alias or local:MODEL_ID')
