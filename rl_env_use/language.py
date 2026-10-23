"""Language-specific output templates; immutable context is isolated per async workflow."""

import inspect
import json
import os
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path

import pycountry

# Small translated output templates compose with the shared domain instructions.
# Other ISO 639-3 languages use the named-language template below.
DATASET = {
	'eng': 'Write generated dataset questions, natural-language answers, answer choices and derivations in English.',
	'nob': 'Skriv genererte spørsmål, svar i naturlig språk, svaralternativer og utledninger i datasettet på norsk bokmål.',
	'nno': 'Skriv genererte spørsmål, svar i naturleg språk, svaralternativ og utleiingar i datasettet på norsk nynorsk.',
	'nor': 'Skriv genererte spørsmål, svar i naturlig språk, svaralternativer og utledninger i datasettet på norsk. Bruk bokmål konsekvent.',
	'spa': 'Escribe las preguntas, las respuestas en lenguaje natural, las opciones y las explicaciones del conjunto de datos en español.',
}
PROBABILITY = {
	'eng': 'Answer with the probability (0 to 1) that it did.',
	'nob': 'Oppgi sannsynligheten (0 til 1) for at dette skjedde.',
	'nno': 'Oppgje sannsynet (0 til 1) for at dette skjedde.',
	'nor': 'Oppgi sannsynligheten (0 til 1) for at dette skjedde.',
	'spa': 'Indica la probabilidad (de 0 a 1) de que haya ocurrido.',
}
_CURRENT = ContextVar('artifact_language', default=None)


def language_code(value: str) -> str:
	code = value.lower() if isinstance(value, str) else ''
	if len(code) != 3 or pycountry.languages.get(alpha_3=code) is None:
		raise ValueError('language must be a three-letter ISO 639-3 code, e.g. eng, nob, nno or spa')
	return code


def settings(spec: dict | None = None) -> dict:
	spec = spec or {}
	code = language_code(spec.get('language', os.getenv('RLENV_LANGUAGE', 'eng')))
	flag = spec.get('localize_diagnostics')
	if flag is None:
		value = os.getenv('RLENV_LOCALIZE_DIAGNOSTICS', '0')
		if value not in ('0', '1'):
			raise ValueError('RLENV_LOCALIZE_DIAGNOSTICS must be 0 or 1')
		flag = value == '1'
	if not isinstance(flag, bool):
		raise ValueError('localize_diagnostics must be a boolean')
	return {'language': code, 'localize_diagnostics': flag}


def current() -> dict:
	return _CURRENT.get() or settings()


@contextmanager
def language_scope(spec: dict):
	token = _CURRENT.set(settings(spec))
	try:
		yield current()
	finally:
		_CURRENT.reset(token)


def environment_language(function):
	"""Scope stages with a spec argument, or workflows with an environment directory."""

	@wraps(function)
	async def wrapped(*args, **kwargs):
		bound = inspect.signature(function).bind(*args, **kwargs).arguments
		spec = bound.get('spec')
		if spec is None:
			path = Path(bound['env'] if 'env' in bound else bound['self'].dir) / 'spec.json'
			spec = json.loads(path.read_text()) if path.exists() else {}
		with language_scope(spec) as options:
			if 'spec' in bound:
				spec.update(options)
			return await function(*args, **kwargs)

	return wrapped


def language_prompt(audience: str = 'diagnostics') -> str:
	options = current()
	code = options['language']
	name = pycountry.languages.get(alpha_3=code).name
	diagnostic_language = name if options['localize_diagnostics'] else 'English'
	if audience == 'dataset':
		instruction = DATASET.get(code, f'Write all generated dataset questions, natural-language answers, answer choices and derivations in {name} (ISO 639-3: {code}).')
		instruction += f' Write why_hard in {name} too. For probability tasks include a target-language instruction to answer with a probability from 0 to 1. Keep original source answer forms as aliases when translating categorical gold answers; explain translated set answers in derivation.'
	elif audience == 'policy':
		instruction = (
			f'Write natural-language final answers in {name} (ISO 639-3: {code}). Write explicit reasoning, memory, evaluation and next-goal explanations in {diagnostic_language}.'
		)
	else:
		instruction = f'Write generated feedback, mining notes, observations, explanations, critiques and other diagnostic prose in {diagnostic_language}. Recomputed natural-language answer fields must use {name}, matching the dataset.'
	return (
		'\n\nOUTPUT LANGUAGE:\n'
		+ instruction
		+ (
			' Preserve source quotations and source materials verbatim, proper names, URLs, identifiers, JSON keys, enum values, arithmetic expressions, numeric/date formats and canonical grading tokens such as yes/no.'
			' Language requirements never change the meaning of an answer, schema or evidence. This overrides any earlier instruction to copy a natural-language gold verbatim: preserve its original form as an alias instead.'
		)
	)
