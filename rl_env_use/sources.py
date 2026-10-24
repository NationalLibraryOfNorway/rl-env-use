"""Verified discovery routes for trial registries; never rewrite outcome evidence."""

import re
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit

CTG = 'https://clinicaltrials.gov/api/v2/studies'
HELSENORGE = 'https://www.helsenorge.no/api/v1/kliniskestudier'


def clinical_trials_url(url: str) -> str:
	parts = urlsplit(url)
	if parts.hostname in ('www.helsenorge.no', 'helsenorge.no') and parts.path.startswith('/api/'):
		if 'clinicaltrial' in parts.path.lower() or 'kliniskestudier' in parts.path.lower():
			return HELSENORGE
	if parts.hostname not in ('clinicaltrials.gov', 'www.clinicaltrials.gov') or parts.path.rstrip('/') != '/api/v2/studies':
		return url
	query = parse_qs(parts.query)
	# Use full study records: invented field names caused 400s and partial snapshots
	# cannot establish pre-results provenance. The miner must still inspect history.
	allowed = ('query.term', 'query.cond', 'query.intr', 'query.spons', 'query.id', 'query.locn', 'filter.overallStatus', 'pageToken')
	params = {k: query[k][0] for k in allowed if k in query}
	if 'country' in query:
		params['query.locn'] = query['country'][0]
	if 'filter.geo' in query and 'NO' in query['filter.geo'][0].split('|'):
		params['query.locn'] = 'Norway'
	params.setdefault('query.locn', 'Norway')
	advanced = query.get('filter.advanced', ['AREA[StudyType]INTERVENTIONAL'])[0]
	if 'filter.advanced' not in query and (query.get('results') == ['with'] or query.get('filter.hasResults') == ['true']):
		advanced += ' AND AREA[HasResults]true'
	params.update({'filter.advanced': advanced, 'pageSize': '3', 'format': 'json'})
	return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(params), ''))


def normalize_mission(mission):
	mission.start_url = clinical_trials_url(mission.start_url)
	mission.instructions = re.sub(r'https?://[^\s<>"\']+', lambda match: clinical_trials_url(match[0]), mission.instructions)
	return mission


def discovery_guidance(sources: list[str] | None) -> str:
	hosts = {urlsplit(url).hostname for url in sources or []}
	guidance = []
	if hosts & {'clinicaltrials.gov', 'www.clinicaltrials.gov'}:
		guidance.append('ClinicalTrials.gov: verified discovery query: ' + clinical_trials_url(CTG) + '. Use query.locn for location; never invent country, results, filter.geo or fields parameters. For condition use query.cond. Do not require HasResults unless registry-posted results are specifically needed: trials with published primary results may have no registry results table. Fetch full records and follow protocol/result publication references, /study/NCT... plus history for dated pre-results inputs; current API data alone does not establish a pre-results snapshot.')
	if hosts & {'www.helsenorge.no', 'helsenorge.no'}:
		guidance.append('Helsenorge: its frontend uses the verified public JSON listing ' + HELSENORGE + '. Fetch without invented filters or pagination; filter returned records by kategorier, status, tittel, sammendrag and utfort_av locally. Follow lenke to hospital pages for trial IDs and registry/results links. nvaProjectId is not an NCT ID. This API is discovery metadata, not primary-endpoint outcome evidence. Use headless Chromium for linked dynamic pages if HTTP is insufficient. Never use /api/clinicaltrials (404).')
	return '\n'.join(guidance)
