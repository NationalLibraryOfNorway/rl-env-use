from urllib.parse import parse_qs, urlsplit

from rl_env_use.scout import Mission
from rl_env_use.sources import HELSENORGE, clinical_trials_url, discovery_guidance, normalize_mission


def test_bad_clinical_trials_query_is_replaced_with_valid_parameters():
	url = clinical_trials_url('https://clinicaltrials.gov/api/v2/studies?query.term=oncology&filter.geo=country%7CNO&results=with&fields=StudyDesign,Locations&country=Norway&pageSize=1000')
	query = parse_qs(urlsplit(url).query)
	assert query['query.locn'] == ['Norway']
	assert query['query.term'] == ['oncology']
	assert query['filter.advanced'] == ['AREA[StudyType]INTERVENTIONAL AND AREA[HasResults]true']
	assert not {'fields', 'results', 'country', 'filter.geo'} & query.keys()
	assert clinical_trials_url(url) == url


def test_helsenorge_listing_and_mission_links_are_corrected():
	bad = 'https://www.helsenorge.no/api/clinicaltrials?status=completed&page=1'
	mission = normalize_mission(Mission(angle='cases', instructions=f'Start at {bad} then inspect linked trials', start_url=bad))
	assert mission.start_url == HELSENORGE
	assert bad not in mission.instructions and HELSENORGE in mission.instructions
	assert clinical_trials_url('https://clinicaltrials.gov/study/NCT00437645') == 'https://clinicaltrials.gov/study/NCT00437645'
	assert clinical_trials_url('https://unrelated.test/api/clinicaltrials') == 'https://unrelated.test/api/clinicaltrials'


def test_publication_backed_discovery_does_not_require_registry_results():
	url = clinical_trials_url('https://clinicaltrials.gov/api/v2/studies?query.locn=Norway')
	assert parse_qs(urlsplit(url).query)['filter.advanced'] == ['AREA[StudyType]INTERVENTIONAL']
	assert clinical_trials_url(url) == url
	advanced = 'AREA[StudyType]INTERVENTIONAL AND AREA[Phase]PHASE3'
	url = clinical_trials_url('https://clinicaltrials.gov/api/v2/studies?filter.advanced=' + advanced)
	assert parse_qs(urlsplit(url).query)['filter.advanced'] == [advanced]


def test_old_has_results_parameter_is_translated_only_when_requested():
	url = clinical_trials_url('https://clinicaltrials.gov/api/v2/studies?filter.hasResults=true')
	query = parse_qs(urlsplit(url).query)
	assert 'filter.hasResults' not in query
	assert query['filter.advanced'] == ['AREA[StudyType]INTERVENTIONAL AND AREA[HasResults]true']


def test_guidance_requires_provenance_and_never_invents_outcome_evidence():
	text = discovery_guidance(['https://clinicaltrials.gov/', 'https://www.helsenorge.no/en/clinical-trials/'])
	assert HELSENORGE in text and 'nvaProjectId is not an NCT ID' in text
	assert 'current API data alone does not establish a pre-results snapshot' in text
	assert discovery_guidance(['https://unrelated.test']) == ''


def test_global_run_deadline_cancels_work():
	import asyncio

	import pytest

	from rl_env_use.__main__ import run

	cancelled = []
	async def work():
		try:
			await asyncio.sleep(10)
		finally:
			cancelled.append(True)
	with pytest.raises(TimeoutError):
		run(work(), timeout=0.01)
	assert cancelled == [True]
