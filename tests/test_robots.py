import httpx
import pytest

from rl_env_use.robots import RobotsGuard, rules_allow


def test_longest_rule_wins_including_allow_exceptions_and_wildcards():
	policy = 'User-agent: *\nDisallow: /api/\nAllow: /api/seo/\nDisallow: /*/private/*\nAllow: /same\nDisallow: /same\nCrawl-delay: 1'
	assert rules_allow(policy, 'https://a.org/api/v2/studies') == (False, 1)
	assert rules_allow(policy, 'https://a.org/api/seo/studies') == (True, 1)
	assert rules_allow(policy, 'https://a.org/x/private/file.pdf') == (False, 1)
	assert rules_allow(policy, 'https://a.org/same') == (True, 1)
	assert rules_allow('User-agent: *\nDisallow: /x$', 'https://a.org/xy')[0]


def test_specific_agent_and_encoded_paths():
	policy = 'User-agent: *\nDisallow: /\nUser-agent: rl-env-use\nDisallow: /private\nAllow: /'
	assert rules_allow(policy, 'https://a.org/public')[0]
	assert not rules_allow(policy, 'https://a.org/%70rivate')[0]


@pytest.mark.asyncio
async def test_disallowed_download_is_never_requested_and_redirects_are_checked(tmp_path):
	calls = []

	def handle(request):
		calls.append(str(request.url))
		if request.url.path == '/robots.txt':
			return httpx.Response(200, text='User-agent: *\nDisallow: /blocked')
		return httpx.Response(302, headers={'location': '/blocked'})

	async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
		guard = RobotsGuard(tmp_path, client, 0)
		with pytest.raises(PermissionError):
			await guard.get('https://a.org/blocked')
		assert calls == ['https://a.org/robots.txt']
		with pytest.raises(PermissionError):
			await guard.get('https://a.org/allowed')
		assert calls == ['https://a.org/robots.txt', 'https://a.org/allowed']


@pytest.mark.asyncio
@pytest.mark.parametrize('code,allowed', [(404, True), (403, False), (503, False)])
async def test_missing_policy_differs_from_unavailable_policy(tmp_path, code, allowed):
	async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(code))) as client:
		guard = RobotsGuard(tmp_path, client, 0)
		if allowed:
			assert (await guard.check('https://a.org/study'))['allowed']
		else:
			with pytest.raises(PermissionError):
				await guard.check('https://a.org/study')
