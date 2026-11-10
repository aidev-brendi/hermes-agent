"""A detail-only Codex 400 has one bounded identical retry, not context recovery."""

from unittest.mock import MagicMock, patch

import httpx
import openai
import pytest

from agent.error_classifier import FailoverReason, classify_api_error
from agent.turn_retry_state import TurnRetryState


def rejection(body=None, status=400):
    body = {'detail': 'Bad Request'} if body is None else body
    response = httpx.Response(status, json=body, request=httpx.Request('POST', 'https://chatgpt.com/backend-api/codex/responses'))
    return openai.BadRequestError('HTTP 400: ' + response.text, response=response, body=body)


def test_generic_codex_400_is_transient_even_on_large_context():
    verdict = classify_api_error(rejection(), provider='openai-codex', approx_tokens=247000, context_length=900000, num_messages=100)
    assert verdict.reason == FailoverReason.server_error
    assert verdict.retryable and not verdict.should_compress


@pytest.mark.parametrize('body', [
    {'detail': 'Bad Request', 'code': 'content_policy_blocked'},
    {'detail': 'blocked by our safety system'},
    {'error': {'code': 'context_length_exceeded', 'message': 'maximum context length exceeded'}},
    {'detail': 'invalid_encrypted_content'},
    {'detail': 'Unsupported parameter: tools'},
])
def test_detailed_400_does_not_use_generic_retry(body):
    from agent.codex_bad_request import is_generic_codex_bad_request
    assert not is_generic_codex_bad_request(rejection(body), provider='openai-codex')


@pytest.mark.parametrize('provider,status', [('openai', 400), ('anthropic', 400), ('openai-codex', 401), ('openai-codex', 403), ('openai-codex', 429)])
def test_retry_is_scoped_to_codex_and_400(provider, status):
    from agent.codex_bad_request import is_generic_codex_bad_request
    assert not is_generic_codex_bad_request(rejection(status=status), provider=provider)


def test_one_retry_then_failure_persists_without_mutating_history():
    from agent.codex_bad_request import recover_generic_codex_bad_request
    agent = MagicMock(provider='openai-codex')
    messages = [{'role': 'user', 'content': 'fixture'}, {'role': 'assistant', 'tool_calls': [{'id': 'read', 'type': 'function', 'function': {'name': 'read_file', 'arguments': '{}'}}]}, {'role': 'tool', 'tool_call_id': 'read', 'content': 'already executed'}]
    snapshot = repr(messages)
    history = []
    state = TurnRetryState()
    kwargs = dict(api_error=rejection(), classified=classify_api_error(rejection(), provider='openai-codex'), retry=state, messages=messages, conversation_history=history, api_call_count=31)
    with patch('agent.codex_bad_request.interruptible_backoff_sleep', return_value=None) as sleep:
        assert recover_generic_codex_bad_request(agent, **kwargs) == ('continue', None)
        action, result = recover_generic_codex_bad_request(agent, **kwargs)
    assert sleep.call_count == 1 and sleep.call_args.args[1] == 1.0
    assert action == 'return' and result['failed'] and not result['completed']
    assert result['failure_reason'] == 'server_error' and result['failure_retryable'] is False
    assert result['messages'] is messages and result['api_calls'] == 31
    assert repr(messages) == snapshot
    agent._persist_session.assert_called_once_with(messages, history)


@pytest.mark.parametrize('override', [
    {'reason': FailoverReason.content_policy_blocked},
    {'reason': FailoverReason.format_error},
    {'retryable': False},
    {'should_compress': True},
    {'should_rotate_credential': True},
])
def test_explicit_classifier_guards_are_not_bypassed(override):
    from dataclasses import replace
    from agent.codex_bad_request import recover_generic_codex_bad_request
    classified = replace(classify_api_error(rejection(), provider='openai-codex'), **override)
    with patch('agent.codex_bad_request.interruptible_backoff_sleep') as sleep:
        assert recover_generic_codex_bad_request(
            MagicMock(provider='openai-codex'), api_error=rejection(), classified=classified,
            retry=TurnRetryState(), messages=[], conversation_history=[], api_call_count=1,
        ) is None
    sleep.assert_not_called()


@pytest.mark.parametrize('persistent', [False, True])
def test_real_codex_loop_retries_once_with_identical_payload(monkeypatch, persistent):
    import copy
    from tests.agent.test_run_agent_codex_responses import _build_agent, _codex_message_response

    agent = _build_agent(monkeypatch)
    agent.provider = 'openai-codex'
    assert agent.api_mode == 'codex_responses'
    wires = []

    def call(kwargs):
        wires.append(copy.deepcopy(kwargs))
        if len(wires) == 1 or persistent:
            raise rejection()
        return _codex_message_response('Recovered.')

    monkeypatch.setattr(agent, '_interruptible_api_call', call)
    monkeypatch.setattr('agent.codex_bad_request.interruptible_backoff_sleep', lambda *a, **kw: None)
    result = agent.run_conversation('Continue the fixture investigation.')
    assert len(wires) == 2
    assert wires[0] == wires[1]
    assert result['failed'] if persistent else result['completed']
    if not persistent:
        assert result['final_response'] == 'Recovered.'



def test_interrupt_and_redirect_are_not_retried():
    from agent.codex_bad_request import recover_generic_codex_bad_request
    agent = MagicMock(provider='openai-codex')
    kwargs = dict(api_error=rejection(), classified=classify_api_error(rejection(), provider='openai-codex'), retry=TurnRetryState(), messages=[], conversation_history=[], api_call_count=1)
    with patch('agent.codex_bad_request.interruptible_backoff_sleep', return_value={'interrupted': True}):
        assert recover_generic_codex_bad_request(agent, **kwargs) == ('return', {'interrupted': True})
    state = TurnRetryState()
    def steer(*a, **kw):
        state.restart_with_redirected_messages = True
    with patch('agent.codex_bad_request.interruptible_backoff_sleep', side_effect=steer):
        assert recover_generic_codex_bad_request(agent, **dict(kwargs, retry=state)) == ('break', None)
