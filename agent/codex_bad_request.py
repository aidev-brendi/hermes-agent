"""Bound the identical retry of Codex's detail-only HTTP 400 response.

No payload repair, credential rotation, model fallback or tool replay belongs here.
"""
from __future__ import annotations

import logging

from agent.turn_recovery import interruptible_backoff_sleep

logger = logging.getLogger(__name__)


def is_generic_codex_bad_request(error, *, provider: str) -> bool:
    from agent.error_classifier import _extract_error_body, _extract_status_code

    return (
        provider == 'openai-codex'
        and _extract_status_code(error) == 400
        and _extract_error_body(error) == {'detail': 'Bad Request'}
    )


def recover_generic_codex_bad_request(
    agent, *, api_error, classified, retry, messages, conversation_history, api_call_count,
):
    """Return a loop action only for the exact observed response, at most once per call."""
    from agent.error_classifier import FailoverReason

    if (classified.reason != FailoverReason.server_error or not classified.retryable
            or classified.should_compress or classified.should_rotate_credential
            or not is_generic_codex_bad_request(api_error, provider=getattr(agent, 'provider', ''))):
        return None
    if retry.codex_generic400_retry_attempted:
        from agent.turn_recovery import _failed_turn_result

        logger.warning('Codex detail-only HTTP 400 persisted after one identical retry')
        agent._persist_session(messages, conversation_history)
        summary = 'HTTP 400: {"detail":"Bad Request"}; one identical retry exhausted.'
        result = _failed_turn_result(
            'The provider rejected the request twice with a detail-only Bad Request. '
            'The turn history was preserved; no further automatic retry was attempted.',
            messages, api_call_count, summary,
        )
        # The error may be transient, but this call's explicit one-retry budget is spent.
        result.update(failure_reason='server_error', failure_retryable=False)
        return 'return', result
    retry.codex_generic400_retry_attempted = True
    logger.warning('Codex detail-only HTTP 400: retrying the identical request once after 1s')
    stopped = interruptible_backoff_sleep(
        agent, 1.0, retry, messages=messages, conversation_history=conversation_history,
        api_call_count=api_call_count, abort_message='Stopped during the bounded Codex retry.',
        interrupt_text='Operation interrupted before the Codex retry.',
        activity_label='Bounded Codex HTTP 400 retry',
    )
    if stopped is not None:
        return 'return', stopped
    if retry.restart_with_redirected_messages:
        return 'break', None
    return 'continue', None
