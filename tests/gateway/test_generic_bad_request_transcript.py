"""A bare HTTP 400 and history size do not establish a context overflow."""
import pytest

from gateway.run_turn import is_context_overflow_failure_result


@pytest.mark.parametrize('reason', [None, 'server_error', 'format_error'])
def test_generic_400_does_not_discard_long_transcript(reason):
    result = {'failed': True, 'error': 'HTTP 400: {"detail":"Bad Request"}'}
    if reason:
        result['failure_reason'] = reason
    assert not is_context_overflow_failure_result(result, history_len=200)


def test_generic_exception_keeps_native_transcript(tmp_path):
    import asyncio
    from gateway.config import GatewayConfig, Platform
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource, SessionStore

    async def check():
        store = SessionStore(tmp_path / 'sessions', GatewayConfig())
        runner = object.__new__(GatewayRunner)
        runner.session_store = store

        async def stop_typing(event, source):
            return None

        runner._hmwa_stop_typing_for_turn = stop_typing
        source = SessionSource(platform=Platform.TELEGRAM, chat_id='fixture', user_id='fixture')
        entry = store.get_or_create_session(source)
        db = store._db_for_session_id(entry.session_id)
        prepared = runner._PreparedTurn(
            [{'role': 'user', 'content': 'fixture'}] * 100, '', 'new question', 'new question',
            None, None, entry.session_id, 'owner-generic-400',
        )
        error = RuntimeError('HTTP 400: {"detail":"Bad Request"}')
        error.status_code = 400
        try:
            await runner._hmwa_agent_error_reply(
                error, MessageEvent(text='new question', source=source, message_id='fixture-message'),
                source, entry, entry.session_key, prepared,
            )
            assert store.has_input_owner(entry.session_id, 'owner-generic-400')
            assert db.message_count() == 2
        finally:
            db.close()

    asyncio.run(check())


@pytest.mark.parametrize('result', [
    {'failed': True, 'compression_exhausted': True, 'error': 'Bad Request'},
    {'failed': True, 'failure_reason': 'context_overflow', 'error': 'Bad Request'},
    {'failed': True, 'error': 'maximum context length exceeded'},
])
def test_real_context_overflow_still_does_not_grow_session(result):
    assert is_context_overflow_failure_result(result, history_len=200)
