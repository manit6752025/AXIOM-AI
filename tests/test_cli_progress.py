from __future__ import annotations

import signal

import pytest

from axiom.cli.errors import EXIT_CODE_CANCELLED
from axiom.cli.progress import cancellable_byte_progress, cancellable_scan_progress, report_cancelled
from axiom.datasets.progress import DatasetScanCancelled, check_scan_cancelled


def test_scan_progress_sets_cancel_event_on_first_ctrl_c_and_restores_handler():
    original = signal.getsignal(signal.SIGINT)
    with cancellable_scan_progress("test scan", 1024, enabled=False) as (cancel_event, update):
        assert update is None
        handler = signal.getsignal(signal.SIGINT)
        handler(signal.SIGINT, None)
        assert cancel_event.is_set()
        with pytest.raises(DatasetScanCancelled, match="cancelled"):
            check_scan_cancelled(cancel_event)
        with pytest.raises(KeyboardInterrupt):
            handler(signal.SIGINT, None)
    assert signal.getsignal(signal.SIGINT) is original


def test_download_progress_sets_cancel_event_on_ctrl_c():
    original = signal.getsignal(signal.SIGINT)
    with cancellable_byte_progress("test download", 2048, enabled=False) as (cancel_event, update):
        assert update is None
        signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
        assert cancel_event.is_set()
        with pytest.raises(DatasetScanCancelled):
            check_scan_cancelled(cancel_event)
    assert signal.getsignal(signal.SIGINT) is original



def test_report_cancelled_uses_documented_exit_code_and_stderr_only(capsys):
    with pytest.raises(SystemExit) as result:
        report_cancelled("Operation cancelled.")

    assert result.value.code == EXIT_CODE_CANCELLED
    assert result.value.code == 130
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Operation cancelled." in captured.err
