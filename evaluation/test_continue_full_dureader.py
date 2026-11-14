import pytest
from evaluation.continue_full_dureader import wait_protocol


def test_missing_producer_fails_without_creating_capture(tmp_path):
    directory=tmp_path/'capture'
    with pytest.raises(RuntimeError,match='stopped before publishing'):
        wait_protocol(directory,999999999,'evaluation.capture_compact_bm25')
    assert not directory.exists()
