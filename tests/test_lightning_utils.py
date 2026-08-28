import time

import pytest

from nsbi.utils.lightning_utils import find_latest_checkpoint


def test_missing_path_returns_none(tmp_path):
    assert find_latest_checkpoint(tmp_path / "does_not_exist") is None


def test_empty_dir_returns_none(tmp_path):
    assert find_latest_checkpoint(tmp_path) is None


def test_picks_most_recently_created(tmp_path):
    older = tmp_path / "epoch=0.ckpt"
    older.write_text("older")
    time.sleep(0.05)
    newer = tmp_path / "sub" / "epoch=1.ckpt"
    newer.parent.mkdir()
    newer.write_text("newer")

    assert find_latest_checkpoint(tmp_path) == newer


def test_accepts_str_path(tmp_path):
    ckpt = tmp_path / "best.ckpt"
    ckpt.write_text("x")
    assert find_latest_checkpoint(str(tmp_path)) == ckpt


def test_string_template(tmp_path):
    (tmp_path / "model.pt").write_text("x")
    assert find_latest_checkpoint(tmp_path, templates="*.pt") == tmp_path / "model.pt"
    # default template only matches *.ckpt
    assert find_latest_checkpoint(tmp_path) is None


def test_invalid_templates_type_raises(tmp_path):
    with pytest.raises(ValueError, match="Templates"):
        find_latest_checkpoint(tmp_path, templates=123)
