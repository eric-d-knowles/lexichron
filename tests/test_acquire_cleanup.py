from ngramprep.ngram_acquire.utils.cleanup import safe_db_cleanup


def test_removes_directory(tmp_path):
    d = tmp_path / "db"
    (d / "sub").mkdir(parents=True)
    (d / "sub" / "f").write_text("x")
    assert safe_db_cleanup(d) is True
    assert not d.exists()


def test_missing_path_is_fine(tmp_path):
    assert safe_db_cleanup(tmp_path / "nope") is True


def test_symlink_is_unlinked_but_target_kept(tmp_path):
    target = tmp_path / "real_db"
    target.mkdir()
    (target / "CURRENT").write_text("x")
    link = tmp_path / "db_link"
    link.symlink_to(target)

    assert safe_db_cleanup(link) is True
    assert not link.exists() and not link.is_symlink()
    assert (target / "CURRENT").exists()
