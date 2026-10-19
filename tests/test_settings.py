from vroidhunter import settings as S


def test_settings_round_trip(tmp_path):
    assert S.load(str(tmp_path)) == {}
    values = {"armor": "pl123_0000", "second_armor": "", "height": 105, "physics": False}
    assert S.save(values, str(tmp_path))
    assert S.load(str(tmp_path)) == values


def test_broken_settings_file_is_ignored(tmp_path):
    (tmp_path / S.FILE_NAME).write_text("{not json", encoding="utf-8")
    assert S.load(str(tmp_path)) == {}
