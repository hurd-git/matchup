import matchup


def test_public_api_is_available() -> None:
    assert matchup.__version__
    assert callable(matchup.picture_match)
    assert callable(matchup.picture_match_64)
    assert callable(matchup.resize_image)
    assert matchup.PreparedImage is not None
