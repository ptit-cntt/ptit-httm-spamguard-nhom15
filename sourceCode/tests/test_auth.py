from app import auth


def test_is_session_token_valid_fresh_token_returns_true():
    token = auth.create_session_token(auth.ADMIN_USERNAME, now=1_000)
    assert auth.is_session_token_valid(token, now=1_000 + 60)


def test_is_session_token_valid_expired_token_returns_false():
    token = auth.create_session_token(auth.ADMIN_USERNAME, now=1_000)
    assert not auth.is_session_token_valid(token, now=1_000 + auth.SESSION_TTL_SECONDS + 1)


def test_is_session_token_valid_malformed_token_returns_false():
    assert not auth.is_session_token_valid("not-a-token")
    assert not auth.is_session_token_valid("abc.def")


def test_is_admin_token_valid_static_admin_token_returns_true():
    assert auth.is_admin_token_valid(auth.ADMIN_TOKEN)
    assert not auth.is_admin_token_valid("")
