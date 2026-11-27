ADMIN = {"X-Admin-Token": "test-token"}


def test_health_and_ready(client):
    assert client.get("/health").json()["status"] == "ok"
    ready = client.get("/ready").json()
    assert ready["model_type"] == "svm" and ready["model_version"] == 1


def test_classify_spam_message_returns_spam_label(client):
    response = client.post("/v1/messages/classify", json={"text": "win free cash prize now, claim now"})
    body = response.json()
    assert response.status_code == 200
    assert body["label"] == "spam"
    assert set(body["scores"]) == {"ham", "spam", "toxic"}
    assert abs(sum(body["scores"].values()) - 1) < 1e-3


def test_classify_blank_text_rejected(client):
    assert client.post("/v1/messages/classify", json={"text": "   "}).status_code == 422


def test_classify_too_long_text_rejected(client):
    assert client.post("/v1/messages/classify", json={"text": "a" * 2001}).status_code == 422


def test_classify_batch_returns_one_result_per_item(client):
    items = [{"text": "đồ ngu như bò", "channel": "comment"}, {"text": "see you at lunch", "channel": "chat"}]
    results = client.post("/v1/messages/classify-batch", json={"items": items}).json()["results"]
    assert len(results) == 2


def test_feedback_with_unknown_message_returns_404(client):
    assert client.post("/v1/feedback", json={"message_id": 999999, "correct_label": "ham"}).status_code == 404


def test_feedback_by_message_id_accepted(client):
    message_id = client.post("/v1/messages/classify", json={"text": "hello friend"}).json()["id"]
    response = client.post("/v1/feedback", json={"message_id": message_id, "correct_label": "ham"})
    assert response.status_code == 201


def test_admin_endpoints_require_token(client):
    assert client.get("/v1/admin/stats").status_code == 401
    assert client.get("/v1/admin/stats", headers={"X-Admin-Token": "wrong"}).status_code == 401


def test_admin_stats_counts_messages(client):
    client.post("/v1/messages/classify", json={"text": "you stupid idiot", "sender_id": "u1"})
    stats = client.get("/v1/admin/stats?days=1", headers=ADMIN).json()
    assert stats["total"] >= 1 and stats["model"]["version"] == 1


def test_admin_review_creates_admin_feedback(client):
    message_id = client.post("/v1/messages/classify", json={"text": "nhận quà miễn phí"}).json()["id"]
    response = client.post(f"/v1/admin/messages/{message_id}/review", json={"label": "spam"}, headers=ADMIN)
    assert response.status_code == 200
    feedback = client.get("/v1/admin/feedback", headers=ADMIN).json()
    assert any(f["source"] == "admin" and f["label"] == "spam" for f in feedback)


def test_admin_models_lists_registry(client):
    index = client.get("/v1/admin/models", headers=ADMIN).json()
    assert index["aliases"]["production"] == 1


def test_metrics_endpoint_exposes_prometheus_metrics(client):
    body = client.get("/metrics").text
    assert "spamguard_predictions_total" in body and "spamguard_model_info" in body


def test_login_with_valid_credentials_returns_session_token(client):
    response = client.post("/v1/auth/login", json={"username": "admin", "password": "1234"})
    body = response.json()
    assert response.status_code == 200
    assert body["username"] == "admin" and body["expires_in"] > 0
    assert client.get("/v1/admin/stats", headers={"X-Admin-Token": body["access_token"]}).status_code == 200


def test_login_with_wrong_password_returns_401(client):
    assert client.post("/v1/auth/login", json={"username": "admin", "password": "wrong"}).status_code == 401
    assert client.post("/v1/auth/login", json={"username": "root", "password": "1234"}).status_code == 401


def test_admin_endpoint_with_tampered_session_token_returns_401(client):
    token = client.post("/v1/auth/login", json={"username": "admin", "password": "1234"}).json()["access_token"]
    payload, signature = token.split(".")
    tampered = f"{payload}.{signature[:-2]}xx"
    assert client.get("/v1/admin/stats", headers={"X-Admin-Token": tampered}).status_code == 401
