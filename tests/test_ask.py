from types import SimpleNamespace

from app import ask


def test_ask_prints_token_usage_and_returns_response(monkeypatch, capsys):
    response = SimpleNamespace(
        usage_metadata=SimpleNamespace(prompt_token_count=12, candidates_token_count=34),
        text="回答",
    )

    class FakeModels:
        def generate_content(self, **kwargs):
            return response

    class FakeClient:
        def __init__(self, **kwargs):
            self.models = FakeModels()

    monkeypatch.setattr(ask.genai, "Client", FakeClient)
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")

    assert ask.ask("質問") == "回答"
    assert capsys.readouterr().out == "[トークン数] 入力: 12, 出力: 34\n"