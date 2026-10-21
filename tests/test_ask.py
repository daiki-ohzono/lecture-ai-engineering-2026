from types import SimpleNamespace

from app import ask


def test_ask_prints_token_usage_and_returns_response(monkeypatch, capsys):
    response = SimpleNamespace(
        usage_metadata=SimpleNamespace(prompt_token_count=12, candidates_token_count=34),
        text="回答",
    )

    class FakeChat:
        def __init__(self, client):
            self.client = client

        def send_message(self, question):
            assert self.client.is_open
            assert question == "質問"
            return response

    class FakeChats:
        def __init__(self, client):
            self.client = client

        def create(self, **kwargs):
            assert kwargs == {"model": ask.MODEL}
            return FakeChat(self.client)

    class FakeClient:
        def __init__(self, **kwargs):
            assert kwargs == {"api_key": "test-key"}
            self.is_open = False
            self.chats = FakeChats(self)

        def __enter__(self):
            self.is_open = True
            return self

        def __exit__(self, *args):
            self.is_open = False

    clients = []

    def make_client(**kwargs):
        client = FakeClient(**kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(ask.genai, "Client", make_client)
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")

    assert ask.ask("質問") == "回答"
    assert not clients[0].is_open
    assert capsys.readouterr().out == "[トークン数] 入力: 12, 出力: 34\n"


def test_ask_sends_follow_up_to_the_same_chat(capsys):
    response = SimpleNamespace(
        usage_metadata=None,
        text="回答",
    )

    class FakeChat:
        questions = []

        def send_message(self, question):
            self.questions.append(question)
            return response

    chat = FakeChat()

    assert ask.ask("最初の質問", chat) == "回答"
    assert ask.ask("続けて質問", chat) == "回答"
    assert chat.questions == ["最初の質問", "続けて質問"]
    assert capsys.readouterr().out == (
        "[トークン数] 入力: 不明, 出力: 不明\n"
        "[トークン数] 入力: 不明, 出力: 不明\n"
    )


def test_main_keeps_client_open_for_interactive_chat(monkeypatch, capsys):
    response = SimpleNamespace(usage_metadata=None, text="回答")

    class FakeChat:
        def __init__(self, client):
            self.client = client
            self.questions = []

        def send_message(self, question):
            assert self.client.is_open
            self.questions.append(question)
            return response

    class FakeChats:
        def __init__(self, client):
            self.client = client
            self.chat = FakeChat(client)

        def create(self, **kwargs):
            assert kwargs == {"model": ask.MODEL}
            return self.chat

    class FakeClient:
        def __init__(self, **kwargs):
            assert kwargs == {"api_key": "test-key"}
            self.is_open = False
            self.chats = FakeChats(self)

        def __enter__(self):
            self.is_open = True
            return self

        def __exit__(self, *args):
            self.is_open = False

    clients = []

    def make_client(**kwargs):
        client = FakeClient(**kwargs)
        clients.append(client)
        return client

    questions = iter(["続けて質問", ""])
    monkeypatch.setattr(ask.genai, "Client", make_client)
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")
    monkeypatch.setattr("builtins.input", lambda _prompt: next(questions))
    monkeypatch.setattr("sys.argv", ["app.ask", "--interactive", "最初の質問"])

    ask.main()

    client = clients[0]
    assert client.chats.chat.questions == ["最初の質問", "続けて質問"]
    assert not client.is_open
    assert capsys.readouterr().out.count("[トークン数]") == 2
