"""第1回: 素のLLMに質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする
    uv run python -m app.ask --interactive "LoRAとは何ですか？"

ツールを持たない「素の」呼び出し。--interactive では同じ会話の履歴を保って
続けて質問できる。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse

from google import genai
from google.genai.chats import Chat
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def create_chat(client: genai.Client) -> Chat:
    return client.chats.create(model=MODEL)


def ask(question: str, chat: Chat | None = None) -> str:
    if chat is None:
        with genai.Client(api_key=get_api_key()) as client:
            response = create_chat(client).send_message(question)
    else:
        response = chat.send_message(question)

    usage = response.usage_metadata
    input_tokens = usage.prompt_token_count if usage else None
    output_tokens = usage.candidates_token_count if usage else None
    print(
        f"[トークン数] 入力: {input_tokens if input_tokens is not None else '不明'}, "
        f"出力: {output_tokens if output_tokens is not None else '不明'}"
    )

    return response.text


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="回答後に続けて質問する（空の質問で終了）",
    )
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    try:
        with genai.Client(api_key=get_api_key()) as client:
            chat = create_chat(client)
            question = args.question
            while True:
                print(f"[質問] {question}\n")
                print(ask(question, chat))
                if not args.interactive:
                    break
                try:
                    question = input("\n続けて質問してください（終了するには空欄）: ").strip()
                except EOFError:
                    break
                if not question:
                    break
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
