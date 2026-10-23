"""Phase 1: interactive chat with history, streamed replies, and error handling.

The model is stateless, so we keep the history here and resend all of it
on every turn.

Usage:
    python chat.py
Commands:
    /reset  start a new conversation
    /quit   exit
"""

from research_agent import config
from research_agent.llm_client import LLMError, chat
from research_agent.prompts import SYSTEM_PROMPT


def new_conversation() -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT}]


def main() -> None:
    history = new_conversation()
    print(f"Chatting with {config.MODEL}. Type /reset for a new conversation, /quit to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input == "/quit":
            break
        if user_input == "/reset":
            history = new_conversation()
            print("(conversation cleared)\n")
            continue

        history.append({"role": "user", "content": user_input})
        print("\nAssistant: ", end="", flush=True)
        try:
            # Print each piece the moment it arrives; flush so it shows immediately.
            response = chat(history, on_token=lambda piece: print(piece, end="", flush=True))
        except LLMError as e:
            # Drop the unanswered question so the history stays user/assistant pairs.
            history.pop()
            print(f"\n\n[error] {e}\n")
            continue
        print("\n")
        if response.truncated:
            print(f"(reply cut off at {config.MAX_OUTPUT_TOKENS} tokens; raise LLM_MAX_OUTPUT_TOKENS for longer answers)\n")
        # Save the reply too, so the next turn includes it.
        history.append({"role": "assistant", "content": response.text})

        used = response.input_tokens + response.output_tokens
        print(
            f"[turn {len(history) // 2} | tokens in: {response.input_tokens} | "
            f"out: {response.output_tokens} | context used: {used}/{config.CONTEXT_WINDOW} | "
            f"first token: {response.time_to_first_token:.2f}s | total: {response.total_seconds:.1f}s | "
            f"{response.tokens_per_second:.1f} tok/s]\n"
        )
        if used > 0.8 * config.CONTEXT_WINDOW:
            print("(warning: context window almost full; older messages may be dropped. Use /reset.)\n")

    print("Bye!")


if __name__ == "__main__":
    main()
