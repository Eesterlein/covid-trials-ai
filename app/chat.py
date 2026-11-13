"""Terminal chat for testing the agent.

  python -m app.chat                      interactive
  python -m app.chat "your question"      one question, then exit
"""
import sys
import time

from app.agent import MODEL, ask, cost


def show(kind, data):
    if kind == "tool":
        print(f"  → {data['name']}({', '.join(f'{k}={v!r}' for k, v in data['input'].items())})", flush=True)
    elif kind == "code":
        print("  → running Python:\n" + "\n".join("      " + l for l in str(data.get("command") or data.get("code") or data).splitlines()[:25]), flush=True)
    elif kind == "code_result":
        print("  ← output: " + str(data)[:400].replace("\n", "\n            "), flush=True)
    elif kind == "refusal":
        print(f"  ! refusal: {data}", flush=True)


def turn(messages, q):
    messages.append({"role": "user", "content": q})
    t0 = time.time()
    text, messages, usage = ask(messages, on_event=show)
    print("\n" + text)
    print(f"\n[{MODEL} | {usage['steps']} steps | in {usage['input']:,} out {usage['output']:,} "
          f"cache read {usage['cache_read']:,} write {usage['cache_write']:,} | ~${cost(usage):.3f} | {time.time() - t0:.0f}s]\n")
    return messages


if __name__ == "__main__":
    messages = []
    if len(sys.argv) > 1:
        turn(messages, " ".join(sys.argv[1:]))
    else:
        while (q := input("you> ").strip()) not in ("", "exit", "quit"):
            messages = turn(messages, q)
