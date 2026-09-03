"""
Stdin prompt (background thread, daemon -- no separate process needed here:
unlike main.py's multiprocessing split, CoppeliaSim's client and the WS link
both have to live in this same process, so raylib and stdin just share it).
"""
import queue

from runner.constants import HELP, CONTROLS, COMMANDS


def cli_thread(commands: "queue.Queue"):
    print(HELP)
    while True:
        try:
            raw = input("[maze] ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            commands.put(("quit", []))
            return

        if not raw:
            continue
        parts = raw.split()
        name, args = parts[0].lower(), parts[1:]

        if name in ("help", "ajuda", "h", "?"):
            print(HELP)
            print(CONTROLS)
            continue
        if name in ("sair", "exit", "q"):
            name = "quit"
        if name not in COMMANDS:
            print(f"[ERROR] comando desconhecido: {name}  (help para a lista)")
            continue

        low, high = COMMANDS[name]
        if not low <= len(args) <= high:
            print(f"[ERROR] '{name}' espera de {low} a {high} argumento(s)")
            continue

        commands.put((name, args))
        if name == "quit":
            return
