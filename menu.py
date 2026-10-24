"""Arrow-key lists. Number keys work too. A pipe falls back to typed numbers."""

import os
import select
import sys
import termios
import tty


def _next_byte(fd):
    if not select.select([fd], [], [], 0.05)[0]:
        return b""
    return os.read(fd, 1)


def _read_key():
    # sys.stdin.read() buffers past the one byte we asked for, so the rest of
    # an arrow sequence (ESC [ A) never reaches select() and looks like Esc.
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd, termios.TCSADRAIN)
        raw = os.read(fd, 1)
        if not raw:
            return "q"
        char = raw.decode("utf-8", "replace")
        if char == "\x1b":
            kind = _next_byte(fd)
            arrow = _next_byte(fd)
            if kind == b"[" and arrow == b"A" or kind == b"O" and arrow == b"A":
                return "up"
            if kind == b"[" and arrow == b"B" or kind == b"O" and arrow == b"B":
                return "down"
            return "esc"
        if char in ("\r", "\n"):
            return "enter"
        if char in ("q", "Q"):
            return "q"
        if char in ("n", "N"):
            return "n"
        if char.isdigit():
            return char
        return char
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _draw(title, labels, selected, hint):
    sys.stdout.write("\033[H\033[J")
    sys.stdout.write(title + "\n\n")
    for index, label in enumerate(labels):
        mark = ">" if index == selected else " "
        sys.stdout.write(f" {mark} {index + 1}  {label}\n")
    sys.stdout.write("\n" + hint + "\n")
    sys.stdout.flush()


def choose(title, labels, selected=0, hint="up/down, enter. q quits."):
    if not labels:
        raise SystemExit("Nothing to choose.")
    selected = max(0, min(selected, len(labels) - 1))
    if not sys.stdin.isatty():
        print(title)
        for index, label in enumerate(labels, start=1):
            print(f"{index}  {label}")
        print("Number, or q:")
        line = sys.stdin.readline().strip().lower()
        if line == "q" or line == "":
            return None
        if line.isdigit() and 1 <= int(line) <= len(labels):
            return int(line) - 1
        raise SystemExit(f"Not a choice: {line}")
    while True:
        _draw(title, labels, selected, hint)
        key = _read_key()
        if key == "up":
            selected = (selected - 1) % len(labels)
        elif key == "down":
            selected = (selected + 1) % len(labels)
        elif key == "enter":
            sys.stdout.write("\n")
            return selected
        elif key == "q" or key == "esc":
            sys.stdout.write("\n")
            return None
        elif key.isdigit():
            pick = int(key) - 1
            if 0 <= pick < len(labels):
                sys.stdout.write("\n")
                return pick


def wait_action():
    """enter runs, n picks again, q quits."""
    if not sys.stdin.isatty():
        line = sys.stdin.readline().strip().lower()
        if line == "n":
            return "n"
        if line in ("q", ""):
            return "q"
        return "enter"
    while True:
        key = _read_key()
        if key == "enter":
            return "enter"
        if key == "n":
            return "n"
        if key in ("q", "esc"):
            return "q"
