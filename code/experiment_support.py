"""Single-run lock and subprocess cleanup shared by the GPU entry points."""

import contextlib
import fcntl
import os
import signal
import subprocess
import sys
import time


@contextlib.contextmanager
def exclusive_run(root):
    with (root / ".tts_run.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another TTS runner is active in this folder. Wait for it to finish.") from None
        stream.seek(0)
        stream.truncate()
        stream.write(str(os.getpid()) + "\n")
        stream.flush()
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def run_child(root, arguments, log, *, show_tail=True, patch_retry=False):
    log.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(2 if patch_retry else 1):
        with log.open("w" if attempt == 0 else "a", encoding="utf-8") as stream:
            child = subprocess.Popen([sys.executable, "-u", *arguments], cwd=root,
                                     stdout=stream, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            previous = signal.getsignal(signal.SIGTERM)

            def interrupted(signum, frame):
                raise KeyboardInterrupt

            signal.signal(signal.SIGTERM, interrupted)
            started = last_message = time.monotonic()
            try:
                while child.poll() is None:
                    time.sleep(1)
                    if time.monotonic() - last_message >= 30:
                        print(f"Still running ({int(time.monotonic() - started)}s); log: {log.relative_to(root)}", flush=True)
                        last_message = time.monotonic()
            except BaseException:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                raise
            finally:
                signal.signal(signal.SIGTERM, previous)
        output = log.read_text(encoding="utf-8", errors="replace")
        needs_restart = any(message in output for message in
                            ("The model's own code needed the length fix", "f5_tts was imported before the length fix"))
        if child.returncode and patch_retry and attempt == 0 and needs_restart:
            print("Supplied one-time length patch applied; restarting the worker.", flush=True)
            continue
        if show_tail:
            print("\n".join(output.splitlines()[-20:]), flush=True)
        if child.returncode:
            raise RuntimeError(f"Worker exited {child.returncode}; inspect {log.relative_to(root)}")
        return
