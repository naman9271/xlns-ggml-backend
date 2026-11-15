import glob
import json
import os
import shutil
import signal
import subprocess
import sys


class Interrupted(Exception):
    pass


_child = None
_cleanups = []


def _on_signal(signum, frame):
    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, signal.SIG_IGN)
    _stop_child()
    raise Interrupted(signal.Signals(signum).name)


def _stop_child():
    global _child
    if _child is not None and _child.poll() is None:
        _child.terminate()
        try:
            _child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _child.kill()
            _child.wait()
    _child = None


def on_cleanup(fn):
    _cleanups.append(fn)


def remove(*patterns):
    for pattern in patterns:
        for path in glob.glob(pattern):
            if os.path.isdir(path) and not os.path.islink(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass


def main(fn):
    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, _on_signal)
    code = 0
    try:
        fn()
    except Interrupted as e:
        print(f"\nstopped ({e}); every finished result is saved, run again to resume", file=sys.stderr, flush=True)
        code = 130
    except SystemExit as e:
        if e.code is None or isinstance(e.code, int):
            code = e.code or 0
        else:
            print(e.code, file=sys.stderr)
            code = 1
    finally:
        for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(s, signal.SIG_IGN)
        _stop_child()
        for cleanup in reversed(_cleanups):
            try:
                cleanup()
            except Exception as e:
                print(f"cleanup failed: {e}", file=sys.stderr)
    sys.exit(code)


def run(cmd, env=None, stdout=True, stderr=True, check=False):
    global _child
    _child = subprocess.Popen(cmd, env=env, text=True,
                              stdout=subprocess.PIPE if stdout else subprocess.DEVNULL,
                              stderr=subprocess.PIPE if stderr else subprocess.DEVNULL)
    out, err = _child.communicate()
    rc = _child.returncode
    _child = None
    if check and rc != 0:
        if err:
            sys.stderr.write(err[-3000:])
        raise SystemExit(f"failed ({rc}): {' '.join(cmd)}")
    return subprocess.CompletedProcess(cmd, rc, out, err)


def append_jsonl(path, obj):
    append_jsonl_many(path, [obj])


def append_jsonl_many(path, objs):
    if not objs:
        return
    line = "".join(json.dumps(o) + "\n" for o in objs).encode()
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        if os.write(fd, line) != len(line):
            raise OSError(f"short write to {path}")
        os.fsync(fd)
    finally:
        os.close(fd)


def load_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"{path}:{n}: skipping a damaged line (it will be rerun)", file=sys.stderr)
    return rows


def publish(tmp, final):
    os.replace(tmp, final)
