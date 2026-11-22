#!/usr/bin/env python3
"""GPU-free regression for timeout cleanup of the inference process tree."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

sys.dont_write_bytecode = True
from hip_native_check import run_process


@unittest.skipUnless(os.name == "posix", "HIP native process-group tests require POSIX")
class NativeHarnessTests(unittest.TestCase):
    def test_normal_exit_preserves_status_and_output(self):
        with tempfile.TemporaryFile(mode='w+') as log:
            status, _, _ = run_process(
                [sys.executable, '-c', "print('child output'); raise SystemExit(7)"],
                os.environ.copy(), log)
            log.seek(0)
            self.assertEqual(status, 7)
            self.assertIn('child output', log.read())

    def test_timeout_kills_wrapper_and_inference_child(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryFile(mode='w+') as log:
            pidfile = Path(tmp) / 'child.pid'
            # Stand in for /usr/bin/time plus its long-running GPU child.
            # Both ignore SIGTERM, so graceful termination alone is insufficient.
            child = 'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)'
            wrapper = (
                'import os,signal,subprocess,sys; from pathlib import Path; '
                'signal.signal(signal.SIGTERM, signal.SIG_IGN); '
                f'child=subprocess.Popen([sys.executable,"-c",{child!r}]); '
                'Path(sys.argv[1]).write_text(str(child.pid)); child.wait()')
            pid = None
            stopped = False
            try:
                with self.assertRaises(TimeoutError):
                    run_process([sys.executable, '-c', wrapper, str(pidfile)],
                                os.environ.copy(), log, timeout=1)
                pid = int(pidfile.read_text())
                # An orphan killed with its wrapper may briefly be a zombie
                # until init reaps it. It must no longer execute or hold GPU work.
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    state = subprocess.run(['ps', '-o', 'stat=', '-p', str(pid)],
                                           capture_output=True, text=True).stdout.strip()
                    if not state or state.startswith('Z'):
                        stopped = True
                        break
                    time.sleep(.01)
                else:
                    self.fail('inference child survived timeout cleanup')
            finally:
                if pid is None and pidfile.exists():
                    pid = int(pidfile.read_text())
                if pid is not None and not stopped:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass


if __name__ == '__main__':
    unittest.main()
