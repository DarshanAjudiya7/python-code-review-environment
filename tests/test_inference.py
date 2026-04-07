import subprocess
import sys


def test_inference_emits_structured_stdout():
    completed = subprocess.run(
        [sys.executable, "inference.py", "--task", "syntax-fix-easy"],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )

    assert completed.returncode == 0
    assert "[START] task=syntax-fix-easy" in completed.stdout
    assert "[STEP] task=syntax-fix-easy" in completed.stdout
    assert "[END] task=syntax-fix-easy" in completed.stdout
