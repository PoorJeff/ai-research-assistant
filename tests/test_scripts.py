import subprocess
import sys


def test_real_paper_evaluation_script_help_runs():
    result = subprocess.run(
        [sys.executable, "scripts/run_real_paper_evaluation.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Run real arXiv PDF retrieval evaluation" in result.stdout
