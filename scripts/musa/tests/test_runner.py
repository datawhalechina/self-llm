"""Check process startup semantics without importing a GPU framework."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "scripts/musa/run.py"
BOOTSTRAP = ROOT / "scripts/musa/bootstrap"


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        (self.folder / "torchada.py").write_text(
            "import os\nassert os.environ['TORCHADA_FORCE_CUDA_AVAILABLE'] == '1'\n"
            "os.environ['ADAPTER_IMPORTED'] = '1'\n"
        )
        (self.folder / "torch_musa.py").write_text("import torchada\n")
        (self.folder / "torch.py").write_text(
            "from types import SimpleNamespace\n"
            "musa = cuda = SimpleNamespace(is_available=lambda: True)\n"
        )
        self.env = os.environ.copy()
        self.env.pop("SELF_LLM_MUSA", None)
        self.env.pop("ADAPTER_IMPORTED", None)
        self.env["PYTHONPATH"] = str(self.folder)

    def launch(self, *args):
        return subprocess.run(
            [sys.executable, str(RUNNER), *args], env=self.env,
            cwd=self.folder, text=True, capture_output=True,
        )

    def test_arguments_cwd_and_exit_status(self):
        script = self.folder / "中文 空格.py"
        script.write_text(
            "import os, sys\nassert os.environ['ADAPTER_IMPORTED'] == '1'\n"
            "assert sys.argv[1:] == ['--label', 'a b']\n"
            "assert os.path.realpath(os.getcwd()) == os.path.realpath(os.path.dirname(__file__))\nraise SystemExit(7)\n"
        )
        result = self.launch(str(script), "--", "--label", "a b")
        self.assertEqual(result.returncode, 7, result.stderr)

    def test_module_and_child_process_inherit_bootstrap(self):
        (self.folder / "example.py").write_text(
            "import os, subprocess, sys\nassert os.environ['ADAPTER_IMPORTED'] == '1'\n"
            "child_env = os.environ.copy(); child_env.pop('ADAPTER_IMPORTED')\n"
            "subprocess.run([sys.executable, '-c', "
            "\"import os; assert os.environ['ADAPTER_IMPORTED'] == '1'; print('child OK')\"], "
            "env=child_env, check=True)\n"
        )
        result = self.launch("--module", "example")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("child OK", result.stdout)

    def test_build_subprocess_skips_hook_but_user_subprocess_keeps_it(self):
        (self.folder / "torchada.py").write_text(
            "import os, subprocess, sys\n"
            "if os.environ.get('BUILD_HELPER') == '1':\n"
            "    raise RuntimeError('torchada recursively imported by build helper')\n"
            "assert os.environ['TORCHADA_FORCE_CUDA_AVAILABLE'] == '1'\n"
            "build_env = os.environ.copy()\n"
            "build_env['BUILD_HELPER'] = '1'\n"
            "build_env.pop('ADAPTER_IMPORTED', None)\n"
            "subprocess.run([sys.executable, '-c', "
            "\"import os, sys; assert os.environ['SELF_LLM_MUSA'] == '0'; "
            "assert 'torchada' not in sys.modules; "
            "assert 'ADAPTER_IMPORTED' not in os.environ; print('build hook disabled')\"], "
            "env=build_env, check=True, timeout=5)\n"
            "os.environ['ADAPTER_IMPORTED'] = '1'\n"
        )
        script = self.folder / "target.py"
        script.write_text(
            "import os, subprocess, sys\n"
            "assert os.environ['SELF_LLM_MUSA'] == '1'\n"
            "assert os.environ['ADAPTER_IMPORTED'] == '1'\n"
            "child_env = os.environ.copy()\n"
            "child_env.pop('ADAPTER_IMPORTED')\n"
            "subprocess.run([sys.executable, '-c', "
            "\"import os, sys; assert os.environ['SELF_LLM_MUSA'] == '1'; "
            "assert 'torchada' in sys.modules; "
            "assert os.environ['ADAPTER_IMPORTED'] == '1'; print('user child hook enabled')\"], "
            "env=child_env, check=True, timeout=10)\n"
        )
        result = self.launch(str(script))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("build hook disabled"), 2, result.stdout)
        self.assertIn("user child hook enabled", result.stdout)

    def test_failed_adapter_prevents_target_execution(self):
        (self.folder / "torchada.py").write_text("raise ImportError('broken adapter')\n")
        script = self.folder / "target.py"
        script.write_text("print('TARGET EXECUTED')\n")
        result = self.launch(str(script))
        self.assertEqual(result.returncode, 1)
        self.assertIn("broken adapter", result.stderr)
        self.assertNotIn("TARGET EXECUTED", result.stdout)

    def test_bootstrap_does_nothing_without_opt_in(self):
        self.env["PYTHONPATH"] = os.pathsep.join([str(BOOTSTRAP), str(self.folder)])
        result = subprocess.run(
            [sys.executable, "-c", "import sys; assert 'torchada' not in sys.modules"],
            env=self.env, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
