"""Tests for the MUSA container launcher without starting Docker or a GPU."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = ROOT / "scripts/musa/container.sh"


class ContainerLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.bin = Path(self.temp.name) / "bin"
        self.bin.mkdir()
        self.args_file = Path(self.temp.name) / "docker-args"
        fake_docker = self.bin / "docker"
        fake_docker.write_text(
            "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$FAKE_DOCKER_ARGS\"\n"
        )
        fake_docker.chmod(0o755)

    def launch(self, **updates):
        env = os.environ.copy()
        env.pop("GPU_LEASE_ID", None)
        env.pop("GPU_ALLOW_UNLEASED", None)
        env.pop("MUSA_MODELS_ROOT", None)
        env["GPU_INDICES"] = "0"
        env["FAKE_DOCKER_ARGS"] = str(self.args_file)
        env["PATH"] = os.pathsep.join([str(self.bin), env.get("PATH", "")])
        env.update(updates)
        return subprocess.run(
            ["bash", str(LAUNCHER), "bash", "-lc", "true"],
            env=env, text=True, capture_output=True,
        )

    def docker_args(self):
        return self.args_file.read_text().splitlines()

    def test_lease_required_by_default(self):
        result = self.launch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("GPU_LEASE_ID", result.stderr)
        self.assertFalse(self.args_file.exists())

    def test_unleased_mode_is_explicit_and_has_no_serverlist_labels(self):
        result = self.launch(GPU_ALLOW_UNLEASED="1", GPU_INDICES="2")
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.docker_args()
        self.assertIn("agent-self-llm-LOCAL-26-unleased", args)
        self.assertIn("MTHREADS_VISIBLE_DEVICES=2", args)
        self.assertNotIn("serverlist.managed=1", args)
        self.assertFalse(any("serverlist.lease_id=" in arg for arg in args))

    def test_lease_mode_keeps_serverlist_labels(self):
        result = self.launch(GPU_LEASE_ID="lease_123", GPU_OWNER="tester")
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.docker_args()
        self.assertIn("serverlist.managed=1", args)
        self.assertIn("serverlist.lease_id=lease_123", args)
        self.assertIn("serverlist.owner=tester", args)
        self.assertIn("serverlist.ticket=LOCAL-26", args)
        self.assertIn("agent-self-llm-LOCAL-26-lease_12", args)

    def test_invalid_unleased_flag_is_rejected(self):
        result = self.launch(GPU_ALLOW_UNLEASED="yes")
        self.assertEqual(result.returncode, 2)
        self.assertIn("GPU_ALLOW_UNLEASED must be 0 or 1", result.stderr)
        self.assertFalse(self.args_file.exists())

    def test_local_models_are_mounted_readonly(self):
        result = self.launch(GPU_ALLOW_UNLEASED="1", MUSA_MODELS_ROOT=self.temp.name)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"type=bind,src={self.temp.name},dst=/models,readonly", self.docker_args())

    def test_missing_model_directory_is_rejected(self):
        result = self.launch(GPU_ALLOW_UNLEASED="1", MUSA_MODELS_ROOT=str(self.bin / "missing"))
        self.assertEqual(result.returncode, 2)
        self.assertIn("MUSA_MODELS_ROOT", result.stderr)
        self.assertFalse(self.args_file.exists())


if __name__ == "__main__":
    unittest.main()
