import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

import _bootstrap  # noqa: F401
from physicar_e2e.expert_driver import build_parser, main


class CliTests(unittest.TestCase):
    @staticmethod
    def run_config(payload):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as stream:
            json.dump(payload, stream)
            stream.flush()
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                result = main(["--config", stream.name, "--preflight-only"])
        return result, stderr.getvalue()

    def test_installed_entry_point_requires_explicit_config(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                build_parser().parse_args([])
        self.assertEqual(raised.exception.code, 2)

    def test_malformed_fixed_speed_has_controlled_cli_error(self):
        config_path = Path(__file__).resolve().parents[1] / "configs" / "expert_driver_v1.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["fixed_speed_mps"] = None
        result, stderr = self.run_config(config)
        self.assertEqual(result, 2)
        self.assertIn("ERROR:", stderr)
        self.assertNotIn("Traceback", stderr)

    def test_non_object_json_has_controlled_cli_error(self):
        for payload in (None, [], "config", 42, True):
            with self.subTest(payload=payload):
                result, stderr = self.run_config(payload)
                self.assertEqual(result, 2)
                self.assertIn("configuration root must be a JSON object", stderr)
                self.assertNotIn("Traceback", stderr)

    def test_missing_field_has_controlled_cli_error(self):
        config_path = Path(__file__).resolve().parents[1] / "configs" / "expert_driver_v1.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        del config["wheelbase_m"]
        result, stderr = self.run_config(config)
        self.assertEqual(result, 2)
        self.assertIn("wheelbase_m", stderr)
        self.assertNotIn("Traceback", stderr)

    def test_unknown_field_has_controlled_cli_error(self):
        config_path = Path(__file__).resolve().parents[1] / "configs" / "expert_driver_v1.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["mystery_parameter"] = 1
        result, stderr = self.run_config(config)
        self.assertEqual(result, 2)
        self.assertIn("mystery_parameter", stderr)
        self.assertNotIn("Traceback", stderr)

    def test_source_launcher_supplies_canonical_checkout_config(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "run_expert_driver_v1.py"
        spec = importlib.util.spec_from_file_location("source_launcher", script)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        args = module.source_checkout_args(["--preflight-only"])
        self.assertEqual(args[0], "--config")
        self.assertEqual(Path(args[1]), script.parents[1] / "configs" / "expert_driver_v1.json")
        self.assertEqual(args[2:], ["--preflight-only"])

    def test_source_launcher_preserves_explicit_config(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "run_expert_driver_v1.py"
        spec = importlib.util.spec_from_file_location("source_launcher_explicit", script)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        supplied = ["--config", "/tmp/custom.json", "--dry-run", "1"]
        self.assertEqual(module.source_checkout_args(supplied), supplied)


if __name__ == "__main__":
    unittest.main()
