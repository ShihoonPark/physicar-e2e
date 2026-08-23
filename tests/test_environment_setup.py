import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "setup_lane_follow_environment_v1.py"
SPEC = importlib.util.spec_from_file_location("lane_follow_environment_setup", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
setup = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = setup
SPEC.loader.exec_module(setup)


class EnvironmentSetupTests(unittest.TestCase):
    canonical = "canonical_test_world"
    derived = "canonical_test_world_e2e_lane_follow_v1"
    cones = tuple(f"cone{number}" for number in range(1, 7))

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.sim_root = Path(self.temporary.name) / "sim"
        self.share = self.sim_root / "src" / "physicar-sim" / "share"
        self.config_path = Path(self.temporary.name) / "environment.json"
        self.config_path.write_text(
            json.dumps(
                {
                    "canonical_world": self.canonical,
                    "derived_world": self.derived,
                    "removed_models": list(self.cones),
                }
            ),
            encoding="utf-8",
        )
        self.canonical_paths = setup.asset_paths(self.share, self.canonical)
        self.derived_paths = setup.asset_paths(self.share, self.derived)
        self._write_canonical()

    def _world_xml(self, cone_names=None):
        cone_names = self.cones if cone_names is None else cone_names
        models = [f'  <model name="{name}"><pose>{index} 0 0 0 0 0</pose></model>' for index, name in enumerate(cone_names)]
        models.insert(2, '  <model name="light1"><pose>1 2 0 0 0 0</pose></model>')
        models.extend(
            [
                '  <model name="wall_0"><static>true</static></model>',
                '  <model name="wall_1"><static>true</static></model>',
            ]
        )
        return (
            "<?xml version='1.0' encoding='utf-8'?>\n"
            '<sdf version="1.4">\n'
            f'<world name="{self.canonical}">\n'
            "  <gravity>0 0 -9.8</gravity>\n"
            + "\n".join(models)
            + "\n</world>\n</sdf>\n"
        )

    def _write_canonical(self, xml=None):
        self.canonical_paths.world.parent.mkdir(parents=True, exist_ok=True)
        self.canonical_paths.route.parent.mkdir(parents=True, exist_ok=True)
        self.canonical_paths.model.mkdir(parents=True, exist_ok=True)
        self.canonical_paths.world.write_text(xml or self._world_xml(), encoding="utf-8")
        self.canonical_paths.route.write_bytes(b"\x93NUMPY\x00test-route-bytes\xff")
        (self.canonical_paths.model / "model.config").write_text("<model/>\n", encoding="utf-8")
        (self.canonical_paths.model / f"{self.canonical}.sdf").write_text("<sdf/>\n", encoding="utf-8")

    def run_cli(self, *extra):
        stdout = io.StringIO()
        stderr = io.StringIO()
        argv = ["--sim-root", str(self.sim_root), "--config", str(self.config_path), *extra]
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = setup.main(argv)
        return result, stdout.getvalue(), stderr.getvalue()

    def generate(self):
        result, stdout, stderr = self.run_cli()
        self.assertEqual((result, stderr), (0, ""))
        self.assertIn("generated and validated", stdout)

    def test_successful_generation_removes_exactly_six_cones_and_preserves_non_cones(self):
        self.generate()
        _, world = setup.parse_world(self.derived_paths.world)
        names = [model.get("name") for model in world.findall("model")]
        self.assertFalse(set(names) & set(self.cones))
        self.assertEqual(names, ["light1", "wall_0", "wall_1"])

    def test_internal_world_name_changed(self):
        self.generate()
        _, world = setup.parse_world(self.derived_paths.world)
        self.assertEqual(world.get("name"), self.derived)

    def test_route_is_copied_byte_identically(self):
        self.generate()
        self.assertEqual(self.derived_paths.route.read_bytes(), self.canonical_paths.route.read_bytes())
        self.assertEqual(setup.sha256(self.derived_paths.route), setup.sha256(self.canonical_paths.route))

    def test_model_directory_is_copied(self):
        self.generate()
        self.assertEqual(
            setup.directory_manifest(self.derived_paths.model),
            setup.directory_manifest(self.canonical_paths.model),
        )

    def test_repeated_invocation_is_idempotent(self):
        self.generate()
        before = (
            self.derived_paths.world.stat().st_mtime_ns,
            self.derived_paths.route.stat().st_mtime_ns,
            self.derived_paths.model.stat().st_mtime_ns,
        )
        result, stdout, stderr = self.run_cli()
        after = (
            self.derived_paths.world.stat().st_mtime_ns,
            self.derived_paths.route.stat().st_mtime_ns,
            self.derived_paths.model.stat().st_mtime_ns,
        )
        self.assertEqual((result, stderr), (0, ""))
        self.assertIn("already valid", stdout)
        self.assertEqual(after, before)

    def test_verify_only_succeeds_on_valid_environment(self):
        self.generate()
        result, stdout, stderr = self.run_cli("--verify-only")
        self.assertEqual((result, stderr), (0, ""))
        self.assertIn("cone count 0", stdout)
        self.assertIn("canonical asset integrity unchanged", stdout)

    def test_corrupted_derived_asset_is_detected_without_overwrite(self):
        self.generate()
        self.derived_paths.route.write_bytes(b"corrupt")
        result, _, stderr = self.run_cli("--verify-only")
        self.assertEqual(result, 2)
        self.assertIn("not byte-identical", stderr)
        result, _, stderr = self.run_cli()
        self.assertEqual(result, 2)
        self.assertIn("--force", stderr)
        self.assertEqual(self.derived_paths.route.read_bytes(), b"corrupt")

    def test_missing_canonical_asset_is_controlled_failure(self):
        self.canonical_paths.route.unlink()
        result, _, stderr = self.run_cli()
        self.assertEqual(result, 2)
        self.assertIn("canonical route is missing", stderr)
        self.assertNotIn("Traceback", stderr)

    def test_malformed_canonical_xml_is_controlled_failure(self):
        self.canonical_paths.world.write_text("<sdf><world>", encoding="utf-8")
        result, _, stderr = self.run_cli()
        self.assertEqual(result, 2)
        self.assertIn("malformed", stderr)
        self.assertNotIn("Traceback", stderr)

    def test_missing_expected_cone_is_controlled_failure(self):
        self.canonical_paths.world.write_text(self._world_xml(self.cones[:-1]), encoding="utf-8")
        result, _, stderr = self.run_cli()
        self.assertEqual(result, 2)
        self.assertIn("cone6=0", stderr)

    def test_duplicated_expected_cone_is_controlled_failure(self):
        self.canonical_paths.world.write_text(self._world_xml((*self.cones, "cone3")), encoding="utf-8")
        result, _, stderr = self.run_cli()
        self.assertEqual(result, 2)
        self.assertIn("cone3=2", stderr)

    def test_force_regenerates_invalid_derived_environment(self):
        self.generate()
        self.derived_paths.world.write_text("<sdf><world name='wrong'/></sdf>", encoding="utf-8")
        result, stdout, stderr = self.run_cli("--force")
        self.assertEqual((result, stderr), (0, ""))
        self.assertIn("generated and validated", stdout)
        result, _, stderr = self.run_cli("--verify-only")
        self.assertEqual((result, stderr), (0, ""))

    def test_canonical_source_files_remain_unchanged(self):
        config = setup.load_config(self.config_path)
        before = setup.canonical_fingerprint(self.canonical_paths)
        self.generate()
        self.assertEqual(setup.canonical_fingerprint(self.canonical_paths), before)
        setup.verify_derived(config, self.canonical_paths, self.derived_paths)
        self.assertEqual(setup.canonical_fingerprint(self.canonical_paths), before)

    def test_world_edit_preserves_non_model_content_bytes(self):
        self.generate()
        derived = self.derived_paths.world.read_text(encoding="utf-8")
        self.assertIn("  <gravity>0 0 -9.8</gravity>\n", derived)
        self.assertIn('  <model name="light1"><pose>1 2 0 0 0 0</pose></model>\n', derived)


if __name__ == "__main__":
    unittest.main()
