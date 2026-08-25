import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
from PIL import Image

import _bootstrap  # noqa: F401
from physicar_e2e.high_speed_temporal import (
    MAX_TOTAL_ATTEMPTS, build_sequences, classify_temporal_run, load_dataset_config,
    load_training_config, run_attempts,
)
from physicar_e2e.pilotnet import preprocess_png
from physicar_e2e.pilotnet_temporal import (
    CausalFrameBuffer, TemporalInputError, build_temporal_pilotnet,
    preprocess_temporal_paths,
)

ROOT = Path(__file__).resolve().parents[1]


def live_result(result="PASS", temporal=False):
    return {"result": result, "temporal_input_failure": temporal, "api_failures": 0,
            "liveness_failures": 0, "safe_stop_success": True}


class Client:
    def safe_stop(self): return []


class TemporalContracts(unittest.TestCase):
    def test_causal_order_gap_boundary_and_same_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "images/source").mkdir(parents=True)
            manifest = root / "source.csv"
            fields = ["episode_id", "sample_index", "image_path", "camera_header_time_ns", "steering_rad", "source_mcap_sha256"]
            with manifest.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
                for i, timestamp in enumerate((100_000_000, 166_000_000, 232_000_000, 400_000_000)):
                    path = root / f"images/source/{i}.png"; Image.new("RGB", (200,66), (i,i,i)).save(path)
                    writer.writerow({"episode_id":"source", "sample_index":i, "image_path":f"images/source/{i}.png",
                                     "camera_header_time_ns":timestamp, "steering_rad":0, "source_mcap_sha256":"one"})
            rows, stats = build_sequences("source", "train", root, manifest)
            self.assertEqual((stats["temporal_candidates"], stats["accepted"], stats["rejected_gap"], stats["rejected_boundary"]), (2,1,1,2))
            self.assertLess(rows[0]["timestamp_t_minus_2_ns"], rows[0]["timestamp_t_minus_1_ns"])
            self.assertLess(rows[0]["timestamp_t_minus_1_ns"], rows[0]["timestamp_t_ns"])
            self.assertEqual(rows[0]["source_id"], "source")

    def test_maximum_gap_is_exactly_120ms(self):
        config = load_dataset_config(ROOT / "configs/high_speed_temporal_dataset_v1.json")
        self.assertEqual(config["maximum_adjacent_gap_s"], .120)
        with self.assertRaises(ValueError): CausalFrameBuffer(.121)

    def test_channel_order_oldest_to_current_and_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            paths=[]
            for i, color in enumerate(((255,0,0),(0,255,0),(0,0,255))):
                path=Path(directory)/f"{i}.png"; Image.new("RGB",(200,66),color).save(path); paths.append(path)
            temporal=preprocess_temporal_paths(paths)
            self.assertEqual(temporal.shape,(9,66,200))
            self.assertTrue(np.array_equal(temporal[:3],preprocess_png(paths[0])))
            self.assertTrue(np.array_equal(temporal[6:],preprocess_png(paths[2])))

    def test_parameter_count_exact(self):
        self.assertEqual(sum(p.numel() for p in build_temporal_pilotnet().parameters()),255_819)

    def test_training_sources_match_v8_and_all_b_are_excluded(self):
        config=load_training_config(ROOT/"configs/pilotnet_training_v9_high_speed_temporal.json")
        self.assertEqual(config["train_episodes"],[f"episode_{i:03d}" for i in range(1,9)])
        self.assertEqual([config[f"dagger{i}_training_rollout"] for i in (1,2,3)],
                         ["high_speed_dagger_rollout_A","high_speed_dagger_iter2_rollout_A","high_speed_dagger_iter3_rollout_A"])
        self.assertTrue(all(config[f"dagger{i}_holdout_rollout"].endswith("_B") for i in (1,2,3)))
        self.assertEqual(config["initialization"],"from_scratch")

    def test_no_new_collection_or_dagger(self):
        config=load_dataset_config(ROOT/"configs/high_speed_temporal_dataset_v1.json")
        self.assertFalse(config["new_training_data_collection_permitted"])
        self.assertFalse(config["dagger_iteration4_permitted"])
        source=(ROOT/"src/physicar_e2e/high_speed_temporal.py").read_text()
        self.assertNotIn("DockerRosBackend",source)
        self.assertNotIn("start_recorder",source)

    def test_live_buffer_waits_for_three_real_frames_without_padding(self):
        buffer=CausalFrameBuffer(); frame=np.zeros((3,66,200),dtype=np.float32)
        buffer.append(1.0,frame); buffer.append(1.06,frame)
        self.assertFalse(buffer.ready)
        with self.assertRaises(TemporalInputError): buffer.tensor()
        buffer.append(1.12,frame); self.assertTrue(buffer.ready); self.assertEqual(buffer.tensor().shape,(9,66,200))

    def test_live_gap_failure_has_dedicated_classification(self):
        buffer=CausalFrameBuffer(); frame=np.zeros((3,66,200),dtype=np.float32); buffer.append(1.0,frame)
        with self.assertRaises(TemporalInputError): buffer.append(1.121,frame)
        self.assertEqual(classify_temporal_run(live_result("FAIL",True)),"TEMPORAL_INPUT_FAIL")


class TemporalLiveGates(unittest.TestCase):
    def execute(self,outcomes):
        run=Mock(side_effect=outcomes); before=Mock(return_value=(object(),{"result":"PASS"}))
        with tempfile.TemporaryDirectory() as directory:
            attempts,result=run_attempts(Client(),object(),object(),Path(directory),preflight_one=before,run_one=run)
        return attempts,result,run

    def test_run1_policy_failure_stops(self):
        attempts,result,run=self.execute([live_result("FAIL"),live_result()])
        self.assertEqual((len(attempts),result,run.call_count),(1,"FAIL",1))

    def test_run2_policy_failure_stops(self):
        attempts,result,run=self.execute([live_result(),live_result("FAIL"),live_result()])
        self.assertEqual((len(attempts),result,run.call_count),(2,"FAIL",2))

    def test_three_valid_runs_and_five_attempt_budget(self):
        attempts,result,run=self.execute([live_result(),live_result(),live_result(),live_result()])
        self.assertEqual((len(attempts),result,run.call_count),(3,"PASS",3)); self.assertEqual(MAX_TOTAL_ATTEMPTS,5)

    def test_temporal_failure_is_replaceable(self):
        attempts,result,run=self.execute([live_result("FAIL",True),live_result(),live_result(),live_result()])
        self.assertEqual((len(attempts),result,run.call_count),(4,"PASS",4))
        self.assertEqual(attempts[0]["classification"],"TEMPORAL_INPUT_FAIL")


if __name__=="__main__": unittest.main()
