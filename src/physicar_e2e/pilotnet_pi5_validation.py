"""Offline parity gate between canonical V4 and the Pi deployment core."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from .pilotnet import clamp_steering_rad, preprocess_live_jpeg, steering_normalized_to_rad
from .pilotnet_inference import CameraOnlyOnnxModel
from .pilotnet_pi5 import PilotNetPi5, preprocess_image_bytes

def file_hash(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def run_parity(model_path: Path, images: list[Path]) -> dict:
    if not images: raise ValueError("parity requires representative images")
    canonical=CameraOnlyOnnxModel(model_path); deployment=PilotNetPi5(model_path); rows=[]
    for path in images:
        payload=path.read_bytes(); expected=preprocess_live_jpeg(payload); actual=preprocess_image_bytes(payload)
        tensor_diff=np.abs(expected-actual[0])
        canonical_rad=clamp_steering_rad(float(steering_normalized_to_rad(canonical.predict(expected))))
        deployment_rad=deployment.infer_tensor(actual); diff=abs(canonical_rad-deployment_rad)
        rows.append({"image":str(path),"image_sha256":file_hash(path),"tensor_exact_equal":bool(np.array_equal(expected,actual[0])),
                     "tensor_mean_absolute_difference":float(np.mean(tensor_diff)),"tensor_max_absolute_difference":float(np.max(tensor_diff)),
                     "canonical_steering_rad":canonical_rad,"deployment_steering_rad":deployment_rad,"absolute_steering_difference_rad":diff})
    differences=np.asarray([row["absolute_steering_difference_rad"] for row in rows])
    passed=all(row["tensor_exact_equal"] for row in rows) and float(np.max(differences)) <= 1e-7
    return {"version":"pilotnet_pi5_deployment_v1_parity","result":"PASS" if passed else "FAIL","sample_count":len(rows),
            "tensor_requirement":"exact equality","all_tensors_exact_equal":all(row["tensor_exact_equal"] for row in rows),
            "steering_tolerance_rad":1e-7,"steering_difference_rad":{"mean":float(np.mean(differences)),"p95":float(np.percentile(differences,95)),"max":float(np.max(differences))},"samples":rows}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--model",required=True,type=Path); p.add_argument("--images",nargs="+",required=True,type=Path); p.add_argument("--output",required=True,type=Path); a=p.parse_args(argv)
    result=run_parity(a.model,a.images); a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); print(json.dumps(result,indent=2,sort_keys=True)); return 0 if result["result"]=="PASS" else 1
if __name__=="__main__": raise SystemExit(main())
