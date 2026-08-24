import importlib.util, io, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import _bootstrap  # noqa: F401
from physicar_e2e.pilotnet import preprocess_live_jpeg
from physicar_e2e.pilotnet_pi5 import (
    EXPECTED_ONNX_SHA256, MAX_STEERING_RAD, PilotNetPi5, ROI, decode_rgb_image,
    preprocess_image_bytes, preprocess_rgb_image, steering_normalized_to_rad, verify_model,
)

def jpeg_bytes(size=(480,360),mode="RGB"):
    image=Image.new(mode,size,128); stream=io.BytesIO(); image.save(stream,format="JPEG"); return stream.getvalue()

class FakeSession:
    def __init__(self,value=.25): self.value=value
    def get_inputs(self): return [SimpleNamespace(name="camera_yuv",shape=["batch",3,66,200])]
    def get_outputs(self): return [SimpleNamespace(name="steering_normalized",shape=["batch",1])]
    def run(self,*_): return [np.asarray([[self.value]],dtype=np.float32)]

class Pi5DeploymentTests(unittest.TestCase):
    def test_hash_constant_and_wrong_model_rejected(self):
        self.assertEqual(EXPECTED_ONNX_SHA256,"5dd2b88b50c43aed44361229dea34e8981cb0a34d05b01eef81a9ccdf63f396a")
        with tempfile.NamedTemporaryFile() as f:
            f.write(b"wrong"); f.flush()
            with self.assertRaisesRegex(RuntimeError,"SHA-256 mismatch"): verify_model(f.name)

    def test_input_contract_roi_resize_yuv_and_shape_match_canonical(self):
        payload=jpeg_bytes(); actual=preprocess_image_bytes(payload)
        self.assertEqual(ROI,(0,160,480,360)); self.assertEqual(actual.shape,(1,3,66,200)); self.assertEqual(actual.dtype,np.float32)
        self.assertTrue(np.array_equal(actual[0],preprocess_live_jpeg(payload)))

    def test_rgb_input_validation(self):
        with self.assertRaisesRegex(ValueError,"480x360"): preprocess_rgb_image(Image.new("RGB",(479,360)))
        with self.assertRaisesRegex(ValueError,"must be RGB"): preprocess_rgb_image(Image.new("L",(480,360)))

    def test_malformed_image_rejected(self):
        with self.assertRaisesRegex(ValueError,"cannot decode"): decode_rgb_image(b"not an image")

    def test_deterministic_preprocessing(self):
        payload=jpeg_bytes(); self.assertTrue(np.array_equal(preprocess_image_bytes(payload),preprocess_image_bytes(payload)))

    def test_physical_output_clamps_to_contract(self):
        self.assertEqual(steering_normalized_to_rad(100),MAX_STEERING_RAD); self.assertEqual(steering_normalized_to_rad(-100),-MAX_STEERING_RAD)

    def test_deterministic_repeated_inference_and_output_parity_helper(self):
        with tempfile.NamedTemporaryFile() as f:
            f.write(b"fake model"); f.flush(); import hashlib
            expected=hashlib.sha256(b"fake model").hexdigest()
            model=PilotNetPi5(f.name,expected_sha256=expected,session_factory=lambda _:FakeSession())
            tensor=preprocess_image_bytes(jpeg_bytes()); outputs=[model.infer_tensor(tensor) for _ in range(3)]
            self.assertEqual(outputs[0],outputs[1]); self.assertAlmostEqual(outputs[0],steering_normalized_to_rad(.25))

    def test_standalone_bundle_preprocessing_parity_and_light_dependencies(self):
        root=Path(__file__).resolve().parents[1]; core=root/"deploy/pi5/pilotnet_pi5_core.py"
        spec=importlib.util.spec_from_file_location("standalone_pi5_core",core); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        payload=jpeg_bytes(); self.assertTrue(np.array_equal(module.preprocess_rgb_image(module.decode_rgb_image(payload)),preprocess_image_bytes(payload)))
        source=core.read_text(encoding="utf-8")
        for forbidden in ("torch","cv2","rclpy","rospy","cuda"):
            self.assertNotIn(f"import {forbidden}",source)

if __name__=="__main__": unittest.main()
