"""Self-contained Raspberry Pi PilotNet V4 inference and benchmark core."""
from __future__ import annotations
import argparse, hashlib, io, json, math, time
from pathlib import Path
import numpy as np
from PIL import Image, UnidentifiedImageError

EXPECTED_ONNX_SHA256="5dd2b88b50c43aed44361229dea34e8981cb0a34d05b01eef81a9ccdf63f396a"
EXPECTED_ONNX_SIZE_BYTES=1012518; SOURCE_SIZE=(480,360); ROI=(0,160,480,360); MODEL_SIZE=(200,66); MAX_STEERING_RAD=.349066
M=np.asarray([[.299,.587,.114],[-.14713,-.28886,.436],[.615,-.51499,-.10001]],dtype=np.float32)

def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()

def verify_model(path):
    path=Path(path); actual=sha256_file(path)
    if actual!=EXPECTED_ONNX_SHA256: raise RuntimeError(f"V4 ONNX SHA-256 mismatch: expected {EXPECTED_ONNX_SHA256}, got {actual}")
    if path.stat().st_size!=EXPECTED_ONNX_SIZE_BYTES: raise RuntimeError(f"V4 ONNX size mismatch: expected {EXPECTED_ONNX_SIZE_BYTES}, got {path.stat().st_size}")
    return {"result":"PASS","sha256":actual,"size_bytes":path.stat().st_size}

def decode_rgb_image(payload):
    try:
        with Image.open(io.BytesIO(payload)) as image: image.load(); rgb=image.convert("RGB")
    except (UnidentifiedImageError,OSError,ValueError) as exc: raise ValueError(f"cannot decode input image: {exc}") from exc
    if rgb.size!=SOURCE_SIZE: raise ValueError(f"input image must be 480x360, got {rgb.size}")
    return rgb

def preprocess_rgb_image(image):
    if image.mode!="RGB" or image.size!=SOURCE_SIZE: raise ValueError(f"input must be RGB 480x360, got {image.mode} {image.size}")
    rgb=np.asarray(image.crop(ROI).resize(MODEL_SIZE,Image.Resampling.BILINEAR),dtype=np.uint8).astype(np.float32)/np.float32(255)
    yuv=rgb@M.T; yuv[...,1:]+=np.float32(.5); chw=np.ascontiguousarray(((yuv-np.float32(.5))*np.float32(2)).transpose(2,0,1),dtype=np.float32)
    return np.expand_dims(chw,0)

class PilotNetPi5:
    def __init__(self,model_path):
        self.identity=verify_model(model_path)
        import onnxruntime as ort
        self.session=ort.InferenceSession(str(model_path),providers=["CPUExecutionProvider"])
        i=self.session.get_inputs()[0]; o=self.session.get_outputs()[0]
        if i.name!="camera_yuv" or list(i.shape)[-3:]!=[3,66,200] or o.name!="steering_normalized" or list(o.shape)[-1:]!=[1]: raise RuntimeError("unexpected ONNX I/O contract")
    def infer_tensor(self,tensor):
        if tensor.shape!=(1,3,66,200) or tensor.dtype!=np.float32: raise ValueError("expected float32 1x3x66x200")
        normalized=float(self.session.run(["steering_normalized"],{"camera_yuv":tensor})[0].reshape(-1)[0])
        rad=float(np.float32(normalized)*np.float32(MAX_STEERING_RAD))
        if not math.isfinite(rad): raise RuntimeError("non-finite steering")
        return max(-MAX_STEERING_RAD,min(MAX_STEERING_RAD,rad))
    def infer_rgb(self,image): return self.infer_tensor(preprocess_rgb_image(image))
    def infer_bytes(self,payload): return self.infer_rgb(decode_rgb_image(payload))

def timing(values):
    a=np.asarray(values)*1000
    return {"count":len(values),"mean_ms":float(np.mean(a)),"median_ms":float(np.median(a)),"p95_ms":float(np.percentile(a,95)),"max_ms":float(np.max(a))}

def run(model,image_path,warmup=0,iterations=1):
    if iterations>1 and iterations<200: raise ValueError("benchmark requires at least 200 iterations")
    path=Path(image_path)
    for _ in range(warmup): model.infer_tensor(preprocess_rgb_image(decode_rgb_image(path.read_bytes())))
    ds=[]; ps=[]; ins=[]; totals=[]; outputs=[]
    for _ in range(iterations):
        total=time.perf_counter(); start=total; image=decode_rgb_image(path.read_bytes()); ds.append(time.perf_counter()-start)
        start=time.perf_counter(); tensor=preprocess_rgb_image(image); ps.append(time.perf_counter()-start)
        start=time.perf_counter(); outputs.append(model.infer_tensor(tensor)); ins.append(time.perf_counter()-start); totals.append(time.perf_counter()-total)
    if iterations==1:
        rad=outputs[0]; return {"steering_rad":rad,"steering_deg":math.degrees(rad),"decode_ms":ds[0]*1000,"preprocess_ms":ps[0]*1000,"inference_ms":ins[0]*1000,"total_ms":totals[0]*1000}
    total_stats=timing(totals)
    platform=__import__("platform"); ort=__import__("onnxruntime")
    return {"benchmark_label":"RASPBERRY PI 5 BENCHMARK" if platform.machine().lower() in ("aarch64","arm64") else "HOST CPU BENCHMARK — NOT RASPBERRY PI 5","runtime_environment":{"system":platform.system(),"machine":platform.machine(),"python":platform.python_version(),"numpy":np.__version__,"Pillow":Image.__version__,"onnxruntime":ort.__version__},"warmup_iterations":warmup,"measured_iterations":iterations,"decode":timing(ds),"preprocessing":timing(ps),"onnx_inference":timing(ins),"total_pipeline":total_stats,"control_period_budget_ms":1000/15,"total_p95_headroom_ms":1000/15-total_stats["p95_ms"],"deterministic_output_range_rad":[min(outputs),max(outputs)]}

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--model",required=True,type=Path); p.add_argument("--image",required=True,type=Path); p.add_argument("--benchmark",action="store_true"); p.add_argument("--warmup",type=int,default=20); p.add_argument("--iterations",type=int,default=250); p.add_argument("--output",type=Path); a=p.parse_args(argv)
    try:
        model=PilotNetPi5(a.model); result=run(model,a.image,a.warmup if a.benchmark else 0,a.iterations if a.benchmark else 1); result["model_identity"]=model.identity
    except Exception as exc: print(f"ERROR: {exc}",file=__import__("sys").stderr); return 2
    if a.output: a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,indent=2,sort_keys=True)); return 0
