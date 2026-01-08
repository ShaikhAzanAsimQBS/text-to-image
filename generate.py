import torch
import time
import pynvml  # For GPU memory stats
from diffusers import StableDiffusionPipeline

# ----------------------------
# MODEL AND PIPELINE SETTINGS
# ----------------------------
MODEL_PATH = r"models/sd15/Realistic_Vision_V6.0_NV_B1_fp16.safetensors"

pipe = StableDiffusionPipeline.from_single_file(
    MODEL_PATH,
    torch_dtype=torch.float16,
    safety_checker=None,
    requires_safety_checker=False
)

# Optimizations for low VRAM GPUs
pipe.enable_attention_slicing()
pipe.enable_model_cpu_offload()

pipe.to("cuda")

# ----------------------------
# PROMPT AND GENERATION SETTINGS
# ----------------------------
prompt = "a man standing in front of a lake at noon, Sunny, soft lighting"
height = 512
width = 512
num_inference_steps = 20
guidance_scale = 6.5

# ----------------------------
# PERFORMANCE METRICS
# ----------------------------
start_time = time.time()

# Optional: Initialize NVML for GPU memory tracking
gpu_stats_enabled = False
try:
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # GPU 0
    gpu_stats_enabled = True
except Exception as e:
    print("Could not initialize NVML for GPU stats:", e)

if gpu_stats_enabled:
    mem_before = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024**2  # MB

# Generate image
image = pipe(
    prompt,
    height=height,
    width=width,
    num_inference_steps=num_inference_steps,
    guidance_scale=guidance_scale
).images[0]

end_time = time.time()
total_time = end_time - start_time

if gpu_stats_enabled:
    mem_after = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024**2  # MB
    mem_used = mem_after - mem_before
    pynvml.nvmlShutdown()
else:
    mem_used = None

# ----------------------------
# SAVE IMAGE
# ----------------------------
output_path = r"output/output.png"
image.save(output_path)

# ----------------------------
# PRINT PERFORMANCE STATS
# ----------------------------
print(f"Image generated successfully and saved to {output_path}")
print(f"Time taken: {total_time:.2f} seconds")
print(f"Height x Width: {height}x{width}")
print(f"Inference steps: {num_inference_steps}")
print(f"Guidance scale: {guidance_scale}")
if mem_used is not None:
    print(f"Approx. GPU memory used for generation: {mem_used:.2f} MB")
