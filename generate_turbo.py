import torch
import time
try:
    import pynvml  # NVIDIA GPU stats
    nvml_available = True
except ImportError:
    print("pynvml not installed, GPU memory stats disabled.")
    nvml_available = False

from diffusers import AutoPipelineForText2Image

# ----------------------------
# MODEL AND PIPELINE SETTINGS
# ----------------------------
pipe = AutoPipelineForText2Image.from_pretrained(
    r"models/turbo",
    torch_dtype=torch.float16,
    safety_checker=None
)

# Optimizations for low VRAM GPUs
pipe.enable_attention_slicing()
pipe.enable_model_cpu_offload()
pipe = pipe.to("cuda")

# ----------------------------
# PROMPT AND GENERATION SETTINGS
# ----------------------------
prompt = (
    "a man wearing a black hoodie standing in front of a lake at noon, Sunny, soft lighting and his face is visible"
)
height = 576
width = 576
num_inference_steps = 1
guidance_scale = 0.0

# ----------------------------
# PERFORMANCE METRICS
# ----------------------------
start_time = time.time()

if nvml_available:
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # GPU 0
        mem_before = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024**2  # MB
    except Exception as e:
        print("Could not initialize NVML:", e)
        nvml_available = False
        mem_before = None
else:
    mem_before = None

# ----------------------------
# IMAGE GENERATION
# ----------------------------
image = pipe(
    prompt,
    height=height,
    width=width,
    num_inference_steps=num_inference_steps,
    guidance_scale=guidance_scale
).images[0]

end_time = time.time()
total_time = end_time - start_time

if nvml_available:
    try:
        mem_after = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024**2  # MB
        mem_used = mem_after - mem_before
        pynvml.nvmlShutdown()
    except:
        mem_used = None
else:
    mem_used = None

# ----------------------------
# SAVE IMAGE
# ----------------------------
output_path = "output_image_3.png"
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
