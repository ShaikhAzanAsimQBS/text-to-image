import torch
import time
try:
    import pynvml
    nvml_available = True
except ImportError:
    print("pynvml not installed, GPU memory stats disabled.")
    nvml_available = False

from diffusers import StableDiffusionPipeline

# ----------------------------
# MODEL PATH (put Flux.1 [schnell] here)
# ----------------------------
MODEL_PATH = "models/flux1_schnell"  # Replace with your local Flux.1 model path

# ----------------------------
# LOAD PIPELINE WITH 8-BIT AND CPU OFFLOAD
# ----------------------------
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",   # Automatically maps layers to GPU/CPU
    load_in_8bit=True,   # Reduce VRAM usage for large models
    safety_checker=None
)

# Reduce VRAM by computing attention in slices
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()  # Move unused layers to CPU

pipe = pipe.to("cuda")  # Ensure main GPU is used

# ----------------------------
# PROMPT AND IMAGE SETTINGS
# ----------------------------
prompt = (
    "a man wearing a black hoodie in front of an isle in a super store, "
    "indoor lighting, photo realistic, soft lighting, great face details, "
    "as seen from CCTV camera footage, full body shot"
)

height = 576    # 16:9 aspect ratio
width = 1024
num_inference_steps = 25
guidance_scale = 7.5

# ----------------------------
# PERFORMANCE METRICS
# ----------------------------
start_time = time.time()

if nvml_available:
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_before = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024**2
    except:
        nvml_available = False
        mem_before = None
else:
    mem_before = None

# ----------------------------
# GENERATE IMAGE
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
        mem_after = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024**2
        mem_used = mem_after - mem_before
        pynvml.nvmlShutdown()
    except:
        mem_used = None
else:
    mem_used = None

# ----------------------------
# SAVE IMAGE
# ----------------------------
output_path = "flux_output.png"
image.save(output_path)

# ----------------------------
# PRINT PERFORMANCE STATS
# ----------------------------
print(f"Image generated successfully: {output_path}")
print(f"Time taken: {total_time:.2f} seconds")
print(f"Resolution: {height}x{width}")
print(f"Inference steps: {num_inference_steps}")
print(f"Guidance scale: {guidance_scale}")
if mem_used is not None:
    print(f"Approx. GPU memory used: {mem_used:.2f} MB")
