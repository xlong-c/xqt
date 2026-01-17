import torch
import diffusers
from sdnq import SDNQConfig # import sdnq to register it into diffusers and transformers
from sdnq.common import use_torch_compile as triton_is_available
from sdnq.loader import apply_sdnq_options_to_model

pipe = diffusers.Flux2KleinPipeline.from_pretrained("Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic", torch_dtype=torch.bfloat16)

# Enable INT8 MatMul for AMD, Intel ARC and Nvidia GPUs:
if triton_is_available and (torch.cuda.is_available() or torch.xpu.is_available()):
    pipe.transformer = apply_sdnq_options_to_model(pipe.transformer, use_quantized_matmul=True)
    pipe.text_encoder = apply_sdnq_options_to_model(pipe.text_encoder, use_quantized_matmul=True)
    # pipe.transformer = torch.compile(pipe.transformer) # optional for faster speeds

pipe.enable_model_cpu_offload()

prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    guidance_scale=1.0,
    num_inference_steps=4,
    generator=torch.manual_seed(0)
).images[0]

image.save("flux-klein-sdnq-4bit-dynamic.png")
