import torch
# import sdnq to register it into diffusers and transformers
from sdnq.common import use_torch_compile as triton_is_available
from sdnq.loader import apply_sdnq_options_to_model
from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline
# from diffusers import Flux2KleinPipeline
pipe = Flux2KleinPipeline.from_pretrained(
    "Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic", torch_dtype=torch.float16)

# Enable INT8 MatMul for AMD, Intel ARC and Nvidia GPUs:
if triton_is_available and (torch.cuda.is_available() or torch.xpu.is_available()):
    pipe.transformer = apply_sdnq_options_to_model(
        pipe.transformer, use_quantized_matmul=True)
    pipe.text_encoder = apply_sdnq_options_to_model(
        pipe.text_encoder, use_quantized_matmul=True)
    pipe.transformer = torch.compile(pipe.transformer) # optional for faster speeds

pipe.enable_model_cpu_offload()

prompt = "A cat holding a sign that says 'hello world'"
image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    guidance_scale=1.0,
    num_inference_steps=20,
).images[0]  # type: ignore[union-attr]
image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    guidance_scale=1.0,
    num_inference_steps=20,
).images[0]  # type: ignore[union-attr]
image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    guidance_scale=1.0,
    num_inference_steps=4,
).images[0]  # type: ignore[union-attr]
image.save("others/flux-klein-sdnq-4bit-dynamic.png")
