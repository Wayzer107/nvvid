#최대 7720프레임 = 321.6초 x 24fps 

import os
import spaces
import torch
from diffusers.pipelines.wan.pipeline_wan_i2v import WanImageToVideoPipeline
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
from diffusers.utils.export_utils import export_to_video
import gradio as gr
import tempfile
import numpy as np
from PIL import Image
import random
import gc

from torchao.quantization import quantize_
from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, Int8WeightOnlyConfig
import aoti

# =========================================================
# MODEL CONFIGURATION
# =========================================================
MODEL_ID = os.getenv("MODEL_ID")
HF_TOKEN = os.environ.get("HF_TOKEN")

MAX_DIM = 832
MIN_DIM = 480
SQUARE_DIM = 640
MULTIPLE_OF = 16

MAX_SEED = np.iinfo(np.int32).max

FIXED_FPS = 24
MIN_FRAMES_MODEL = 8
MAX_FRAMES_MODEL = 7720

MIN_DURATION = 0.5
MAX_DURATION = 10.0

# =========================================================
# LOAD PIPELINE
# =========================================================
print("Loading pipeline...")
pipe = WanImageToVideoPipeline.from_pretrained(
    MODEL_ID,
    transformer=WanTransformer3DModel.from_pretrained(
        MODEL_ID,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        token=HF_TOKEN
    ),
    transformer_2=WanTransformer3DModel.from_pretrained(
        MODEL_ID,
        subfolder="transformer_2",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        token=HF_TOKEN
    ),
    torch_dtype=torch.bfloat16,
).to("cuda")

# =========================================================
# LOAD LORA ADAPTERS
# =========================================================
print("Loading LoRA adapters...")
pipe.load_lora_weights(
    "Kijai/WanVideo_comfy",
    weight_name="Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors",
    adapter_name="lightx2v"
)
pipe.load_lora_weights(
    "Kijai/WanVideo_comfy",
    weight_name="Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors",
    adapter_name="lightx2v_2",
    load_into_transformer_2=True
)

pipe.set_adapters(["lightx2v", "lightx2v_2"], adapter_weights=[1., 1.])
pipe.fuse_lora(adapter_names=["lightx2v"], lora_scale=3., components=["transformer"])
pipe.fuse_lora(adapter_names=["lightx2v_2"], lora_scale=1., components=["transformer_2"])
pipe.unload_lora_weights()

# =========================================================
# QUANTIZATION & AOT OPTIMIZATION
# =========================================================
print("Applying quantization...")
quantize_(pipe.text_encoder, Int8WeightOnlyConfig())
quantize_(pipe.transformer, Float8DynamicActivationFloat8WeightConfig())
quantize_(pipe.transformer_2, Float8DynamicActivationFloat8WeightConfig())

print("Loading AOTI blocks...")
aoti.aoti_blocks_load(pipe.transformer, 'zerogpu-aoti/Wan2', variant='fp8da')
aoti.aoti_blocks_load(pipe.transformer_2, 'zerogpu-aoti/Wan2', variant='fp8da')

# =========================================================
# DEFAULT PROMPTS
# =========================================================
default_prompt_i2v = "Generate a video with smooth and natural movement. Objects should have visible motion while maintaining fluid transitions."
default_negative_prompt = "low quality, worst quality, blurry, distorted, deformed, ugly, bad anatomy"

# =========================================================
# IMAGE RESIZING LOGIC
# =========================================================
def resize_image(image: Image.Image) -> Image.Image:
    width, height = image.size
    if width == height:
        return image.resize((SQUARE_DIM, SQUARE_DIM), Image.LANCZOS)

    aspect_ratio = width / height
    MAX_ASPECT_RATIO = MAX_DIM / MIN_DIM
    MIN_ASPECT_RATIO = MIN_DIM / MAX_DIM

    image_to_resize = image

    if aspect_ratio > MAX_ASPECT_RATIO:
        crop_width = int(round(height * MAX_ASPECT_RATIO))
        left = (width - crop_width) // 2
        image_to_resize = image.crop((left, 0, left + crop_width, height))
    elif aspect_ratio < MIN_ASPECT_RATIO:
        crop_height = int(round(width / MIN_ASPECT_RATIO))
        top = (height - crop_height) // 2
        image_to_resize = image.crop((0, top, width, top + crop_height))

    if width > height:
        target_w = MAX_DIM
        target_h = int(round(target_w / aspect_ratio))
    else:
        target_h = MAX_DIM
        target_w = int(round(target_h * aspect_ratio))

    final_w = round(target_w / MULTIPLE_OF) * MULTIPLE_OF
    final_h = round(target_h / MULTIPLE_OF) * MULTIPLE_OF

    final_w = max(MIN_DIM, min(MAX_DIM, final_w))
    final_h = max(MIN_DIM, min(MAX_DIM, final_h))

    return image_to_resize.resize((final_w, final_h), Image.LANCZOS)

# =========================================================
# UTILITY FUNCTIONS
# =========================================================
def get_num_frames(duration_seconds: float):
    return 1 + int(np.clip(int(round(duration_seconds * FIXED_FPS)), MIN_FRAMES_MODEL, MAX_FRAMES_MODEL))

def get_duration(
    input_image, prompt, steps, negative_prompt,
    duration_seconds, guidance_scale, guidance_scale_2,
    seed, randomize_seed, progress,
):
    if input_image is None:
        return 120

    BASE_FRAMES_HEIGHT_WIDTH = 81 * 832 * 624
    BASE_STEP_DURATION = 15
    width, height = resize_image(input_image).size
    frames = get_num_frames(duration_seconds)
    factor = frames * width * height / BASE_FRAMES_HEIGHT_WIDTH
    step_duration = BASE_STEP_DURATION * factor ** 1.5
    return 10 + int(steps) * step_duration

# =========================================================
# MAIN GENERATION FUNCTION
# =========================================================
@spaces.GPU(duration=get_duration)
def generate_video(
    input_image,
    prompt,
    steps=4,
    negative_prompt=default_negative_prompt,
    duration_seconds=3.5,
    guidance_scale=1,
    guidance_scale_2=1,
    seed=42,
    randomize_seed=False,
    progress=gr.Progress(track_tqdm=True),
):
    if input_image is None:
        raise gr.Error("Please upload an image.")

    num_frames = get_num_frames(duration_seconds)
    current_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)
    resized_image = resize_image(input_image)

    output_frames_list = pipe(
        image=resized_image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=resized_image.height,
        width=resized_image.width,
        num_frames=num_frames,
        guidance_scale=float(guidance_scale),
        guidance_scale_2=float(guidance_scale_2),
        num_inference_steps=int(steps),
        generator=torch.Generator(device="cuda").manual_seed(current_seed),
    ).frames[0]

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpfile:
        video_path = tmpfile.name
    export_to_video(output_frames_list, video_path, fps=FIXED_FPS)
    return video_path, current_seed

# =========================================================
# GRADIO UI
# =========================================================
with gr.Blocks() as demo:
    
    gr.HTML("""
    <style>
        .gradio-container {
            background: linear-gradient(135deg, #fef9f3 0%, #f0e6fa 50%, #e6f0fa 100%) !important;
        }
        footer {display: none !important;}
    </style>
    <div style="text-align: center; margin-bottom: 20px;">
        <h1 style="color: #6b5b7a; font-size: 2.2rem; font-weight: 700; margin-bottom: 0.3rem;">
            🎬 NSFW Uncensored "Image to Video"
        </h1>
        <p style="color: #8b7b9b; font-size: 1rem;">Powered by Wan 2.2 Model</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            input_image_component = gr.Image(
                type="pil", 
                label="📷 Upload Image",
                height=350
            )
            prompt_input = gr.Textbox(
                label="✏️ Prompt", 
                value=default_prompt_i2v,
                placeholder="Describe the motion you want...",
                lines=3
            )
            duration_seconds_input = gr.Slider(
                minimum=MIN_DURATION, 
                maximum=MAX_DURATION, 
                step=0.5, 
                value=3.5,
                label="⏱️ Duration (seconds)"
            )

            with gr.Accordion("⚙️ Options", open=False):
                negative_prompt_input = gr.Textbox(
                    label="Negative Prompt", 
                    value=default_negative_prompt, 
                    lines=2
                )
                steps_slider = gr.Slider(
                    minimum=1, 
                    maximum=30, 
                    step=1, 
                    value=6, 
                    label="Inference Steps"
                )
                guidance_scale_input = gr.Slider(
                    minimum=0.0, 
                    maximum=10.0, 
                    step=0.5, 
                    value=1, 
                    label="Guidance Scale"
                )
                guidance_scale_2_input = gr.Slider(
                    minimum=0.0, 
                    maximum=10.0, 
                    step=0.5, 
                    value=1, 
                    label="Guidance Scale 2"
                )
                seed_input = gr.Slider(
                    label="Seed", 
                    minimum=0, 
                    maximum=MAX_SEED, 
                    step=1, 
                    value=42
                )
                randomize_seed_checkbox = gr.Checkbox(
                    label="Randomize Seed", 
                    value=True
                )

            generate_button = gr.Button(
                "✨ Generate Video", 
                variant="primary"
            )

        with gr.Column(scale=1):
            video_output = gr.Video(
                label="🎥 Generated Video", 
                autoplay=True,
                height=450
            )

    ui_inputs = [
        input_image_component, prompt_input, steps_slider,
        negative_prompt_input, duration_seconds_input,
        guidance_scale_input, guidance_scale_2_input,
        seed_input, randomize_seed_checkbox
    ]
    
    generate_button.click(
        fn=generate_video, 
        inputs=ui_inputs, 
        outputs=[video_output, seed_input]
    )

if __name__ == "__main__":
    demo.queue().launch()