import torch
import gradio as gr
from PIL import Image
from diffusers import QwenImageEditPlusPipeline
from pathlib import Path

# Setup storage for output
RESULT_DIR = Path("./static")
RESULT_DIR.mkdir(exist_ok=True)

# Load pipeline (basic mode: bfloat16 for efficiency, to CUDA)
pipe = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2509",
    torch_dtype=torch.bfloat16
).to("cuda")

def process_edit(target_img, ref_img, prompt):
    if target_img is None or ref_img is None:
        return None
    
    # No manual resize – pipeline auto-handles to ~1MP
    inputs = {
        "image": [target_img.convert("RGB"), ref_img.convert("RGB")],
        "prompt": prompt,
        "negative_prompt": " ",  # Required for CFG >1
        "num_inference_steps": 30,  # Basic: 30 steps for speed/quality
        "true_cfg_scale": 4.0,
    }
    
    with torch.inference_mode():
        output = pipe(**inputs).images[0]
    
    # Save for Gradio access
    save_path = RESULT_DIR / "result.jpg"
    output.save(save_path, "JPEG")
    
    return output

# Basic Gradio demo
with gr.Blocks() as demo:
    gr.Markdown("# Basic Qwen-Image-Edit-2509 Demo")
    with gr.Row():
        with gr.Column():
            slot1 = gr.Image(label="Target Image", type="pil")
            slot2 = gr.Image(label="Reference Image", type="pil")
            prompt = gr.Textbox(label="Edit Prompt", value="Use image 2 to edit image 1.")
            btn = gr.Button("Generate")
        with gr.Column():
            out = gr.Image(label="Output")
    btn.click(process_edit, inputs=[slot1, slot2, prompt], outputs=out)

demo.launch(server_name="0.0.0.0", server_port=8000, allowed_paths=["./static"])
