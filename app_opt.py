import gradio as gr
import torch
from diffusers import HunyuanDiTPipeline
from transformers import T5EncoderModel
import random
import gc
from PIL import Image
import os

def flush():
    gc.collect()
    torch.cuda.empty_cache()

class HunyuanDiTGenerator:
    def __init__(self):
        self.model_id = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = None
        self.text_encoder_2 = None
        self.encoder_pipeline = None
        self.default_negative_prompt = ''

    def load_model(self, model_id):
        if self.model_id == model_id and self.pipeline is not None:
            return "Model already loaded."

        self.model_id = model_id
        flush()

        # Load text encoder
        self.text_encoder_2 = T5EncoderModel.from_pretrained(
            self.model_id,
            subfolder="text_encoder_2",
            load_in_8bit=True,
            device_map="auto",
        )

        # Load encoder pipeline
        self.encoder_pipeline = HunyuanDiTPipeline.from_pretrained(
            self.model_id, 
            text_encoder_2=self.text_encoder_2,
            transformer=None,
            vae=None,
            torch_dtype=torch.float16,
            device_map="balanced",
        )

        # Load main pipeline
        self.pipeline = HunyuanDiTPipeline.from_pretrained(
            self.model_id,
            text_encoder=None,
            text_encoder_2=None,
            torch_dtype=torch.float16,
        ).to(self.device)

        return f"Model {model_id} loaded successfully!"

    def get_text_emb(self, prompts):
        with torch.no_grad():
            TEXT_ENCODER_CONF = {
                "negative_prompt": self.default_negative_prompt,
                "prompt_embeds": None,
                "negative_prompt_embeds": None,
                "prompt_attention_mask": None,
                "negative_prompt_attention_mask": None,
                "max_sequence_length": 256,
                "text_encoder_index": 1,
            }
            prompt_emb1 = self.encoder_pipeline.encode_prompt(prompts, negative_prompt=self.default_negative_prompt)
            prompt_emb2 = self.encoder_pipeline.encode_prompt(prompts, **TEXT_ENCODER_CONF)
        return prompt_emb1, prompt_emb2

    def generate(self, prompt, negative_prompt, seed, infer_steps, guidance_scale):
        if self.pipeline is None:
            return None, "Please load a model first!"

        if seed is None or seed == "":
            seed = random.randint(0, 1_000_000)
        seed = int(seed)
        generator = torch.Generator(device=self.device).manual_seed(seed)

        self.default_negative_prompt = negative_prompt if negative_prompt else ''

        prompt_emb1, prompt_emb2 = self.get_text_emb(prompt)
        prompt_embeds, negative_prompt_embeds, prompt_attention_mask, negative_prompt_attention_mask = prompt_emb1
        prompt_embeds_2, negative_prompt_embeds_2, prompt_attention_mask_2, negative_prompt_attention_mask_2 = prompt_emb2

        samples = self.pipeline(
            prompt_embeds=prompt_embeds,
            prompt_embeds_2=prompt_embeds_2,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_2=negative_prompt_embeds_2,
            prompt_attention_mask=prompt_attention_mask,
            prompt_attention_mask_2=prompt_attention_mask_2,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            negative_prompt_attention_mask_2=negative_prompt_attention_mask_2,
            num_images_per_prompt=1,
            guidance_scale=guidance_scale,
            num_inference_steps=infer_steps,
            generator=generator, 
        ).images[0]

        return samples, f"Image generated successfully! Seed: {seed}"

# Initialize generator and image history
generator = HunyuanDiTGenerator()
image_history = []

MODEL_CHOICES = [
    "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers",
    "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled",
    "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers",
    "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled"
]

def load_model(model_choice):
    return generator.load_model(model_choice)

def generate_image(prompt, negative_prompt, seed, infer_steps, guidance_scale):
    global image_history
    
    try:
        image, message = generator.generate(prompt, negative_prompt, seed, infer_steps, guidance_scale)
        
        if image is not None:
            image_history.append(image)
            image_history = image_history[-10:]  # Keep only the last 10 images
        
        return image, message
    except Exception as e:
        return None, f"Error generating image: {str(e)}"

def save_image(image):
    if image is not None:
        os.makedirs("generated_images", exist_ok=True)
        save_path = os.path.join("generated_images", f"generated_{random.randint(0, 10000)}.png")
        image.save(save_path)
        return f"Image saved to {save_path}"
    return "No image to save"

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# HyDiT-Live:")
    gr.Markdown("## Minimal Hunyuan-DiT Image Generation")
    
    with gr.Row():
        with gr.Column():
            model_dropdown = gr.Dropdown(choices=MODEL_CHOICES, label="Select Model")
            load_button = gr.Button("Load Model")
            model_status = gr.Textbox(label="Model Status", interactive=False)
            
            prompt_input = gr.Textbox(label="Prompt")
            negative_prompt_input = gr.Textbox(label="Negative Prompt")
            seed_input = gr.Textbox(label="Seed (optional)", placeholder="Random if left empty")
            infer_steps_slider = gr.Slider(minimum=30, maximum=100, value=30, step=1, label="Inference Steps")
            guidance_scale_slider = gr.Slider(minimum=1, maximum=20, value=7, step=1, label="Guidance Scale")
            
            generate_button = gr.Button("Generate Image")
        
        with gr.Column():
            image_output = gr.Image(label="Generated Image")
            save_button = gr.Button("Save Image")
            save_status = gr.Textbox(label="Save Status", interactive=False)
            generation_status = gr.Textbox(label="Generation Status", interactive=False)
    
    with gr.Row():
        gr.Markdown("## Image History (Last 10 Generated Images)")
        image_gallery = gr.Gallery(
            label="Image History",
            show_label=False,
            elem_id="gallery",
            columns=[5],
            rows=[2],
            object_fit="contain",
            height="auto"
        )

    # Event handlers
    load_button.click(load_model, inputs=[model_dropdown], outputs=[model_status])
    generate_button.click(generate_image, 
                          inputs=[prompt_input, negative_prompt_input, seed_input, infer_steps_slider, guidance_scale_slider],
                          outputs=[image_output, generation_status])
    save_button.click(save_image, inputs=[image_output], outputs=[save_status])
    
    # Update gallery after each generation
    generate_button.click(lambda: image_history, outputs=[image_gallery])

# Launch the interface
if __name__ == "__main__":
    demo.launch(share=True)
