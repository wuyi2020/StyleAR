from inference_solver import FlexARInferenceSolver
from PIL import Image
import os
from transformers.image_utils import load_image


def merge_images(images, output_path):  
    width, height = images[0].size  
    total_width = width * len(images)  
    result_img = Image.new('RGB', (total_width, height))  
    x_offset = 0  
    for image in images:  
        result_img.paste(image, (x_offset, 0))  
        x_offset += width  
    result_img.save(output_path)  


# ********************* Omni-Potent *********************
inference_solver = FlexARInferenceSolver(
    model_path="Alpha-VLLM/Lumina-mGPT-7B-768",
    precision="fp32",
    lora_params_path="/apdcephfs_nj7/share_303172354/aniwu/Lumina-lora/lumina_mgpt/output/laion_InstantStyle_0311_ratio01_noise01/epoch9",
    target_size=768,
)


special_tokens = "S* " * 15 + "S*"
q1 = f"Instant style Generation. Given image style is {special_tokens}. Please generate an same style image according to my instruction: "


test_prompt = "a dog"


input_image_paths = [""]

save_root = ""

os.makedirs(save_root, exist_ok=True)

images = []
counter = 0
for input_image_path in input_image_paths:
    input_image = load_image(input_image_path)
    input_image.save(f"{save_root}/input_{counter}.png")
    images.append(input_image)
    counter += 1

qas = [[q1 + test_prompt, None]]

generated = inference_solver.generate(
    images=[],
    qas=qas,
    clip_image=images,
    max_gen_len=8192,
    temperature=1.0,
    logits_processor=inference_solver.create_logits_processor(cfg=4.0, image_top_k=2000),
)

a1 = generated[0]
new_image = generated[1][0]

new_image.save(f"{save_root}/generated_image.png")
