from inference_solver import FlexARInferenceSolver
from PIL import Image
import os
from transformers.image_utils import load_image
import argparse


def main(args):
    inference_solver = FlexARInferenceSolver(
        model_path="Alpha-VLLM/Lumina-mGPT-7B-768",
        precision="fp32",
        lora_params_path=args.params_path,
        target_size=768,
    )

    if args.seg is None:
        input_image_paths = [args.style]
    else:
        input_image_paths = [args.style, args.seg]

    save_root = args.save_path

    os.makedirs(save_root, exist_ok=True)

    images = []
    counter = 0
    for input_image_path in input_image_paths:
        input_image = load_image(input_image_path)
        input_image.save(f"{save_root}/input_{counter}.png")
        images.append(input_image)
        counter += 1

    generated = inference_solver.generate(
        images=[],
        test_prompt=args.prompt,
        clip_image=images,
        noise_strength=args.noise_strength,
        max_gen_len=8192,
        temperature=1.0,
        logits_processor=inference_solver.create_logits_processor(cfg=4.0, image_top_k=2000),
    )

    a1 = generated[0]
    new_image = generated[1][0]

    new_image.save(f"{save_root}/generated_image.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params_path", type=str)
    parser.add_argument("--style", type=str)
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--noise_strength", type=float, default=0.1)
    parser.add_argument("--seg", type=str, default=None)
    parser.add_argument("--save_path", type=str, default="output")
    args = parser.parse_args()
    main(args)
