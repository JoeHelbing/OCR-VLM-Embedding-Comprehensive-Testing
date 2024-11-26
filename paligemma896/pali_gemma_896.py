import os

import torch
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

if __name__ == "__main__":
    os.chdir("..")

    image_dir = "pdf_resumes/CV_Images"
    model_id = "google/paligemma-3b-pt-896"
    device = "cuda:0"
    dtype = torch.bfloat16

    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device,
        revision="bfloat16",
    ).eval()
    processor = AutoProcessor.from_pretrained(model_id)

    # Update prompt format with special tokens
    prompt = "<image> <bos> Give a YAML representation of the text in the image"

    image_files = [
        f for f in os.listdir(image_dir) if f.endswith(".png") or f.endswith(".jpg")
    ]

    for image_file in image_files:
        image_location = os.path.join(image_dir, image_file)
        image = Image.open(image_location)

        model_inputs = processor(text=prompt, images=image, return_tensors="pt")
        model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}
        model_inputs["input_ids"] = model_inputs["input_ids"].to(torch.long)
        model_inputs = {
            k: v.to(dtype=dtype) if k != "input_ids" else v
            for k, v in model_inputs.items()
        }
        input_len = model_inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = model.generate(
                **model_inputs, max_new_tokens=100, do_sample=False
            )
            generation = generation[0][input_len:]
            decoded = processor.decode(generation, skip_special_tokens=True)
            print(f"Result for {image_file}: {decoded}")
