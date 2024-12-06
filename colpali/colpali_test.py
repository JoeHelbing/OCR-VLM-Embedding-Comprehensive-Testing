# Get the parent directory of the current file
import sys
import tkinter as tk
from io import BytesIO
from pathlib import Path
from typing import Any, List, cast

import requests
import torch
from colpali_engine.models import ColQwen2, ColQwen2Processor
from colpali_engine.utils.torch_utils import get_torch_device
from peft import LoraConfig
from PIL import Image, ImageTk
from transformers.models.qwen2_vl import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
)

# Get the parent directory of the current file and add to Python path
current_file = Path(__file__).resolve()
parent_dir = current_file.parent.parent
sys.path.insert(0, str(parent_dir))

from helpers import get_image_files


def show_images(images, filepaths=None):
    root = tk.Tk()
    root.title("Image Viewer")
    current_idx = [0]  # Using list to make it mutable in nested functions

    # Create and configure frame
    frame = tk.Frame(root)
    frame.pack(padx=10, pady=10)

    # Initialize label for image
    label = tk.Label(frame)
    label.pack()

    def update_image():
        scaled_img = scale_image(images[current_idx[0]], new_height=2000)
        photo = ImageTk.PhotoImage(scaled_img)
        label.configure(image=photo)
        label.image = photo  # Keep reference
        # Update counter display with filename if available
        display_text = f"Image {current_idx[0] + 1} of {len(images)}"
        if filepaths:
            filename = Path(filepaths[current_idx[0]]).name
            display_text += f"\n{filename}"
        counter_label.config(text=display_text)

    def next_image():
        if current_idx[0] < len(images) - 1:
            current_idx[0] += 1
            update_image()

    def prev_image():
        if current_idx[0] > 0:
            current_idx[0] -= 1
            update_image()

    # Button frame
    button_frame = tk.Frame(root)
    button_frame.pack(pady=5)

    # Previous button
    prev_button = tk.Button(button_frame, text="Previous", command=prev_image)
    prev_button.pack(side=tk.LEFT, padx=5)

    # Counter label
    counter_label = tk.Label(button_frame, text="")
    counter_label.pack(side=tk.LEFT, padx=10)

    # Next button
    next_button = tk.Button(button_frame, text="Next", command=next_image)
    next_button.pack(side=tk.LEFT, padx=5)

    # Display first image
    update_image()

    # Key bindings
    root.bind("<Left>", lambda e: prev_image())
    root.bind("<Right>", lambda e: next_image())

    root.mainloop()


def load_image_from_url(url: str) -> Image.Image:
    """
    Load a PIL image from a valid URL.
    """
    response = requests.get(url)
    return Image.open(BytesIO(response.content))


def scale_image(image: Image.Image, new_height: int = 2000) -> Image.Image:
    """
    Scale an image to a new height while maintaining the aspect ratio.
    """
    width, height = image.size
    aspect_ratio = width / height
    new_width = int(new_height * aspect_ratio)

    scaled_image = image.resize((new_width, new_height))

    return scaled_image


def scale_image_to_a4(image: Image.Image, max_dimension: int = 1024) -> Image.Image:
    """
    Scale an image to match A4 proportions while fitting within max_dimension.
    A4 ratio is 1:√2 (approximately 1:1.4142)
    """
    A4_RATIO = 1 / 1.4142

    # Calculate new dimensions maintaining A4 ratio
    if max_dimension * A4_RATIO <= max_dimension:
        new_width = int(max_dimension * A4_RATIO)
        new_height = max_dimension
    else:
        new_width = max_dimension
        new_height = int(max_dimension / A4_RATIO)

    scaled_image = image.resize((new_width, new_height))
    return scaled_image


class ColQwen2ForRAG(ColQwen2):
    """
    ColQwen2 model implementation that can be used both for retrieval and generation.
    Allows switching between retrieval and generation modes.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_retrieval_enabled = True

    def forward(self, *args, **kwargs) -> Any:
        """
        Forward pass that calls either Qwen2VLForConditionalGeneration.forward for generation
        or ColQwen2.forward for retrieval based on the current mode.
        """
        if self.is_retrieval_enabled:
            return ColQwen2.forward(self, *args, **kwargs)
        else:
            return Qwen2VLForConditionalGeneration.forward(self, *args, **kwargs)

    def generate(self, *args, **kwargs):
        """
        Generate text using Qwen2VLForConditionalGeneration.generate.
        """
        if not self.is_generation_enabled:
            raise ValueError(
                "Set the model to generation mode by calling `enable_generation()` before calling `generate()`."
            )
        return super().generate(*args, **kwargs)

    @property
    def is_retrieval_enabled(self) -> bool:
        return self._is_retrieval_enabled

    @property
    def is_generation_enabled(self) -> bool:
        return not self.is_retrieval_enabled

    def enable_retrieval(self) -> None:
        """
        Switch to retrieval mode.
        """
        self.enable_adapters()
        self._is_retrieval_enabled = True

    def enable_generation(self) -> None:
        """
        Switch to generation mode.
        """
        self.disable_adapters()
        self._is_retrieval_enabled = False


model_name = "vidore/colqwen2-v1.0"
device = get_torch_device("auto")

print(f"Using device: {device}")

# Get the LoRA config from the pretrained retrieval model
lora_config = LoraConfig.from_pretrained(model_name)

# Load the processors
processor_retrieval = cast(
    ColQwen2Processor, ColQwen2Processor.from_pretrained(model_name)
)
processor_generation = cast(
    Qwen2VLProcessor,
    Qwen2VLProcessor.from_pretrained(lora_config.base_model_name_or_path),
)

# Load the model with the loaded pre-trained adapter for retrieval
model = cast(
    ColQwen2ForRAG,
    ColQwen2ForRAG.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    ),
)

# Load the image and query
images_filepaths = get_image_files()
# load the images from the filepaths and convert to PIL images
images: List[Image.Image] = [Image.open(image) for image in images_filepaths]
# queries = [
#     "Which candidate has skills in Python, Java, and C++?",  # changed queries for resume search
#     "Who is applying for an engineering internship?",
# ]

# # Cut list to first two images
# images = images[:5]

query = "Which candidate has a background in Manufacturing?"

# NOTE: Because ColQWen2 uses dynamic resolution, we will scale down the images to prevent VRAM overload and faster
# inference times for both indexing and generation. From my experiments, a scale of 512 pixels is a good default for
# document tasks. Feel free to experiment with higher resolutions, especially if the text on your document is small.
images = [scale_image_to_a4(image, max_dimension=800) for image in images]

for image in images:
    print(image.size)


# Process the inputs in batches of 2
batch_size = 2
num_batches = (len(images) + batch_size - 1) // batch_size
all_image_embeddings = []

# Process queries seperately
batch_queries = processor_retrieval.process_queries([query]).to(model.device)
query_embeddings = model.forward(**batch_queries)

for i in range(num_batches):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, len(images))
    batch_images = processor_retrieval.process_images(images[start_idx:end_idx]).to(
        model.device
    )

    # Forward pass
    model.enable_retrieval()

    with torch.no_grad():
        # Process batch and resize output
        image_embeddings = model.forward(**batch_images)

        print(image_embeddings.shape)
        all_image_embeddings.append(image_embeddings)

# Concatenate the embeddings
image_embeddings = torch.cat(all_image_embeddings, dim=0)
print(image_embeddings.shape)

# Calculate similarity scores
scores = processor_retrieval.score_multi_vector(query_embeddings, image_embeddings)
print(scores)
print(len(scores))

# Get the top-1 page image
retrieved_image_index = scores.argmax().item()
print(retrieved_image_index)
retrieved_image = images[retrieved_image_index]

# Clear the cache
torch.cuda.empty_cache()

# Preprocess the inputs
conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
            },
            {
                "type": "text",
                "text": f"Answer the following question using the input image: {query}",
            },
        ],
    }
]
text_prompt = processor_generation.apply_chat_template(
    conversation, add_generation_prompt=True
)
inputs_generation = processor_generation(
    text=[text_prompt],
    images=[image],
    padding=True,
    return_tensors="pt",
).to(device)

# Generate the RAG response
model.enable_generation()
output_ids = model.generate(**inputs_generation, max_new_tokens=128)

# Ensure that only the newly generated token IDs are retained from output_ids
generated_ids = [
    output_ids[len(input_ids) :]
    for input_ids, output_ids in zip(inputs_generation.input_ids, output_ids)
]

# Decode the RAG response
output_text = processor_generation.batch_decode(
    generated_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=True,
)
print("RAG response:")
print(output_text)
retrieved_image = [retrieved_image]
retrieved_filepath = [images_filepaths[retrieved_image_index]]
print(f"Image retrieved for the following query: `{query}`")
show_images(retrieved_image, retrieved_filepath)
