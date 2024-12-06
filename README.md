## System
- 3090 24GB VRAM
- AMD 5600X
- 32GB System RAM
- WSL2 -- Ubuntu 24.04 LTS
- CUDA 12.6

Must authenticate Huggingface token for many of these.

## Datasets

### Resume Dataset
https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset

Converted to 200 DPI pngs, then downselected for english language resumes via 
Paligemma detecting langauge. Verified 10.

WSL poppler setup for pdf2image:
```bash
sudo apt-get install -y poppler-utils
export PATH="/usr/bin:$PATH"
```

## OCR Solutions

### PaddleOCR

#### License
Apache 2.0

#### Install and running the code
https://github.com/PaddlePaddle/PaddleOCR

Additional resources:
- https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/pipeline_usage/tutorials/information_extraction_pipelines/document_scene_information_extraction_en.md
- https://paddlepaddle.github.io/PaddleOCR/latest/en/ppocr/quick_start.html

Setup notes:
- Using CUDA 12.6 requires specific parameters
- image_orientation must be on CPU

CUDA Setup:
https://developer.nvidia.com/cudnn

```bash
wget https://developer.download.nvidia.com/compute/cudnn/9.5.1/local_installers/cudnn-local-repo-ubuntu2404-9.5.1_1.0-1_amd64.deb
sudo dpkg -i cudnn-local-repo-ubuntu2404-9.5.1_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2404-9.5.1/cudnn-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cudnn
```

WSL poppler setup for pdf2image:
```bash
sudo apt-get install -y poppler-utils
export PATH="/usr/bin:$PATH"
```

PaddlePaddle parameter setup:
```python
table_engine = PPStructure(
    use_gpu=True,
    max_batch_size=10,
    gpu_mem=1000,
    cpu_threads=1,
    lang="en",
    image_orientation_predictor_kwargs={
        'use_gpu': False
    },
    layout=True,
    table=True,
    ocr=True,
)
```

#### Thoughts
- Pretty good, not perfect
- Good structured output similar to Amazon Textract, though not as advanced or adaptable
- Misses figures (pictures)
- Able to do bounding boxes with location information
- A bit hit or miss on some documents, sometimes it fails to pull anything relevant from a document at all
### GOT-OCR2
https://github.com/Ucas-HaoranWei/GOT-OCR2.0
[arxiv paper](https://arxiv.org/pdf/2409.01704)
#### License
#### Source pages
#### Tested model
#### Install and running the code
#### Example Outputs
#### Thoughts

### Qwen-2-VL
https://github.com/QwenLM/Qwen2-VL

#### License
#### Source pages
#### Tested model
#### Install and running the code
#### Example Outputs
#### Thoughts

### TrOCR
https://arxiv.org/abs/2109.10282

#### License
#### Source pages
#### Tested model
#### Install and running the code
#### Example Outputs
#### Thoughts

### EasyOCR
https://github.com/JaidedAI/EasyOCR

#### License
#### Source pages
#### Tested model
#### Install and running the code
#### Example Outputs
#### Thoughts

### Pixtral
https://huggingface.co/mistralai/Pixtral-12B-2409

#### License
#### Source pages
#### Tested model
#### Install and running the code
#### Example Outputs
#### Thoughts

### Mini-cpm
https://github.com/OpenBMB/MiniCPM-V

#### License
#### Source pages
#### Tested model
#### Install and running the code
#### Example Outputs
#### Thoughts

### Florence
https://huggingface.co/microsoft/Florence-2-large-ft

#### License
#### Source pages
#### Tested model
#### Install and running the code
#### Example Outputs
#### Thoughts

### Surya
https://github.com/VikParuchuri/surya

#### License
#### Source pages
#### Tested model
#### Install and running the code
#### Example Outputs
#### Thoughts

### DocTR
https://github.com/mindee/doctr

#### License
#### Source pages
#### Tested model
#### Install and running the code
#### Example Outputs
#### Thoughts

### Docling
https://github.com/DS4SD/docling

#### License
#### Source pages
#### Tested model
#### Install and running the code
#### Example Outputs
#### Thoughts

### Tesseract
https://github.com/tesseract-ocr/tesseract

#### License
#### Source pages
#### Tested model
#### Install and running the code
#### Example Outputs
#### Thoughts

### Molmo

#### License
#### Source pages
#### Tested model
#### Install and running the code
#### Example Outputs
#### Thoughts

### Unstructured
https://docs.unstructured.io/welcome

#### License
#### Source pages
#### Tested model
#### Install and running the code
#### Example Outputs
#### Thoughts

### Yolo8m (tables extraction)
Turn pages into OCR and .csv strings, then LLM pass into table structure
https://huggingface.co/keremberke/yolov8m-table-extraction

#### License
#### Source pages
#### Tested model
#### Install and running the code
#### Example Outputs
#### Thoughts

### GPT-4o

#### License
#### Source pages
#### Tested model
#### Install and running the code
#### Example Outputs
#### Thoughts

### Claude

#### License
#### Source pages
#### Tested model
#### Install and running the code
#### Example Outputs
#### Thoughts

### Gemini

#### License
#### Source pages
#### Tested model
#### Install and running the code
#### Example Outputs
#### Thoughts

### Nougat
https://github.com/facebookresearch/nougat

#### License
#### Source pages
#### Tested model
#### Install and running the code
#### Example Outputs
#### Thoughts

### PDF Extract Kit
https://github.com/opendatalab/PDF-Extract-Kit

#### License
#### Source pages
#### Tested model
#### Install and running the code
#### Example Outputs
#### Thoughts

### gmft
https://github.com/conjuncts/gmft
https://github.com/conjuncts/gmft/blob/main/notebooks/bulk_extract.ipynb

#### License
#### Source pages
#### Tested model
#### Install and running the code
#### Example Outputs
#### Thoughts

## Embeddings
### ColPali
ColPali model - vision retriever based on the ColBERT architecture and the PaliGemma model.

https://github.com/illuin-tech/colpali
https://arxiv.org/abs/2407.01449
[vidore (ILLUIN Vidore)](https://huggingface.co/vidore)

- Multiple variations, originally paligemma, then Qwen2-VL-2B-Instruct.
- Lots of resources for fine tuning / retraining

#### License
Depends on base model, Qwen2 is Apache 2.0 though
#### Source pages
#### Tested model
#### Install and running the code
#### Example Outputs
#### Thoughts
## VLMs as OCR or Structured Extraction
### PaliGemma

#### License
https://ai.google.dev/gemma/terms
No real limitations in use for regular use cases including commercial.

#### Source pages
- https://arxiv.org/abs/2407.07726
- https://huggingface.co/docs/transformers/main/en/model_doc/paligemma

**3 sizes**
- https://huggingface.co/google/paligemma-3b-pt-224
- https://huggingface.co/google/paligemma-3b-pt-448
- https://huggingface.co/google/paligemma-3b-pt-896

#### Tested model
Limitations:
- Transformers PaliGemma 3B weights
- Pre-trained with 896*896 input images
- 512 token input/output text sequences
- Available in float32, bfloat16 and float16 formats for fine-tuning

#### Install and running the code
Important note on prompt: Need to structure the prompt with the <image> tag for number of images passed (i.e. 1 <image> tag for a single image) followed by <bos> for the text instructions.

```python
prompt = "<image> <bos> Give a YAML representation of the text in the image"
```

Full implementation:
```python
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
```

#### Example Outputs
```text
Result for CV_Oumaima_Dahib_page_1.png: master's student in consulting and change with a strong focus on developing strategic consulting skills and change management capabilities . prepared to drive effective change and deliver sustainable results in dynamic business environments

professional experience

inditex - zara

sales assistante

april 2023 march 2024

customer - focused service : provided excellent customer service to build strong relationships and ensure positive shopping experiences .

visual merchandising : utilized visual merchandising techniques to enhance store aesthetics and drive sales . inventory management + business

Result for CV_Valentin_Gerard_page_1.png: example of a cv

Result for CV_Melvil_Delahaye_page_1.png: exemple de cv
```

#### Thoughts
- Really poor as an OCR or structured extraction system
- Likely good as a custom trainable solution with labeled data
- Very limited context window of 512 tokens (input and output)
- Out of the box performance is not worth much

### LLaVA-CoT
[PKU-YuanGroup/LLaVA-CoT: LLaVA-CoT, a visual language model capable of spontaneous, systematic reasoning](https://github.com/PKU-YuanGroup/LLaVA-CoT)

#### License
#### Source pages
#### Tested model
#### Install and running the code
#### Example Outputs
#### Thoughts

### SmolVLM
[smollm/finetuning/Smol_VLM_FT.ipynb at main · huggingface/smollm](https://github.com/huggingface/smollm/blob/main/finetuning/Smol_VLM_FT.ipynb)
[HuggingFaceTB/SmolVLM-Instruct · Hugging Face](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct)
#### License
#### Source pages
#### Tested model
#### Install and running the code
#### Example Outputs
#### Thoughts
