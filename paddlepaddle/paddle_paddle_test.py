import glob
import json
import os

import cv2
import numpy as np
import pdf2image
from paddleocr import PPStructure, draw_structure_result, save_structure_res
from PIL import Image


def convert_pdf_to_images(pdf_path):
    """Convert PDF to a list of PIL Images."""
    try:
        # Convert PDF to list of images
        images = pdf2image.convert_from_path(pdf_path)
        return images
    except Exception as e:
        print(f"Error converting PDF: {e}")
        return None


def save_results(
    result, image, base_filename, json_folder, text_folder, image_folder, font_path
):
    """Save processing results in multiple formats."""
    # Save structured results
    save_structure_res(result, json_folder, base_filename)

    # Save JSON with indentation
    json_path = os.path.join(json_folder, f"{base_filename}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        # Create a copy of results without image data for JSON
        json_result = []
        for item in result:
            item_copy = item.copy()
            if "img" in item_copy:
                item_copy.pop("img")
            json_result.append(item_copy)
        json.dump(json_result, f, indent=4, ensure_ascii=False)

    # Save text output
    text_path = os.path.join(text_folder, f"{base_filename}.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        for item in result:
            f.write(item.get("text", "") + "\n")

    # Save visualization
    im_show = draw_structure_result(image, result, font_path=font_path)
    im_show = Image.fromarray(im_show)
    im_show.save(os.path.join(image_folder, f"result_{base_filename}.jpg"))


def process_document(
    file_path, table_engine, json_folder, text_folder, image_folder, font_path
):
    """Process either PDF or image file."""
    file_extension = os.path.splitext(file_path)[1].lower()
    base_filename = os.path.basename(file_path).split(".")[0]

    if file_extension == ".pdf":
        print("Processing PDF file...")
        images = convert_pdf_to_images(file_path)
        if images is None:
            return

        all_results = []
        for i, image in enumerate(images):
            # Convert PIL Image to cv2 format
            cv2_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            result = table_engine(cv2_image)

            # Save results
            page_filename = f"{base_filename}_page_{i+1}"
            save_results(
                result,
                image,
                page_filename,
                json_folder,
                text_folder,
                image_folder,
                font_path,
            )

            all_results.extend(result)

        return all_results
    else:
        print("Processing image file...")
        # Handle image file
        img = cv2.imread(file_path)
        if img is None:
            print(f"Error: Could not read image file {file_path}")
            return None

        result = table_engine(img)
        save_results(
            result,
            Image.open(file_path).convert("RGB"),
            base_filename,
            json_folder,
            text_folder,
            image_folder,
            font_path,
        )

        return result


def main():
    # Initialize PPStructure
    table_engine = PPStructure(
        use_gpu=True,
        max_batch_size=10,
        gpu_mem=1000,
        cpu_threads=1,
        lang="en",
        image_orientation_predictor_kwargs={"use_gpu": False},
        layout=True,
        table=True,
        ocr=True,
    )

    # Setup paths
    input_folder = "data/pdf_resumes/CV_Images_EN"
    json_folder = "paddlepaddle/output/json"
    text_folder = "paddlepaddle/output/text"
    image_folder = "paddlepaddle/output/result_image"
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

    # Create output directories if they don't exist
    os.makedirs(json_folder, exist_ok=True)
    os.makedirs(text_folder, exist_ok=True)
    os.makedirs(image_folder, exist_ok=True)

    # Process each image in the input folder
    for file_path in glob.glob(os.path.join(input_folder, "*.*")):
        results = process_document(
            file_path, table_engine, json_folder, text_folder, image_folder, font_path
        )
        if results:
            print(
                f"Processing of {os.path.basename(file_path)} completed successfully!"
            )
        else:
            print(f"Processing of {os.path.basename(file_path)} failed!")


if __name__ == "__main__":
    main()
