import os

from pdf2image import convert_from_path

pdf_dir = "pdf_resumes/CV_Brut"
output_dir = "pdf_resumes/CV_Images"


def convert_pdfs_to_images(pdf_dir, output_dir, dpi=200):
    """
    Convert all PDFs in a directory to images
    Args:
        pdf_dir (str): Directory containing PDFs
        output_dir (str): Directory to save converted images
        dpi (int): DPI for conversion quality
    """
    os.makedirs(output_dir, exist_ok=True)

    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        try:
            images = convert_from_path(pdf_path, dpi=dpi)

            for i, image in enumerate(images):
                output_name = f"{os.path.splitext(pdf_file)[0]}_page_{i+1}.png"
                output_path = os.path.join(output_dir, output_name)

                # Save as PNG
                image.save(output_path, "PNG")
                print(f"Converted {pdf_file} page {i+1} to {output_name}")

        except Exception as e:
            print(f"Error converting {pdf_file}: {str(e)}")


if __name__ == "__main__":
    convert_pdfs_to_images(pdf_dir, output_dir)
