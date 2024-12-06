import base64
from io import BytesIO
from pathlib import Path

import ollama
from PIL import Image


class ResumeImageParser:
    def __init__(self, model_name="llama3.2-vision"):
        self.model_name = model_name

    def load_image(self, image_path):
        """Load PNG image from path."""
        try:
            return Image.open(image_path)
        except Exception as e:
            print(f"Error loading image: {str(e)}")
            return None

    def encode_image_base64(self, image):
        """Convert PIL Image to base64 string."""
        try:
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
        except Exception as e:
            print(f"Error encoding image: {str(e)}")
            return None

    def parse_resume_with_vision(self, image_path):
        """Use Ollama vision model to extract information from resume image."""
        try:
            # Load the image
            image = self.load_image(image_path)
            if not image:
                return None

            # Convert the image to base64
            image_base64 = self.encode_image_base64(image)
            if not image_base64:
                return None

            # Prepare the prompt for resume information extraction
            prompt = """
            Analyze this resume image and extract the following information:
            - Full Name
            - Email
            - Phone
            - Education (including institution, degree, graduation date)
            - Work Experience (including company names, titles, dates, and key responsibilities)
            - Skills
            - Certifications (if any)
            
            Please provide the information in a structured format.
            """

            response = ollama.chat(
                model=self.model_name,
                format="json",
                messages=[
                    {"role": "user", "content": prompt, "images": [image_base64]}
                ],
            )

            return response["message"]["content"]
        except Exception as e:
            print(f"Error parsing resume with vision model: {str(e)}")
            return None

    def process_resume(self, image_path):
        """Process a single resume image and return JSON content."""
        # Parse the image using vision model
        llm_response = self.parse_resume_with_vision(image_path)
        if llm_response:
            return llm_response
        return None


def main():
    # Configure the parser
    parser = ResumeImageParser()

    # Set up input and output directories
    input_dir = Path("data/pdf_resumes/CV_Images_EN/")
    output_dir = Path("llama3_2/parsed_resumes/")
    output_dir.mkdir(exist_ok=True)

    # Process all PNG files in the input directory
    for image_file in input_dir.glob("*.png"):
        print(f"Processing {image_file.name}...")

        # Generate output file path
        output_file = output_dir / f"{image_file.stem}.json"

        # Process the resume
        json_content = parser.process_resume(image_file)

        # Save the JSON output
        if json_content:
            try:
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(json_content)
                print(f"Successfully processed {image_file.name}")
            except Exception as e:
                print(f"Error saving JSON for {image_file.name}: {str(e)}")
        else:
            print(f"Failed to process {image_file.name}")


if __name__ == "__main__":
    main()
