import logging
import os
from enum import Enum
from typing import List, Optional

from ollama import chat
from pydantic import BaseModel, EmailStr, Field, HttpUrl

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ProficiencyLevel(str, Enum):
    NATIVE = "Native"
    FLUENT = "Fluent"
    ADVANCED = "Advanced"
    INTERMEDIATE = "Intermediate"
    BASIC = "Basic"


class ContactInfo(BaseModel):
    full_name: Optional[str] = Field(
        None,
        description="The complete name of the person, including first name, middle name (if any), and last name",
    )
    emails: Optional[List[EmailStr]] = Field(
        None,
        description="List of email addresses found in the resume, typically professional emails",
    )
    phones: Optional[List[str]] = Field(
        None,
        description="List of phone numbers in any format (e.g., +1-123-456-7890, (123) 456-7890)",
    )
    locations: Optional[List[str]] = Field(
        None,
        description="Physical locations mentioned, including city, state, country, or full addresses",
    )
    linkedin: Optional[HttpUrl] = Field(
        None,
        description="LinkedIn profile URL, typically in format: linkedin.com/in/username",
    )
    github: Optional[HttpUrl] = Field(
        None, description="GitHub profile URL, typically in format: github.com/username"
    )
    portfolio: Optional[HttpUrl] = Field(
        None, description="Personal website or portfolio URL"
    )


class EducationItem(BaseModel):
    institution: Optional[str] = Field(
        None,
        description="Name of the educational institution (university, college, etc.)",
    )
    degree: Optional[str] = Field(
        None,
        description="Type of degree (e.g., Bachelor of Science, Master of Arts, Ph.D.)",
    )
    field_of_study: Optional[str] = Field(
        None,
        description="Major or field of study (e.g., Computer Science, Business Administration)",
    )
    location: Optional[str] = Field(
        None, description="Location of the institution (city, state, country)"
    )
    start_date: Optional[str] = Field(
        None, description="Start date of education in any format (MM/YYYY, YYYY, etc.)"
    )
    end_date: Optional[str] = Field(
        None, description="End date of education or 'Present' if ongoing"
    )
    gpa: Optional[str] = Field(
        None, description="GPA or academic achievements, if mentioned"
    )


class ExperienceItem(BaseModel):
    job_title: Optional[str] = Field(
        None, description="Official job title or role name"
    )
    company: Optional[str] = Field(None, description="Company or organization name")
    location: Optional[str] = Field(
        None, description="Job location (city, state, country, or 'Remote')"
    )
    start_date: Optional[str] = Field(
        None, description="Start date in any format (MM/YYYY, YYYY, etc.)"
    )
    end_date: Optional[str] = Field(
        None, description="End date or 'Present' if current position"
    )
    description: Optional[List[str]] = Field(
        None,
        description="List of bullet points describing responsibilities and achievements",
    )
    technologies: Optional[List[str]] = Field(
        None,
        description="List of technologies, tools, or skills specifically mentioned in this role",
    )


class ProjectItem(BaseModel):
    name: Optional[str] = Field(None, description="Project title or name")
    description: Optional[List[str]] = Field(
        None,
        description="List of bullet points describing the project, its goals, and outcomes",
    )
    link: Optional[HttpUrl] = Field(
        None, description="Project URL (GitHub, live demo, etc.)"
    )
    start_date: Optional[str] = Field(
        None, description="Project start date in any format"
    )
    end_date: Optional[str] = Field(
        None, description="Project end date or 'Present' if ongoing"
    )
    technologies: Optional[List[str]] = Field(
        None,
        description="List of technologies, frameworks, or tools used in the project",
    )


class LanguageItem(BaseModel):
    name: Optional[str] = Field(None, description="Name of the language")
    proficiency: Optional[ProficiencyLevel] = Field(
        None, description="Proficiency level in the language"
    )


class Resume(BaseModel):
    contact: Optional[ContactInfo] = Field(
        None, description="Personal and contact information from the resume header"
    )
    summary: Optional[str] = Field(
        None,
        description="Professional summary or objective statement at the beginning of the resume",
    )
    education: Optional[List[EducationItem]] = Field(
        None, description="Educational background and qualifications"
    )
    experience: Optional[List[ExperienceItem]] = Field(
        None, description="Professional work experience, including internships"
    )
    projects: Optional[List[ProjectItem]] = Field(
        None, description="Personal or professional projects"
    )
    skills: Optional[List[str]] = Field(
        None, description="Technical skills, soft skills, and competencies"
    )
    languages: Optional[List[LanguageItem]] = Field(
        None, description="Language proficiencies"
    )
    certifications: Optional[List[str]] = Field(
        None, description="Professional certifications and licenses"
    )
    achievements: Optional[List[str]] = Field(
        None, description="Notable achievements, awards, or recognition"
    )


def get_images_in_directory(directory: str) -> List[str]:
    """
    Get a list of image file paths in the specified directory.
    """
    logging.debug(f"Getting images in directory: {directory}")
    image_paths = []
    for file in os.listdir(directory):
        if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
            image_paths.append(os.path.join(directory, file))
    logging.debug(f"Found {len(image_paths)} images")
    return image_paths


def extract_markdown_from_image(image_path: str) -> str:
    """
    Use Ollama to extract text from an image as raw Markdown (no commentary).
    """
    logging.debug(f"Extracting markdown from image: {image_path}")
    prompt = (
        "You are a highly accurate OCR tool. "
        "Please extract all text from this image of a resume and format it as Markdown. "
        "Return ONLY the Markdown text, with no extra commentary."
    )
    response = chat(
        model="llama3.2-vision",  # Adjust to your actual model
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [image_path],
            },
        ],
        options={"temperature": 0.0},  # For best possible accuracy/determinism
    )
    md_text = response.message.content
    logging.debug(f"Extracted markdown: {md_text[:100]}...")  # Log first 100 characters
    return md_text


def structured_response_from_image(image_path: str) -> Resume:
    logging.debug(f"Extracting structured response from image: {image_path}")
    parsing_prompt = """
    Analyze this resume image and extract information according to these guidelines:

    1. Look for contact information typically at the top of the resume
    2. Identify distinct sections (Education, Experience, etc.)
    3. Extract dates in their original format
    4. Preserve bullet points in experience and project descriptions as separate list items
    5. Look for skills mentioned throughout the document, not just in a skills section
    6. Extract any technologies mentioned in experience or project descriptions

    Return the information in valid JSON matching the provided schema structure.
    """
    response = chat(
        model="llama3.2-vision",
        format=Resume.model_json_schema(),  # Pass in the schema for the response
        messages=[
            {
                "role": "user",
                "content": parsing_prompt,
                "images": [image_path],
            },
        ],
        options={
            "temperature": 0
        },  # Set temperature to 0 for more deterministic output
    )
    try:
        image_description = Resume.model_validate_json(response.message.content)
        logging.debug(f"Extracted structured response: {image_description}")
    except Exception as e:
        logging.error(f"Error parsing structured response: {e}")
        image_description = Resume()
    return image_description


def save_markdown(md_text: str, output_dir: str, image_path: str):
    logging.debug(f"Saving markdown for image: {image_path}")
    stem = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{stem}.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    logging.debug(f"Saved markdown to: {output_path}")


def save_json(resume: Resume, output_dir: str, image_path: str):
    logging.debug(f"Saving JSON for image: {image_path}")
    stem = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{stem}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(resume.model_dump_json(indent=4))
    logging.debug(f"Saved JSON to: {output_path}")


if __name__ == "__main__":
    # Get images
    IMAGE_PATH = "/home/joe/ttop/nlp_proj/data/pdf_resumes/CV_Images_EN"
    EXTRACTIONS_PATH = "extractions_v3"
    os.makedirs(EXTRACTIONS_PATH, exist_ok=True)
    logging.info(f"Starting processing for images in: {IMAGE_PATH}")
    image_paths = get_images_in_directory(IMAGE_PATH)
    for image_path in image_paths:
        stem = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = os.path.join(EXTRACTIONS_PATH, stem)
        os.makedirs(output_dir, exist_ok=True)

        md_result = extract_markdown_from_image(image_path)
        save_markdown(md_result, output_dir, image_path)

        img_result = structured_response_from_image(image_path)
        save_json(img_result, output_dir, image_path)

    logging.info("First pass complete. Saved extracted Markdown to resume_raw.md")
