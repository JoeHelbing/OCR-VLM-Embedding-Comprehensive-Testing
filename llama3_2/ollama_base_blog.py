from typing import List, Literal, Optional

from ollama import chat
from pydantic import BaseModel


class Object(BaseModel):
    name: str
    confidence: float
    attributes: str


class ImageDescription(BaseModel):
    summary: str
    objects: List[Object]
    scene: str
    colors: List[str]
    time_of_day: Literal["Morning", "Afternoon", "Evening", "Night"]
    setting: Literal["Indoor", "Outdoor", "Unknown"]
    text_content: Optional[str] = None


path = "/home/joe/ttop/nlp_proj/llama3_2/image.png"

response = chat(
    model="llama3.2-vision:latest",
    format=ImageDescription.model_json_schema(),  # Pass in the schema for the response
    messages=[
        {
            "role": "user",
            "content": "Analyze this image and describe what you see, including any objects, the scene, colors and any text you can detect.",
            "images": [path],
        },
    ],
    options={"temperature": 0},  # Set temperature to 0 for more deterministic output
)

image_description = ImageDescription.model_validate_json(response.message.content)
print(image_description)
