import os
import json
from pathlib import Path
import PyPDF2
import ollama
import argparse
from typing import Dict, Any, List
import logging

class PDFProcessor:
    def __init__(self, model_name: str = "qwen2", host: str = "http://localhost:11434"):
        """
        Initialize the PDF processor
        
        Args:
            model_name (str): Name of the Ollama model to use
            host (str): Ollama host address
        """
        self.model_name = model_name
        ollama.set_host(host)
        self.setup_logging()
    
    def setup_logging(self):
        """Configure logging settings"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('pdf_processing.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text content from a PDF file
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            str: Extracted text from the PDF
        """
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            self.logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
            return ""

    def get_json_from_ollama(self, text: str) -> Dict[str, Any]:
        """
        Send text to Ollama and get JSON response
        
        Args:
            text (str): Text to process
            
        Returns:
            dict: JSON response from Ollama
        """
        prompt = """
        Please analyze the following text and create a JSON object with these fields:
        - title: The main title or subject of the document
        - summary: A brief summary of the main points (max 200 words)
        - key_points: List of key points (max 5)
        - entities: List of important named entities mentioned
        - document_type: Type of document (report, article, etc.)
        
        Text to analyze:
        {text}
        
        Return only valid JSON without any additional text or explanations.
        """
        
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt.format(text=text),
                format="json"
            )
            return json.loads(response['response'])
        except Exception as e:
            self.logger.error(f"Error getting response from Ollama: {str(e)}")
            return {
                "error": str(e),
                "title": "Error processing document",
                "summary": "Failed to process document",
                "key_points": [],
                "entities": [],
                "document_type": "unknown"
            }

    def process_directory(self, input_dir: str, output_dir: str) -> None:
        """
        Process all PDFs in a directory and save JSON outputs
        
        Args:
            input_dir (str): Directory containing PDF files
            output_dir (str): Directory to save JSON outputs
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        pdf_files = list(input_path.glob("*.pdf"))
        self.logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        for pdf_file in pdf_files:
            try:
                self.logger.info(f"Processing {pdf_file}")
                text = self.extract_text_from_pdf(str(pdf_file))
                if not text:
                    continue
                    
                json_output = self.get_json_from_ollama(text)
                
                output_file = output_path / f"{pdf_file.stem}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(json_output, f, indent=2, ensure_ascii=False)
                
                self.logger.info(f"Successfully processed {pdf_file}")
                
            except Exception as e:
                self.logger.error(f"Error processing {pdf_file}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Process PDFs using Ollama and generate JSON outputs")
    parser.add_argument("--input", "-i", required=True, help="Input directory containing PDFs")
    parser.add_argument("--output", "-o", required=True, help="Output directory for JSON files")
    parser.add_argument("--model", "-m", default="llama2", help="Ollama model name (default: llama2)")
    parser.add_argument("--host", default="http://localhost:11434", 
                        help="Ollama host address (default: http://localhost:11434)")
    
    args = parser.parse_args()
    
    processor = PDFProcessor(model_name=args.model, host=args.host)
    processor.process_directory(args.input, args.output)

if __name__ == "__main__":
    main()