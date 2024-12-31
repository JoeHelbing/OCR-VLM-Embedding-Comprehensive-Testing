import json
import logging
import os
import shutil
from datetime import datetime
from typing import Any, Dict, List, Tuple

import fitz  # PyMuPDF for PDF processing
from ollama import chat
from pydantic import BaseModel, Field, validator

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


def cleanup_temp_folder(temp_dir: str) -> None:
    """
    Delete the temporary directory and its contents
    """
    try:
        shutil.rmtree(temp_dir)
        logging.debug(f"Deleted temporary directory: {temp_dir}")
    except Exception as e:
        logging.error(f"Failed to delete temporary directory {temp_dir}: {str(e)}")


def convert_pdf_pages_to_images(pdf_path: str) -> Tuple[List[str], str]:
    """
    Convert each page of a PDF to an image
    Returns a tuple of (list of image file paths, temp directory path)
    """
    logging.debug(f"Converting PDF to images: {pdf_path}")
    image_paths = []
    temp_dir = os.path.join(os.getcwd(), "image_conversion")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # Open PDF
        pdf_document = fitz.open(pdf_path)

        # Convert each page
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]

            # Convert page to image
            pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))  # 300 DPI

            # Save image
            image_path = os.path.join(temp_dir, f"page_{page_num + 1}.png")
            pix.save(image_path)
            image_paths.append(image_path)

            logging.debug(f"Converted page {page_num + 1} to {image_path}")

        return image_paths, temp_dir

    except Exception as e:
        logging.error(f"Failed to convert PDF: {str(e)}")
        cleanup_temp_folder(temp_dir)
        return [], ""

    finally:
        if "pdf_document" in locals():
            pdf_document.close()


class TableCell(BaseModel):
    """Represents a single cell in a table"""

    content: str = Field(default="", description="The text content of the cell")
    row_index: int = Field(description="Zero-based row index of the cell")
    col_index: int = Field(description="Zero-based column index of the cell")
    is_header: bool = Field(
        default=False, description="Whether this cell is part of the header"
    )
    spans_rows: int = Field(default=1, description="Number of rows this cell spans")
    spans_cols: int = Field(default=1, description="Number of columns this cell spans")


class Table(BaseModel):
    """Model representing a table structure"""

    num_rows: int = Field(default=0)
    num_cols: int = Field(default=0)
    page_number: int
    location: str = Field(default="")
    title: str = Field(default="")
    cells: List[TableCell] = Field(default_factory=list)

    @validator("num_rows", "num_cols", pre=True)
    def set_default_dimensions(cls, v):
        return v or 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any], page_number: int) -> "Table":
        """Create a Table instance from dictionary data"""
        return cls(
            num_rows=len(data.get("data", [])),
            num_cols=len(data.get("headers", [])),
            page_number=page_number,
            location=data.get("location", ""),
            title=data.get("title", ""),
            cells=[],  # Initialize empty, cells can be added later
        )

    def to_markdown(self) -> str:
        """Convert the table to markdown format"""
        if not self.cells:
            return ""

        # Initialize empty grid
        grid = [["" for _ in range(self.num_cols)] for _ in range(self.num_rows)]

        # Fill in the grid
        for cell in self.cells:
            grid[cell.row_index][cell.col_index] = cell.content

        # Generate markdown
        md = f"### {self.title}\n\n" if self.title else ""

        # Create header row
        md += "|" + "|".join(grid[0]) + "|\n"
        md += "|" + "|".join(["---" for _ in range(self.num_cols)]) + "|\n"

        # Create data rows
        for row in grid[1:]:
            md += "|" + "|".join(row) + "|\n"

        return md

    def to_dict(self) -> Dict[str, Any]:
        """Convert the table to a dictionary format"""
        grid = [["" for _ in range(self.num_cols)] for _ in range(self.num_rows)]
        headers = []

        # Fill in the grid and identify headers
        for cell in self.cells:
            grid[cell.row_index][cell.col_index] = cell.content
            if cell.is_header:
                headers.append(cell.content)

        # If no explicit headers were marked, use first row
        if not headers:
            headers = grid[0]
            data_rows = grid[1:]
        else:
            data_rows = grid

        # Convert to dictionary
        return {
            "title": self.title,
            "page_number": self.page_number,
            "location": self.location,
            "headers": headers,
            "data": data_rows,
        }


class TableDetectionResponse(BaseModel):
    """Response model for table detection"""

    tables_found: bool = Field(description="Whether any tables were found on the page")
    message: str = Field(
        description="Description of what was found or why no tables were detected"
    )
    table_count: int = Field(default=0, description="Number of tables found")
    tables: List[Table] = Field(
        default_factory=list, description="Details of tables if found"
    )


def extract_tables_from_page(
    image_path: str, page_number: int
) -> TableDetectionResponse:
    """Extract tables from a single page image using VLM"""
    logging.debug(f"Extracting tables from page {page_number}: {image_path}")

    # First pass: Detect tables and their structure
    structure_prompt = """
    Analyze this page for tables. 
    First determine if there are any clear, structured tables (not lists or text blocks).
    
    If NO tables are found, return:
    {
        "tables_found": false,
        "message": "Explain why no tables were detected (e.g., 'Page contains only paragraphs of text')",
        "table_count": 0,
        "tables": []
    }
    
    If tables ARE found, for each table:
    1. Identify the number of rows and columns
    2. Locate any table title or caption
    3. Determine if there are header rows
    4. Note the table's location on the page
    
    Then return:
    {
        "tables_found": true,
        "message": "Description of tables found",
        "table_count": number_of_tables,
        "tables": [array of Table objects matching schema]
    }
    """

    try:
        # Get table structure response from VLM
        structure_response = chat(
            model="llama3.2-vision:latest",
            format=TableDetectionResponse.model_json_schema(),
            messages=[
                {
                    "role": "user",
                    "content": structure_prompt,
                    "images": [image_path],
                }
            ],
            options={"temperature": 0},
        )

        # Parse the structure response
        tables_found = []
        raw_tables = Table.model_validate_json(structure_response.message.content)

        # Second pass: Extract content for each table
        for table_structure in raw_tables:
            content_prompt = f"""
            Focus on the table {table_structure.location} of the page.
            Extract the content of each cell in the table.
            Include spans for merged cells if present.
            Format as a list of TableCell objects matching the schema.
            """

            content_response = chat(
                model="llama3.2-vision:latest",
                format=Table.model_json_schema(),
                messages=[
                    {
                        "role": "user",
                        "content": content_prompt,
                        "images": [image_path],
                    }
                ],
                options={"temperature": 0},
            )

            # Create complete table with content
            table = Table(
                cells=content_response.cells,
                num_rows=table_structure.num_rows,
                num_cols=table_structure.num_cols,
                title=table_structure.title,
                page_number=page_number,
                location=table_structure.location,
            )
            tables_found.append(table)

        # When no tables found
        if not tables_found:
            return TableDetectionResponse(
                tables_found=False,
                message="No tables detected on page",
                table_count=0,
                tables=[],
            )

        # When tables are found, create proper Table objects
        detected_tables = []
        for table_data in raw_tables:
            table = Table.from_dict(table_data, page_number)
            detected_tables.append(table)

        return TableDetectionResponse(
            tables_found=True,
            message=f"Found {len(detected_tables)} tables",
            table_count=len(detected_tables),
            tables=detected_tables,
        )

    except Exception as e:
        logging.error(f"Error processing page {page_number}: {str(e)}")
        return TableDetectionResponse(
            tables_found=False,
            message=f"Error processing page: {str(e)}",
            table_count=0,
            tables=[],
        )


def save_table_extractions(tables: List[Table], output_dir: str, base_filename: str):
    """Save extracted tables in both markdown and JSON formats"""
    logging.debug(f"Saving {len(tables)} tables to {output_dir}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save each table
    for i, table in enumerate(tables):
        # Save as markdown
        md_filename = f"{base_filename}_table_{i+1}.md"
        md_path = os.path.join(output_dir, md_filename)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(table.to_markdown())

        # Save as JSON
        json_filename = f"{base_filename}_table_{i+1}.json"
        json_path = os.path.join(output_dir, json_filename)
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(table.model_dump_json(indent=4))


def process_pdf_tables(pdf_path: str, output_dir: str) -> List[Table]:
    """Process all pages in a PDF for tables"""
    logging.info(f"Starting table extraction from {pdf_path}")

    # Convert PDF pages to images (assuming you have this function)
    image_paths, temp_dir = convert_pdf_pages_to_images(pdf_path)

    # Create a summary log
    summary_log = []
    all_tables = []

    try:
        for page_num, image_path in enumerate(image_paths, 1):
            result = extract_tables_from_page(image_path, page_num)

            # Log the result for this page
            summary_log.append(
                {
                    "page_number": page_num,
                    "tables_found": result.tables_found,
                    "message": result.message,
                    "table_count": result.table_count,
                }
            )

            # Save tables if any were found
            if result.tables_found:
                base_filename = f"page_{page_num}"
                save_table_extractions(result.tables, output_dir, base_filename)
                all_tables.extend(result.tables)

        # Save the summary log
        summary_path = os.path.join(output_dir, "table_extraction_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "total_pages": len(image_paths),
                    "total_tables_found": len(all_tables),
                    "page_summaries": summary_log,
                },
                f,
                indent=4,
            )

        logging.info(
            f"Extracted {len(all_tables)} tables total from {len(image_paths)} pages"
        )
        return all_tables

    finally:
        cleanup_temp_folder(temp_dir)


if __name__ == "__main__":
    # Input and output paths
    PDF_PATH = "/home/joe/ttop/nlp_proj/data/tables/pdfs/IDU14c2805901442814b1f1927712d8316eff02f.pdf"
    EXTRACTIONS_PATH = "table_extractions"

    # Create output directory
    os.makedirs(EXTRACTIONS_PATH, exist_ok=True)

    logging.info(f"Starting table extraction from PDF: {PDF_PATH}")

    try:
        # Process the PDF
        all_tables = process_pdf_tables(PDF_PATH, EXTRACTIONS_PATH)

        # Save overall summary
        summary_path = os.path.join(EXTRACTIONS_PATH, "extraction_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "pdf_path": PDF_PATH,
                    "total_tables_extracted": len(all_tables),
                    "extraction_timestamp": datetime.now().isoformat(),
                },
                f,
                indent=4,
            )

        logging.info(f"Table extraction complete. Found {len(all_tables)} tables.")
        logging.info(f"Results saved to: {EXTRACTIONS_PATH}")

    except Exception as e:
        logging.error(f"Failed to process PDF: {str(e)}")
