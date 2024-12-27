import argparse
import json
import logging
from pathlib import Path
from typing import List

import yaml
from docling.datamodel.settings import settings
from docling.document_converter import (
    DocumentConverter,
)

_log = logging.getLogger(__name__)

# Turn on inline debug visualizations:
settings.debug.visualize_layout = True
settings.debug.visualize_ocr = True
settings.debug.visualize_tables = True
settings.debug.visualize_cells = True


def get_input_files(input_dir: Path) -> List[Path]:
    """Get all PDF and image files from the input directory."""
    supported_extensions = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif"}
    return [
        Path(f) for f in input_dir.iterdir() if f.suffix.lower() in supported_extensions
    ]


def main(input_dir, output_dir):
    doc_converter = DocumentConverter()

    conv_results = doc_converter.convert_all(get_input_files(input_dir))

    for res in conv_results:
        out_path = output_dir
        print(
            f"Document {res.input.file.name} converted."
            f"\nSaved markdown output to: {str(out_path)}"
        )
        _log.debug(res.document._export_to_indented_text(max_text_len=16))
        # Export Docling document format to markdowndoc:
        with (out_path / f"{res.input.file.stem}.md").open("w") as fp:
            fp.write(res.document.export_to_markdown())

        with (out_path / f"{res.input.file.stem}.json").open("w") as fp:
            fp.write(json.dumps(res.document.export_to_dict()))

        with (out_path / f"{res.input.file.stem}.yaml").open("w") as fp:
            fp.write(yaml.safe_dump(res.document.export_to_dict()))


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process documents with Docling")
    parser.add_argument(
        "input_dir", type=str, help="Input directory containing documents"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="scratch/",
        help="Output directory for processed files (default: scratch/)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set the logging level",
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Convert string paths to Path objects
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Validate input directory
    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    main(input_dir, output_dir)
