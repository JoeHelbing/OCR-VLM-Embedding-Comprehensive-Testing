tesseract image_filepath output_filepath txt hocr # don't add file extension (.txt, .hocr) to output_filepath, saves both txt and hocr files

# inserts script into hocr file to display in browser as html file
sed 's|</body>|<script src="https://unpkg.com/hocrjs"></script>\n</body>|' hocr_filepath.hocr > output_filepath.html # make sure to match output filepath name