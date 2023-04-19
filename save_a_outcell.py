import os
import argparse
import nbformat
from nbconvert.exporters import PDFExporter
from nbconvert.preprocessors import TagRemovePreprocessor

# create an ArgumentParser object to handle command-line arguments
parser = argparse.ArgumentParser(description='Export a Jupyter Notebook cell as a PDF file')
parser.add_argument('notebook_file', help='path to the Jupyter Notebook file')
parser.add_argument('cell_idx', type=int, help='index of the cell to export (starting from 0)')
parser.add_argument('output_file', help='name of the output PDF file')
args = parser.parse_args()

# load the Jupyter Notebook file as a notebook object
with open(args.notebook_file) as f:
    nb = nbformat.read(f, as_version=4)

# remove all tags from the notebook except the "print-pdf" tag
tag_processor = TagRemovePreprocessor(remove_all=True)
tag_processor.remove_cell_tags = ("print-pdf",)
nb = tag_processor.preprocess(nb)

# select the cell you want to save
cell = nb.cells[args.cell_idx]

# create a PDFExporter object and configure it
exporter = PDFExporter()
exporter.preprocessors = [tag_processor]
exporter.exclude_input_prompt = True
exporter.exclude_output_prompt = True
exporter.template_file = 'classic'

# export the cell as a PDF file
output, _ = exporter.from_notebook_node(cell)

# save the PDF file to disk
with open(args.output_file, 'wb') as f:
    f.write(output)
    
# open the PDF file with the default PDF viewer
os.system(f'start {args.output_file}')
