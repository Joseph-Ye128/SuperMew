from io import BytesIO
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas


source = Path(r"C:\Users\bubble\Desktop\叶柏辰_AI Agent开发实习生简历.pdf")
output = Path("output") / "叶柏辰_AI Agent开发实习生简历_无水印.pdf"
output.parent.mkdir(parents=True, exist_ok=True)

reader = PdfReader(str(source))
writer = PdfWriter()

for page in reader.pages:
    width = float(page.mediabox.width)
    height = float(page.mediabox.height)
    overlay_stream = BytesIO()
    overlay = canvas.Canvas(overlay_stream, pagesize=(width, height))
    # The template watermark is located in the otherwise empty lower-right margin.
    overlay.setFillColorRGB(1, 1, 1)
    overlay.rect(width - 105, 0, 105, 58, fill=1, stroke=0)
    overlay.save()
    overlay_stream.seek(0)
    page.merge_page(PdfReader(overlay_stream).pages[0])
    writer.add_page(page)

with output.open("wb") as stream:
    writer.write(stream)

print(output.resolve())
