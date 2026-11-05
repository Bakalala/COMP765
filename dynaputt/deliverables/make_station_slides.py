"""Build the two-slide PDF for the interactive station session."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "report" / "figures"
OUT = HERE / "station_slides.pdf"
PAGE_W, PAGE_H = landscape(letter)
MARGIN = 0.55 * inch
INK = colors.HexColor("#1f1f1d")
MUTED = colors.HexColor("#5a5954")


def heading(pdf: canvas.Canvas, title: str) -> None:
    pdf.setFillColor(INK)
    pdf.setFont("Helvetica-Bold", 25)
    pdf.drawString(MARGIN, PAGE_H - 0.85 * inch, title)


def bullets(pdf: canvas.Canvas, lines: list[str], x: float, y: float, size: float = 16.5) -> None:
    pdf.setFillColor(INK)
    pdf.setFont("Helvetica", size)
    for line in lines:
        pdf.drawString(x, y, line)
        y -= size * 1.95


def picture(pdf: canvas.Canvas, name: str, x: float, y: float, width: float) -> float:
    image = ImageReader(str(FIGURES / name))
    image_w, image_h = image.getSize()
    height = width * image_h / image_w
    pdf.drawImage(image, x, y - height, width, height)
    return y - height


def footer(pdf: canvas.Canvas, text: str) -> None:
    pdf.setFillColor(MUTED)
    pdf.setFont("Helvetica", 10.5)
    pdf.drawString(MARGIN, 0.35 * inch, text)


def main() -> None:
    pdf = canvas.Canvas(str(OUT), pagesize=landscape(letter))
    pdf.setTitle("Dyna Putt station slides")
    width = (PAGE_W - 2 * MARGIN - 0.3 * inch) / 2
    right = MARGIN + width + 0.3 * inch
    top = PAGE_H - 1.15 * inch

    heading(pdf, "More imagined practice means fewer real shots")
    bottom = picture(pdf, "course_maps.png", MARGIN, top, width)
    picture(pdf, "planning_budget.png", right, top, width)
    bullets(pdf, [
        "Q-learning updates once per real shot.",
        "Dyna-Q also replays k remembered shots after each real one.",
        "Wall-gate real shots to 80% success: 5063, 1848, 1607 for k = 0, 5, 20 (5 seeds).",
        "Most of the gain comes from the first few updates (100 seeds, right).",
    ], MARGIN, bottom - 0.45 * inch)
    footer(pdf, "Threshold: first 20-episode window with at least 80% sinks. Bars show 95% intervals.")
    pdf.showPage()

    heading(pdf, "But it depends on where the values start")
    bottom = picture(pdf, "init_control.png", MARGIN, top, width)
    picture(pdf, "route_length.png", right, top, width)
    bullets(pdf, [
        "Every shot costs 1, so a zero Q table is optimistic. Planning partly drains that.",
        "Without optimism the benefit disappears on the 2-shot course and shrinks on Wall-gate.",
        "With the lowest start value, most agents never solve the 8-shot course.",
        "A fair baseline for a planner is a well-initialized learner.",
    ], MARGIN, bottom - 0.45 * inch)
    footer(pdf, "100 seeds, separate from the primary five. Exploratory: designed after the first results.")
    pdf.showPage()
    pdf.save()
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
