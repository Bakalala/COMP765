"""Vector mathematical notation for the ReportLab assignment report.

Matplotlib supplies mathematical glyph layout; ReportLab embeds the fonts and
draws the glyphs and fraction bars directly. Equations remain sharp at any zoom.
Matrices use the same math renderer for each entry and vector square brackets.
"""

from dataclasses import dataclass
from functools import lru_cache
import hashlib

from matplotlib import rc_context
from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import MathTextParser
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Flowable


PARSER = MathTextParser("path")


@lru_cache(maxsize=512)
def _parse(expression, size):
    with rc_context({"mathtext.fontset": "stix"}):
        expression = expression.replace(r"\frac", r"\dfrac")
        return PARSER.parse(f"${expression}$", dpi=72,
                            prop=FontProperties(size=size))


@lru_cache(maxsize=32)
def _font_name(path):
    name = "Math_" + hashlib.sha1(path.encode()).hexdigest()[:12]
    pdfmetrics.registerFont(TTFont(name, path))
    return name


def _draw_math(canvas, layout, x, y):
    baseline = y + layout.depth
    for font, size, codepoint, offset_x, offset_y in layout.glyphs:
        canvas.setFont(_font_name(font.fname), size)
        canvas.drawString(x + offset_x, baseline + offset_y, chr(codepoint))
    for rect_x, rect_y, width, height in layout.rects:
        canvas.rect(x + rect_x, baseline + rect_y, width, height,
                    stroke=0, fill=1)


@dataclass(frozen=True)
class Matrix:
    rows: tuple


def matrix(rows):
    return Matrix(tuple(tuple(str(entry) for entry in row) for row in rows))


class MathBlock(Flowable):
    def __init__(self, parts, size=11, number=None):
        super().__init__()
        self.parts = parts
        self.size = size
        self.number = number
        self.spaceBefore = 3
        self.spaceAfter = 7
        self.hAlign = "CENTER"

    def _layout(self, size):
        components = []
        for part in self.parts:
            if isinstance(part, Matrix):
                cells = [[_parse(entry, size) for entry in row]
                         for row in part.rows]
                widths = [max(row[j].width for row in cells) + 12
                          for j in range(len(cells[0]))]
                heights = [max(cell.height for cell in row) + 6
                           for row in cells]
                components.append(("matrix", sum(widths) + 12, sum(heights),
                                   (cells, widths, heights)))
            else:
                parsed = _parse(part, size)
                components.append(("math", parsed.width, parsed.height, parsed))
        width = sum(item[1] for item in components) + 7 * (len(components) - 1)
        return components, width, max(item[2] for item in components)

    def wrap(self, available_width, available_height):
        size = self.size
        components, width, height = self._layout(size)
        limit = available_width - (30 if self.number is not None else 0)
        while width > limit and size > 8:
            size -= 0.25
            components, width, height = self._layout(size)
        if width > limit:
            raise ValueError(f"Equation too wide: {width:.1f} > {limit:.1f}")
        self.layout = components
        self.content_width = width
        self.content_height = height
        self.width = available_width
        self.height = height + 4
        return self.width, self.height

    def draw(self):
        canvas = self.canv
        canvas.saveState()
        canvas.setFillColor(colors.HexColor("#0f172a"))
        canvas.setStrokeColor(colors.HexColor("#0f172a"))
        canvas.setLineWidth(0.65)
        x = (self.width - self.content_width) / 2
        for kind, width, height, data in self.layout:
            y = 2 + (self.content_height - height) / 2
            if kind == "math":
                _draw_math(canvas, data, x, y)
            else:
                cells, widths, heights = data
                canvas.lines([(x + 4, y, x + 1, y),
                              (x + 1, y, x + 1, y + height),
                              (x + 1, y + height, x + 4, y + height),
                              (x + width - 4, y, x + width - 1, y),
                              (x + width - 1, y, x + width - 1, y + height),
                              (x + width - 1, y + height, x + width - 4, y + height)])
                top = y + height
                for row, row_height in zip(cells, heights):
                    top -= row_height
                    column_x = x + 6
                    for cell, column_width in zip(row, widths):
                        _draw_math(canvas, cell,
                                   column_x + (column_width - cell.width) / 2,
                                   top + (row_height - cell.height) / 2)
                        column_x += column_width
            x += width + 7
        if self.number is not None:
            canvas.setFont("Body", 9)
            canvas.setFillColor(colors.HexColor("#64748b"))
            canvas.drawRightString(self.width, self.height / 2 - 3,
                                   f"({self.number})")
        canvas.restoreState()


def equation(*parts, size=11, number=None):
    return MathBlock(parts, size=size, number=number)
