from dataclasses import dataclass


@dataclass
class LineSpan:
    text: str        # full line text, words joined with single space
    page_no: int
    source: str      # "native" or "surya"
    x0: float        # left edge of the line
    y0: float        # top edge of the line
    x1: float        # right edge of the line
    y1: float        # bottom edge of the line

    @property
    def bbox(self):
        return (self.x0, self.y0, self.x1, self.y1)

    def is_valid(self):
        return bool(self.text.strip()) and self.x1 > self.x0 and self.y1 > self.y0