from dataclasses import dataclass, field


# BIO label scheme for token classification
# O       - correct, no error
# B-SPELL - beginning of a spelling error span
# I-SPELL - continuation of a spelling error span
# B-GRAM  - beginning of a grammar error span
# I-GRAM  - continuation of a grammar error span
# B-CITE  - beginning of a wrong citation span
# I-CITE  - continuation of a wrong citation span

LABELS = ["O", "B-SPELL", "I-SPELL", "B-GRAM", "I-GRAM", "B-CITE", "I-CITE"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

# error type derived from BIO prefix
ERROR_TYPES = {
    "SPELL": "spelling",
    "GRAM": "grammar",
    "CITE": "citation",
}


@dataclass
class ErrorSpan:
    text: str           # the flagged text e.g. "Section 302 IPC"
    error_type: str     # "spelling", "grammar", or "citation"
    page_no: int
    x0: float
    y0: float
    x1: float
    y1: float
    suggestion: str = ""      # suggested correction, empty until we have a correction model
    confidence: float = 0.0   # model confidence score for this span

    @property
    def bbox(self):
        return (self.x0, self.y0, self.x1, self.y1)

    @property
    def highlight_color(self):
        # consistent color per error type for the frontend
        colors = {
            "spelling": "#FFD700",    # yellow
            "grammar": "#FFA500",     # orange
            "citation": "#FF4444",    # red
        }
        return colors.get(self.error_type, "#CCCCCC")