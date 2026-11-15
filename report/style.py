"""Colors and matplotlib style of the report's figures."""
import matplotlib

# color = meaning: teal = data / score, orange = class / guidance, gray = noise
INK = "#1F2A37"      # text
MUTED = "#5B6676"    # captions, labels
TEAL = "#0F766E"     # data, score
ORANGE = "#D9480F"   # class "ship", guidance
GRAY = "#9AA3AE"     # noise
LIGHT = "#E9EDF1"    # faint fills, grid
WHITE = "#FFFFFF"

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def setup_matplotlib():
    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "font.family": ["Calibri", "DejaVu Sans"],   # DejaVu if Calibri is missing
        "mathtext.fontset": "cm",          # math looks like LaTeX
        "font.size": 15,
        "axes.edgecolor": LIGHT,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "savefig.dpi": 300,
        "savefig.transparent": True,
    })
