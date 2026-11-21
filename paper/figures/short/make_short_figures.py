"""Re-render the figures of paper_short.tex in the template fonts (Charis SIL text, STIX math)."""
import sys, types, runpy
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
FONTS, OUT, DUMMY = map(Path, sys.argv[1:4])
for f in FONTS.glob("*.ttf"):
    font_manager.fontManager.addfont(str(f))
CHARIS = '["Charis SIL"]'
REPO = Path(__file__).resolve().parents[3]

# make_figures.py: swap the font list, write PDFs to OUT, thesis copies to a dummy folder
src = (REPO / "paper/figures/make_figures.py").read_text()
old = '"font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"]'
assert old in src
src = src.replace(old, f'"font.sans-serif": {CHARIS}, "mathtext.fontset": "stix"')
src = src.replace('if __name__ == "__main__":', 'if False:')
# Short paper: legend names as in Table 2 of paper_short.tex; no frequency-prior curve.
for old, new in [
    ('"PN, Waze-timed police-confirmed labels"', '"PN"'),
    ('("nnPU, register prior", phase6', '(r"nnPU, $\\pi_{\\mathrm{rec}}$", phase6'),
    ('"nnPU + reporting propensity"', 'r"nnPU, $\\pi_{\\mathrm{rec}}$, propensity weights"'),
    ('        ("Frequency prior alone", phase7["scores"]["frequency_prior_alone"]["deciles"], VIOLET, "D"),\n', ''),
    ('"PN (U as negative)"', '"PN"'),
    ('"PN rescaled by the known prior"', 'r"PN $\\times\\,\\pi/\\pi_L$"'),
    ('(AQUA, "^", "nnPU, register prior")', '(AQUA, "^", r"nnPU, $\\pi_{\\mathrm{rec}}$")'),
    ('"nnPU + estimated propensity"', 'r"nnPU, $\\pi_{\\mathrm{rec}}$, propensity weights"'),
]:
    assert src.count(old) == 1, old
    src = src.replace(old, new)
mod = types.ModuleType("make_figures"); mod.__file__ = str(REPO / "paper/figures/make_figures.py")
exec(compile(src, mod.__file__, "exec"), mod.__dict__)
mod.OUT, mod.THESIS = OUT, DUMMY
mod.calibration(); mod.label_hiding()
if len(sys.argv) > 4 and sys.argv[4] == "--no-maps":
    sys.exit(0)

# mapbase.py / thesis_maps.py: same font swap, maps written to DUMMY
mb_src = (REPO / "viz/mapbase.py").read_text()
old = '"font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"]'
assert old in mb_src
mb_src = mb_src.replace(old, f'"font.sans-serif": {CHARIS}, "mathtext.fontset": "stix"')
mb = types.ModuleType("mapbase"); mb.__file__ = str(REPO / "viz/mapbase.py")
exec(compile(mb_src, mb.__file__, "exec"), mb.__dict__)
mb.OUT = DUMMY
sys.modules["mapbase"] = mb
sys.path.insert(0, str(REPO / "viz"))
runpy.run_path(str(REPO / "viz/thesis_maps.py"), run_name="__main__")
