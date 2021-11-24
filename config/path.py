from pathlib import Path

src: Path = Path(__file__).absolute().parent.parent
data: Path = src / 'data'
plots: Path = src / 'plots'
