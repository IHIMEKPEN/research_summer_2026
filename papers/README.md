# Papers

| Path | Content |
|------|---------|
| [`icra2027/`](icra2027/) | ICRA 2027 draft (`main.tex`, `references.bib`, `main.pdf`) |
| [`neurips2026_wrl/`](neurips2026_wrl/) | NeurIPS 2026 WRL workshop draft (non-archival); see `COMPLETION.md` + `EMAIL_TO_ORGANIZERS.md` |
| [`reference_papers/`](reference_papers/) | Local PDF references (optional) |

## Build

ICRA 2027:
```bash
cd papers/icra2027
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

NeurIPS 2026 WRL:
```bash
cd papers/neurips2026_wrl
# Prefer official NeurIPS author-kit neurips.sty before final upload.
latexmk -pdf main.tex
# or: pdflatex main && bibtex main && pdflatex main && pdflatex main
```
