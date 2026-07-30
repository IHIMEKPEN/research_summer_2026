# Papers

| Path | Content |
|------|---------|
| [`icra2027/`](icra2027/) | ICRA 2027 draft (`main.tex`, `references.bib`, `main.pdf`) |
| [`reference_papers/`](reference_papers/) | Local PDF references (optional) |

Build:
```bash
cd papers/icra2027
pdflatex main && bibtex main && pdflatex main && pdflatex main
```
