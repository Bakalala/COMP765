# Report draft

- `build_report.py` is the editable ReportLab source. Its data tables read
  `results/summary.csv` and the summaries in `results/extended/`.
- `make_figures.py` draws the report figures at printed size into `figures/`.
- `report.pdf` is generated from that source.

Section 2.4 compares the project with Assignment 1 but names no peer papers, because
those are not recorded here. Add them if you have them, and read the reflection in
Section 8 to confirm it matches your view. Tables read the result files directly, but numbers in the
prose are typed by hand, so recheck them if any experiment is rerun.

Rebuild from this directory with the project requirements installed:

```bash
python3 make_figures.py
python3 build_report.py
```
