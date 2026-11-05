# Report draft

- `build_report.py` is the editable ReportLab source. Its data tables read
  `results/summary.csv` and the summaries in `results/extended/`.
- `make_figures.py` draws the report figures at printed size into `figures/`.
- `report.pdf` is generated from that source.

Before submission, replace the author placeholder and complete the two red
`[STUDENT TO WRITE ...]` slots: the peer comparison in Section 2.4 and the
reflection in Section 8. Tables read the result files directly, but numbers in the
prose are typed by hand, so recheck them if any experiment is rerun.

Rebuild from this directory with the project requirements installed:

```bash
python3 make_figures.py
python3 build_report.py
```
