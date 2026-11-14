# PARD — Plan-Referenced Execution Reconstruction for FFF

Research code, archived numerical results, figures, and manuscript source for **PARD: A plan-referenced framework for quantitative process execution reconstruction without additional sensors in fused filament fabrication** (NAMRC 55 submission).

## Repository contents

| Path | Contents |
| --- | --- |
| `src/prof.py` | Reads telemetry and the planned G-code layer table; generates intermediate files |
| `src/final_pipeline.py` | Evaluates channel routing, benchmarks, ablations, robustness, and clustered bootstrap |
| `src/make_assets.py` | Regenerates tables, manuscript macros, and plots |
| `src/exploratory/` | Historical exploratory analysis; not part of the final execution workflow |
| `results/` | Archived numerical results and LaTeX tables from the study |
| `figs/` | Archived manuscript figures |
| `paper/` | Manuscript PDF and LaTeX source |

## Data requirements

**Raw data are not included in this code repository.** Place exactly one telemetry Excel workbook (`.xlsx`, containing `Training_Data` and `Run_Audit` sheets) and one corresponding plan-reference layer CSV (`.csv`) inside a newly created `data/` directory. These two input files are required for reproduction. The plan-reference CSV must contain the baseline layer durations, extrusion lengths, and geometry fields expected by `src/final_pipeline.py`. Obtain the inputs from the associated dataset release when available; a code release alone does not supply the inputs.

## Run

From the root of the repository, with Python 3 and the two inputs in `data/`:

```bash
python -m pip install -r requirements.txt
python src/prof.py
python src/final_pipeline.py
python src/make_assets.py
```

The pipeline writes intermediate files and regenerated numerical results to `gen/` and figures to `figs/`. The archived publication results remain in `results/`. The final scripts have been made path-portable, but full regeneration has **not been tested against the missing raw inputs** in this package. Dependency versions are not pinned.

## Interpretation and scope

The evaluation covers one LulzBot Mini 3 printer, PLA, and one geometry. It includes an exploratory model-development component described in the manuscript. Fan state and extrusion length are firmware reports and not independent physical measurements of airflow or deposited material.

## Citation and data availability

See the associated paper and its data-availability statement for publication and dataset citation details when released.

## License

No software license is specified in this archive. Reuse permissions must be confirmed with the code owners before adding a license.
