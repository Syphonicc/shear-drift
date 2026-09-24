# Notice: what is covered by which terms

The [`LICENSE`](LICENSE) file applies the MIT License to the **source code** in
this repository: everything under `analysis/`, `web/`, the Python under
`geometry/`, and the build scripts.

## Derived data and figures

The reduced arrays under `data/` and `runs/`, the sweep results in
`analysis/womersley_sweep.csv`, the PNG figures, and the built page in `docs/`
are released under
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
Please cite as given in [`CITATION.cff`](CITATION.cff).

## Third-party inputs

These are **not** covered by either licence above and remain subject to the
terms of their original providers.

- **Cerebral aneurysm geometry.** Derived from anonymised images distributed
  for the 2015 International Aneurysm CFD Challenge. Institutional review
  board approval for sharing those anonymised images was obtained by the
  challenge organisers from Wakayama Rosai Hospital, as reported in
  Valen-Sendstad K. et al., *Real-world variability in the prediction of
  intracranial aneurysm wall shear stress: the 2015 International Aneurysm CFD
  Challenge*, Cardiovascular Engineering and Technology **9** (2018) 544–564;
  data at [doi:10.6084/m9.figshare.6383516](https://doi.org/10.6084/m9.figshare.6383516).
  No raw images are redistributed here.
- **AAA042 surface.** From the Vascular Model Repository. Not redistributed
  here; only derived biomarker fields and analysis scripts are included.
- **three.js** (`web/vendor/three.min.js`), r128, MIT License, © three.js
  authors.
