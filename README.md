# Schizophrenia EEG leakage-free benchmark

Analysis code and results for a leakage-free evaluation of class-conditional
Tucker decomposition for schizophrenia detection from EEG, with subject-wise
baselines on three public cohorts.

Author: Sibghatullah I. Khan, Department of Electronics and Communication
Engineering, Sreenidhi Institute of Science and Technology, Hyderabad, India.
ORCID 0000-0003-1263-8100. Manuscript under review.

## Layout

- `code/` analysis pipeline and figure-generation scripts
- `figures/` manuscript figures
- `results/` per-fold metrics and the tables reported in the manuscript

## Data

The raw EEG recordings are not redistributed here. Download them from the
original depositors:

- RepOD (Olejarczyk and Jernajczyk): https://repod.icm.edu.pl
- Moscow State University adolescent dataset: http://brain.bio.msu.ru
- Button-tone corollary discharge dataset: Kaggle, `broach/button-tone-sz`

Feature matrices and per-fold prediction dumps are archived separately on
Zenodo (DOI to be added).

## Licence

MIT Licence for the code. The datasets remain under the licences set by their
original depositors.
