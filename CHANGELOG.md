# Changelog

## 1.2.0 — 2026-09-06

- aligned training, validation, and Figure 2 PR-AUC with the manuscript's trapezoidal definition;
- added explicit final-master exclusion filtering and 489/309/180 assertions to Figure 2;
- separated the 497-image preprocessing pool from the final 489-patient analytic cohort in executable split and training checks;
- added exact calibration and decision-curve regeneration from frozen predictions;
- added corrected Layer4 Grad-CAM extraction compatible with the Figure 4 assembly pipeline;
- replaced non-functional literal path placeholders with required environment variables;
- fixed the undefined Figure 2 audit object and made its metric fingerprint recompute from predictions;
- moved Figure 4 example-case identifiers and historical preprocessing audit identifiers to environment configuration;
- completed the MIT license and added citation, dependency, schema, privacy, and reproducibility files;
- clarified that the 497-case notebook is an optional historical preprocessing stage and neutralized its preprocessing-only exclusion terminology;
- aligned software citation metadata with the manuscript title;
- cleared notebook outputs and normalized notebook metadata to Python 3.10.
