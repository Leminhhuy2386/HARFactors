# Review addendum, 2026-10-16 (after iteration 3)

This is not a fourth review iteration. It records what the downloads made on the user's PC resolved. The session "IEEE
Sensors Letters template and resources" put 36 files and `SOURCES.txt` in
`/work/huy.leminh/.cache/harfactors_scratch/pc_downloads/2026-10-16/`. All 36 sha256 values were re-checked on a
compute node (Slurm job 86788). The iteration 3 report is unchanged.

## Resolved

- **Mandatory check 13: now passes.** Figure 6 (page 14 of arXiv:2606.13194v1, sha256 2fcc4b3d…) was read by eye
  against `docs/sources/arena_extracted.json`. Four cells match (`docs/sources/figure6_spot_checks.json`):
  - SAD / CNN-HAR, 95.60 ± 5.91;
  - SAD / TinyHAR, 97.67 ± 5.22;
  - UCI-HAR / Random Forest, 84.69 (mean only);
  - UTD-MHAD / Random Forest, 51.20 (mean only).

  The column order matches, and so does the CNN-HAR mean of 67.72 % in the paper's Mean row and text.
- **Official class.** `paper/IEEE_lsens.cls` now comes from the IEEE Template Selector's 2023 LSENS author templates.
  - The manuscript follows `regular_lsens.tex`: subject header, author block, graphical abstract, Index Terms.
  - The abstract was cut from about 178 to 140 words, to meet the template's limit of 150.
  - G7 rerun3 PASS: 3 pages, 2 figures, 0 problems. `audit --all` passes 10/10.
- **Source sentences confirmed on the official pages:**
  - UTD-MHAD: "After removing three corrupted sequences, the dataset includes 861 data sequences"; the inertial rate
    is 50 Hz.
  - IEEE Sensors Letters: the 4-page limit, including a single column of references.
  - The AI/ML scope statement ("will be declined without review").

## Still blocking (unchanged)

- **Check 20, label-shuffle control:** not run (it needs training). The user's decision is pending; the gap is
  disclosed in Limitations.
- **Check 16, re-execution:** 360 of 828 confirmatory runs were re-executed; the rest is disclosed.
- **Formalities:**
  - the author block and affiliations;
  - the subject category (set to "Sensor Applications", to be confirmed);
  - the graphical abstract (Fig. 1 now, replaceable);
  - at least 2 suggested reviewers, ORCID and an EDICS category at submission.
- **Scope risk:** IEEE Sensors Letters declines AI/ML or data-analysis work with no measurable impact on sensor systems.
  The editor may judge this letter out of scope despite its sensing-study framing.

## Verdict

Still `NOT_READY_FOR_SUBMISSION`, now for checks 16 and 20 and the formalities only.
