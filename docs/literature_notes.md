# Reference notes: HARFactors letter (checked 2026-10-15)

## How the references were verified (read first)

- From this host, WebFetch gets **HTTP 403** from almost every bibliographic host: api.crossref.org,
  arxiv.org / export.arxiv.org, dl.acm.org, dblp.org, openalex, semanticscholar, mdpi.com, jmlr.org, ijcai.org,
  proceedings.mlsys.org, link.springer.com, PMC, Wikipedia, ieee-sensorsletters.org, ieee-sensors.org,
  overleaf.com and ieeexplore. The hosts that worked were **doi.org** (handle API), **huggingface.co/papers**,
  **github.com / raw.githubusercontent.com** and **pypi.org**. No browser was connected, and curl fails on TLS
  (exit 60). curl -k was not used.
- So every entry was confirmed by at least two independent lookups:
  1. `https://doi.org/api/handles/<DOI>`: proves the DOI is registered and gives the publisher landing URL.
     For MDPI, OUP and IEEE that URL also encodes the volume, issue or page, or the document number.
  2. A metadata source: the Hugging Face paper page (arXiv works), the authors' own README, BibTeX or
     CITATION.cff on GitHub, the WHAR Arena reference list (read through its HF page), or the WebSearch result
     from the publisher or a repository domain.
- Fields that no lookup confirmed were **left out** of `references.bib`. Each case is listed in the JSON
  `note`. Example: the IJCAI end page of hammerla2016deep.
- One integrity catch: **Holm 1979 has no DOI.** `10.2307/4615733`, which an R package cites, returns 404 from
  doi.org. Cite the JSTOR stable URL instead.
- The quotes below are short phrases from the abstracts or text that the tools returned (≤ 25 words each).
  Text without quote marks is a paraphrase of a search snippet.

## Verified works (22)

### Benchmark, library and dataset

**burzer2026arena**: WHAR Arena, arXiv:2606.13194 (v1, June 2026). Authors: Burzer, King, Riedel, Beigl,
Röddiger (KIT; IPAI Foundation).
- What it is: an open benchmark of 30 datasets, a shared cross-subject protocol, 17 architectures and 4,760 runs.
  The authors conclude that "the WHAR state of the art is distributed rather than dominated by a single
  architecture".
- Protocol, from the HF text: "we use a k-subject-groups (KSG) splitter with k=10"; "Final predictive performance
  is reported exclusively on the test set containing unseen subjects"; and "Leave-one-subject-out cross-validation
  is not used due to the large number of resulting splits".
- Reporting: they give the "mean and standard deviation over all splits" and warn that small differences "should
  not be overinterpreted as clear evidence of superiority".
- Seeds: the paper reports no number of training seeds, only model-split runs. This is relevant to our
  seed-vs-subject decomposition.
- No venue or DOI was found. Cite it as an arXiv preprint.

**burzer2025whar**: WHAR Datasets, UbiComp Companion '25, pp. 1315–1322, doi 10.1145/3714394.3756254.
- An open-source library that standardises WHAR data handling through a standardized data format and a
  configuration-driven design.
- The paper version supports 9 datasets, integrates with PyTorch and TensorFlow, and reports preprocessing
  speedups of up to 3.8× (search snippet of the arXiv abstract).
- Subject and activity labels are kept in central tables, so subject-wise splitting needs no re-reading of the raw
  data.
- Cite it as the source of the paper-era caches and splits.

**chen2015utdmhad**: UTD-MHAD, IEEE ICIP 2015, pp. 168–172, doi 10.1109/ICIP.2015.7350781.
- A public dataset of four synchronised modalities (RGB, depth, skeleton, inertial) for 27 actions. Experiments
  show how it "can be used to study fusion approaches that involve using both depth camera data and inertial
  sensor data" (search snippet of the abstract).
- **Dataset facts and where each came from.** The official page could not be fetched (403). The sentences below
  were returned by WebSearch for the official page, https://personal.utdallas.edu/~kehtar/UTD-MHAD.html, in a
  search restricted to `personal.utdallas.edu`:
  - Sequences: "after removing three corrupted sequences, the dataset includes 861 data sequences"
    (attributed to the official page).
  - Rate: "The sampling rate of this wearable inertial sensor is 50 Hz." (official page; the same sentence was
    returned in a second, unrestricted search).
  - Subjects, actions and repetitions: "27 actions performed by 8 subjects (4 females and 4 males) with each
    subject performing each action 4 times". This was returned as the dataset description in searches whose
    hits were the official page and papers using it, e.g. https://arxiv.org/pdf/1612.09401.
  - Sensor placement: the sensor "is fixed on the right wrist or thigh of the subject according to the type of
    action".
- Cross-check, not a citable source: the harness config
  `third_party/whar-datasets/src/whar_datasets/config/cfg_utd_mhad.py` has `sampling_freq=50`,
  `num_of_subjects=8` and `num_of_activities=27`.
- Caveat: an exact-phrase web search for the 861 sentence found no page. Open the official page once by hand to
  confirm the wording before quoting it in the letter.

### Evaluation methodology and person heterogeneity

**hammerla2015lets**: Hammerla & Plötz, UbiComp '15, pp. 1041–1051, doi 10.1145/2750858.2807551.
- Random cross-validation on segmented time series is biased, because neighbouring segments are not independent.
  The paper proposes "meta-segmented cross validation".
- Removing this bias gives a notable drop in measured performance, and user-dependent performance should not be
  read as an upper bound (search snippet of the abstract).
- Use it to motivate person-disjoint evaluation.

**lockhart2014limitations**: Lockhart & Weiss, UbiComp '14 Adjunct, pp. 747–756, doi 10.1145/2638728.2641306.
- Finds that "most AR data sets are sufficiently limited as to impact the reliability of existing research
  results".
- Its main focus is model type (personal / hybrid / impersonal), which "determines how AR training and test data
  are partitioned". These model types perform very differently.
- Supports the claim that *who is tested* drives the reported accuracy.

**qian2021latent**: Qian, Pan & Miao, AAAI 2021 (vol. 35, no. 13), doi 10.1609/aaai.v35i13.17416.
- Cross-person HAR. Training and test data are usually assumed to share one distribution, which ignores how
  activity patterns differ between people.
- The proposed GILE learns domain-agnostic and domain-specific features without access to the target person.
  It is evaluated on three benchmarks (search snippet of the abstract).
- Use it as the citation for inter-subject distribution shift.
- Pages are not verified, so they are left out.

**stisen2015smart**: Stisen et al., SenSys '15, pp. 127–140, doi 10.1145/2809695.2809718.
- Sensor and device heterogeneity across 36 smartphones and smartwatches (13 models, 4 manufacturers, 9 users)
  significantly degrades HAR performance (search snippet of the abstract).
- Introduces the HHAR dataset. This is the sensor-side counterpart to the person factor, and a good hook for the
  Sensors Letters scope (see below).

**bulling2014tutorial**: Bulling, Blanke & Schiele, ACM CSUR 46(3), 33:1–33:33, doi 10.1145/2499621.
- The standard HAR tutorial. It introduces the "activity recognition chain (ARC)" as a "general-purpose framework
  for designing and evaluating activity recognition systems".
- Use it as the general background citation.

**bouthillier2021accounting**: Bouthillier et al., MLSys 2021 (Proc. MLSys vol. 3), arXiv 2103.03098.
- The abstract opens: "Strong empirical evidence that one machine-learning algorithm A outperforms another one B"
  "calls for multiple trials optimizing the learning pipeline over sources of variation", but "corners are cut to
  reach conclusions".
- This is the ML-benchmark precedent for splitting variance into data-split, seed and other sources.
- Pages are not verified.

### Models in the population

**ordonez2016deep**: Ordóñez & Roggen, Sensors 16(1), 115, doi 10.3390/s16010115.
- DeepConvLSTM: convolutional layers plus LSTM units for multimodal wearable HAR, with no hand-designed features.
- Beats non-recurrent deep networks by about 4% on the challenge dataset (search snippet of the abstract).

**hammerla2016deep**: Hammerla, Halloran & Plötz, IJCAI 2016, arXiv 1604.08880.
- "rigorously explore deep, convolutional, and recurrent approaches across three representative datasets" and
  introduce a regularisation approach.
- **Start page 1533 is verified (ijcai.org header). The end page is not**, so pages are not in the .bib.

**gu2022efficiently**: Gu, Goel & Ré, ICLR 2022, arXiv 2111.00396 (S4).
- Conventional models "still struggle to scale to very long sequences of 10000 or more steps". S4 is a structured
  state-space layer that addresses this.
- Cite it for the S4 family in the population.

**gu2022parameterization**: Gu, Gupta, Goel & Ré, NeurIPS 35 (2022), arXiv 2206.11893 (S4D).
- Diagonal state-space models. The abstract notes that "The first version to show this potential was the S4
  model".
- S4D simplifies S4's prescribed HiPPO state matrix.

**gu2023mamba**: Gu & Dao, arXiv:2312.00752 (Mamba).
- Selective state spaces. The abstract places SSMs among subquadratic architectures "developed to address
  Transformers' computational inefficiency on long sequences".
- The authors' own README still cites it as an arXiv preprint.

**breiman2001random**: Breiman, Machine Learning 45(1), 5–32, doi 10.1023/A:1010933404324.
- The Random Forest. WHAR Arena uses RF as its classical baseline and finds that "compact neural models such as
  TinierHAR and classical Random Forests" define the efficiency Pareto frontier (search summary of the Arena
  abstract).
- No abstract of Breiman's paper was fetched.

**christ2018tsfresh**: Christ et al., Neurocomputing 307, 72–77, doi 10.1016/j.neucom.2018.03.067.
- The tsfresh feature-extraction package. Cite it only if the feature-based baseline or data-factor features use
  tsfresh.

### Statistics and software

**seabold2010statsmodels**: Seabold & Perktold, SciPy 2010, pp. 92–96, doi 10.25080/Majora-92bf1922-011.
- "statsmodels is a library for statistical and econometric analysis in Python".
- Cite it for MixedLM / REML and OLS.

**patterson1971recovery** (extra): Patterson & Thompson, Biometrika 58(3), 545–554, doi 10.1093/biomet/58.3.545.
- Estimates the variance weights by maximising the likelihood of a set of error contrasts rather than of the full
  data. This is the origin of REML.
- Optional if space is tight.

**field2007bootstrapping**: Field & Welsh, JRSS-B 69(3), 369–390, doi 10.1111/j.1467-9868.2007.00593.x.
- Bootstrap methods for clustered data. Key line: "the cluster bootstrap is consistent under both the
  transformation and random-effect models".
- This directly justifies resampling subjects, splits or datasets as clusters.

**wilcoxon1945individual** (extra): Wilcoxon, Biometrics Bulletin 1(6), 80–83, doi 10.2307/3001968.
- Introduces the rank-sum and signed-rank tests. Optional; Demšar can stand in for it.

**holm1979simple**: Holm, Scand. J. Statist. 6(2), 65–70 (JSTOR 4615733, **no DOI**).
- The sequentially rejective procedure: hypotheses are rejected one at a time until no further rejection is
  possible.

**demsar2006statistical**: Demšar, JMLR 7, 1–30 (no DOI).
- Recommends the Wilcoxon signed-ranks test for comparing two classifiers over multiple data sets, and the
  Friedman test with post-hoc tests for more than two.
- This supports the paired, non-parametric tests across datasets.

## Not added (and why)

- **Weiss & Lockhart 2012 (personalization).** Not looked up. lockhart2014limitations covers the
  personal-vs-impersonal point and was verified.
- **Chang et al. 2020, IMWUT (10.1145/3380985).** Verified by search, but it studies sensor *placement*
  ("wearing diversity"), not people, so it is weaker for candidate 15.
- **Napoli et al. 2024, Sci. Data (10.1038/s41597-024-03951-4).** A domain adaptation and generalisation
  benchmark, cited by WHAR Arena. Its DOI was not checked against doi.org, so it is not in the .bib.
- **Dehghani et al. 2019 (arXiv 1904.02666), "Subject cross validation in HAR".** Relevant, but arXiv-only. The HF
  page returned 404 and arXiv returned 403, so it could not be verified.

## IEEE Sensors Letters: format facts

Neither official page (ieee-sensorsletters.org, ieee-sensors.org) could be fetched (403). The sentences below come
from WebSearch snippets of those pages. Each is followed by the URL the snippet was attributed to. Re-check them on
the live page before submission.

1. **Page limit.** "Papers submitted to this journal will be limited to a maximum length of 4 pages which includes
   a minimum of single column restricted for references."
   - Source: https://ieee-sensorsletters.org/information-for-authors/
   - Same rule: "Papers are limited to 4 pages with the stipulation that at least one column of each paper be
     devoted exclusively to references." (https://ieee-sensorsletters.org/about/)
   - Over-length papers: "In such cases, the paper will be transferred to IEEE Sensors Journal." (Associate Editor
     Manual, https://ieee-sensorsletters.org/wp-content/uploads/2025/03/SL-Associate-Editor-Manual-2024.pdf)
   - Consequence: about one of the eight columns goes to references, so 12–18 references fit only if kept short.
2. **Template.** "Authors are required to prepare manuscripts employing the double-column style template for IEEE
   Sensors Letters." Word and LaTeX versions exist.
   - Source: https://ieee-sensors.org/sensors-letters/senslet-information-for-authors/
3. **SCOPE: a risk to check now.** Two independent searches returned this sentence from the Information for
   Authors page: "Submissions that focus solely on AI/ML algorithms, data analysis, or signal processing – without
   a demonstrated and measurable impact on sensor devices or sensor systems – are out of scope for IEEE Sensors
   Letters."
   - Source: https://ieee-sensorsletters.org/information-for-authors/
   - A third, exact-phrase search did not find the sentence, so confirm it on the live page.
   - If it holds, the letter must be framed around the *sensor system*: channel count, sensor placement and
     modality as data factors, the per-subject sensing variability, and deployment implications. It should not
     read as pure ML benchmarking.
4. **Abstract.** No word limit was found on any Sensors Letters page. The Associate Editor Manual asks for "an
   abstract stating the scope of the paper and summarizing the author's conclusions".
   - Do not borrow the 100–175-word figure: that belongs to IEEE *Signal Processing* Letters, a different
     journal.
5. **Index Terms and subject category.**
   - Index Terms: the template renders "Abstract—" and "Index Terms—" blocks; its placeholder terms are
     "IEEE, IEEE Sensors Letters, LaTeX, paper, template".
   - Subject: the template sets the article subject with `\IEEELSENSarticlesubject{...}`, with "Sensor
     Applications" as the example (seen in a paper built from the template, https://arxiv.org/html/2406.16723v1).
   - EDICS: "Each submission must be classified by the author with one category of the Journal's EDICS"; a keyword
     list is provided at submission (https://ieee-sensorsletters.org/information-for-authors/). A keyword list page
     exists at https://ieee-sensors.org/ieee-sensors-letters/keywords/.
6. **LaTeX class.** The Overleaf template "IEEE Sensors Letters LaTeX Template" is
   https://www.overleaf.com/latex/templates/ieee-sensors-letters-latex-template/sdwszcmkkmdk. Its skeleton
   `bare_lsens.tex` (V1.0, 2017/07/12, Michael Shell) describes itself as a "skeleton file demonstrating the use of
   IEEE_lsens.cls for an IEEE Sensors Letters paper".
   - It loads `\documentclass{IEEE_lsens}`.
   - Its comments say "the only text size is 9pt", so no class options are needed, and that margins and fonts must
     not be changed unless the journal's editors instruct it.
   - This matches the believed names, **IEEE_lsens.cls / bare_lsens.tex**.
7. **Reference style.** The template's demo text cites numerically ("[1]"). No sentence on the official pages
   names a reference style; IEEE numeric (IEEEtran.bst) is the implied default but is **not stated** in the
   sources found.
8. **Other requirements** (Information for Authors, per the search snippet):
   - submit through the IEEE Author Portal;
   - suggest at least two reviewers from other institutions;
   - ORCID is required.
