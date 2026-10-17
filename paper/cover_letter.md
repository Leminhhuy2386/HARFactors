# Cover letter (draft of 2026-10-17)

Fields in square brackets are for the authors to fill in or confirm before submission.

---

[Date]

The Editor-in-Chief
IEEE Sensors Letters

**Re: Submission of the letter "Who Is Tested or Which Network? A Pre-Registered Variance Decomposition and Error
Budget for Person-Disjoint Wearable Activity Recognition"**

Dear Editor,

We submit the above letter for consideration in IEEE Sensors Letters, subject category [Sensor Applications; to be
confirmed].

## What the letter does

Wearable activity-recognition systems are chosen on macro-F1 measured on people unseen in training. The letter
measures how much of that number belongs to the people tested and how much to the network. It does so under one fixed
person-disjoint protocol, with a crossed variance-components model and a pre-registered test on held-out datasets.

It shows that, for networks of similar design, the test-subject group carries most of the variance. It shows that
this ratio depends on which networks are compared. And it turns the variance components into an error budget, which
says how many test subjects a sensing study needs, and that pairing by split matters more than further training
seeds. It also finds that the benchmark protocol's validation set, drawn from the training subjects, ranks networks
poorly. These results bear directly on how wearable sensing studies record subjects and report accuracy.

## Related manuscript and overlap (declaration)

1. **Companion manuscript.** M. Le [and co-authors], "RILA-Net: Recurrence-Plot Layers and Invariant Window Laws for
   Microcontroller-Sized Wearable Activity Recognition, Tested on Pre-Registered Held-Out Datasets", in preparation for
   the IEEE Sensors Journal [status at submission: not yet submitted / submitted, manuscript ID ...]. The letter cites
   it. [We attach it for the reviewers as supplementary material.]

2. **Shared runs.** The letter analyses stored training runs that we produced while developing and evaluating the
   networks of the companion manuscript:
   - set S1: 2,460 runs of ten networks on ten development datasets, three seeds;
   - set S2: 948 runs of four networks on eight held-out datasets, three seeds, from the companion manuscript's
     pre-registered held-out evaluation.

   The companion manuscript reports these networks' accuracies, their paired comparisons, their component
   contrasts and what they learn.

3. **What the letter does not repeat.** The letter reuses the runs for a different question: how the variance of
   person-disjoint macro-F1 divides between test-subject groups, networks, their interactions and training seeds. It
   reports no comparison between the networks. The component contrasts and the analysis of learned parameters were
   removed from the letter and are cited to the companion manuscript.

4. **Prior knowledge before the pre-registration.** The letter's only confirmatory hypothesis (H1) was registered,
   and its analysis code sealed by hash, before any variance analysis of S2. S2's per-dataset means and paired network
   differences were known at that time from the companion work. The letter states this (Section II-D). Its variance
   components were not known.

5. **Different values for a few runs.** The companion manuscript retrained the runs it could not re-execute on its
   machines. The letter analyses the stored records, which it re-executed from their checkpoints. The two manuscripts
   therefore use different values for:
   - three of the 828 runs of the confirmatory test;
   - the seed-2 runs of three networks on one development dataset (DSADS);
   - runs of two datasets that the letter's pre-registered inclusion rule excludes (MotionSense, Daphnet).

   The letter states this in its verification paragraph.

6. **Benchmark results.** The comparison population (S3) and the 30-dataset analysis use the WHAR Arena benchmark
   (arXiv:2606.13194):
   - its 17 published networks, re-run per split with the benchmark's paper-era code;
   - its published table, which the letter quotes and re-analyses as published single runs.

7. **Other related work in preparation.** [A manuscript on deploying wearable-HAR networks on Cortex-M
   microcontrollers, in preparation for the IEEE Embedded Systems Letters, uses the same evaluation harness; authors to
   confirm that it shares no result with this letter.]

Apart from the companion manuscript above, this letter has not been published and is not under consideration
elsewhere [authors to confirm].

## Reproducibility

- **Pre-registration:** `docs/preregistration.md`, frozen on 2026-10-15; sha256
  91487bd455f70451c2be613dc80ba581b8d1d64636f14ac92471e43c6f7cf3c4, as written with CRLF line endings.
- **Repository:** the analysis code, the stored run records, every derived number and the verification gates are in
  the project repository [link, or "available to the reviewers on request"].

[Suggested reviewers: at least two, with affiliations and e-mail addresses.]
[Conflicts of interest: none / ...]

Sincerely,

[Corresponding author, affiliation, ORCID, e-mail]
