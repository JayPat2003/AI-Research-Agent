# Evidence Grading and the Demo Trials Catalog

This platform is a general research analyst. The demo knowledge base happens to be cardiometabolic outcomes evidence because that is a high-demand briefing domain for health systems, payers, and medical affairs. Uploading a different corpus (oncology, health policy, finance) does not require changing the agents.

## How to grade claims in a briefing

- Completed, independently published CVOTs outrank marketing slides and small A1c studies for event claims.
- Secondary endpoints and subgroups are hypotheses until replicated.
- Withdrawn or non-marketed agents (for example albiglutide in HARMONY Outcomes) can support biology but not formulary recommendations.
- Ongoing trials (for example SURPASS-CVOT in the demo table) must be labeled ongoing. Never invent a result.

## Structured table `clinical_trials`

The Data Agent can query a demo catalog with columns: trial_name, condition, intervention, drug_class, phase, n_participants, region, status, primary_endpoint, start_year.

Example questions:

- How many completed Phase 3 GLP-1 trials list 3-point MACE as the primary endpoint?
- Which SGLT2 trials in the catalog enroll a heart-failure population?
- What is the largest completed trial by n_participants?

The table is **not** a live ClinicalTrials.gov extract. Counts only reflect the seeded public demo subset. If the user needs a live registry, say so rather than hallucinating additional NCT numbers.

## What this knowledge base is not

It is not a diagnostic engine, not a dosing calculator, and not a substitute for regulatory labeling. The critic agent should fail drafts that tell a patient which drug to start tonight.
