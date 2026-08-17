# GLP-1 versus SGLT2 as Add-on Therapy: Decision Framing

The example comparison this corpus is built to support is:

“Compare GLP-1 receptor agonists and SGLT2 inhibitors for adults with type 2 diabetes and established cardiovascular disease. What do the major outcomes trials show, which populations were studied, and which class is more suitable as the first add-on in a research briefing for a clinical-operations team?”

## What the evidence clusters show

| Axis | SGLT2 inhibitors | GLP-1 receptor agonists |
| --- | --- | --- |
| ASCVD / MACE | Benefit in several CVOTs, strongest in secondary-prevention diabetes cohorts; not uniform across every agent | Benefit in several CVOTs (LEADER, SUSTAIN-6, REWIND, SELECT) |
| Heart-failure hospitalization | Consistent class-level benefit, including dedicated HF trials with and without diabetes | Not the primary evidence base; do not treat as HF disease-modifying substitutes |
| CKD progression | CREDENCE, DAPA-CKD, EMPA-KIDNEY | FLOW (semaglutide) plus metabolic effects; thinner class-wide kidney-outcome catalog |
| Body weight | Modest loss | Larger average loss, especially semaglutide and tirzepatide |
| Route | Oral, once daily | Weekly injection or daily oral semaglutide |
| Persistence issues | Genital infections, volume symptoms, rare DKA | GI intolerance, supply/cost, injection burden |

## Production suitability for a briefing system

For a **clinical-operations or medical-affairs briefing**, not a bedside order:

1. If HF or progressive CKD is the dominant comorbidity, SGLT2 evidence is the default first cluster to present, with GLP-1 as complementary for residual ASCVD/weight/glycemic need.
2. If obesity and ASCVD dominate and HF/CKD are absent or secondary, GLP-1 evidence (including SELECT for non-diabetes obesity with CVD) is the default first cluster.
3. Combination therapy is common in practice; exclusive “winner” language is usually the wrong output. State overlap, sequencing, and evidence gaps.
4. Use trial names, inclusion criteria, and endpoints. Do not pool every GLP-1 or every SGLT2 into one hazard ratio.

A decision-intelligence product should retrieve both trial narratives and the structured `clinical_trials` rows (phase, n, endpoint, status) so the analyst can see which claims are backed by completed CVOTs versus ongoing studies such as SURPASS-CVOT.
