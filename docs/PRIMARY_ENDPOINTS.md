# Primary-endpoint assessment

Use `--mode endpoint-assessment` for a domain-specific **retrospective SFT** dataset:
registration/protocol evidence plus results → assessment of whether each prespecified
primary endpoint was met. The model receives the results. This is not prospective
forecasting or an RL environment, and it does not establish predictive accuracy.

The existing `--mode sft` remains general clinical-trial evidence reading, including
endpoint extraction and evidence sufficiency. Its examples, collection checkpoints,
exports and critic schedule are unchanged. Use a separate environment directory for
the new mode. Do not relabel the general dataset as endpoint assessment. CLI mode
changes on environments with candidate artifacts are refused.

## Start a separate Norwegian environment

From the repository root, create the directory and copy the checked-in spec:

```sh
mkdir -p envs/norwegian-primary-endpoints
cp examples/norwegian-primary-endpoints.spec.json envs/norwegian-primary-endpoints/spec.json
.venv/bin/rl-env-use build envs/norwegian-primary-endpoints --only scout
```

Do not overwrite an existing spec when resuming. The example requires named routes
`local:writer` and `local:critic-thinking` in the ignored `.env`. Each can use a
different model/endpoint/API key. Configure the latter with `enable_thinking=true`
if you want provider reasoning captured. Alternatively select your configured pair:

```sh
.venv/bin/rl-env-use build envs/norwegian-primary-endpoints \
  --writer local:writer --critic local:critic-thinking \
  --language nob --no-localize-diagnostics
```

Providing only one role flag fills both roles; omitting both reuses saved roles.
`new --mode endpoint-assessment` also creates a spec without model calls; `create
--mode endpoint-assessment` plans a spec using the writer first. For this workflow,
start with the example above so the evidence requirements are explicit.

Discovery uses Helsenorge and Norwegian hospital links, NorTrials, NORCRIN/Helse Vest
and the ClinicalTrials.gov Norway query. Norwegian pages are discovery/context;
they do not establish success merely by describing recruitment or hoped-for benefit.
The collector reuses the general mode's permitted-download machinery **inside the
new environment**, including its `sft-frontier.json` and `sft-documents.jsonl`.
Full current registry responses are paired by exact NCT ID; posted primary outcome
objects include their analyses. Registry `RESULT` bibliography links and explicit
publication PMIDs fetch complete indexed Europe PMC abstracts. The critic must
still confirm the publication, population and endpoint match. An abstract is not
claimed to be full text; missing details require fuller evidence or exclusion.

## Supply reviewed historical evidence

Current registration metadata cannot prove what was prespecified before results
were known. This mode does not scrape a private record-history API or bypass robots
restrictions. A curator must identify permitted historical registrations and, where
needed, protocols/SAPs and primary results reports. Automatic discovery alone will
usually produce an evidence queue rather than accepted examples.

Add entries to `endpoint_cases` in the spec. This schema illustration uses placeholder
URLs and identifiers; replace them with actual documents and reviewed facts:

```json
{
  "endpoint_cases": {
    "NCT00000001": {
      "sources": [
        {
          "url": "https://example.invalid/archived-registration.html",
          "role": "registration",
          "version_date": "2018-06-01",
          "first_results_date": "2021-02-15",
          "reviewed_by": "curator-name",
          "verification_note": "Verified the dated archived registration and amendment history against the earliest public results availability."
        },
        {
          "url": "https://example.invalid/prespecified-analysis-plan.pdf",
          "role": "sap",
          "version_date": "2018-07-01",
          "first_results_date": "2021-02-15",
          "reviewed_by": "curator-name",
          "verification_note": "Confirmed this analysis plan predates results and specifies the decision rules."
        },
        {
          "url": "https://example.invalid/primary-results.html",
          "role": "results"
        }
      ],
      "publication_pmids": ["12345678"]
    }
  }
}
```

Allowed roles: `registration`, `protocol`, `sap`, `results`. Downloaded manifest
sources must contain the exact NCT ID. For indexed publications whose abstracts
omit it, `publication_pmids` or the registry bibliography supplies an explicit
identity link, checked by the critic. Do not use this to link unrelated papers.
An archived registration must be identified as `registration`; a protocol or SAP
alone does not replace the registration requirement.

Dates use ISO `YYYY-MM-DD`. The current ClinicalTrials.gov study API cannot be relabeled as a historical source.
A source is marked `prespecification_verified` only when
`reviewed_by` and `verification_note` are nonempty and `version_date` is strictly
before `first_results_date`. **This is a recorded curator attestation, not automatic
proof of historical authenticity.** Review the version, public availability,
amendments, outcome switching and the earliest results disclosure; do not simply
copy the registry's study start date or a later publication date. The critic checks
whether the source actually supports the claimed prespecification and rules.
Conflicting historical versions remain visible to the critic; coverage checks fail
closed when structured versions list different primary endpoints, requiring curation.

Supported sources are HTML, JSON, XML, plain text and text-bearing PDFs. Full raw
responses and SHA-256 checksums are retained. PDF text is extracted from all pages;
check tables, minus signs and reading order manually. Scanned PDFs need reviewed
OCR outside this collector. Empty documents, sources over 10 MB, and extracted
texts over 150,000 characters are rejected rather than silently truncated. These
per-source limits do not guarantee the combined bundle fits a model's context.

After updating the manifest, rebuild the paired sources and then generate:

```sh
.venv/bin/rl-env-use build envs/norwegian-primary-endpoints --only scout
.venv/bin/rl-env-use build envs/norwegian-primary-endpoints --only synth
.venv/bin/rl-env-use export envs/norwegian-primary-endpoints
```

Scout downloads sources but makes no model calls. `synth`/`verify` operate on existing
bundles; repeat scout after changing the evidence manifest. Completed reviews are
checkpointed. Changed evidence, writer/critic routes or language produce a new
candidate key; identical reviewed candidates are reused. Failed calls and structural
rejections can be retried. Before export, raw source checksums/permission provenance
are checked again. `endpoint-reviews.jsonl` supports manual exclusions:

```json
{"id":"candidate-id-from-endpoint-candidates.jsonl","verdict":"reject","reason":"Endpoint/result population mismatch"}
```

The latest review per ID wins. Re-export after adding a rejection; original
candidates are preserved.

## Decisions and acceptance

Each example covers all registered primary endpoints with the exact registered
name, timepoint, population, comparison/direction, metric, hypothesis, success rule,
reported result, label, explanation and exact citations by source ID. Narrative
fields use `language` (ISO 639-3, e.g. `nob`); source names/quotations and schema enums
are preserved. Diagnostics stay English without `--localize-diagnostics`.

Per-endpoint labels are `met`, `not_met`, `insufficient_evidence`. The derived
`endpoint_summary` is `mixed` when fully resolved endpoints have different labels;
it is `insufficient_evidence` if any endpoint remains unresolved. `trial_success`
is a separate field: its definitive labels require a cited, prespecified overall
success rule. A mixed set of findings does not by itself decide overall success.

The writer and critic must check superiority/noninferiority/equivalence, analysis
population, contrast direction, margins, timepoints, estimands, adjusted thresholds,
multiplicity, hierarchy and co-primary rules. No default p<0.05 rule is supplied.
A secondary endpoint cannot substitute for a primary endpoint; no posted results
never means failure; failure to establish noninferiority does not prove inferiority.
See the [FDA multiple-endpoints guidance](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/multiple-endpoints-clinical-trials)
for the distinction between individual findings and study decision rules.

Deterministic gates check exact quotes and source roles, reviewed historical
provenance, decision-field completeness, raw snapshots, and endpoint-name coverage
when historical registration is structured ClinicalTrials.gov JSON. For HTML/PDF
registrations, complete endpoint coverage depends on the independent critic and
curator review. Statistical/clinical semantic correctness is model-reviewed, not
formally proven by these gates. Validate a clinical/statistical expert-reviewed
sample before treating exported labels as gold.

All seven critic checks must pass, and **any `blocking_issues` reject the example**,
including grammar or unsupported elaboration. Corrections are not silently applied.
Raw provider reasoning is saved in each example's separate `critic_reasoning` field;
it is never included in training messages. Reasoning is captured when the endpoint
returns it; this mode does not require it or schedule a later backfill.

`endpoint_include_insufficient=false` (default) queues incomplete registrations or
results without spending model calls, and rejects drafts with unresolved endpoints.
Set it to `true` only for a deliberately labeled uncertainty subset. These examples
still require a citable registered endpoint; the flag does not permit generic
recruitment or unrelated evidence-reading tasks. Report resolved endpoint counts,
uncertainty examples and case/trial groups separately.

## Configuration and artifacts

| Setting | Meaning |
|---|---|
| `target_tasks` | Accepted trial-assessment target; one candidate per evidence bundle, not multiple paraphrases |
| `sft_discovery_limit` | Discovery batch size inherited from the shared collector |
| `endpoint_continuous` | Continue source batches; default false |
| `endpoint_max_batches` | Batch cap per invocation; zero/missing means no cap |
| `endpoint_include_insufficient` | Allow explicitly unresolved endpoint examples; default false |
| `endpoint_model_timeout` | Seconds per writer/critic call; default 1200 |
| `endpoint_critic_max_tokens` | Critic output ceiling; default 32768; writer ceiling 16384 |
| `endpoint_cases` | Per-NCT reviewed registration/protocol/SAP/results manifest |

Collection stops at the target, source exhaustion, batch cap or a requested single
stage. Source exhaustion can still leave evidence gaps requiring curation. Repeated
builds with changed evidence can create multiple versions for one trial: deduplicate
by NCT ID/evidence version and split train/validation/test by NCT ID, not row.

| Artifact | Purpose |
|---|---|
| `endpoint-bundles.jsonl` | Paired full evidence and provenance per trial |
| `endpoint-source-errors.jsonl` | Blocked, unavailable, malformed or oversized sources |
| `endpoint-needs-evidence.jsonl` | Trials missing historical registration or results |
| `endpoint-candidates.jsonl` | Drafts, exact citations, critic checks, rejections/errors and reasoning |
| `endpoint-reviews.jsonl` | Optional manual reject/accept overrides; acceptance cannot bypass model/source gates |
| `endpoint-status.json` | Current stage and progress |
| `dataset-endpoints.jsonl` | Messages, structured assessment, labels and separate critic reasoning |
| `dataset-endpoints.chat.jsonl` | Messages-only trainer export; assistant emits assessment JSON |
| `dataset-endpoints.records.jsonl` | Full records plus source provenance |
| `dataset-endpoints.audit.json` | Per-example provenance and critic verdicts |
| `source-captures/`, `robots-captures/` | Raw downloaded evidence and permission records; keep with dataset |

Robots checks and redirect checks remain enforced. The existing documented public
ClinicalTrials.gov v2 API exception is retained with its NLM authorization record;
it does not authorize private history endpoints, other sources or arbitrary scraping.
The EU CTIS interactive portal is not a supported automated source in this collector.
A zero-example result accompanied by evidence gaps is an incomplete run, not a
successful endpoint dataset.
