# Norwegian administrative source collection

`collect-sources` builds an evidence corpus without any writer, critic, browser
agent or training-generation calls. It can run alongside an existing model-bound
clinical pipeline. This phase collects sources, not accepted SFT examples.

## SimpleAudit source review

Source inventory inspected at SimulaMet/SimpleAudit commit
[`6636ae8883033e1acaaf326a975e826e7c7bee37`](https://github.com/SimulaMet/SimpleAudit/tree/6636ae8883033e1acaaf326a975e826e7c7bee37).
The relevant scenario modules and their accompanying READMEs identify these domains:

| SimpleAudit pack | Primary sources to collect | Coverage |
|---|---|---|
| `nav_aap` | nav.no | AAP, eligibility, deadlines, G and dated rates |
| `skatteetaten`, `skatteetaten_legitimasjon` | skatteetaten.no | Tax, deductions, appeals, identity documents, citizenship and service/channel exceptions |
| `helfo` | helfo.no; related patient guidance at helsenorge.no | Co-payments, exemption cards, prescription reimbursement and routing |
| `lanekassen` | lanekassen.no; laws/regulations at lovdata.no | Loans/grants, appeals, income/assets, debt relief and academic-year changes |
| `toll_reisegodskvote` | toll.no; underlying regulations | Travel duration, age/residency and transport-personnel exceptions |
| `arbeidstilsynet_arbeidstid` | arbeidstilsynet.no; arbeidsmiljøloven | Working time, age groups, exempt positions and rest requirements |
| `nb_kryss_ordning` | nb.no/tjenester | ISBN/ISSN/ISMN and legal deposit; closely related schemes with different rules |
| `ung` | ung.no | Youth information; this collection uses public articles and excludes `/oss/` Q&A |

Direct pack citations seed the starter spec. Additional NAV benefit pages,
Lånekassen appeal/conversion guidance, Helsenorge patient guidance, public laws and
Ung article hubs extend those seeds; not every seed is an explicit SimpleAudit URL.
The runtime `simpleaudit-source-references.json` records the exact pack/file/line
for directly extracted URLs. It does not copy scenario questions or expected answers.
The benchmark remains separate from training collection; source overlap still needs
to be disclosed when evaluating against SimpleAudit.

The scenario READMEs document disagreements between administrative guidance and
underlying legislation. The National Library pack also records inconsistent copy
counts within a page. These are reasons to retain complete dated snapshots and
source identity, not silently collapse all rules into one authoritative answer.
Do not propagate historical scenario rates as current facts; obtain the current
source and the applicable year/version independently.

UDI, Altinn, Sivilombudet, Datatilsynet and municipal guidance are possible later
extensions. They are not all identified by the above packs and are not part of the
initial active allowlist. Add them deliberately when the task scope is defined.

## Start and resume

From the repository root, for a new environment:

```sh
mkdir -p envs/norwegian-administrative-sources
cp examples/norwegian-administrative-sources.spec.json envs/norwegian-administrative-sources/spec.json
.venv/bin/rl-env-use collect-sources envs/norwegian-administrative-sources
```

Do not copy over an existing spec when resuming. Re-run the final command to resume
the frontier. The environment must have `mode=administrative-sources`; it cannot
reuse a clinical dataset's spec. It is not a generation mode for `create`/`build`.
No model endpoint or API key is needed for collection. Source text remains in its
original language; `language=nob` records the intended future dataset language.

The checked-in configuration starts from NAV, Skatteetaten, Helfo, Lånekassen,
Tolletaten, Arbeidstilsynet, NB, Lovdata, Helsenorge and Ung. It follows same-allowlist
links up to depth two, caps total attempted URLs at 2,000 and each agency at 250,
and waits at least one second between requests per origin (or a longer robots delay).
These are collection bounds, not expected example counts. Seeds are attempted before
discovered links. An agency whose cap is reached keeps its pending URLs; raising the
cap and rerunning resumes them. Both errors and robots refusals count toward caps.

`administrative_sources` maps bare domain names to `paths` and `seeds`. Exact hosts
and their `www.` aliases are accepted, with slash-boundary path checks. Redirects
are checked for scope and robots permission before following. HTTPS is required;
credentials, query URLs, login/search paths, obvious binary assets and youth Q&A
paths are excluded. No logged-in or personal-record APIs are used. Ung article
coverage excludes its real-person Q&A; any future Q&A collection needs a separate
scope/reuse review.

Robots permission is checked and captured before downloading. A 404/410 robots
response is recorded as no policy file; unavailable policies fail closed. A UTF-8
BOM is ignored when parsing rules so it cannot erase a site's first User-agent group.
No institutional override is inferred from the user's National Library affiliation.
An agency-specific agreement or supported export can be added later with its actual
scope and provenance; collection currently uses the ordinary robots gate.

HTTP 429/5xx retry/backoff behavior is inherited from `RobotsGuard`. Blocked or failed
URLs remain in the audit and are not repeatedly retried on resume. To retry after a
policy or transient failure is resolved, retain an audit copy, remove the selected
row from `admin-collection.jsonl`, and add that URL to the configured seeds. Do not
remove genuine robots refusals merely to force retries.

## Evidence artifacts

| Artifact | Contents |
|---|---|
| `admin-documents.jsonl` | Full extracted evidence, title, declared language/dates, URLs, hashes and download provenance |
| `admin-collection.jsonl` | Every attempted URL, collection/block/error status, discovery parent and depth |
| `admin-frontier.json` | Pending URLs and crawl depth; survives interruption |
| `admin-status.json` | Current URL, counts by status and successful documents by agency |
| `source-captures/` | Complete raw permitted HTML/PDF responses |
| `robots-captures/`, `robots-audit.jsonl` | Policy snapshots, permission decisions and crawl delays |
| `simpleaudit-source-references.json` | Optional source-discovery record created for the current run |

HTML extraction keeps the main/article/body text, removing navigation, scripts,
forms and explicitly hidden elements. It does not render JavaScript or retrieve
hidden accordion content. Dynamic/incomplete documents require a supported content
API or a later rendering adapter. Text-bearing PDFs are extracted across all pages;
scanned PDFs need OCR, and tables/minus signs need verification. Documents under
300 or over 150,000 extracted characters are excluded; raw responses over 10 MB are
rejected. No evidence text is truncated to make it pass the bounds.

Declared page dates are stored without being equated to rule-effective dates.
`captured_at` proves when we fetched a snapshot, not when its rules took effect.
Keep tax year, academic year, effective date, population, nationality, age, service,
channel and scheme distinctions explicit in the later task pipeline. Determining
these fields requires source interpretation and checking, not a guessed date.

`reuse_status=not_assessed` is recorded separately from robots/download permission.
Do not treat robots allowance as permission to publish full source copies or train
on them. The later administrative writer/critic and SFT export adapter are not part
of this source-only phase. Inspect corpus quality, resolve access/reuse questions
and define task schemas before generation.
