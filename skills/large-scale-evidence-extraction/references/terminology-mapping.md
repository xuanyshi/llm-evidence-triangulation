# Online mapping and review

The `both` batch stage automatically queries missing vocabulary fields after saving the source-grounded entities. A failure in one terminology service does not prevent requests to the others. Existing index mappings, including their broader relations and original versions, are preserved.

## Sources and acceptance

| Field | Online source | Acceptance rule |
| --- | --- | --- |
| MeSH | NLM MeSH RDF | Unique exact descriptor label, or exact MSH source synonym in a unique NCIm concept, with the target descriptor/supplementary identifier confirmed by NLM |
| UMLS | NCI Metathesaurus | One active CUI with an exact preferred name or source synonym, with complete search results |
| ICD | NLM Clinical Tables ICD-10-CM, sourced from CMS | Unique exact diagnosis name, or exact ICD10CM source synonym in the uniquely matched NCIm concept, with the code and current target name independently confirmed |

New ICD mappings are specifically **ICD-10-CM**. They must not be described as WHO ICD-10. Supplied index mappings from other ICD versions remain labelled by their original provenance. Some high-level ICD categories are absent from the Clinical Tables search dataset; a proposed parent code that cannot be confirmed stays in review. Never choose an unspecified-complication code just to fill an empty field.

NCI Metathesaurus supplies shared UMLS CUIs; the recorded version describes the **NCIm release**, not a separately verified UMLS release. Only `C` plus seven digits is accepted from the NCIm endpoint. A unique concept must match the queried name or an explicitly listed source synonym. Multiple exact CUIs, truncated searches and differing target-code candidates remain unresolved. Similarity ranking alone cannot select a code. For MeSH/ICD, the target-vocabulary synonym itself must match the original query before a crosswalk is accepted.

Normalization changes Unicode compatibility forms, case and repeated whitespace. It preserves signs, punctuation and digits, so HER2+ and HER2− or type 1 and type 2 diabetes are not merged. All accepted mappings record the code, target label, code system, source URL, retrieval time, matching method, available version, and crosswalk evidence when used. If the service does not expose its release version, the version remains null and retrieval time is retained.

The public source contracts are documented by [NLM MeSH RDF](https://id.nlm.nih.gov/mesh/lookup), [NLM Clinical Tables ICD-10-CM](https://clinicaltables.nlm.nih.gov/apidoc/icd10cm/v3/doc.html), and [NCI's description of NCIm source mappings and CUIs](https://evs.nci.nih.gov/ftp1/NCI_Thesaurus/Neoplasm/About_Core.html).

## Stored decisions

Each entity includes `mapping_checks` for `mesh`, `icd` and `umls`. Status values are `matched`, `index`, `not_found`, `ambiguous`, or `lookup_error`. Review records retain candidate codes/labels and a reason where available. Query records retain the source URL, attempt outcome and response hash. This separates an empty successful search from a service outage.

`mapping_status=mapped` means all vocabulary fields are populated. `mapping_status=needs_review` lists missing fields in `review_vocabularies`, even when another field has already matched. A CUI conflicting with a supplied CUI prevents filling missing fields from that conflicting lookup. Concepts such as age groups, measurement variables or study designs may have no applicable ICD diagnosis; leave them for review rather than inventing disease codes. A failed lookup does not establish that no mapping exists.

## Cache, limits and recovery

Results are stored in `terminology.sqlite3`, keyed by normalized label and resolver version. Per-label locking prevents concurrent studies from issuing the same query repeatedly. Caches persist across process restarts. Successful/ambiguous/no-match searches are retained for 30 days by default; results containing a service error are reused for at most 60 seconds. `--terminology-cache-days` changes the normal cache lifetime.

Terminology request starts are paced independently of model calls, default 120 per minute across this process. Default timeout is 15 seconds and at most two attempts are made for each source request, including target verification. Retryable HTTP errors honor Retry-After up to 60 seconds. Both label queries and crosswalk confirmations consume this rate allowance. Parallel runs in different directories must share the upstream rate allowance deliberately.

Use the original run command with `--retry-mappings` to refresh missing mappings. Each distinct label is refreshed at most once during that process; previously populated fields are preserved. This mode uses saved entities and does not require or call the extraction model. It also supports compatible historical run databases that lack online mapping; the old manifest is archived before the mapping-only upgrade. The evidence table, JSONL, review list and summary are regenerated after completion.

The service sees the entity label or target code being queried, not the study's full abstract. Review files retain the local original mention and quote so ambiguity can be adjudicated in context.
