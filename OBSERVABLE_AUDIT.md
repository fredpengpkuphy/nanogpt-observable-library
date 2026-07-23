# Observable formula and description audit

Implementation of record: `observable_lib.py`

- Audited: 1821 / 1821 catalog records
- Structural or formula errors after corrections: 0
- Exact and unqualified: 1783
- Correct with an explicit limitation: 38
- Both manifests match the catalog definition set: true

## Limitation groups

- 12 x Formula is exact; T scaling uses 1/T as a full-length reference, not the causal-uniform baseline.
- 12 x Formula is exact; unavailable causal pairs count as zero, giving earlier keys a structural exposure advantage.
- 12 x Correctly defined, but structurally near zero because causal query q=0 has only one valid key.
- 1 x Historical series is absent because the legacy EMA accepted delta's first NaN; runtime now skips non-finite warm-up inputs.
- 1 x Historical definition is structurally degenerate: batch-centering makes the following global mean zero.

The complete per-observable result, including ordinal, canonical id, parsed pipeline, status, findings, and errors, is in `data/observable_audit.json`.

## Validation rules

- canonical-id fields equal source, selector, transforms, reduction, and parsed temporal metadata
- all ids are unique and all operators are registered by the Python implementation
- selector, UI module, layer, and role metadata agree
- attention-, GELU-, matrix-, and effective-rank reductions are attached only to compatible sources
- every catalog reduction has an explicit exact-formula branch
- both run manifests contain exactly the same 1821 definitions as the reference catalog
