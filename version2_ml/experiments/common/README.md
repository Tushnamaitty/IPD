# `experiments/common` — shared A1/A4/A7 utilities (aggregate-only)

New utilities introduced by the approved rework plan (Phase 0c), coordinated
with the handoff Section 8 output contract. No real data, identifiers,
predictions or row-level values are stored here.

| Module | Purpose |
|---|---|
| `aggregate_schema.py` | Locked 31-column output schema; absent values written as `not_applicable`; disclosure suppression helper (<10). |
| `manifests.py` | SHA-256 feature-manifest and split-manifest hashes; row-order independent. |
| `checksums.py` | Privacy-safe row-membership digests proving matched comparisons (digests only, memory only). |
| `synthetic.py` | Deterministic synthetic NFHS-shaped fixtures for tests (no real data). |

Tests: `tests/test_common_utils.py` (synthetic only).

Rules honored here: aggregate outputs only; no small-cell releases (<10);
no survey-design inference claims; no absolute paths or identifiers written.
