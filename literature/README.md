# Literature database

This directory contains the literature inventory used by the project. It covers:

1. every entry currently present in references.bib; and
2. conversation-only references recorded in manual_entries.json, including the
   antecedents discussed for finite-time singularities and variable-elasticity
   production functions.

## Files

- literature_browser.html: searchable local interface. Open it in any browser.
- literature_database.csv: UTF-8 CSV for Excel, R, Stata, or Python.
- literature_database.json: machine-readable version of the same records.
- manual_entries.json: conversation additions and audited metadata overrides.
- metadata_cache.json: cached DOI metadata for reproducible offline rebuilds.
- validation_report.json: coverage, duplicate, link, and abstract-status checks.
- ../scripts/build_literature_database.py: updater and validator.

## Abstract policy

The field abstract_type is essential:

- openalex_indexed_abstract or crossref_registered_abstract: abstract recovered
  from an academic metadata service.
- source_abstract: abstract taken from the work's official page.
- source_summary: summary supplied by the issuing organization when the item has
  no conventional academic abstract.
- editorial_summary: synopsis written for this project because the source does
  not publish an abstract. It must not be quoted as author-written text.
- unavailable: no abstract or responsible summary is available.

The database never silently presents an editorial synopsis as an original abstract.
abstract_source_url records the provenance of the text, and document_url points to a
PDF when a stable one is available.

## Updating

The script uses only the Python standard library. Run:

    python scripts/build_literature_database.py

This rebuilds every output from the checked-in cache and therefore works offline.
To refresh DOI metadata from OpenAlex and Crossref, run:

    python scripts/build_literature_database.py --refresh

Review changes to manual_entries.json, metadata_cache.json, and
validation_report.json before committing. A successful run exits with an error if a
citation used in the manuscript is absent from the database or if duplicate DOI
records are detected.

## Adding a reference mentioned in conversation

Add a record to the additions array in manual_entries.json. Use a stable
citation_key, include a DOI when one exists, and set source_group to
conversation_addition. If the source has no abstract, provide a concise synopsis
and label it editorial_summary.
