#!/usr/bin/env python3
"""Build an auditable literature database from BibTeX and verified additions."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BIB_PATH = ROOT / "references.bib"
MANUAL_PATH = ROOT / "literature" / "manual_entries.json"
OUT_DIR = ROOT / "literature"
CACHE_PATH = OUT_DIR / "metadata_cache.json"

FIELDS = [
    "citation_key", "source_group", "cited_in_manuscript", "citation_locations",
    "entry_type", "authors", "title", "year", "venue", "volume", "number",
    "pages", "doi", "document_url", "landing_page_url", "abstract",
    "abstract_type", "abstract_source_url", "metadata_source",
    "verification_status", "topics", "notes", "last_verified",
]


def latex_to_text(value: str) -> str:
    replacements = {
        r"\&": "&", r"\%": "%", r"\_": "_", r"\url": "",
        r"{\'a}": "á", r"{\'e}": "é", r"{\'i}": "í",
        r"{\'o}": "ó", r"{\'u}": "ú", r"{\'A}": "Á",
        r"{\'E}": "É", r"{\'I}": "Í", r"{\'O}": "Ó",
        r"{\'U}": "Ú", r"{\~n}": "ñ", r"{\~N}": "Ñ",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    value = value.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", value).strip()


def parse_bibtex(path: Path) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8")
    entries: list[dict[str, str]] = []
    start_re = re.compile(r"@(?P<type>\w+)\s*\{\s*(?P<key>[^,\s]+)\s*,")
    pos = 0
    while match := start_re.search(text, pos):
        depth, i = 1, match.end()
        while i < len(text) and depth:
            depth += (text[i] == "{") - (text[i] == "}")
            i += 1
        body = text[match.end(): i - 1]
        entry = {"entry_type": match.group("type"), "citation_key": match.group("key")}
        entry.update(parse_bib_fields(body))
        entries.append(entry)
        pos = i
    return entries


def parse_bib_fields(body: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    i = 0
    while i < len(body):
        match = re.search(r"([A-Za-z][\w-]*)\s*=", body[i:])
        if not match:
            break
        key = match.group(1).lower()
        i += match.end()
        while i < len(body) and body[i].isspace():
            i += 1
        if i >= len(body):
            break
        if body[i] in "{\"":
            opener = body[i]
            closer = "}" if opener == "{" else "\""
            depth = 1
            j = i + 1
            while j < len(body) and depth:
                if opener == "{":
                    depth += (body[j] == "{") - (body[j] == "}")
                elif body[j] == closer and body[j - 1] != "\\":
                    depth = 0
                j += 1
            value = body[i + 1:j - 1]
            i = j
        else:
            j = body.find(",", i)
            j = len(body) if j < 0 else j
            value, i = body[i:j], j
        fields[key] = latex_to_text(value.strip())
    return fields


def normalize_bib_entry(entry: dict[str, str]) -> dict[str, str]:
    venue = (
        entry.get("journal") or entry.get("booktitle")
        or entry.get("institution") or entry.get("howpublished") or ""
    )
    url = entry.get("url", "")
    doi = entry.get("doi", "").replace("https://doi.org/", "")
    return {
        "citation_key": entry["citation_key"],
        "source_group": "manuscript_bibliography",
        "entry_type": entry.get("entry_type", ""),
        "authors": entry.get("author", ""),
        "title": entry.get("title", ""),
        "year": entry.get("year", ""),
        "venue": venue,
        "volume": entry.get("volume", ""),
        "number": entry.get("number", ""),
        "pages": entry.get("pages", ""),
        "doi": doi,
        "document_url": url,
        "landing_page_url": f"https://doi.org/{doi}" if doi else url,
        "topics": "",
        "notes": entry.get("note", ""),
    }


def citation_locations(root: Path) -> dict[str, list[str]]:
    locations: dict[str, list[str]] = {}
    pattern = re.compile(r"\\cite\w*\s*(?:\[[^\]]*\]\s*)*\{([^}]+)\}")
    for path in sorted(root.rglob("*.tex")):
        if any(part in {"build", "tmp", "output"} for part in path.parts):
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for match in pattern.finditer(line):
                for key in match.group(1).split(","):
                    locations.setdefault(key.strip(), []).append(
                        f"{path.relative_to(root).as_posix()}:{lineno}"
                    )
    return locations


def get_json(url: str, attempts: int = 2, timeout: int = 12) -> dict:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "AI-Future-Paper literature updater (academic use)"},
    )
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            if attempt + 1 == attempts:
                raise
            time.sleep(1.5 * (attempt + 1))
    return {}


def invert_abstract(index: dict[str, list[int]] | None) -> str:
    if not index:
        return ""
    words: list[tuple[int, str]] = []
    for word, positions in index.items():
        words.extend((position, word) for position in positions)
    return " ".join(word for _, word in sorted(words))


def fetch_openalex(doi: str) -> dict[str, str]:
    encoded = urllib.parse.quote(f"https://doi.org/{doi}", safe=":/")
    data = get_json(f"https://api.openalex.org/works/{encoded}")
    primary = data.get("primary_location") or {}
    best_oa = data.get("best_oa_location") or {}
    source = primary.get("source") or {}
    authors = "; ".join(
        item.get("author", {}).get("display_name", "")
        for item in data.get("authorships", [])
        if item.get("author", {}).get("display_name")
    )
    abstract = invert_abstract(data.get("abstract_inverted_index"))
    return {
        "openalex_id": data.get("id", ""),
        "title": data.get("title", ""),
        "authors": authors,
        "year": str(data.get("publication_year") or ""),
        "venue": source.get("display_name", ""),
        "abstract": abstract,
        "abstract_type": "openalex_indexed_abstract" if abstract else "",
        "abstract_source_url": data.get("id", "") if abstract else "",
        "document_url": best_oa.get("pdf_url") or primary.get("pdf_url") or "",
        "landing_page_url": primary.get("landing_page_url") or f"https://doi.org/{doi}",
        "metadata_source": "OpenAlex",
    }


def clean_markup(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", value or "")
    return html.unescape(re.sub(r"\s+", " ", value)).strip()


def fetch_crossref_abstract(doi: str) -> dict[str, str]:
    encoded = urllib.parse.quote(doi, safe="")
    data = get_json(f"https://api.crossref.org/works/{encoded}").get("message", {})
    abstract = clean_markup(data.get("abstract", ""))
    return {
        "abstract": abstract,
        "abstract_type": "crossref_registered_abstract" if abstract else "",
        "abstract_source_url": f"https://api.crossref.org/works/{encoded}" if abstract else "",
    }


def fetch_semantic_scholar_abstract(doi: str) -> dict[str, str]:
    encoded = urllib.parse.quote(f"DOI:{doi}", safe=":")
    fields = "title,abstract,url,openAccessPdf"
    data = get_json(
        f"https://api.semanticscholar.org/graph/v1/paper/{encoded}?fields={fields}"
    )
    abstract = (data.get("abstract") or "").strip()
    pdf = (data.get("openAccessPdf") or {}).get("url") or ""
    return {
        "abstract": abstract,
        "abstract_type": "semantic_scholar_abstract" if abstract else "",
        "abstract_source_url": data.get("url", "") if abstract else "",
        "document_url": pdf,
    }


def merge_nonempty(base: dict[str, str], extra: dict[str, str], overwrite: bool = False) -> None:
    for key, value in extra.items():
        if value and (overwrite or not base.get(key)):
            base[key] = str(value)


def enrich_entry(entry: dict[str, str], cache: dict, refresh: bool) -> dict[str, str]:
    doi = entry.get("doi", "").lower()
    if not doi:
        entry.setdefault("verification_status", "manual_url")
        return entry
    cached = cache.get(doi, {})
    if refresh or not cached:
        fetched: dict[str, str] = {}
        try:
            fetched = fetch_openalex(doi)
        except Exception as exc:  # Continue with other providers and record the failure.
            fetched["openalex_error"] = type(exc).__name__
        if not fetched.get("abstract"):
            try:
                merge_nonempty(fetched, fetch_crossref_abstract(doi))
            except Exception as exc:
                fetched["crossref_error"] = type(exc).__name__
        cached = fetched
        cache[doi] = cached
        time.sleep(0.1)
    merge_nonempty(entry, cached)
    entry["verification_status"] = (
        "doi_metadata_matched" if cached.get("title") else "doi_unresolved"
    )
    return entry


def finalize_entry(entry: dict[str, str], locations: dict[str, list[str]]) -> dict[str, str]:
    key = entry["citation_key"]
    refs = locations.get(key, [])
    entry["cited_in_manuscript"] = "yes" if refs else "no"
    entry["citation_locations"] = "; ".join(refs)
    if not entry.get("document_url"):
        entry["document_url"] = entry.get("landing_page_url", "")
    if not entry.get("landing_page_url") and entry.get("doi"):
        entry["landing_page_url"] = f"https://doi.org/{entry['doi']}"
    entry.setdefault("abstract", "")
    if not entry.get("abstract"):
        entry["abstract_type"] = "unavailable"
        entry["abstract_source_url"] = ""
    elif not entry.get("abstract_type"):
        entry["abstract_type"] = "unavailable"
    entry.setdefault("abstract_source_url", "")
    entry.setdefault("metadata_source", "manual")
    entry.setdefault("verification_status", "manual")
    entry.setdefault("topics", "")
    entry.setdefault("notes", "")
    entry["last_verified"] = date.today().isoformat()
    return {field: str(entry.get(field, "")) for field in FIELDS}


def write_csv(entries: list[dict[str, str]]) -> None:
    with (OUT_DIR / "literature_database.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(entries)


def write_json(entries: list[dict[str, str]]) -> None:
    payload = {
        "generated_on": date.today().isoformat(),
        "record_count": len(entries),
        "records": entries,
    }
    (OUT_DIR / "literature_database.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_html(entries: list[dict[str, str]]) -> None:
    data = json.dumps(entries, ensure_ascii=False).replace("</", "<\\/")
    template = """<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Literatura — AI Future Paper</title>
<style>
:root{--ink:#17231d;--muted:#607067;--line:#dce4df;--paper:#f7f9f7;--card:#fff;
--accent:#176b4d;--tag:#e7f3ed}*{box-sizing:border-box}body{margin:0;background:var(--paper);
color:var(--ink);font:16px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
header{background:#123d2f;color:white;padding:2.4rem max(5vw,1rem)}
header h1{margin:0 0 .35rem;font:clamp(2rem,5vw,3.5rem)/1.1 Georgia,serif}
header p{margin:0;color:#dbeae3;max-width:65rem}.controls{position:sticky;top:0;z-index:2;
display:grid;grid-template-columns:2fr 1fr 1fr;gap:.7rem;padding:1rem max(5vw,1rem);
background:#f7f9f7ef;border-bottom:1px solid var(--line);backdrop-filter:blur(8px)}
input,select{width:100%;padding:.8rem;border:1px solid #aebbb4;border-radius:.45rem;
background:white;font:inherit}.summary{padding:1rem max(5vw,1rem) 0;color:var(--muted)}
main{display:grid;gap:1rem;padding:1rem max(5vw,1rem) 3rem}article{background:var(--card);
border:1px solid var(--line);border-radius:.65rem;padding:1.2rem;box-shadow:0 2px 10px #163c2910}
h2{font:1.35rem/1.25 Georgia,serif;margin:.2rem 0}.meta{color:var(--muted);margin:.3rem 0 .8rem}
.abstract{max-width:78rem}.tags,.links{display:flex;gap:.4rem;flex-wrap:wrap}
.tag{background:var(--tag);color:#14583f;border-radius:999px;padding:.15rem .55rem;font-size:.78rem}
a{color:var(--accent);font-weight:650}.links{gap:1rem;margin-top:.8rem}.empty{text-align:center;color:var(--muted)}
@media(max-width:720px){.controls{grid-template-columns:1fr;position:static}}
</style></head><body><header><h1>Literatura del proyecto</h1>
<p>Referencias del manuscrito y trabajos adicionales citados durante la conversación.
Cada registro distingue un abstract o resumen de la fuente de una síntesis editorial.</p></header>
<section class="controls"><input id="q" type="search" placeholder="Buscar autor, título, abstract, tema…">
<select id="group"><option value="">Todos los conjuntos</option>
<option value="manuscript_bibliography">Bibliografía del manuscrito</option>
<option value="conversation_addition">Adiciones de la conversación</option></select>
<select id="atype"><option value="">Todos los tipos de resumen</option>
<option value="abstract">Abstract o resumen de la fuente</option><option value="editorial_summary">Resumen editorial</option>
<option value="unavailable">No disponible</option></select></section>
<div class="summary" id="summary"></div><main id="results"></main>
<script>const records=__DATA__;
const esc=s=>String(s||"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}[c]));
const isEditorial=x=>x.abstract_type==="editorial_summary";
const isAbstract=x=>x.abstract_type&&!["editorial_summary","unavailable"].includes(x.abstract_type);
function render(){const q=document.querySelector("#q").value.toLowerCase();
const g=document.querySelector("#group").value,a=document.querySelector("#atype").value;
const rows=records.filter(x=>{const hay=Object.values(x).join(" ").toLowerCase();
return(!q||hay.includes(q))&&(!g||x.source_group===g)&&(!a||
(a==="editorial_summary"&&isEditorial(x))||(a==="abstract"&&isAbstract(x))||
(a==="unavailable"&&x.abstract_type==="unavailable"));});
document.querySelector("#summary").textContent=rows.length+" de "+records.length+" referencias";
document.querySelector("#results").innerHTML=rows.length?rows.map(x=>"<article><div class='tags'>"+
"<span class='tag'>"+esc(x.year)+"</span><span class='tag'>"+esc(x.source_group)+"</span>"+
"<span class='tag'>"+esc(x.abstract_type)+"</span></div><h2>"+esc(x.title)+"</h2>"+
"<div class='meta'>"+esc(x.authors)+" · "+esc(x.venue)+"</div><div class='abstract'>"+
(esc(x.abstract)||"<em>Abstract no disponible.</em>")+"</div><div class='links'>"+
(x.document_url?"<a href='"+esc(x.document_url)+"' target='_blank'>Documento</a>":"")+
(x.landing_page_url&&x.landing_page_url!==x.document_url?"<a href='"+esc(x.landing_page_url)+"' target='_blank'>Ficha</a>":"")+
(x.abstract_source_url?"<a href='"+esc(x.abstract_source_url)+"' target='_blank'>Fuente del abstract</a>":"")+
"</div></article>").join(""):"<div class='empty'>No hay resultados para estos filtros.</div>"}
["q","group","atype"].forEach(id=>document.querySelector("#"+id).addEventListener("input",render));render();
</script></body></html>"""
    (OUT_DIR / "literature_browser.html").write_text(
        template.replace("__DATA__", data), encoding="utf-8"
    )


def validation_report(
    entries: list[dict[str, str]], locations: dict[str, list[str]]
) -> dict:
    keys = {entry["citation_key"] for entry in entries}
    cited_keys = set(locations)
    dois: dict[str, list[str]] = {}
    for entry in entries:
        if entry["doi"]:
            dois.setdefault(entry["doi"].lower(), []).append(entry["citation_key"])
    return {
        "generated_on": date.today().isoformat(),
        "record_count": len(entries),
        "manuscript_bibliography_count": sum(
            entry["source_group"] == "manuscript_bibliography" for entry in entries
        ),
        "conversation_addition_count": sum(
            entry["source_group"] == "conversation_addition" for entry in entries
        ),
        "source_abstract_count": sum(
            entry["abstract_type"] in {
                "openalex_indexed_abstract",
                "crossref_registered_abstract",
                "source_abstract",
            }
            for entry in entries
        ),
        "source_summary_count": sum(
            entry["abstract_type"] == "source_summary" for entry in entries
        ),
        "editorial_summary_count": sum(
            entry["abstract_type"] == "editorial_summary" for entry in entries
        ),
        "abstract_unavailable_count": sum(
            entry["abstract_type"] == "unavailable" for entry in entries
        ),
        "missing_document_url": [
            entry["citation_key"] for entry in entries if not entry["document_url"]
        ],
        "cited_keys_missing_from_database": sorted(cited_keys - keys),
        "duplicate_dois": {
            doi: doi_keys for doi, doi_keys in dois.items() if len(doi_keys) > 1
        },
        "unresolved_dois": [
            entry["citation_key"]
            for entry in entries
            if entry["doi"] and entry["verification_status"] == "doi_unresolved"
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh", action="store_true",
        help="Refresh cached DOI metadata from OpenAlex and Crossref.",
    )
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manual = json.loads(MANUAL_PATH.read_text(encoding="utf-8"))
    cache = (
        json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        if CACHE_PATH.exists() else {}
    )
    locations = citation_locations(ROOT)
    entries = [normalize_bib_entry(item) for item in parse_bibtex(BIB_PATH)]
    entries.extend(manual.get("additions", []))
    overrides = manual.get("overrides", {})
    finished: list[dict[str, str]] = []
    for entry in entries:
        entry = enrich_entry(dict(entry), cache, args.refresh)
        merge_nonempty(entry, overrides.get(entry["citation_key"], {}), overwrite=True)
        finished.append(finalize_entry(entry, locations))
    finished.sort(key=lambda item: (-int(item["year"] or 0), item["authors"], item["title"]))
    write_csv(finished)
    write_json(finished)
    write_html(finished)
    CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = validation_report(finished, locations)
    (OUT_DIR / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["cited_keys_missing_from_database"] or report["duplicate_dois"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
