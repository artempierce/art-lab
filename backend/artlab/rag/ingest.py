"""
rag/ingest.py — puts documents into the knowledge base. A command you run, not part of the chat.

Run it:
    cd backend && uv run python -m artlab.rag.ingest                          # the defaults: knowledge/ + 2 docs
    cd backend && uv run python -m artlab.rag.ingest ../notes/ https://example.com/guide

For each source (a local .md/.txt file, or a web page):

    1. load    read the file, or fetch the page safely (rag/web.py), as plain text
    2. skip?   fingerprint the text (SHA-256); if the knowledge base already holds this exact text,
               skip the rest — re-running ingest only re-embeds what changed
    3. split   cut into chunks of ~800 characters, each labelled with the heading it sits under
    4. scan    run the input guard's injection rules on every chunk. A match is stored but *flagged*:
               it shows in the report, and search never returns it
    5. store   embed every chunk locally and save it to Chroma, replacing that source's old chunks

Then, for folders and files you named, anything that disappeared from disk is removed from the
knowledge base too, so it mirrors your files.

Why flagged chunks are kept rather than dropped: you can see exactly what the scanner caught (and
spot false positives) in the report and in the database. Search skips them either way.
"""

import argparse
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from artlab.config import DEFAULT_SOURCES, REPO_ROOT
from artlab.guards.input import find_injection
from artlab.rag.knowledge import KnowledgeBase
from artlab.rag.web import fetch_page

# ~800 characters is about 200 tokens: enough to hold a rule together with its context, small enough
# that a search returns just the relevant part of a document rather than all of it.
CHUNK_SIZE = 800
# Neighbouring chunks share 100 characters, so a sentence cut at a chunk boundary is whole in one of them.
CHUNK_OVERLAP = 100
FILE_TYPES = {".md", ".txt"}

# Injection rules NOT run on documents. "fake-tags" catches our own delimiter tags (like
# <untrusted_retrieval>). In a chat message that matters, because your text is never wrapped. In a
# document it doesn't: every chunk is wrapped when the agent reads it, and any such tag inside is
# escaped then (tools/untrusted.py), so it can't break out. Running the rule here only produced false
# alarms — our own docs *describe* these tags (the first real ingest flagged 4 chunks of architecture.md).
DOCUMENT_SKIP_RULES = frozenset({"fake-tags"})


@dataclass
class Document:
    """One source, loaded as plain text and ready to split.

    source    its name in the knowledge base: a repo-relative path, or the URL
    kind      "file" or "web"
    markdown  True when the text has markdown headings to split on (.md files, and web pages,
              whose headings web.py converts to markdown)
    """

    source: str
    title: str
    text: str
    kind: str
    markdown: bool


@dataclass
class IngestReport:
    """What one ingest run did. Printed at the end of the command; also what the tests check."""

    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # unchanged since the last run
    removed: list[str] = field(default_factory=list)  # deleted from disk, so removed from the knowledge base
    chunks: int = 0  # chunks embedded and stored in this run
    flagged: list[tuple[str, str, str]] = field(default_factory=list)  # (source, heading, rule)
    errors: list[tuple[str, str]] = field(default_factory=list)  # (source, what went wrong)


def source_name(path: Path) -> str:
    """A file's name in the knowledge base: relative to the repo when inside it (short and stable across
    machines), otherwise its absolute path. Example: /Users/sol/art-lab/knowledge/a.md -> "knowledge/a.md"."""
    path = path.resolve()
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


def find_files(paths: list[Path]) -> list[Path]:
    """Expand the given files and folders into a sorted list of .md / .txt files (folders are searched recursively)."""
    files: set[Path] = set()
    for path in paths:
        if path.is_dir():
            files.update(p for p in path.rglob("*") if p.suffix in FILE_TYPES and p.is_file())
        elif path.suffix in FILE_TYPES and path.is_file():
            files.add(path)
    return sorted(files)


def load_file(path: Path) -> Document:
    """Read a local file. Its title is its first "# " heading if it has one, otherwise the file name."""
    text = path.read_text(encoding="utf-8")
    title = next((line[2:].strip() for line in text.splitlines() if line.startswith("# ")), path.stem)
    return Document(source_name(path), title, text, kind="file", markdown=path.suffix == ".md")


def split(doc: Document) -> list[tuple[str, str]]:
    """Cut a document into chunks. Returns [(heading, chunk_text), ...].

    Steps:
      1. Markdown is first cut at its headings, so no chunk mixes two sections. Each section remembers
         its "## / ###" headings, joined like "Titles" or "Travel › Hotels".
      2. Sections longer than CHUNK_SIZE are cut again, preferring paragraph, then line, then word breaks.
      3. Every chunk's text starts with "Document title › Heading" — a *contextual header*. A chunk that
         only says "At most 2 per video" is hard to find; with "Sponsorship & Disclosure Policy › What we
         accept" on top, a question about sponsors lands on it.
    """
    if doc.markdown:
        sections = MarkdownHeaderTextSplitter([("#", "h1"), ("##", "h2"), ("###", "h3")]).split_text(doc.text)
        parts = [(" › ".join(v for k, v in s.metadata.items() if k != "h1"), s.page_content) for s in sections]
    else:
        parts = [("", doc.text)]

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = []
    for heading, body in parts:
        for piece in splitter.split_text(body):
            label = " › ".join(p for p in (doc.title, heading) if p)
            chunks.append((heading, f"{label}\n\n{piece}"))
    return chunks


def ingest_document(doc: Document, kb: KnowledgeBase, known: dict[str, dict], report: IngestReport) -> None:
    """Steps 2–5 for one loaded document: skip if unchanged, else split, scan, embed and store it."""
    # 2. Unchanged since last time? Then there's nothing to do.
    content_hash = hashlib.sha256(doc.text.encode()).hexdigest()
    previous = known.get(doc.source)
    if previous and previous["content_hash"] == content_hash:
        report.skipped.append(doc.source)
        return

    # 3 + 4. Split, and scan every chunk with the same rules the input guard uses on chat messages.
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    records = []
    for index, (heading, text) in enumerate(split(doc)):
        rule = find_injection(text, skip=DOCUMENT_SKIP_RULES) or ""
        if rule:
            report.flagged.append((doc.source, heading or doc.title, rule))
        metadata = {
            "source": doc.source, "title": doc.title, "heading": heading, "kind": doc.kind,
            "chunk_index": index, "content_hash": content_hash, "flagged": rule, "ingested_at": now,
        }
        records.append((f"{doc.source}::{index}", text, metadata))

    # 5. Embed and store, replacing whatever this source had before.
    kb.replace_source(doc.source, records)
    (report.updated if previous else report.added).append(doc.source)
    report.chunks += len(records)


def ingest(targets: list[str | Path], kb: KnowledgeBase) -> IngestReport:
    """Ingest files, folders and URLs into `kb`, then remove files that no longer exist on disk.

    Args:
        targets: local paths (files or folders) and/or http(s) URLs, in any mix.
    """
    report = IngestReport()
    known = kb.sources()  # what's stored already, read once for the whole run

    urls = [str(t) for t in targets if str(t).startswith(("http://", "https://"))]
    paths = [Path(t) for t in targets if str(t) not in urls]

    # Local files. One bad file (e.g. not valid UTF-8) is reported and skipped; the rest still ingest.
    files = find_files(paths)
    for path in files:
        try:
            ingest_document(load_file(path), kb, known, report)
        except (OSError, UnicodeDecodeError) as exc:
            report.errors.append((source_name(path), str(exc)))

    # Web pages: fetched safely (see rag/web.py). A refused or failed URL is reported, not fatal.
    for url in urls:
        try:
            final_url, title, text = fetch_page(url)
        except Exception as exc:  # UnsafeURL, FetchError, network and HTTP errors all end up here
            report.errors.append((url, str(exc)))
            continue
        ingest_document(Document(final_url, title, text, kind="web", markdown=True), kb, known, report)

    # Mirror deletions: a stored file under one of the given paths that's no longer on disk is removed.
    present = {source_name(f) for f in files}
    roots = [source_name(p) for p in paths]
    for source, info in known.items():
        under_a_root = any(source == root or source.startswith(root.rstrip("/") + "/") for root in roots)
        if info["kind"] == "file" and under_a_root and source not in present:
            kb.delete_source(source)
            report.removed.append(source)
    return report


def format_report(report: IngestReport) -> str:
    """The summary the command prints, one line per outcome, e.g. "added    6   knowledge/brand-voice.md, …"."""

    def row(label: str, items: list, note: str = "") -> str:
        names = ", ".join(str(i[0] if isinstance(i, tuple) else i) for i in items[:3]) + (", …" if len(items) > 3 else "")
        return f"  {label:<9}{len(items):>3}   {note or names}".rstrip()

    lines = [
        "Knowledge base updated",
        row("added", report.added),
        row("updated", report.updated),
        row("skipped", report.skipped, "(unchanged)" if report.skipped else ""),
        row("removed", report.removed),
        f"  {'chunks':<9}{report.chunks:>3}   embedded and stored",
        row("flagged", report.flagged, " "),
    ]
    lines += [f"             ⚑ {src} › {heading}  [{rule}]" for src, heading, rule in report.flagged]
    lines.append(row("errors", report.errors, " "))
    lines += [f"             ✕ {src}: {message}" for src, message in report.errors]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point: parse the sources, run ingest, print the report."""
    parser = argparse.ArgumentParser(description="Add files, folders and web pages to the Art Lab knowledge base.")
    parser.add_argument("sources", nargs="*", help="files, folders or http(s) URLs (default: knowledge/ and two docs)")
    args = parser.parse_args(argv)
    report = ingest(args.sources or DEFAULT_SOURCES, KnowledgeBase())
    print(format_report(report))


if __name__ == "__main__":
    main()
