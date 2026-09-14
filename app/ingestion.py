from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


HEADING_RE = re.compile(r"^\s*(#{1,6})\s+(.+?)\s*$")


@dataclass
class Chunk:
    chunk_id: str
    document: str
    section: str
    chunk_type: str
    text: str


def extract_text(filename: str, data: bytes) -> str:
    suffix = Path(filename).suffix.lower()

    if suffix in {".md", ".txt"}:
        return data.decode("utf-8", errors="replace")

    if suffix == ".pdf":
        import io
        reader = PdfReader(io.BytesIO(data))
        pages = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(f"<!-- page {page_number} -->\n{text}")
        return "\n\n".join(pages)

    raise ValueError("Unsupported file type. Use .md, .txt, or .pdf.")


def _is_table_line(line: str) -> bool:
    return "|" in line and line.strip().count("|") >= 2


def _looks_like_table_separator(line: str) -> bool:
    stripped = line.strip().replace("|", "").replace(":", "").replace("-", "")
    return stripped == "" and "-" in line


def _flush_prose(
    chunks: list[tuple[str, str, str]],
    section: str,
    text: str,
    max_chars: int = 900,
    overlap: int = 120,
) -> None:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return

    if len(text) <= max_chars:
        chunks.append((section, "prose", text))
        return

    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))

        if end < len(text):
            boundary = max(
                text.rfind(". ", start, end),
                text.rfind("; ", start, end),
                text.rfind(" ", start, end),
            )
            if boundary > start + max_chars // 2:
                end = boundary + 1

        piece = text[start:end].strip()
        if piece:
            chunks.append((section, "prose", piece))

        if end >= len(text):
            break

        start = max(end - overlap, start + 1)


def _flush_table(
    chunks: list[tuple[str, str, str]],
    section: str,
    lines: list[str],
    max_chars: int = 1800,
) -> None:
    if not lines:
        return

    table = "\n".join(line.rstrip() for line in lines).strip()
    if len(table) <= max_chars:
        chunks.append((section, "table", table))
        return

    header = lines[:2] if len(lines) >= 2 else lines[:1]
    current = list(header)
    current_len = len("\n".join(current))

    for row in lines[len(header):]:
        proposed = current_len + len(row) + 1
        if proposed > max_chars and len(current) > len(header):
            chunks.append((section, "table", "\n".join(current)))
            current = list(header)
            current_len = len("\n".join(current))
        current.append(row)
        current_len += len(row) + 1

    if len(current) > len(header):
        chunks.append((section, "table", "\n".join(current)))


def chunk_document(text: str, document: str) -> list[Chunk]:
    if not text.strip():
        raise ValueError("Document is empty or contains no readable text.")

    raw_chunks: list[tuple[str, str, str]] = []
    section = "Document"
    prose_lines: list[str] = []
    table_lines: list[str] = []
    in_table = False

    def flush_prose_buffer() -> None:
        nonlocal prose_lines
        _flush_prose(raw_chunks, section, "\n".join(prose_lines))
        prose_lines = []

    def flush_table_buffer() -> None:
        nonlocal table_lines
        _flush_table(raw_chunks, section, table_lines)
        table_lines = []

    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    for line in lines:
        heading = HEADING_RE.match(line)

        if heading:
            if in_table:
                flush_table_buffer()
                in_table = False
            flush_prose_buffer()
            section = heading.group(2).strip()
            continue

        if _is_table_line(line):
            if prose_lines:
                flush_prose_buffer()
            in_table = True
            table_lines.append(line)
            continue

        if in_table:
            if _looks_like_table_separator(line):
                table_lines.append(line)
                continue
            flush_table_buffer()
            in_table = False

        if line.strip():
            prose_lines.append(line)
        elif prose_lines:
            flush_prose_buffer()

    if in_table:
        flush_table_buffer()
    flush_prose_buffer()

    result: list[Chunk] = []
    for index, (chunk_section, chunk_type, chunk_text) in enumerate(raw_chunks):
        digest = hashlib.sha1(
            f"{document}\0{index}\0{chunk_section}\0{chunk_text}".encode("utf-8")
        ).hexdigest()[:16]
        result.append(
            Chunk(
                chunk_id=digest,
                document=document,
                section=chunk_section,
                chunk_type=chunk_type,
                text=chunk_text,
            )
        )

    return result
