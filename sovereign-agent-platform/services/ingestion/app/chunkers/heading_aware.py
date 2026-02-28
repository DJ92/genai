from __future__ import annotations

import re
from dataclasses import dataclass, field

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class Chunk:
    content: str
    offset_start: int
    offset_end: int
    headings: list[str] = field(default_factory=list)


@dataclass
class _Section:
    headings: list[str]
    offset_start: int
    offset_end: int
    content: str


def _split_by_paragraphs(section: _Section, max_chars: int) -> list[Chunk]:
    paragraphs = [paragraph for paragraph in section.content.split("\n\n") if paragraph.strip()]
    chunks: list[Chunk] = []
    cursor = section.offset_start
    buffer: list[str] = []
    current_len = 0
    prefix = ""
    if section.headings:
        prefix = " > ".join(section.headings) + "\n\n"

    for paragraph in paragraphs:
        paragraph_len = len(paragraph) + 2
        if buffer and current_len + paragraph_len > max_chars:
            text = "\n\n".join(buffer).strip()
            chunk_text = f"{prefix}{text}".strip()
            chunk_end = cursor + len(text)
            chunks.append(
                Chunk(
                    content=chunk_text,
                    offset_start=max(section.offset_start, cursor - len(text)),
                    offset_end=min(section.offset_end, chunk_end),
                    headings=section.headings.copy(),
                )
            )
            buffer = []
            current_len = 0

        buffer.append(paragraph)
        current_len += paragraph_len
        cursor += paragraph_len

    if buffer:
        text = "\n\n".join(buffer).strip()
        chunk_text = f"{prefix}{text}".strip()
        chunk_start = max(section.offset_start, section.offset_end - len(text))
        chunks.append(
            Chunk(
                content=chunk_text,
                offset_start=chunk_start,
                offset_end=section.offset_end,
                headings=section.headings.copy(),
            )
        )
    return chunks


def _flush_section(
    sections: list[_Section], headings: list[str], lines: list[str], start: int, end: int
) -> None:
    body = "".join(lines).strip()
    if not body:
        return
    sections.append(
        _Section(
            headings=headings.copy(),
            offset_start=start,
            offset_end=end,
            content=body,
        )
    )


def chunk_document(text: str, max_tokens: int = 512) -> list[Chunk]:
    max_chars = max(256, max_tokens * 4)
    lines = text.splitlines(keepends=True)

    headings: list[str] = []
    sections: list[_Section] = []
    section_lines: list[str] = []
    section_start = 0
    offset = 0

    for line in lines:
        stripped = line.strip()
        match = HEADING_RE.match(stripped)

        if match:
            _flush_section(sections, headings, section_lines, section_start, offset)
            section_lines = []
            level = len(match.group(1))
            title = match.group(2).strip()
            headings = headings[: level - 1]
            headings.append(title)
            section_start = offset
        else:
            if not section_lines:
                section_start = offset
            section_lines.append(line)

        offset += len(line)

    _flush_section(sections, headings, section_lines, section_start, len(text))

    chunks: list[Chunk] = []
    for section in sections:
        if len(section.content) <= max_chars:
            prefix = ""
            if section.headings:
                prefix = " > ".join(section.headings) + "\n\n"
            chunks.append(
                Chunk(
                    content=f"{prefix}{section.content}".strip(),
                    offset_start=section.offset_start,
                    offset_end=section.offset_end,
                    headings=section.headings.copy(),
                )
            )
            continue
        chunks.extend(_split_by_paragraphs(section, max_chars=max_chars))

    return chunks

