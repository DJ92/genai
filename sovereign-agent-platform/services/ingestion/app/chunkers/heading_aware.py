from dataclasses import dataclass


@dataclass
class Chunk:
    content: str
    offset_start: int
    offset_end: int


def chunk_document(text: str, max_tokens: int = 512) -> list[Chunk]:
    max_chars = max_tokens * 4
    chunks: list[Chunk] = []
    cursor = 0

    while cursor < len(text):
        end = min(cursor + max_chars, len(text))
        split = text.rfind("\n\n", cursor, end)
        if split <= cursor:
            split = end
        content = text[cursor:split].strip()
        if content:
            chunks.append(Chunk(content=content, offset_start=cursor, offset_end=split))
        cursor = split if split > cursor else end

    return chunks
