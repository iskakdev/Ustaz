from typing import Iterator
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from database.models import Material

HEADERS = [('#', 'h1'), ('##', 'h2'), ('###', 'h3')]


class MaterialLoader(BaseLoader):
    """Материалы курса из БД -> Document. lazy_load отдаёт по одному, load() собирает список."""

    def __init__(self, materials: list[Material]):
        self.materials = materials

    def lazy_load(self) -> Iterator[Document]:
        for material in self.materials:
            yield Document(page_content=material.text,
                           metadata={'material_id': material.id, 'material': material.title})


def section_title(meta: dict, fallback: str) -> str:
    return ' › '.join(meta[h] for h in ('h1', 'h2', 'h3') if meta.get(h)) or fallback


def split_sections(documents: list[Document]) -> list[Document]:
    """Режем по заголовкам Markdown: раздел лекции - кандидат на понятия, заголовки уходят в metadata.
    Длинный раздел без подзаголовков дорезаем, чтобы он влез в окно модели при поиске понятий."""
    header_splitter = MarkdownHeaderTextSplitter(HEADERS, strip_headers=False)
    size_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    sections = []
    for doc in documents:
        for part in header_splitter.split_text(doc.page_content):
            part.metadata = {**doc.metadata, 'section': section_title(part.metadata, doc.metadata['material'])}
            sections.extend(size_splitter.split_documents([part]))
    return [section for section in sections if len(section.page_content.strip()) > 80]


def split_chunks(sections: list[Document]) -> list[Document]:
    """Куски для поиска: тьютору нужен абзац с ответом, а не весь раздел."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = splitter.split_documents(sections)
    for number, chunk in enumerate(chunks):
        chunk.metadata['chunk'] = number
    return chunks


def format_docs(docs: list[Document]) -> str:
    """Нумеруем куски, чтобы модель могла сослаться на них как [1], [2]."""
    return '\n\n'.join(f'[{n}] ({doc.metadata.get("section", "")})\n{doc.page_content}'
                       for n, doc in enumerate(docs, start=1))
