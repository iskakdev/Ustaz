import os
from typing import Optional
from langchain_core.callbacks import CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import InMemoryVectorStore
from ai import llm
from config import STORE_DIR

MIN_SCORE = 0.45  # ниже этого косинуса кусок считаем не про вопрос (подобрано на bge-m3)

_stores: dict[int, InMemoryVectorStore] = {}


def store_path(course_id: int) -> str:
    return os.path.join(STORE_DIR, f'course_{course_id}.json')


async def build_store(course_id: int, chunks: list[Document]) -> InMemoryVectorStore:
    """Векторы считаем один раз при сборке курса и сохраняем в файл: после перезапуска не пересчитываем."""
    store = InMemoryVectorStore(llm.get_embeddings())
    await store.aadd_documents(chunks)
    os.makedirs(STORE_DIR, exist_ok=True)
    store.dump(store_path(course_id))
    _stores[course_id] = store
    return store


def get_store(course_id: int) -> Optional[InMemoryVectorStore]:
    if course_id not in _stores:
        path = store_path(course_id)
        if not os.path.exists(path):
            return None
        _stores[course_id] = InMemoryVectorStore.load(path, llm.get_embeddings())
    return _stores[course_id]


class CourseRetriever(BaseRetriever):
    """Поиск по материалам одного курса. Score кладём в metadata и отсекаем слабые совпадения:
    пустой результат значит, что в курсе этого нет, и тьютор не будет выдумывать."""
    course_id: int
    k: int = 4
    min_score: float = MIN_SCORE

    def _wrap(self, found: list[tuple[Document, float]]) -> list[Document]:
        return [Document(page_content=doc.page_content, metadata={**doc.metadata, 'score': round(score, 3)})
                for doc, score in found if score >= self.min_score]

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        store = get_store(self.course_id)
        return self._wrap(store.similarity_search_with_score(query, k=self.k)) if store else []

    async def _aget_relevant_documents(self, query: str, *,
                                       run_manager: AsyncCallbackManagerForRetrieverRun) -> list[Document]:
        store = get_store(self.course_id)
        return self._wrap(await store.asimilarity_search_with_score(query, k=self.k)) if store else []
