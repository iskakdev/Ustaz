"""Тесты гоняют весь путь без Ollama: модель и эмбеддинги подменяются фейками на LangChain-интерфейсах."""
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

TMP = tempfile.mkdtemp(prefix='ustaz-test-')
os.environ['DB_URL'] = f'sqlite:///{TMP}/test.db'
os.environ['STORE_DIR'] = f'{TMP}/stores'
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from langchain_core.embeddings import Embeddings  # noqa: E402
from langchain_core.language_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, BaseMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from langchain_core.utils.function_calling import convert_to_openai_tool  # noqa: E402

STEMS = ['функци', 'вызов', 'стек', 'кадр', 'рекурс', 'базов', 'случа', 'глубин', 'лимит', 'recursionerror',
         'мемоиз', 'кэш', 'cache', 'фибонач', 'fib', 'цикл', 'дерев', 'return', 'параметр', 'перемен']


class StemEmbeddings(Embeddings):
    """Игрушечные эмбеддинги: вектор - сколько раз встретились слова по теме курса.
    Похожие по теме тексты близки, а вопрос не по теме даёт нулевой вектор - как нам и нужно в тестах."""

    def vector(self, text: str) -> list[float]:
        words = re.findall(r'\w+', text.lower())
        vec = [float(sum(1 for w in words if w.startswith(stem))) for stem in STEMS] + [0.05]  # без нулевых векторов
        norm = sum(x * x for x in vec) ** 0.5
        return [x / norm for x in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.vector(text)


class ScriptedChatModel(BaseChatModel):
    """Фейковая модель: ответ решает script по сообщениям и по тому, какие tools/схемы к ней привязаны.
    bind_tools настоящий, поэтому with_structured_output и create_agent работают через штатный путь LangChain."""
    script: Callable[[list[BaseMessage], dict], AIMessage]

    @property
    def _llm_type(self) -> str:
        return 'scripted'

    def _generate(self, messages, stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self.script(messages, kwargs))])

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self.bind(tools=[convert_to_openai_tool(t) for t in tools], tool_choice=tool_choice, **kwargs)


def call(name: str, args: dict) -> AIMessage:
    return AIMessage('', tool_calls=[{'name': name, 'args': args, 'id': f'call_{name}'}])


SECTION_CONCEPTS = {
    'Вызов функции': [('Вызов функции', [], ['return обязателен'])],
    'Стек вызовов': [('Стек вызовов', ['Вызов функции'], ['стек работает как очередь'])],
    'Базовый и рекурсивный случай': [
        ('Базовый случай', ['Стек вызовов'], ['рекурсия сама понимает, когда остановиться']),
        ('Рекурсивный случай', ['Базовый случай', 'Вызов функции'], ['все вызовы делят одну переменную'])],
    'Глубина рекурсии и RecursionError': [('RecursionError', ['стек вызовов', 'Рекурсивный случай'],
                                           ['лимит надо просто поднять']),
                                          ('Стек вызовов', [], ['кадр живёт вечно'])],
    'Мемоизация': [('Мемоизация', ['Рекурсивный случай'], ['кэш уменьшает глубину рекурсии'])],
    'Рекурсия или цикл': [('Рекурсия или цикл', ['RecursionError', 'Мемоизация'], ['рекурсия быстрее цикла'])],
}


def last_text(messages: list[BaseMessage]) -> str:
    return messages[-1].text


def course_script(messages: list[BaseMessage], kwargs: dict) -> AIMessage:
    tools = [t['function']['name'] for t in kwargs.get('tools') or []]
    system = messages[0].text if messages[0].type == 'system' else ''
    text = last_text(messages)

    if tools == ['SectionConcepts']:
        section = re.search(r'Раздел: .*?([^›\n]+)\n', text).group(1).strip()
        return call('SectionConcepts', {'concepts': [
            {'name': name, 'summary': f'{name} - суть из лекции.', 'requires': req, 'misconceptions': mis}
            for name, req, mis in SECTION_CONCEPTS.get(section, [])]})
    if tools == ['Prerequisites']:
        # для RecursionError добавляем лишнее транзитивное ребро и выдуманное понятие - код должен их убрать
        name = re.search(r'Понятие: (.+?) - ', text).group(1)
        requires = {n: req for items in SECTION_CONCEPTS.values() for n, req, _ in items if req}.get(name, [])
        if name == 'RecursionError':
            requires = requires + ['Вызов функции', 'Нечто внешнее']
        return call('Prerequisites', {'reason': 'по лекции', 'requires': requires})
    if tools == ['QuestionSet']:
        name = re.search(r'Понятие: (.+)', text).group(1)
        return call('QuestionSet', {'questions': [
            {'kind': kind, 'text': f'[{kind}] Вопрос про «{name}»', 'reference_answer': f'Эталон про {name}.',
             'key_points': [f'{name} пункт один', f'{name} пункт два']} for kind in ('recall', 'apply', 'analyze')]})
    if tools == ['Grade']:
        answer = re.search(r'<answer>(.*)</answer>', text, re.S).group(1).lower()
        if 'верно' in answer and 'неверно' not in answer:
            grade = {'score': 0.9, 'covered': ['всё'], 'missing': [], 'misconception': None, 'feedback': 'Отлично.'}
        elif 'частично' in answer:
            grade = {'score': 0.5, 'covered': ['одно'], 'missing': ['другое'], 'misconception': None,
                     'feedback': 'Наполовину.'}
        else:
            grade = {'score': 0.1, 'covered': [], 'missing': ['всё'],
                     'misconception': 'Рекурсия сама понимает когда остановиться', 'feedback': 'Мимо.'}
        return call('Grade', grade)
    if tools == ['Difficulty']:
        return call('Difficulty', {'pattern': 'Думают, что рекурсия останавливается сама.', 'example': 'неверно'})
    if tools == ['TeacherBrief']:
        return call('TeacherBrief', {'summary': 'Группа путает остановку рекурсии.', 'priorities': [
            {'concept': 'Базовый случай', 'why': 'От него зависит всё остальное.',
             'how': 'Разобрать factorial(-1) на доске.'}]})
    if 'progress_overview' in tools:
        if messages[-1].type == 'human':
            if 'Привет' in text and 'Сначала вызови' not in system:
                return AIMessage('Привет! Чем помочь?')  # ответ «из головы» - ground_first должен переспросить
            return call('progress_overview', {})
        return AIMessage(f'Твой прогресс: {text.splitlines()[0]} Следующий шаг: ответь на вопрос в /next/.')
    if 'Перепиши последний вопрос' in system:
        return AIMessage(text)
    if 'проверочный вопрос' in system:
        return AIMessage('Подумай: что станет со стеком, если результат уже был посчитан раньше?')
    if 'Ты тьютор' in system:
        return AIMessage('Стек вызовов хранит кадры функций [1].')
    return AIMessage('ok')


@pytest.fixture(scope='session')
def fakes():
    from ai import coach, llm
    model = ScriptedChatModel(script=course_script)
    embeddings = StemEmbeddings()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(llm, 'get_chat_model', lambda *args, **kwargs: model)
        mp.setattr(llm, 'get_embeddings', lambda: embeddings)
        coach.reset_coach()
        yield mp


@pytest.fixture(scope='session')
def client(fakes):
    import main
    with TestClient(main.ustaz_app) as test_client:
        yield test_client


def auth(client: TestClient, full_name: str, email: str, role: str) -> dict:
    client.post('/auth/register/', json={'full_name': full_name, 'email': email, 'password': 'secret123',
                                         'role': role})
    token = client.post('/auth/login/', json={'email': email, 'password': 'secret123'}).json()['access_token']
    return {'Authorization': f'Bearer {token}'}


@pytest.fixture(scope='session')
def teacher(client):
    return auth(client, 'Айгуль Преподаватель', 'teacher@test.kg', 'teacher')


@pytest.fixture(scope='session')
def student(client):
    return auth(client, 'Бакыт Студент', 'student@test.kg', 'student')


@pytest.fixture(scope='session')
def course_id(client, teacher):
    course = client.post('/courses/', json={'title': 'Рекурсия в Python'}, headers=teacher).json()
    text = (ROOT / 'materials' / 'recursion.md').read_bytes()
    response = client.post(f'/courses/{course["id"]}/materials/upload/', headers=teacher,
                           files={'file': ('recursion.md', text, 'text/markdown')})
    assert response.status_code == 200, response.text
    response = client.post(f'/courses/{course["id"]}/build/', headers=teacher)
    assert response.status_code == 202, response.text
    return course['id']


def sse(text: str) -> list[tuple[str, Any]]:
    events = []
    for block in text.strip().split('\n\n'):
        lines = dict(line.split(': ', 1) for line in block.splitlines())
        data = lines['data']
        events.append((lines['event'], json.loads(data) if data[:1] in '[{"' else data))
    return events
