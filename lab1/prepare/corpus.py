"""Размеченный корпус Universal Dependencies.

Модуль читает файлы корпуса в формате conllu (допускается сжатие gzip) и превращает
каждое слово в пару из эталонной леммы и тега задания. Знаки препинания и символы
пропускаются, как и в формате ответа лабораторной работы. Корпус нужен для обучения
нейросети, подсчета частот разборов и честной оценки точности на отложенных частях.
Это золотой стандарт для проверки всех методов.
"""

import gzip
import pickle
from dataclasses import dataclass
from pathlib import Path

from ..tagset import SKIP_UPOS, tag_from_upos
from ..text_utils import normalize_word

PROGRESS_STEP = 20_000


@dataclass
class GoldToken:
    """Слово корпуса с эталонной леммой и тегом, None означает неизвестную разметку."""

    text: str
    lemma: str | None
    tag: str | None


@dataclass
class GoldSentence:
    """Предложение корпуса и номер текста, из которого оно взято."""

    text_id: str
    tokens: list[GoldToken]


def open_text(path: Path):
    """Открывает текстовый файл, сжатый или обычный.

    Args:
        path: Путь к файлу, расширение gz означает сжатие gzip.

    Returns:
        Файловый объект для чтения текста в кодировке utf-8.
    """
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8-sig")


def read_conllu_token(line: str) -> GoldToken | None:
    """Читает одну строку разметки conllu.

    Args:
        line: Строка из десяти колонок, разделенных табуляцией.

    Returns:
        Слово с эталонной разметкой, слово без разметки, если часть речи неизвестна,
        или None для знаков препинания, символов, составных токенов и пустых узлов.
    """
    columns = line.split("\t")
    if len(columns) < 4 or "-" in columns[0] or "." in columns[0]:
        return None
    text = columns[1].replace(" ", "")
    upos = columns[3] # часть речи 
    if upos in SKIP_UPOS or not text:
        return None
    tag = tag_from_upos(upos)
    if tag is None:
        return GoldToken(text, None, None)
    lemma = columns[2] if columns[2] not in ("_", "") else text
    return GoldToken(text, normalize_word(lemma), tag)


def parse_conllu(path: Path) -> list[GoldSentence]:
    """Разбирает один файл корпуса в формате conllu.

    Args:
        path: Путь к файлу, допускается сжатие gzip.

    Returns:
        Список предложений с эталонной разметкой.
    """
    sentences: list[GoldSentence] = []
    sent_id = ""
    tokens: list[GoldToken] = []
    with open_text(path) as file:
        for raw_line in file:
            line = raw_line.rstrip("\n")
            if line.startswith("#"):
                if line.startswith("# sent_id"):
                    sent_id = line.split("=", 1)[1].strip()
                continue
            if not line:
                if tokens:
                    sentences.append(GoldSentence(sent_id, tokens))
                    if len(sentences) % PROGRESS_STEP == 0:
                        print(f"Прочитано предложений: {len(sentences)}")
                tokens = []
                continue
            token = read_conllu_token(line)
            if token is not None:
                tokens.append(token)
    if tokens:
        sentences.append(GoldSentence(sent_id, tokens))
    return sentences


def parse_corpus(paths: list[Path]) -> list[GoldSentence]:
    """Разбирает несколько файлов корпуса и объединяет их предложения.

    Args:
        paths: Пути к файлам в формате conllu.

    Returns:
        Список предложений всех файлов по порядку.
    """
    sentences: list[GoldSentence] = []
    for path in paths:
        print(f"Разбор файла {path.name}")
        sentences.extend(parse_conllu(path))
    print(f"Всего предложений: {len(sentences)}")
    return sentences


def lemma_key(form: str, lemma: str, tag: str) -> str:
    """Строит ключ для таблицы частот разборов.

    Args:
        form: Нормализованная словоформа.
        lemma: Нормализованная лемма.
        tag: Тег разбора.

    Returns:
        Строка из трех полей, разделенных символом табуляции.
    """
    return f"{form}\t{lemma}\t{tag}"


def count_lemmas(sentences: list[GoldSentence]) -> dict[str, int]:
    """Считает, сколько раз в корпусе встречался каждый разбор слова.

    Args:
        sentences: Предложения с эталонной разметкой.

    Returns:
        Таблица, где ключ построен функцией lemma_key, а значение это число повторов.
    """
    counts: dict[str, int] = {}
    for sentence in sentences:
        for token in sentence.tokens:
            if token.tag is None or token.lemma is None:
                continue
            key = lemma_key(normalize_word(token.text), normalize_word(token.lemma), token.tag)
            counts[key] = counts.get(key, 0) + 1
    return counts


def count_tags(sentences: list[GoldSentence]) -> dict[str, int]:
    """Считает частоты тегов в корпусе.

    Args:
        sentences: Предложения с эталонной разметкой.

    Returns:
        Таблица, где ключ это тег, а значение число слов с этим тегом.
    """
    counts: dict[str, int] = {}
    for sentence in sentences:
        for token in sentence.tokens:
            if token.tag is not None:
                counts[token.tag] = counts.get(token.tag, 0) + 1
    return counts


def save_corpus(parts: tuple[list[GoldSentence], ...], path: Path) -> None:
    """Сохраняет разобранные части корпуса в файл кэша.

    Args:
        parts: Кортеж списков предложений: обучение, отладка и тест.
        path: Путь к файлу кэша.

    Returns:
        Ничего не возвращает.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(parts, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_corpus_cache(path: Path) -> tuple[list[GoldSentence], ...]:
    """Загружает разобранные части корпуса из файла кэша.

    Args:
        path: Путь к файлу кэша.

    Returns:
        Кортеж списков предложений: обучение, отладка и тест.
    """
    with open(path, "rb") as file:
        return pickle.load(file)
