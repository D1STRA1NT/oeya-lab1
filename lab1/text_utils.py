"""Работа с текстом на уровне строк и токенов.

Модуль содержит нормализацию слов (регистр и буква е), разбиение строки на токены
без знаков препинания, определение чисел, восстановление регистра лемм
и чтение и запись текстовых файлов в кодировке utf-8. Им пользуются почти все
остальные модули проекта.
"""

import string
import sys
from pathlib import Path

PUNCT_CHARS = set(string.punctuation) | set("«»“”„…—–")


def normalize_word(word: str) -> str:
    """Приводит слово к виду, удобному для поиска в словаре.

    Args:
        word: Исходное слово.

    Returns:
        Слово в нижнем регистре, в котором буква е с двумя точками заменена на обычную е.
    """
    return word.lower().replace("ё", "е")


def capitalize_first(word: str) -> str:
    """Делает первую букву слова заглавной, не трогая остальные.

    Args:
        word: Исходное слово.

    Returns:
        Слово с заглавной первой буквой.
    """
    return word[:1].upper() + word[1:]


def strip_punctuation(token: str) -> str:
    """Убирает знаки препинания с начала и конца токена.

    Args:
        token: Токен вместе со знаками препинания.

    Returns:
        Токен без знаков препинания по краям. Внутренние дефисы и точки сохраняются.
    """
    start = 0
    end = len(token)
    while start < end and token[start] in PUNCT_CHARS:
        start += 1
    while end > start and token[end - 1] in PUNCT_CHARS:
        end -= 1
    return token[start:end]


def tokenize_line(line: str) -> list[str]:
    """Разбивает строку на токены и удаляет знаки препинания.

    Args:
        line: Строка с предложением.

    Returns:
        Список токенов без знаков препинания. Токены из одних знаков отбрасываются.
    """
    tokens = (strip_punctuation(part) for part in line.split())
    return [token for token in tokens if token]


def special_tag(token: str) -> str | None:
    """Определяет тег чисел, не требующий словаря.

    Args:
        token: Токен без знаков препинания.

    Returns:
        Тег NUM для токена, в котором есть цифры и нет букв, иначе None.
    """
    has_digit = any(char.isdigit() for char in token)
    if has_digit and not any(char.isalpha() for char in token):
        return "NUM"
    return None


def restore_lemma_case(lemma: str, token: str, tag: str, index: int) -> str:
    """Возвращает заглавную букву лемме, найденной для имени собственного.

    Args:
        lemma: Лемма в нижнем регистре.
        token: Исходный токен из текста.
        tag: Тег, определенный для токена.
        index: Номер токена в предложении, начиная с нуля.

    Returns:
        Лемма с заглавной буквой для существительного в середине предложения,
        которое написано с заглавной буквы, но не целиком заглавными, иначе лемма без изменений.
    """
    if tag == "S" and index > 0 and token[:1].isupper() and not (len(token) > 1 and token.isupper()):
        return capitalize_first(lemma)
    return lemma


def configure_console_utf8() -> None:
    """Включает кодировку utf-8 для ввода и вывода консоли.

    Args:
        Нет.

    Returns:
        Ничего не возвращает.
    """
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def read_lines(path: Path) -> list[str]:
    """Читает текстовый файл построчно.

    Args:
        path: Путь к файлу в кодировке utf-8, допускается метка порядка байтов.

    Returns:
        Список строк без символов перевода строки.
    """
    with open(path, "r", encoding="utf-8-sig") as file:
        return file.read().splitlines()


def write_lines(path: Path, lines: list[str]) -> None:
    """Записывает строки в текстовый файл.

    Args:
        path: Путь к выходному файлу, он будет создан в кодировке utf-8.
        lines: Строки для записи.

    Returns:
        Ничего не возвращает.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as file:
        file.write("\n".join(lines))
        if lines:
            file.write("\n")
