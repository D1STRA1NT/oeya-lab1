"""Загрузка корпуса и словаря с использованием кэша.

Модуль находит файлы данных в папке lab1/data/raw (или рядом с проектом), разбирает их
один раз и сохраняет результат в папку кэша, а при следующих запусках читает кэш
за несколько секунд. Остальные модули получают данные только через функции
этого модуля.
"""

from pathlib import Path

from ..config import (
    CORPUS_CACHE,
    DEV_NAMES,
    DICT_CACHE,
    PROJECT_DIR,
    RAW_DIR,
    TEST_NAMES,
    TRAIN_NAMES,
    UNIMORPH_NAME,
)
from .corpus import GoldSentence, load_corpus_cache, parse_corpus, save_corpus
from .dictionary import Dictionary, build_dictionary


def find_file(name: str) -> Path | None:
    """Ищет файл данных в папке lab1/data/raw и в корне проекта.

    Args:
        name: Имя файла без суффикса сжатия.

    Returns:
        Путь к файлу или к его сжатой версии с суффиксом gz, None если файла нет.
    """
    for folder in (RAW_DIR, PROJECT_DIR):
        for suffix in ("", ".gz"):
            path = folder / (name + suffix)
            if path.exists():
                return path
    return None


def find_files(names: list[str]) -> list[Path]:
    """Ищет сразу несколько файлов данных и сообщает, если какого-то нет.

    Args:
        names: Имена файлов без суффикса сжатия.

    Returns:
        Список найденных путей в том же порядке, что и имена.
    """
    paths = []
    for name in names:
        path = find_file(name)
        if path is None:
            raise FileNotFoundError(
                f"Нет файла {name}. Выполните команду download или положите файл в папку {RAW_DIR}."
            )
        paths.append(path)
    return paths


def load_corpus(rebuild: bool = False) -> tuple[list[GoldSentence], list[GoldSentence], list[GoldSentence]]:
    """Возвращает обучающую, отладочную и тестовую части корпуса.

    Args:
        rebuild: Если True, кэш игнорируется и корпус читается из файлов заново.

    Returns:
        Три списка предложений: обучение, отладка и тест. Деление берется из
        самого корпуса, в нем тексты разных частей не пересекаются.
    """
    if CORPUS_CACHE.exists() and not rebuild:
        print("Загрузка корпуса из кэша")
        return load_corpus_cache(CORPUS_CACHE)
    parts = (
        parse_corpus(find_files(TRAIN_NAMES)),
        parse_corpus(find_files(DEV_NAMES)),
        parse_corpus(find_files(TEST_NAMES)),
    )
    save_corpus(parts, CORPUS_CACHE)
    return parts


def load_dictionary(train: list[GoldSentence] | None = None, rebuild: bool = False) -> Dictionary:
    """Возвращает словарь словоформ, при необходимости строит его заново.

    Args:
        train: Обучающие предложения, из них берутся слова корпуса. Если None и кэша
            словаря нет, корпус загружается сам.
        rebuild: Если True, кэш игнорируется и словарь строится заново.

    Returns:
        Словарь словоформ.
    """
    if DICT_CACHE.exists() and not rebuild:
        print("Загрузка словаря из кэша")
        return Dictionary.load(DICT_CACHE)
    if train is None:
        train = load_corpus()[0]
    unimorph_path = find_file(UNIMORPH_NAME)
    if unimorph_path is None:
        print("Файл UniMorph не найден, словарь строится только по корпусу")
    dictionary = build_dictionary(unimorph_path, train)
    dictionary.save(DICT_CACHE)
    return dictionary


def load_data(
    rebuild: bool = False,
) -> tuple[Dictionary, list[GoldSentence], list[GoldSentence], list[GoldSentence]]:
    """Возвращает словарь и все три части корпуса.

    Args:
        rebuild: Если True, кэши игнорируются и все строится заново.

    Returns:
        Словарь, обучающая, отладочная и тестовая части корпуса.
    """
    train, dev, test = load_corpus(rebuild)
    print(f"Предложений: обучение {len(train)}, отладка {len(dev)}, тест {len(test)}")
    dictionary = load_dictionary(train, rebuild)
    return dictionary, train, dev, test
