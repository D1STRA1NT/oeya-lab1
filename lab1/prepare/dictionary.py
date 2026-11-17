"""Словарь словоформ.

Модуль строит быстрый поиск: по нормализованной словоформе выдаются все возможные
пары лемма и тег. Словарь собирается из двух источников: таблицы изменяемых слов
UniMorph (существительные, прилагательные и глаголы со всеми формами) и слов
обучающей части корпуса, где есть и служебные слова. Для известных слов
морфологический разбор сводится к поиску в словаре, а неизвестные слова разбирает
нейросеть.
"""

import pickle
from pathlib import Path

from .corpus import GoldSentence, open_text
from ..tagset import TAG_ID, TAGS, UNIMORPH_TO_TAG
from ..text_utils import normalize_word

TAG_STRIDE = 64


class Dictionary:
    """Словарь словоформ с поиском возможных лемм и тегов."""

    def __init__(self) -> None:
        """Создает пустой словарь.

        Args:
            Нет.

        Returns:
            Ничего не возвращает.
        """
        # значение формы это код пары лемма и тег или кортеж таких кодов
        # ключ - нормализованная словоформа
        # значение - код пары лемма и тег или кортеж таких кодов
        # код = номер леммы * 64 + номер тега
        self.forms: dict[str, int | tuple[int, ...]] = {}
        # список всех лемм в словаре
        self.lemma_texts: list[str] = []
        # для проверки, есть ли уже такая лемма в словаре
        self.lemma_index: dict[str, int] = {}

    def __len__(self) -> int:
        """Возвращает число словоформ в словаре.

        Args:
            Нет.

        Returns:
            Количество разных нормализованных словоформ.
        """
        return len(self.forms)

    def intern_lemma(self, text: str) -> int:
        """Возвращает номер леммы, добавляя ее в таблицу при первом появлении.

        Args:
            text: Нормализованный текст леммы.

        Returns:
            Номер леммы в общем списке.
        """
        index = self.lemma_index.get(text)
        if index is None:
            index = len(self.lemma_texts)
            self.lemma_texts.append(text)
            self.lemma_index[text] = index
        return index

    def add(self, form: str, lemma: str, tag: str) -> None:
        """Добавляет разбор словоформы, не создавая повторов.

        Args:
            form: Словоформа в любом написании.
            lemma: Лемма в любом написании.
            tag: Тег задания.

        Returns:
            Ничего не возвращает.
        """
        form = normalize_word(form)
        lemma = normalize_word(lemma)
        if not form or not lemma or " " in form:
            return
        code = self.intern_lemma(lemma) * TAG_STRIDE + TAG_ID[tag]
        current = self.forms.get(form)
        if current is None:
            self.forms[form] = code
        elif isinstance(current, int):
            if current != code:
                self.forms[form] = (current, code)
        elif code not in current:
            self.forms[form] = current + (code,)

    def lookup(self, form: str) -> list[tuple[str, str]]:
        """Ищет все разборы словоформы.

        Args:
            form: Нормализованная словоформа.

        Returns:
            Список пар лемма и тег, пустой список для неизвестного слова.
        """
        value = self.forms.get(form)
        if value is None:
            return []
        codes = (value,) if isinstance(value, int) else value
        return [(self.lemma_texts[code // TAG_STRIDE], TAGS[code % TAG_STRIDE]) for code in codes]

    def lookup_tag_ids(self, form: str) -> tuple[int, ...]:
        """Ищет номера тегов, возможных для словоформы.

        Args:
            form: Нормализованная словоформа.

        Returns:
            Кортеж номеров тегов без повторов, пустой для неизвестного слова.
        """
        value = self.forms.get(form)
        if value is None:
            return ()
        codes = (value,) if isinstance(value, int) else value
        return tuple(dict.fromkeys(code % TAG_STRIDE for code in codes))

    def save(self, path: Path) -> None:
        """Сохраняет словарь в файл для быстрой загрузки.

        Args:
            path: Путь к файлу кэша.

        Returns:
            Ничего не возвращает.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lemma_index = {}
        with open(path, "wb") as file:
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: Path) -> "Dictionary":
        """Загружает словарь из файла кэша.

        Args:
            path: Путь к файлу кэша.

        Returns:
            Загруженный словарь.
        """
        with open(path, "rb") as file:
            return pickle.load(file)


def add_corpus_entries(dictionary: Dictionary, sentences: list[GoldSentence]) -> None:
    """Добавляет в словарь все слова размеченного корпуса.

    Args:
        dictionary: Словарь, который пополняется.
        sentences: Предложения с эталонной разметкой, только обучающая часть.

    Returns:
        Ничего не возвращает.
    """
    for sentence in sentences:
        for token in sentence.tokens:
            if token.tag is not None and token.lemma is not None:
                dictionary.add(token.text, token.lemma, token.tag)


def add_unimorph_entries(dictionary: Dictionary, path: Path) -> None:
    """Добавляет в словарь таблицу форм UniMorph.

    Args:
        dictionary: Словарь, который пополняется.
        path: Путь к файлу (допускается gzip), где в каждой строке лемма, словоформа и признаки
            разделены табуляцией, а первый признак это часть речи.

    Returns:
        Ничего не возвращает.
    """
    with open_text(path) as file:
        for line in file:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            tag = UNIMORPH_TO_TAG.get(parts[2].split(";", 1)[0])
            if tag is not None:
                dictionary.add(parts[1], parts[0], tag)


def build_dictionary(unimorph_path: Path | None, train: list[GoldSentence]) -> Dictionary:
    """Строит словарь из обучающего корпуса и таблицы UniMorph.

    Args:
        unimorph_path: Путь к файлу UniMorph или None, если его нет.
        train: Обучающие предложения. Отладочная и тестовая части в словарь не входят,
            чтобы оценка на них оставалась честной.

    Returns:
        Готовый словарь словоформ.
    """
    dictionary = Dictionary()
    # корпус добавляется первым, его разборы идут в списке кандидатов раньше
    add_corpus_entries(dictionary, train)
    print(f"Из корпуса: словоформ {len(dictionary)}")
    if unimorph_path is not None:
        add_unimorph_entries(dictionary, unimorph_path)
        print(f"После добавления UniMorph: словоформ {len(dictionary)}")
    return dictionary
