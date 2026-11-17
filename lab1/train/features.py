"""Признаки слов и подготовка предложений для моделей.

Модуль превращает слова в числа: номер слова в словаре обучающих данных, номера
символов, признаки регистра и список тегов, возможных по словарю.
Он не зависит от PyTorch, поэтому подготовленные данные можно проверять и
кэшировать отдельно от обучения. Здесь же строится вектор окончания слова,
который использует модель лемматизации неизвестных слов.
"""

from collections import Counter
from dataclasses import dataclass

from ..config import BOS_ID, CHAR_ALPHABET, EOS_ID, IGNORE_LABEL, PAD_ID, UNK_ID
from ..prepare.corpus import GoldSentence
from ..prepare.dictionary import Dictionary
from ..tagset import TAG_ID
from ..text_utils import normalize_word

CHAR_TO_ID = {char: index + 4 for index, char in enumerate(CHAR_ALPHABET)}
HEAD_CHARS = 6


class Vocabulary:
    """Словарь слов обучающей выборки с номерами."""

    def __init__(self, itos: list[str]) -> None:
        """Создает словарь по списку слов, где первые два места заняты служебными.

        Args:
            itos: Список слов, номер слова равен его позиции в списке.

        Returns:
            Ничего не возвращает.
        """
        # список слов в словаре
        self.itos = itos
        # обратная таблица: по написанию слова сразу находится его номер в itos
        self.stoi = {word: index for index, word in enumerate(itos)}

    def __len__(self) -> int:
        """Возвращает размер словаря.

        Args:
            Нет.

        Returns:
            Число слов вместе со служебными.
        """
        return len(self.itos)

    @staticmethod
    def build(counter: Counter, min_freq: int) -> "Vocabulary":
        """Строит словарь из частых слов.

        Args:
            counter: Частоты нормализованных слов.
            min_freq: Минимальная частота слова для попадания в словарь.

        Returns:
            Словарь, где номера 0 и 1 заняты символами заполнения и неизвестного слова.
        """
        words = [word for word, number in counter.most_common() if number >= min_freq]
        return Vocabulary(["<pad>", "<unk>"] + words)

    def encode(self, word: str) -> int:
        """Возвращает номер слова.

        Args:
            word: Нормализованное слово.

        Returns:
            Номер слова или номер неизвестного слова.
        """
        return self.stoi.get(word, UNK_ID)


def encode_chars(word: str, max_chars: int) -> list[int]:
    """Кодирует слово последовательностью номеров символов фиксированной длины.

    Args:
        word: Нормализованное слово.
        max_chars: Длина результата вместе с метками начала и конца слова.

    Returns:
        Список номеров символов, дополненный нулями справа.
    """
    room = max_chars - 2
    if len(word) > room:
        # у длинных слов сохраняем начало и конец, потому что окончание важнее всего
        word = word[:HEAD_CHARS] + word[-(room - HEAD_CHARS):]
    ids = [BOS_ID] + [CHAR_TO_ID.get(char, UNK_ID) for char in word] + [EOS_ID]
    return ids + [PAD_ID] * (max_chars - len(ids))


def encode_suffix(word: str, suffix_len: int) -> list[int]:
    """Кодирует последние символы слова для модели лемматизации.

    Args:
        word: Нормализованное слово.
        suffix_len: Сколько символов с конца слова нужно взять.

    Returns:
        Список из suffix_len номеров символов, дополненный нулями слева.
    """
    ids = [BOS_ID] + [CHAR_TO_ID.get(char, UNK_ID) for char in word]
    ids = ids[-suffix_len:]
    return [PAD_ID] * (suffix_len - len(ids)) + ids


def case_features(token: str, index: int) -> list[float]:
    """Строит признаки регистра слова.
    1. слово с заглавной буквы?
    2. слово из заглавных букв?
    3. первое слово?

    Args:
        token: Слово в исходном написании.
        index: Номер слова в предложении, начиная с нуля.

    Returns:
        Три числа: слово с заглавной буквы, слово из заглавных букв, первое слово.
    """
    is_title = token[:1].isupper() and not token.isupper()
    is_upper = token.isupper() and len(token) > 1
    return [float(is_title), float(is_upper), float(index == 0)]


def gold_label_ids(sentence: GoldSentence) -> list[int]:
    """Переводит эталонные теги предложения в номера.

    Args:
        sentence: Предложение с эталонной разметкой.

    Returns:
        Список номеров тегов, для неоднозначных слов стоит служебное значение игнорирования.
    """
    return [IGNORE_LABEL if token.tag is None else TAG_ID[token.tag] for token in sentence.tokens]


@dataclass
class PreparedSentence:
    """Предложение, переведенное в числовые признаки."""

    word_ids: list[int]
    char_ids: list[list[int]]
    case_feats: list[list[float]]
    cand_tags: list[tuple[int, ...]]
    labels: list[int]


class SentenceEncoder:
    """Преобразует предложения в числовые признаки с учетом словаря."""

    def __init__(self, dictionary: Dictionary, vocab: Vocabulary, max_chars: int) -> None:
        """Запоминает словари и длину кодирования символов.

        Args:
            dictionary: Словарь словоформ.
            vocab: Словарь слов обучающей выборки.
            max_chars: Длина кодирования символов одного слова.

        Returns:
            Ничего не возвращает.
        """
        self.dictionary = dictionary
        self.vocab = vocab
        self.max_chars = max_chars

    def prepare(
        self,
        tokens: list[str],
        labels: list[int] | None = None,
        disabled: set[int] | None = None,
    ) -> PreparedSentence:
        """Строит признаки для всех слов предложения.

        Args:
            tokens: Слова предложения без знаков препинания.
            labels: Номера эталонных тегов или None, если разметки нет.
            disabled: Номера слов, для которых словарь нужно скрыть. Так в оценке
                имитируются неизвестные слова.

        Returns:
            Предложение в числовом виде.
        """
        disabled = disabled or set()
        word_ids: list[int] = []
        char_ids: list[list[int]] = []
        case_feats: list[list[float]] = []
        cand_tags: list[tuple[int, ...]] = []
        for index, token in enumerate(tokens):
            form = normalize_word(token)
            char_ids.append(encode_chars(form, self.max_chars))
            case_feats.append(case_features(token, index))
            # скрываем словарь при оценке результатов
            if index in disabled:
                word_ids.append(UNK_ID)
                cand_tags.append(())
            else:
                word_ids.append(self.vocab.encode(form))
                cand_tags.append(self.dictionary.lookup_tag_ids(form))
        return PreparedSentence(
            word_ids, char_ids, case_feats, cand_tags, labels or [IGNORE_LABEL] * len(tokens)
        )
