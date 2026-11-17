"""Сборка пакетов данных для PyTorch.

Модуль превращает список подготовленных предложений в тензоры с одинаковой длиной.
При обучении он имитирует неизвестные слова: случайно скрывает слово и разбор из
словаря, чтобы модель научилась определять часть речи по окончанию и контексту.
"""

import random
from dataclasses import dataclass
from typing import Iterator

import torch

from ..config import CASE_FEATURE_COUNT, IGNORE_LABEL, PAD_ID, UNK_ID, TaggerConfig
from .features import PreparedSentence


@dataclass
class Batch:
    """Пакет предложений в виде тензоров."""

    word_ids: torch.Tensor
    char_ids: torch.Tensor
    case_feats: torch.Tensor
    cand_mask: torch.Tensor
    labels: torch.Tensor
    lengths: torch.Tensor

    def to(self, device: torch.device) -> "Batch":
        """Переносит все тензоры пакета на устройство.

        Args:
            device: Устройство, например видеокарта или процессор.

        Returns:
            Новый пакет на указанном устройстве.
        """
        return Batch(
            self.word_ids.to(device),
            self.char_ids.to(device),
            self.case_feats.to(device),
            self.cand_mask.to(device),
            self.labels.to(device),
            self.lengths.to(device),
        )


def sentence_length(item) -> int:
    """Возвращает длину предложения в словах.

    Args:
        item: Подготовленное предложение или любой другой элемент списка.

    Returns:
        Число слов для подготовленного предложения, для остальных элементов единица.
    """
    word_ids = getattr(item, "word_ids", None)
    return len(word_ids) if word_ids is not None else 1


def make_batch(
    sentences: list[PreparedSentence],
    num_tags: int,
    cfg: TaggerConfig,
    train: bool,
    rng: random.Random,
) -> Batch:
    """Собирает пакет тензоров из подготовленных предложений.


    Args:
        sentences: Предложения пакета.
        num_tags: Число тегов.
        cfg: Параметры модели разметки.
        train: Если True, часть слов и разборов словаря скрывается случайным образом.
        rng: Генератор случайных чисел.

    Returns:
        Пакет тензоров, дополненных до длины самого длинного предложения.
    """
    max_len = max(len(sentence.word_ids) for sentence in sentences)
    pad_chars = [PAD_ID] * cfg.max_chars
    pad_case = [0.0] * CASE_FEATURE_COUNT
    word_rows, char_rows, case_rows, cand_rows, label_rows, lengths = [], [], [], [], [], []
    for sentence in sentences:
        length = len(sentence.word_ids)
        words = list(sentence.word_ids)
        # формируем маску тегов
        cand = [[0.0] * num_tags for _ in range(length)]
        for index in range(length):
            # у 12% слов затирается номер слова и маска тегов
            # чтобы для неизвестных модель опиралась на буквы и соседние слова
            if train and rng.random() < cfg.dict_drop_prob:
                words[index] = UNK_ID
                continue
            for tag in sentence.cand_tags[index]:
                cand[index][tag] = 1.0
            # еще у 10% слов затирается только номер слова
            # чтобы для редких модель училась читать подсказку из словаря
            if train and rng.random() < cfg.word_unk_prob:
                words[index] = UNK_ID
        pad = max_len - length
        word_rows.append(words + [PAD_ID] * pad)
        char_rows.append(sentence.char_ids + [pad_chars] * pad)
        case_rows.append(sentence.case_feats + [pad_case] * pad)
        cand_rows.append(cand + [[0.0] * num_tags for _ in range(pad)])
        label_rows.append(sentence.labels + [IGNORE_LABEL] * pad)
        lengths.append(length)
    return Batch(
        torch.tensor(word_rows, dtype=torch.long),
        torch.tensor(char_rows, dtype=torch.long),
        torch.tensor(case_rows, dtype=torch.float),
        torch.tensor(cand_rows, dtype=torch.float),
        torch.tensor(label_rows, dtype=torch.long),
        torch.tensor(lengths, dtype=torch.long),
    )



def iterate_chunks(
    items: list, size: int, shuffle: bool, rng: random.Random, max_tokens: int = 0
) -> Iterator[list]:
    """Делит список на пакеты, группируя предложения близкой длины.

    Args:
        items: Исходный список.
        size: Наибольшее число предложений в пакете.
        shuffle: Если True, список предварительно перемешивается, а пакеты выдаются
            в случайном порядке.
        rng: Генератор случайных чисел.
        max_tokens: Наибольшее число слов в пакете вместе с заполнением, ноль означает
            отсутствие ограничения. Ограничение защищает память видеокарты от длинных предложений.

    Returns:
        Итератор по пакетам. Внутри порции из пятидесяти пакетов предложения
        отсортированы по длине, поэтому заполнения почти нет и обучение идет быстрее.
    """
    order = list(items)
    if shuffle:
        rng.shuffle(order)
    pool_size = size * 50
    batches: list[list] = []
    for start in range(0, len(order), pool_size):
        pool = sorted(order[start:start + pool_size], key=sentence_length)
        current: list = []
        longest = 0
        for item in pool:
            length = sentence_length(item)
            too_big = max_tokens and max(longest, length) * (len(current) + 1) > max_tokens
            if current and (len(current) >= size or too_big):
                batches.append(current)
                current = []
                longest = 0
            current.append(item)
            longest = max(longest, length)
        if current:
            batches.append(current)
    if shuffle:
        rng.shuffle(batches)
    yield from batches
