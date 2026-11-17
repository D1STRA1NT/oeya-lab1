"""Правила лемматизации по окончанию слова.

Модуль описывает лемматизацию как правило вида: отрезать несколько букв с конца
слова и дописать другой хвост. Такие правила извлекаются из словаря и позволяют
получать лемму для слов, которых в словаре нет. Это реализация идеи сопоставления
с конца для неизвестных слов.
"""

import random
from collections import Counter

from ..prepare.dictionary import Dictionary
from ..tagset import TAG_ID
from ..text_utils import normalize_word

Rule = tuple[int, str]


def make_rule(form: str, lemma: str) -> Rule:
    """Строит правило, которое превращает словоформу в лемму.

    Args:
        form: Нормализованная словоформа.
        lemma: Нормализованная лемма.

    Returns:
        Пара из числа отрезаемых с конца букв и строки, которую нужно дописать.
    """
    prefix = 0
    limit = min(len(form), len(lemma))
    while prefix < limit and form[prefix] == lemma[prefix]:
        prefix += 1
    return len(form) - prefix, lemma[prefix:]


def apply_rule(form: str, rule: Rule) -> str:
    """Применяет правило к словоформе.

    Args:
        form: Нормализованная словоформа.
        rule: Пара из числа отрезаемых букв и дописываемого хвоста.

    Returns:
        Полученная лемма. Если отрезать нечего, возвращается исходная форма.
    """
    cut, tail = rule
    if cut > len(form):
        return form
    return form[: len(form) - cut] + tail


def collect_pairs(dictionary: Dictionary, count: int, seed: int) -> list[tuple[str, int, Rule]]:
    """Случайно выбирает словоформы словаря и строит для них правила.

    Args:
        dictionary: Словарь словоформ.
        count: Сколько словоформ нужно выбрать.
        seed: Начальное значение генератора случайных чисел.

    Returns:
        Список троек: словоформа, номер тега и правило получения леммы.
    """
    #бер 800 случайных слов из словаря
    rng = random.Random(seed)
    keys = list(dictionary.forms)
    chosen = keys if count >= len(keys) else rng.sample(keys, count)
    pairs: list[tuple[str, int, Rule]] = []
    for form in chosen:
        # оставляем только слова
        if not all(char.isalpha() or char == "-" for char in form):
            continue
        # получаем  пары(лемма, тег) для словоформы
        candidates = dictionary.lookup(form)
        if not candidates:
            continue
        # случайно выбираем одну
        lemma, tag = candidates[rng.randrange(len(candidates))]
        # к форме с номером тега добавляем правило:
        # (число отрезанных у формы букв, добавленное окончание леммы)
        pairs.append((form, TAG_ID[tag], make_rule(form, normalize_word(lemma))))
    return pairs


def build_rule_list(pairs: list[tuple[str, int, Rule]], min_count: int) -> list[Rule]:
    """Отбирает достаточно частые правила.

    Args:
        pairs: Тройки словоформа, номер тега и правило.
        min_count: Минимальное число повторов правила.

    Returns:
        Список правил, отсортированный по убыванию частоты.
    """
    counter = Counter(rule for _, _, rule in pairs)
    return [rule for rule, number in counter.most_common() if number >= min_count]
