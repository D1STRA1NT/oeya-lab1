"""Оценка качества лемматизации и частеречной разметки.

Модуль сравнивает ответы системы с эталонной разметкой тестовой части корпуса и
считает точность по тегам, леммам и по обоим признакам сразу, а также точность,
полноту и F1 по каждому тегу. Результаты делятся на слова из словаря, слова без
словаря и числа. Чтобы проверить работу с неизвестными словами,
словарь можно скрыть у заданной доли слов, и тогда они ведут себя как незнакомые.
"""

import random

from ..prepare.corpus import GoldSentence
from ..prepare.dictionary import Dictionary
from ..tagset import TAGS
from ..text_utils import normalize_word, special_tag

CATEGORY_TITLES = {
    "all": "Все слова",
    "known": "Слова, найденные в словаре",
    "unknown": "Слова без словаря",
    "special": "Числа",
}


def empty_counters() -> dict[str, int]:
    """Создает пустые счетчики верных ответов.

    Args:
        Нет.

    Returns:
        Словарь счетчиков: слова, верные теги, верные леммы, верные оба признака.
    """
    return {"tokens": 0, "tag_ok": 0, "lemma_ok": 0, "both_ok": 0}


def build_disabled_sets(
    sentences: list[list[str]], rate: float, seed: int
) -> list[set[int]]:
    """Случайно выбирает слова, у которых нужно скрыть словарь.

    Args:
        sentences: Предложения как списки токенов.
        rate: Доля слов, которые нужно сделать неизвестными, от нуля до единицы.
        seed: Начальное значение генератора случайных чисел.

    Returns:
        Для каждого предложения множество номеров скрытых слов. При одинаковом seed
        разные системы получают одинаковые множества.
    """
    rng = random.Random(seed)
    if rate <= 0:
        return [set() for _ in sentences]
    return [{index for index in range(len(tokens)) if rng.random() < rate} for tokens in sentences]


def evaluate_analyzer(
    analyzer,
    dictionary: Dictionary,
    sentences: list[GoldSentence],
    oov_rate: float,
    seed: int,
) -> dict:
    """Оценивает систему на размеченных предложениях.

    Args:
        analyzer: Объект с методом analyze_sentences.
        dictionary: Словарь словоформ, нужный для деления слов на категории.
        sentences: Предложения с эталонной разметкой.
        oov_rate: Доля слов, у которых словарь скрывается при разборе.
        seed: Начальное значение генератора случайных чисел.

    Returns:
        Словарь с общими счетчиками по категориям и статистикой по тегам.
    """
    token_lists = [[token.text for token in sentence.tokens] for sentence in sentences]
    disabled = build_disabled_sets(token_lists, oov_rate, seed)
    predictions = analyzer.analyze_sentences(token_lists, disabled)
    counters = {name: empty_counters() for name in CATEGORY_TITLES}
    per_tag = {tag: {"tp": 0, "predicted": 0, "gold": 0} for tag in TAGS}
    for sentence, analyses, hidden in zip(sentences, predictions, disabled):
        for index, (gold, (lemma, tag)) in enumerate(zip(sentence.tokens, analyses)):
            if gold.tag is None or gold.lemma is None:
                continue
            # категория определяет, было ли у слова доступно знание словаря
            if special_tag(gold.text):
                category = "special"
            elif index in hidden or not dictionary.lookup(normalize_word(gold.text)):
                category = "unknown"
            else:
                category = "known"
            tag_ok = tag == gold.tag
            lemma_ok = normalize_word(lemma) == normalize_word(gold.lemma)
            for name in ("all", category):
                bucket = counters[name]
                bucket["tokens"] += 1
                bucket["tag_ok"] += tag_ok
                bucket["lemma_ok"] += lemma_ok
                bucket["both_ok"] += tag_ok and lemma_ok
            per_tag[gold.tag]["gold"] += 1
            per_tag[tag]["predicted"] += 1
            per_tag[gold.tag]["tp"] += tag_ok
    return {"counters": counters, "per_tag": per_tag}


def format_report(title: str, result: dict) -> str:
    """Готовит текстовый отчет об оценке.

    Args:
        title: Заголовок отчета.
        result: Результат функции evaluate_analyzer.

    Returns:
        Многострочный текст с точностью по категориям и таблицей по тегам.
    """
    lines = [title]
    for name, caption in CATEGORY_TITLES.items():
        bucket = result["counters"][name]
        total = bucket["tokens"]
        if not total:
            continue
        lines.append(
            f"  {caption}: слов {total}, тег {bucket['tag_ok'] / total:.4f}, "
            f"лемма {bucket['lemma_ok'] / total:.4f}, оба признака {bucket['both_ok'] / total:.4f}"
        )
    lines.append("  Тег    точность  полнота   F1      слов")
    for tag, stats in result["per_tag"].items():
        if not stats["gold"] and not stats["predicted"]:
            continue
        precision = stats["tp"] / stats["predicted"] if stats["predicted"] else 0.0
        recall = stats["tp"] / stats["gold"] if stats["gold"] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        lines.append(f"  {tag:<6} {precision:<9.4f} {recall:<9.4f} {f1:<7.4f} {stats['gold']}")
    return "\n".join(lines)
