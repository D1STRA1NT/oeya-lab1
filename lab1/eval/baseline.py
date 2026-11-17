"""Базовый метод без нейросетей.

Модуль реализует простое решение для сравнения: поиск в словаре, выбор самого
частого разбора в корпусе для неоднозначных слов и угадывание неизвестных слов
по окончанию. Результаты базового метода показывают, какую пользу дает нейросеть.
"""

from collections import Counter
from operator import itemgetter

from ..config import SEED, LemmaConfig
from ..prepare.corpus import GoldSentence, count_lemmas, count_tags, lemma_key
from ..prepare.dictionary import Dictionary
from ..tagset import TAG_ID, TAGS
from ..text_utils import normalize_word, restore_lemma_case, special_tag
from ..train.lemma_rules import Rule, apply_rule, collect_pairs

SUFFIX_LENGTHS = (3, 2, 1)
MIN_SUFFIX_SUPPORT = 3


class SuffixGuesser:
    """Угадывает тег и правило лемматизации по последним буквам слова."""

    def __init__(self) -> None:
        """Создает пустую таблицу окончаний.

        Args:
            Нет.

        Returns:
            Ничего не возвращает.
        """
        self.table: dict[str, Counter] = {}

    def fit(self, pairs: list[tuple[str, int, Rule]]) -> None:
        """Заполняет таблицу окончаний по примерам из словаря.

        Args:
            pairs: Тройки словоформа, номер тега и правило получения леммы.

        Returns:
            Ничего не возвращает.
        """
        for form, tag_id, rule in pairs:
            # для каждой формы заполняем таблицу окончаний
            for length in SUFFIX_LENGTHS:
                # для ключа-окончанияя из 3, потом 2, потом 1 букв
                if len(form) >= length:
                    # добавляем пары (номер тега, правило) и сколько раз каждая встретилась
                    self.table.setdefault(form[-length:], Counter())[(tag_id, rule)] += 1

    def guess(self, form: str) -> tuple[int, Rule]:
        """Выбирает самый частый разбор для слов с таким же окончанием.

        Args:
            form: Нормализованная словоформа.

        Returns:
            Номер тега и правило. Если окончание не встречалось, предлагается существительное
            без изменения слова.
        """
        for length in SUFFIX_LENGTHS:
            counter = self.table.get(form[-length:])
            if counter and sum(counter.values()) >= MIN_SUFFIX_SUPPORT:
                return counter.most_common(1)[0][0]
        return TAG_ID["S"], (0, "")


class FrequencyBaseline:
    """Словарь, частотный выбор разбора и угадывание по окончанию."""

    def __init__(
        self,
        dictionary: Dictionary,
        lemma_counts: dict[str, int],
        tag_counts: dict[str, int],
        guesser: SuffixGuesser,
    ) -> None:
        """Запоминает все таблицы, нужные для разбора.

        Args:
            dictionary: Словарь словоформ.
            lemma_counts: Частоты разборов слов в обучающем корпусе.
            tag_counts: Частоты тегов в обучающем корпусе.
            guesser: Угадыватель по окончаниям для неизвестных слов.

        Returns:
            Ничего не возвращает.
        """
        self.dictionary = dictionary
        self.lemma_counts = lemma_counts
        self.tag_counts = tag_counts
        self.guesser = guesser

    @staticmethod
    def from_corpus(
        dictionary: Dictionary, train_sentences: list[GoldSentence], cfg: LemmaConfig
    ) -> "FrequencyBaseline":
        """Строит базовый метод по обучающей части корпуса.

        Args:
            dictionary: Словарь словоформ.
            train_sentences: Обучающие предложения.
            cfg: Параметры выборки примеров из словаря.

        Returns:
            Готовый базовый метод.
        """
        guesser = SuffixGuesser()
        guesser.fit(collect_pairs(dictionary, cfg.pairs_count, SEED))
        return FrequencyBaseline(
            dictionary, count_lemmas(train_sentences), count_tags(train_sentences), guesser
        )

    def choose_known(self, form: str, candidates: list[tuple[str, str]]) -> tuple[str, str]:
        """Выбирает самый частый разбор среди вариантов из словаря.

        Args:
            form: Нормализованная словоформа.
            candidates: Пары лемма и тег из словаря.

        Returns:
            Выбранная пара лемма и тег. При равных частотах побеждает более частый тег.
        """
        scored = [
            (
                (
                    self.lemma_counts.get(lemma_key(form, normalize_word(lemma), tag), 0),
                    self.tag_counts.get(tag, 0),
                ),
                lemma,
                tag,
            )
            for lemma, tag in candidates
        ]
        _, lemma, tag = max(scored, key=itemgetter(0))
        return lemma, tag

    def analyze_sentences(
        self, sentences: list[list[str]], disabled: list[set[int]] | None = None
    ) -> list[list[tuple[str, str]]]:
        """Разбирает предложения.

        Args:
            sentences: Предложения как списки токенов без знаков препинания.
            disabled: Для каждого предложения номера слов, у которых словарь скрыт.

        Returns:
            Для каждого слова пара лемма и тег.
        """
        results = []
        for number, tokens in enumerate(sentences):
            hidden = disabled[number] if disabled else set()
            analyses = []
            for index, token in enumerate(tokens):
                special = special_tag(token)
                if special:
                    analyses.append((token.lower(), special))
                    continue
                form = normalize_word(token)
                candidates = [] if index in hidden else self.dictionary.lookup(form)
                if candidates:
                    lemma, tag = self.choose_known(form, candidates)
                    analyses.append((restore_lemma_case(lemma, token, tag, index), tag))
                    continue
                tag_id, rule = self.guesser.guess(form)
                tag = TAGS[tag_id]
                lemma = restore_lemma_case(apply_rule(form, rule), token, tag, index)
                analyses.append((lemma, tag))
            results.append(analyses)
        return results
