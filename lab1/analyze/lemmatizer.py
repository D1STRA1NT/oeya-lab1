"""Итоговый лемматизатор и частеречный теггер.

Модуль собирает все части решения в один конвейер. Для каждого слова словарь
дает возможные разборы, контекстная нейросеть выбирает тег, а лемма
берется из словаря. Если слова в словаре нет, лемма строится по правилу, которое
выбирает вторая нейросеть по окончанию слова. Модуль также форматирует результат
в виде токена с леммой и тегом в фигурных скобках для сдачи работы.
"""

import random
from operator import itemgetter
from pathlib import Path

import torch

from ..config import LemmaConfig, TaggerConfig
from ..prepare.corpus import lemma_key
from ..prepare.dictionary import Dictionary
from ..tagset import TAGS
from ..text_utils import normalize_word, restore_lemma_case, special_tag, tokenize_line
from ..train.batching import make_batch
from ..train.features import SentenceEncoder, Vocabulary, encode_suffix
from ..train.lemma_rules import Rule, apply_rule
from ..train.model import LemmaRuleNet, TaggerNet, apply_candidate_mask

SENTENCE_CHUNK = 16
RULE_CHUNK = 2048


class NeuralLemmatizer:
    """Лемматизатор на основе словаря и двух нейросетевых моделей."""

    def __init__(
        self,
        dictionary: Dictionary,
        tagger: TaggerNet,
        lemma_net: LemmaRuleNet,
        vocab: Vocabulary,
        rules: list[Rule],
        lemma_counts: dict[str, int],
        tagger_cfg: TaggerConfig,
        lemma_cfg: LemmaConfig,
        device: torch.device,
    ) -> None:
        """Запоминает все компоненты конвейера и переводит модели в режим применения.

        Args:
            dictionary: Словарь словоформ.
            tagger: Модель частеречной разметки.
            lemma_net: Модель правил лемматизации.
            vocab: Словарь слов модели разметки.
            rules: Список правил лемматизации.
            lemma_counts: Частоты разборов в обучающем корпусе.
            tagger_cfg: Параметры модели разметки.
            lemma_cfg: Параметры модели лемматизации.
            device: Устройство вычислений.

        Returns:
            Ничего не возвращает.
        """
        self.dictionary = dictionary
        self.tagger = tagger.to(device).eval()
        self.lemma_net = lemma_net.to(device).eval()
        self.rules = rules
        self.lemma_counts = lemma_counts
        self.tagger_cfg = tagger_cfg
        self.lemma_cfg = lemma_cfg
        self.device = device
        self.encoder = SentenceEncoder(dictionary, vocab, tagger_cfg.max_chars)

    def pick_lemma(self, form: str, tag: str, lemmas: list[str]) -> str:
        """Выбирает лемму среди вариантов словаря с одним и тем же тегом.

        Args:
            form: Нормализованная словоформа.
            tag: Выбранный тег.
            lemmas: Варианты лемм из словаря.

        Returns:
            Самая частая в корпусе лемма, а при равенстве частот первая из списка.
        """
        if len(lemmas) == 1:
            return lemmas[0]
        scored = [
            (self.lemma_counts.get(lemma_key(form, normalize_word(lemma), tag), 0), lemma)
            for lemma in lemmas
        ]
        return max(scored, key=itemgetter(0))[1]

    def predict_tags(
        self, sentences: list[list[str]], disabled: list[set[int]]
    ) -> list[list[int]]:
        """Выбирает номера тегов для порции предложений.

        Args:
            sentences: Предложения как списки токенов.
            disabled: Для каждого предложения номера слов со скрытым словарем.

        Returns:
            Для каждого предложения список номеров тегов.
        """
        prepared = [
            self.encoder.prepare(tokens, None, hidden) for tokens, hidden in zip(sentences, disabled)
        ]
        batch = make_batch(prepared, len(TAGS), self.tagger_cfg, False, random.Random(0)).to(self.device)
        with torch.no_grad():
            logits = self.tagger(
                batch.word_ids, batch.char_ids, batch.case_feats, batch.cand_mask, batch.lengths
            )
            predicted = apply_candidate_mask(logits, batch.cand_mask).argmax(dim=-1).cpu().tolist()
        return [row[: len(tokens)] for row, tokens in zip(predicted, sentences)]

    def predict_rules(self, requests: list[tuple[str, int]]) -> list[Rule]:
        """Выбирает правила лемматизации для слов без словаря.

        Args:
            requests: Пары из нормализованной словоформы и номера тега.

        Returns:
            Список правил в том же порядке, что и запросы.
        """
        chosen: list[Rule] = []
        for start in range(0, len(requests), RULE_CHUNK):
            part = requests[start:start + RULE_CHUNK]
            suffixes = torch.tensor(
                [encode_suffix(form, self.lemma_cfg.suffix_len) for form, _ in part],
                dtype=torch.long, device=self.device,
            )
            tags = torch.tensor([tag for _, tag in part], dtype=torch.long, device=self.device)
            with torch.no_grad():
                rule_ids = self.lemma_net(suffixes, tags).argmax(dim=-1).cpu().tolist()
            chosen.extend(tuple(self.rules[rule_id]) for rule_id in rule_ids)
        return chosen

    def analyze_sentences(
        self, sentences: list[list[str]], disabled: list[set[int]] | None = None
    ) -> list[list[tuple[str, str]]]:
        """Разбирает предложения: находит лемму и тег каждого слова.

        Args:
            sentences: Предложения как списки токенов без знаков препинания.
            disabled: Для каждого предложения номера слов, у которых словарь скрыт.

        Returns:
            Для каждого слова пара лемма и тег.
        """
        results: list[list[tuple[str, str]]] = []
        for start in range(0, len(sentences), SENTENCE_CHUNK):
            chunk = sentences[start:start + SENTENCE_CHUNK]
            hidden = [disabled[start + offset] if disabled else set() for offset in range(len(chunk))]
            tag_rows = self.predict_tags(chunk, hidden)
            analyses = [[("", "")] * len(tokens) for tokens in chunk]
            pending: list[tuple[int, int]] = []
            requests: list[tuple[str, int]] = []
            for number, tokens in enumerate(chunk):
                for index, token in enumerate(tokens):
                    tag_id = tag_rows[number][index]
                    tag = TAGS[tag_id]
                    # у чисел лемма совпадает с самим токеном, а тег выбирает модель по контексту,
                    # так как годы в корпусе размечены как прилагательные
                    if special_tag(token):
                        analyses[number][index] = (token.lower(), tag)
                        continue
                    form = normalize_word(token)
                    candidates = [] if index in hidden[number] else self.dictionary.lookup(form)
                    lemmas = [lemma for lemma, cand_tag in candidates if cand_tag == tag]
                    if lemmas:
                        lemma = self.pick_lemma(form, tag, lemmas)
                        analyses[number][index] = (restore_lemma_case(lemma, token, tag, index), tag)
                    else:
                        # слова нет в словаре, лемму строит вторая модель по окончанию
                        pending.append((number, index))
                        requests.append((form, tag_id))
            rules = self.predict_rules(requests) if requests else []
            for (number, index), (form, tag_id), rule in zip(pending, requests, rules):
                tag = TAGS[tag_id]
                lemma = restore_lemma_case(apply_rule(form, rule), chunk[number][index], tag, index)
                analyses[number][index] = (lemma, tag)
            results.extend(analyses)
        return results


def load_lemmatizer(
    path: Path, dictionary: Dictionary, device: torch.device
) -> NeuralLemmatizer:
    """Загружает обученные модели и собирает из них лемматизатор.

    Args:
        path: Путь к файлу модели, созданному при обучении.
        dictionary: Словарь словоформ.
        device: Устройство вычислений.

    Returns:
        Готовый к работе лемматизатор.
    """
    if not path.exists():
        raise FileNotFoundError(f"Нет файла модели {path}. Сначала выполните команду train.")
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    tagger_cfg = TaggerConfig(**bundle["tagger_cfg"])
    lemma_cfg = LemmaConfig(**bundle["lemma_cfg"])
    vocab = Vocabulary(bundle["word_itos"])
    rules = [(int(cut), str(tail)) for cut, tail in bundle["rules"]]
    tagger = TaggerNet(len(vocab), len(TAGS), tagger_cfg)
    tagger.load_state_dict(bundle["tagger_state"])
    lemma_net = LemmaRuleNet(len(TAGS), len(rules), lemma_cfg)
    lemma_net.load_state_dict(bundle["lemma_state"])
    return NeuralLemmatizer(
        dictionary, tagger, lemma_net, vocab, rules, bundle["lemma_counts"],
        tagger_cfg, lemma_cfg, device,
    )


def format_sentence(tokens: list[str], analyses: list[tuple[str, str]]) -> str:
    """Форматирует разбор предложения: за каждым токеном в фигурных скобках идут лемма и тег.

    Args:
        tokens: Токены предложения без знаков препинания.
        analyses: Пары лемма и тег для каждого токена.

    Returns:
        Строка, в которой разборы слов разделены пробелами.
    """
    return " ".join(f"{token}{{{lemma}={tag}}}" for token, (lemma, tag) in zip(tokens, analyses))


def process_lines(analyzer, lines: list[str]) -> list[str]:
    """Обрабатывает строки входного файла задания.

    Args:
        analyzer: Любой объект с методом analyze_sentences, например нейросетевой
            лемматизатор или базовый метод.
        lines: Строки входного файла, в каждой одно предложение.

    Returns:
        Строки результата в формате задания. Пустые входные строки остаются пустыми.
    """
    sentences = [tokenize_line(line) for line in lines]
    non_empty = [tokens for tokens in sentences if tokens]
    analyses = iter(analyzer.analyze_sentences(non_empty))
    output = []
    for tokens in sentences:
        output.append(format_sentence(tokens, next(analyses)) if tokens else "")
    return output
