"""Обучение нейросетевых моделей.

Модуль обучает две модели: контекстную модель частеречной разметки на размеченном
корпусе и модель правил лемматизации на словоформах словаря. Обучение
идет на видеокарте, если PyTorch ее видит, иначе на процессоре. Результаты
сохраняются одним файлом вместе со словарем слов и таблицей частот разборов.
"""

import random
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from ..config import IGNORE_LABEL, MODEL_PATH, SEED, LemmaConfig, TaggerConfig
from ..prepare.corpus import GoldSentence, count_lemmas
from ..prepare.dictionary import Dictionary
from ..tagset import TAGS
from ..text_utils import normalize_word
from .batching import iterate_chunks, make_batch
from .features import PreparedSentence, SentenceEncoder, Vocabulary, encode_suffix, gold_label_ids
from .lemma_rules import Rule, build_rule_list, collect_pairs
from .model import LemmaRuleNet, TaggerNet, apply_candidate_mask

PROGRESS_STEPS = 200


def select_device() -> torch.device:
    """Выбирает устройство для вычислений.

    Args:
        Нет.

    Returns:
        Видеокарта, если она доступна, иначе процессор.
    """
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(
            f"Используется видеокарта: {torch.cuda.get_device_name(0)}, "
            f"свободно памяти {free // 2**20} из {total // 2**20} МБ"
        )
        return torch.device("cuda")
    print("Видеокарта не найдена, обучение пойдет на процессоре. Нужна сборка PyTorch с CUDA.")
    return torch.device("cpu")


def build_vocabulary(sentences: list[GoldSentence], min_freq: int) -> Vocabulary:
    """Строит словарь слов по обучающим предложениям.

    Args:
        sentences: Обучающие предложения.
        min_freq: Минимальная частота слова.

    Returns:
        Словарь слов с номерами.
    """
    counter: Counter = Counter()
    for sentence in sentences:
        counter.update(normalize_word(token.text) for token in sentence.tokens)
    return Vocabulary.build(counter, min_freq)


def prepare_sentences(
    encoder: SentenceEncoder, sentences: list[GoldSentence]
) -> list[PreparedSentence]:
    """Переводит размеченные предложения в числовые признаки.

    Args:
        encoder: Кодировщик предложений.
        sentences: Предложения с эталонной разметкой.

    Returns:
        Список подготовленных предложений.
    """
    return [
        encoder.prepare([token.text for token in sentence.tokens], gold_label_ids(sentence))
        for sentence in sentences
        if sentence.tokens
    ]


def evaluate_tag_accuracy(
    model: TaggerNet, prepared: list[PreparedSentence], cfg: TaggerConfig, device: torch.device
) -> float:
    """Считает долю верно угаданных тегов.

    Args:
        model: Обученная или обучаемая модель разметки.
        prepared: Подготовленные предложения с эталонными тегами.
        cfg: Параметры модели.
        device: Устройство вычислений.

    Returns:
        Доля верных тегов среди слов с однозначной разметкой.
    """
    model.eval()
    rng = random.Random(0)
    correct = 0
    total = 0
    with torch.no_grad():
        for chunk in iterate_chunks(prepared, cfg.batch_size * 2, False, rng, cfg.max_batch_tokens * 2):
            batch = make_batch(chunk, len(TAGS), cfg, False, rng).to(device)
            logits = model(
                batch.word_ids, batch.char_ids, batch.case_feats, batch.cand_mask, batch.lengths
            )
            predicted = apply_candidate_mask(logits, batch.cand_mask).argmax(dim=-1)
            labeled = batch.labels != IGNORE_LABEL
            correct += ((predicted == batch.labels) & labeled).sum().item()
            total += labeled.sum().item()
    return correct / max(total, 1)


def train_tagger(
    dictionary: Dictionary,
    train_sentences: list[GoldSentence],
    dev_sentences: list[GoldSentence],
    cfg: TaggerConfig,
    device: torch.device,
) -> tuple[TaggerNet, Vocabulary]:
    """Обучает модель частеречной разметки.

    Args:
        dictionary: Словарь словоформ.
        train_sentences: Обучающие предложения.
        dev_sentences: Отладочные предложения для выбора лучшей эпохи.
        cfg: Параметры модели и обучения.
        device: Устройство вычислений.

    Returns:
        Обученная модель с лучшей точностью на отладочной выборке и словарь слов.
    """
    # фиксируем генераторы, чтобы перемешивание и инициализация весов повторялись
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    vocab = build_vocabulary(train_sentences, cfg.word_min_freq)
    encoder = SentenceEncoder(dictionary, vocab, cfg.max_chars)
    train_prepared = prepare_sentences(encoder, train_sentences)
    dev_prepared = prepare_sentences(encoder, dev_sentences)
    print(f"Слов в словаре модели: {len(vocab)}")

    model = TaggerNet(len(vocab), len(TAGS), cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.lr_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)
    best_accuracy = -1.0
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        started = time.time()
        for chunk in iterate_chunks(train_prepared, cfg.batch_size, True, rng, cfg.max_batch_tokens):
            batch = make_batch(chunk, len(TAGS), cfg, True, rng).to(device)
            if (batch.labels != IGNORE_LABEL).sum().item() == 0:
                continue
            logits = model(
                batch.word_ids, batch.char_ids, batch.case_feats, batch.cand_mask, batch.lengths
            )
            loss = criterion(logits.reshape(-1, len(TAGS)), batch.labels.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
            steps += 1
            if steps % PROGRESS_STEPS == 0:
                print(f"  эпоха {epoch}, шаг {steps}, средние потери {total_loss / steps:.4f}")
        scheduler.step()
        accuracy = evaluate_tag_accuracy(model, dev_prepared, cfg, device)
        seconds = time.time() - started
        memory = torch.cuda.max_memory_allocated() // 2**20 if device.type == "cuda" else 0
        print(
            f"Эпоха {epoch}: потери {total_loss / max(steps, 1):.4f}, "
            f"точность тегов на отладке {accuracy:.4f}, время {seconds / 60:.1f} мин, "
            f"пик памяти видеокарты {memory} МБ"
        )
        # запоминаем лучшие веса, чтобы последние эпохи не ухудшили результат
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, vocab


def train_lemma_net(
    dictionary: Dictionary, cfg: LemmaConfig, device: torch.device
) -> tuple[LemmaRuleNet, list[Rule]]:
    """Обучает модель выбора правила лемматизации на словоформах словаря.

    Args:
        dictionary: Словарь словоформ.
        cfg: Параметры модели и обучения.
        device: Устройство вычислений.

    Returns:
        Обученная модель и список правил, номера которых она предсказывает.
    """
    torch.manual_seed(SEED)
    pairs = collect_pairs(dictionary, cfg.pairs_count, SEED)
    rules = build_rule_list(pairs, cfg.min_rule_count)
    rule_id = {rule: index for index, rule in enumerate(rules)}
    kept = [(form, tag, rule_id[rule]) for form, tag, rule in pairs if rule in rule_id]
    print(f"Примеров для обучения лемматизации: {len(kept)}, правил: {len(rules)}")

    suffixes = torch.tensor(
        [encode_suffix(form, cfg.suffix_len) for form, _, _ in kept], dtype=torch.long, device=device
    )
    tags = torch.tensor([tag for _, tag, _ in kept], dtype=torch.long, device=device)
    labels = torch.tensor([label for _, _, label in kept], dtype=torch.long, device=device)

    # последние два процента примеров не участвуют в обучении и нужны для контроля
    valid_size = max(len(kept) // 50, 1)
    train_size = len(kept) - valid_size
    model = LemmaRuleNet(len(TAGS), len(rules), cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        order = torch.randperm(train_size, device=device)
        total_loss = 0.0
        steps = 0
        for start in range(0, train_size, cfg.batch_size):
            index = order[start:start + cfg.batch_size]
            logits = model(suffixes[index], tags[index])
            loss = criterion(logits, labels[index])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1
            if steps % PROGRESS_STEPS == 0:
                print(f"  эпоха {epoch}, шаг {steps}, средние потери {total_loss / steps:.4f}")
        model.eval()
        with torch.no_grad():
            logits = model(suffixes[train_size:], tags[train_size:])
            accuracy = (logits.argmax(dim=-1) == labels[train_size:]).float().mean().item()
        print(f"Эпоха {epoch}: потери {total_loss / max(steps, 1):.4f}, точность правил {accuracy:.4f}")
    return model, rules


def save_bundle(
    path: Path,
    tagger: TaggerNet,
    vocab: Vocabulary,
    lemma_net: LemmaRuleNet,
    rules: list[Rule],
    lemma_counts: dict[str, int],
    tagger_cfg: TaggerConfig,
    lemma_cfg: LemmaConfig,
) -> None:
    """Сохраняет обе модели и вспомогательные данные в один файл.

    Args:
        path: Путь к файлу модели.
        tagger: Обученная модель разметки.
        vocab: Словарь слов модели разметки.
        lemma_net: Обученная модель правил лемматизации.
        rules: Список правил лемматизации.
        lemma_counts: Частоты разборов в обучающем корпусе.
        tagger_cfg: Параметры модели разметки.
        lemma_cfg: Параметры модели лемматизации.

    Returns:
        Ничего не возвращает.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "tagger_cfg": asdict(tagger_cfg),
        "lemma_cfg": asdict(lemma_cfg),
        "word_itos": vocab.itos,
        "tagger_state": {key: value.detach().cpu() for key, value in tagger.state_dict().items()},
        "lemma_state": {key: value.detach().cpu() for key, value in lemma_net.state_dict().items()},
        "rules": [[cut, tail] for cut, tail in rules],
        "lemma_counts": lemma_counts,
        "tags": TAGS,
    }
    torch.save(bundle, path)
    print(f"Модель сохранена: {path}")


def train_all(
    dictionary: Dictionary,
    train_sentences: list[GoldSentence],
    dev_sentences: list[GoldSentence],
    path: Path = MODEL_PATH,
) -> None:
    """Обучает обе модели и сохраняет их.

    Args:
        dictionary: Словарь словоформ.
        train_sentences: Обучающие предложения.
        dev_sentences: Отладочные предложения.
        path: Путь к файлу, куда сохраняются модели.

    Returns:
        Ничего не возвращает.
    """
    device = select_device()
    tagger_cfg = TaggerConfig()
    lemma_cfg = LemmaConfig()
    print("Обучение модели частеречной разметки")
    tagger, vocab = train_tagger(dictionary, train_sentences, dev_sentences, tagger_cfg, device)
    print("Обучение модели лемматизации неизвестных слов")
    lemma_net, rules = train_lemma_net(dictionary, lemma_cfg, device)
    lemma_counts = count_lemmas(train_sentences)
    save_bundle(path, tagger, vocab, lemma_net, rules, lemma_counts, tagger_cfg, lemma_cfg)
