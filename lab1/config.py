"""Конфигурация проекта.

Модуль хранит пути к данным и моделям, адреса загрузки данных, алфавит символов
и гиперпараметры обучения. Остальные модули берут настройки отсюда, поэтому
параметры эксперимента меняются в одном месте.
"""

from dataclasses import dataclass
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent
DATA_DIR = PACKAGE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
MODEL_DIR = DATA_DIR / "models"
LOG_DIR = DATA_DIR / "logs"
EXAMPLES_DIR = PACKAGE_DIR / "examples"
RESULT_DIR = PACKAGE_DIR / "results"

DICT_CACHE = CACHE_DIR / "dictionary.pkl"
CORPUS_CACHE = CACHE_DIR / "corpus.pkl"
MODEL_PATH = MODEL_DIR / "lab1_model.pt"

# корпус SynTagRus из Universal Dependencies, файлы лежат на github
SYNTAGRUS_URL = "https://raw.githubusercontent.com/UniversalDependencies/UD_Russian-SynTagRus/master/"
TRAIN_NAMES = [
    "ru_syntagrus-ud-train-a.conllu",
    "ru_syntagrus-ud-train-b.conllu",
    "ru_syntagrus-ud-train-c.conllu",
]
DEV_NAMES = ["ru_syntagrus-ud-dev.conllu"]
TEST_NAMES = ["ru_syntagrus-ud-test.conllu"]
# словарь изменяемых слов UniMorph, составлен по Википедии
UNIMORPH_URL = "https://raw.githubusercontent.com/unimorph/rus/master/rus"
UNIMORPH_NAME = "unimorph_rus.tsv"

SEED = 42

# служебные номера символов и слов, общие для всех моделей
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
CHAR_ALPHABET = "абвгдежзийклмнопрстуфхцчшщъыьэюяabcdefghijklmnopqrstuvwxyz0123456789-"
NUM_CHARS = len(CHAR_ALPHABET) + 4
CASE_FEATURE_COUNT = 3
IGNORE_LABEL = -100


@dataclass
class TaggerConfig:
    """Параметры контекстной модели частеречной разметки."""

    word_dim: int = 128            # длина вектора целого слова
    char_dim: int = 48             # длина вектора одной буквы
    char_filters: int = 64         # сколько признаков снимает каждый фильтр по буквам
    cand_dim: int = 32             # длина вектора тегов, допустимых по словарю
    
    hidden_dim: int = 192          # размер памяти LSTM в одну сторону
    num_layers: int = 2            # сколько слоёв LSTM стоят друг за другом
    dropout: float = 0.33          # доля нейронов, которые случайно выключаются при обучении
    max_chars: int = 24            # сколько символов слова подаётся в сеть
    
    word_min_freq: int = 2         # слово попадает в словарь, если встретилось хотя бы столько раз
    word_unk_prob: float = 0.10    # вероятность спрятать номер слова; теги словаря остаются
    dict_drop_prob: float = 0.12   # вероятность спрятать и номер слова, и теги словаря
    
    batch_size: int = 64           # максимум предложений в одном шаге
    max_batch_tokens: int = 1500   # максимум слов в шаге, чтобы пакет поместился в память
    epochs: int = 12               # сколько раз сеть проходит весь обучающий корпус
    lr: float = 2e-3               # начальная величина шага изменения весов
    lr_decay: float = 0.9          # после каждой эпохи шаг умножается на это число


@dataclass
class LemmaConfig:
    """Параметры модели выбора правила лемматизации для неизвестных слов."""

    suffix_len: int = 12
    char_dim: int = 48
    tag_dim: int = 24
    hidden_dim: int = 512
    dropout: float = 0.2
    pairs_count: int = 800_000
    min_rule_count: int = 3
    batch_size: int = 1024
    epochs: int = 6
    lr: float = 2e-3
