"""Нейросетевые модели на PyTorch.

В модуле две небольшие модели, которые помещаются в память видеокарты на 2 ГБ.
Первая разбирает предложение целиком и выбирает часть речи для каждого слова:
она учитывает слово, его буквы, регистр, соседей и теги, допустимые по словарю.
Вторая для слова без словаря выбирает правило лемматизации по его окончанию и тегу.
"""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ..config import CASE_FEATURE_COUNT, NUM_CHARS, PAD_ID, LemmaConfig, TaggerConfig

CONV_KERNELS = (2, 3, 4, 5)
FORBIDDEN_SCORE = -1e4


class TaggerNet(nn.Module):
    """Двунаправленная рекуррентная модель частеречной разметки."""

    def __init__(self, vocab_size: int, num_tags: int, cfg: TaggerConfig) -> None:
        """Создает слои модели.

        Args:
            vocab_size: Размер словаря слов обучающей выборки.
            num_tags: Число тегов.
            cfg: Параметры модели.

        Returns:
            Ничего не возвращает.
        """
        super().__init__()
        self.word_emb = nn.Embedding(vocab_size, cfg.word_dim, padding_idx=PAD_ID)
        self.char_emb = nn.Embedding(NUM_CHARS, cfg.char_dim, padding_idx=PAD_ID)
        self.convs = nn.ModuleList(
            [nn.Conv1d(cfg.char_dim, cfg.char_filters, kernel, padding=kernel // 2)
             for kernel in CONV_KERNELS]
        )
        self.cand_proj = nn.Linear(num_tags, cfg.cand_dim)
        in_dim = (
            cfg.word_dim + cfg.char_filters * len(CONV_KERNELS)
            + CASE_FEATURE_COUNT + cfg.cand_dim + 1
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.lstm = nn.LSTM(
            in_dim, cfg.hidden_dim, num_layers=cfg.num_layers, batch_first=True,
            bidirectional=True, dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.out = nn.Linear(2 * cfg.hidden_dim, num_tags)

    def encode_chars(self, char_ids: torch.Tensor) -> torch.Tensor:
        """Строит вектор слова по его символам сверточными фильтрами.

        Args:
            char_ids: Тензор номеров символов размера батч, слова, символы.

        Returns:
            Тензор размера батч, слова, число фильтров.
        """
        batch, length, width = char_ids.shape
        flat = char_ids.reshape(batch * length, width)
        emb = self.char_emb(flat).transpose(1, 2)
        pooled = [torch.relu(conv(emb)).max(dim=2).values for conv in self.convs]
        return torch.cat(pooled, dim=1).reshape(batch, length, -1)

    def forward(
        self,
        word_ids: torch.Tensor,
        char_ids: torch.Tensor,
        case_feats: torch.Tensor,
        cand_mask: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Считает оценки тегов для каждого слова.

        Args:
            word_ids: Номера слов, размер батч на слова.
            char_ids: Номера символов, размер батч, слова, символы.
            case_feats: Признаки регистра, размер батч, слова, три.
            cand_mask: Теги, допустимые по словарю, размер батч, слова, теги.
            lengths: Длины предложений без заполнения.

        Returns:
            Оценки тегов размера батч, слова, теги.
        """
        known = (cand_mask.sum(dim=-1, keepdim=True) > 0).float()
        features = torch.cat(
            [
                self.word_emb(word_ids),
                self.encode_chars(char_ids),
                case_feats,
                torch.relu(self.cand_proj(cand_mask)),
                known,
            ],
            dim=-1,
        )
        features = self.dropout(features)
        packed = pack_padded_sequence(
            features, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        output, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(output, batch_first=True, total_length=word_ids.shape[1])
        return self.out(self.dropout(output))


def apply_candidate_mask(logits: torch.Tensor, cand_mask: torch.Tensor) -> torch.Tensor:
    """Запрещает теги, которых нет в словаре, для слов с известными разборами.

    Args:
        logits: Оценки тегов размера батч, слова, теги.
        cand_mask: Теги, допустимые по словарю, той же формы.

    Returns:
        Оценки, где для известных слов недопустимые теги получили очень низкое значение.
    """
    has_candidates = cand_mask.sum(dim=-1, keepdim=True) > 0
    forbidden = has_candidates & (cand_mask == 0)
    return logits.masked_fill(forbidden, FORBIDDEN_SCORE)


class LemmaRuleNet(nn.Module):
    """Модель выбора правила лемматизации по окончанию слова и тегу."""

    def __init__(self, num_tags: int, num_rules: int, cfg: LemmaConfig) -> None:
        """Создает слои модели.

        Args:
            num_tags: Число тегов.
            num_rules: Число правил лемматизации.
            cfg: Параметры модели.

        Returns:
            Ничего не возвращает.
        """
        super().__init__()
        self.char_emb = nn.Embedding(NUM_CHARS, cfg.char_dim, padding_idx=PAD_ID)
        self.tag_emb = nn.Embedding(num_tags, cfg.tag_dim)
        in_dim = cfg.suffix_len * cfg.char_dim + cfg.tag_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, num_rules),
        )

    def forward(self, suffix_ids: torch.Tensor, tag_ids: torch.Tensor) -> torch.Tensor:
        """Считает оценки правил лемматизации.

        Args:
            suffix_ids: Номера последних символов слов, размер батч на длину окончания.
            tag_ids: Номера тегов слов, размер батч.

        Returns:
            Оценки правил размера батч на число правил.
        """
        chars = self.char_emb(suffix_ids).flatten(1)
        return self.net(torch.cat([chars, self.tag_emb(tag_ids)], dim=-1))
