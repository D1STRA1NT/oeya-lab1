"""Демонстрация работы лемматизатора на примерах.

Модуль обрабатывает входные тексты из папки lab1/examples, сохраняет результат в формате
задания и сравнивает его с эталонной разметкой, которая лежит рядом. По итогам
считается точность по тегам, леммам и обоим признакам, отдельно для слов, которых
нет в словаре. Для каждого текста строится картинка png с входом, выходом, разбором
по словам и точностью, ее можно вставить в отчет как скриншот результата.
"""

import html
import re

from ..analyze.lemmatizer import format_sentence, load_lemmatizer
from ..config import EXAMPLES_DIR, MODEL_PATH, RESULT_DIR
from ..prepare.resources import load_dictionary
from ..text_utils import normalize_word, read_lines, tokenize_line, write_lines
from .render import screenshot_page

DEMO_DIR = RESULT_DIR / "demo"
GOLD_RE = re.compile(r"(\S+?)\{(.+)=([A-Z]+)\}$")
SAMPLES = [
    ("example", "Пример из задания"),
    ("new", "Новый текст: е и ё, регистр, неизвестные слова"),
]

STYLE = """
body{font-family:Arial,sans-serif;margin:24px;color:#1a202c;background:white;width:1100px}
h1{font-size:20px;margin:0 0 12px}h2{font-size:15px;margin:18px 0 6px;color:#2b6cb0}
pre{background:#f7fafc;border:1px solid #e2e8f0;padding:10px;white-space:pre-wrap;margin:0;font-size:14px}
table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #cbd5e0;padding:3px 8px}
th{background:#edf2f7}.bad{background:#fed7d7}.unk{background:#fefcbf}.ok{background:#f0fff4}
.sum{font-size:15px;line-height:1.6}
"""


def parse_gold_line(line: str) -> list[tuple[str, str, str]]:
    """Разбирает строку эталонной разметки.

    Args:
        line: Строка вида токен, затем в фигурных скобках лемма, знак равенства и тег.

    Returns:
        Список троек токен, лемма, тег.
    """
    result = []
    for part in line.split():
        match = GOLD_RE.match(part)
        if match:
            result.append((match.group(1), match.group(2), match.group(3)))
    return result


def build_rows(analyzer, lines: list[str], gold_lines: list[str]) -> tuple[list[list[str]], list[dict]]:
    """Разбирает строки текста и сравнивает результат с эталоном.

    Args:
        analyzer: Загруженный лемматизатор.
        lines: Строки входного текста.
        gold_lines: Строки эталонной разметки, по одной на каждую входную строку.

    Returns:
        Выходные строки в формате задания и список записей по словам с полями
        token, lemma, tag, known, gold_lemma, gold_tag, tag_ok, lemma_ok.
    """
    sentences = [tokenize_line(line) for line in lines]
    analyses = analyzer.analyze_sentences(sentences)
    outputs = [format_sentence(tokens, result) for tokens, result in zip(sentences, analyses)]
    records = []
    for tokens, result, gold_line in zip(sentences, analyses, gold_lines):
        gold = parse_gold_line(gold_line)
        if len(gold) != len(tokens):
            raise ValueError(f"Число слов в эталоне ({len(gold)}) и в тексте ({len(tokens)}) не совпадает")
        for token, (lemma, tag), (_, gold_lemma, gold_tag) in zip(tokens, result, gold):
            records.append({
                "token": token, "lemma": lemma, "tag": tag,
                "known": bool(analyzer.dictionary.lookup(normalize_word(token))),
                "gold_lemma": gold_lemma, "gold_tag": gold_tag,
                "tag_ok": tag == gold_tag,
                "lemma_ok": normalize_word(lemma) == normalize_word(gold_lemma),
            })
    return outputs, records


def accuracy_text(records: list[dict]) -> str:
    """Считает точность и оформляет ее в виде строки.

    Args:
        records: Записи по словам.

    Returns:
        Строка с числом верных ответов и процентами по тегу, лемме и обоим признакам.
    """
    if not records:
        return "слов нет"
    total = len(records)
    tag = sum(item["tag_ok"] for item in records)
    lemma = sum(item["lemma_ok"] for item in records)
    both = sum(item["tag_ok"] and item["lemma_ok"] for item in records)
    return (
        f"слов {total}: тег {tag} ({tag / total:.1%}), лемма {lemma} ({lemma / total:.1%}), "
        f"тег и лемма {both} ({both / total:.1%})"
    )


def build_page(title: str, lines: list[str], outputs: list[str], records: list[dict]) -> str:
    """Собирает html страницу с результатами одного текста.

    Args:
        title: Заголовок страницы.
        lines: Строки входного текста.
        outputs: Выходные строки в формате задания.
        records: Записи по словам.

    Returns:
        Текст html страницы.
    """
    esc = html.escape
    rows = []
    for item in records:
        both_ok = item["tag_ok"] and item["lemma_ok"]
        css = "ok" if both_ok else "bad"
        rows.append(
            f'<tr class="{css}"><td>{esc(item["token"])}</td><td>{esc(item["lemma"])}</td>'
            f'<td>{item["tag"]}</td><td class="{"" if item["known"] else "unk"}">'
            f'{"в словаре" if item["known"] else "неизвестное"}</td>'
            f'<td>{esc(item["gold_lemma"])}</td><td>{item["gold_tag"]}</td>'
            f'<td>{"верно" if both_ok else "ошибка"}</td></tr>'
        )
    unknown = [item for item in records if not item["known"]]
    summary = [
        f"Все слова: {accuracy_text(records)}",
        f"Слова из словаря: {accuracy_text([i for i in records if i['known']])}",
        f"Неизвестные слова: {accuracy_text(unknown)}",
    ]
    return (
        f'<!doctype html><meta charset="utf-8"><style>{STYLE}</style><h1>{esc(title)}</h1>'
        f'<h2>Вход</h2><pre>{esc(chr(10).join(lines))}</pre>'
        f'<h2>Выход в формате задания</h2><pre>{esc(chr(10).join(outputs))}</pre>'
        f'<h2>Точность (сравнение с ручной разметкой, регистр и буква ё не учитываются)</h2>'
        f'<div class="sum">{"<br>".join(esc(line) for line in summary)}</div>'
        f'<h2>Разбор по словам (желтым отмечены слова, которых нет в словаре)</h2>'
        f'<table><tr><th>Слово</th><th>Лемма</th><th>Тег</th><th>Словарь</th>'
        f'<th>Эталон: лемма</th><th>Эталон: тег</th><th>Итог</th></tr>{"".join(rows)}</table>'
    )


def run_demo() -> None:
    """Обрабатывает все примеры, сохраняет результаты и картинки.

    Args:
        Нет.

    Returns:
        Ничего не возвращает.
    """
    from ..train import select_device

    dictionary = load_dictionary()
    analyzer = load_lemmatizer(MODEL_PATH, dictionary, select_device())
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    for name, title in SAMPLES:
        lines = read_lines(EXAMPLES_DIR / f"{name}_input.txt")
        gold_lines = read_lines(EXAMPLES_DIR / f"{name}_gold.txt")
        outputs, records = build_rows(analyzer, lines, gold_lines)
        write_lines(EXAMPLES_DIR / f"{name}_output.txt", outputs)
        print(f"\n{title}")
        print("\n".join(outputs))
        print(f"Точность: {accuracy_text(records)}")
        page = DEMO_DIR / f"{name}.html"
        page.write_text(build_page(title, lines, outputs, records), encoding="utf-8")
        height = 520 + 28 * len(records) + 40 * len(lines)
        if screenshot_page(page, DEMO_DIR / f"{name}.png", 1160, height):
            page.unlink()
            print(f"Картинка сохранена: {DEMO_DIR / (name + '.png')}")
