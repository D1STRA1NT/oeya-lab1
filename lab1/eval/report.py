"""Таблицы и графики по результатам обучения и оценки.

Модуль читает журналы train.log и evaluate.log, которые создает обучение, и строит
по ним таблицы в формате markdown и графики в формате svg. Графики рисуются без
сторонних библиотек, поэтому для отчета не нужно ничего устанавливать. Результаты
сохраняются в папку lab1/results.
"""

import re

from ..config import LOG_DIR, RESULT_DIR
from .render import svg_to_png

EPOCH_RE = re.compile(r"^Эпоха (\d+): потери ([\d.]+), (?:точность тегов на отладке|точность правил) ([\d.]+)")
TITLE_RE = re.compile(r"^(Базовый метод|Нейросеть), (.+)$")
CATEGORY_RE = re.compile(
    r"^\s+(.+?): слов (\d+), тег ([\d.]+), лемма ([\d.]+), оба признака ([\d.]+)$"
)
TAG_RE = re.compile(r"^\s+([A-Z]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)$")

COLORS = ["#2b6cb0", "#dd6b20", "#2f855a", "#805ad5"]


def read_log(name: str) -> list[str]:
    """Читает журнал из папки lab1/data/logs.

    Args:
        name: Имя файла журнала.

    Returns:
        Список строк журнала.
    """
    path = LOG_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Нет файла {path}. Сначала выполните обучение и оценку.")
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def parse_training(lines: list[str]) -> tuple[list[tuple[int, float, float]], list[tuple[int, float, float]]]:
    """Разбирает журнал обучения на две таблицы по эпохам.

    Args:
        lines: Строки журнала train.log.

    Returns:
        Две таблицы из троек эпоха, потери, точность: для модели разметки и для модели
        лемматизации. Модели различаются по тому, что у второй номера эпох начинаются заново.
    """
    tagger: list[tuple[int, float, float]] = []
    lemma: list[tuple[int, float, float]] = []
    current = tagger
    for line in lines:
        match = EPOCH_RE.match(line)
        if not match:
            continue
        epoch = int(match.group(1))
        if epoch == 1 and tagger and current is tagger:
            current = lemma
        current.append((epoch, float(match.group(2)), float(match.group(3))))
    return tagger, lemma


def parse_evaluation(lines: list[str]) -> dict[str, dict]:
    """Разбирает журнал оценки.

    Args:
        lines: Строки журнала evaluate.log.

    Returns:
        Словарь, где ключ это заголовок раздела, а значение содержит таблицы
        categories (название, слов, тег, лемма, оба) и tags (тег, точность, полнота, F1, слов).
    """
    sections: dict[str, dict] = {}
    current = None
    for line in lines:
        title = TITLE_RE.match(line)
        if title:
            current = {"method": title.group(1), "mode": title.group(2), "categories": [], "tags": []}
            sections[line.strip()] = current
            continue
        if current is None:
            continue
        category = CATEGORY_RE.match(line)
        if category:
            name, count, tag, lemma, both = category.groups()
            current["categories"].append((name, int(count), float(tag), float(lemma), float(both)))
            continue
        tag_row = TAG_RE.match(line)
        if tag_row:
            tag, precision, recall, f1, count = tag_row.groups()
            current["tags"].append((tag, float(precision), float(recall), float(f1), int(count)))
    return sections


def percent(value: float) -> str:
    """Форматирует долю как проценты.

    Args:
        value: Число от нуля до единицы.

    Returns:
        Строка вида 97,25 процентов с запятой и знаком процента.
    """
    return f"{value * 100:.2f}%".replace(".", ",")


def markdown_table(header: list[str], rows: list[list[str]]) -> str:
    """Строит таблицу в формате markdown.

    Args:
        header: Заголовки столбцов.
        rows: Строки таблицы.

    Returns:
        Текст таблицы.
    """
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def svg_wrap(width: int, height: int, body: str, title: str) -> str:
    """Оборачивает содержимое графика в svg документ.

    Args:
        width: Ширина картинки.
        height: Высота картинки.
        body: Элементы svg.
        title: Заголовок графика.

    Returns:
        Текст svg файла.
    """
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Arial, sans-serif" font-size="13">'
        f'<rect width="{width}" height="{height}" fill="white"/>'
        f'<text x="{width / 2}" y="24" text-anchor="middle" font-size="16" font-weight="bold">{title}</text>'
        f"{body}</svg>"
    )


def line_chart(title: str, x_values: list[int], series: dict[str, list[float]], y_label: str) -> str:
    """Рисует линейный график с несколькими линиями.

    Args:
        title: Заголовок.
        x_values: Значения по горизонтали, например номера эпох.
        series: Название линии и ее значения.
        y_label: Подпись вертикальной оси.

    Returns:
        Текст svg файла.
    """
    width, height, left, right, top, bottom = 640, 380, 70, 30, 50, 60
    values = [value for line in series.values() for value in line]
    low, high = min(values), max(values)
    pad = (high - low) * 0.1 or 0.01
    low, high = low - pad, high + pad
    plot_w, plot_h = width - left - right, height - top - bottom

    def px(index: int) -> float:
        """Переводит номер точки в координату по горизонтали.

        Args:
            index: Номер точки.

        Returns:
            Координата в картинке.
        """
        return left + plot_w * index / max(len(x_values) - 1, 1)

    def py(value: float) -> float:
        """Переводит значение в координату по вертикали.

        Args:
            value: Значение на графике.

        Returns:
            Координата в картинке.
        """
        return top + plot_h * (1 - (value - low) / (high - low))

    body = []
    for step in range(5):
        value = low + (high - low) * step / 4
        y = py(value)
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#e2e8f0"/>')
        body.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">{value:.3f}</text>')
    for index, x in enumerate(x_values):
        body.append(f'<text x="{px(index):.1f}" y="{height - bottom + 20}" text-anchor="middle">{x}</text>')
    body.append(f'<text x="{width / 2}" y="{height - 15}" text-anchor="middle">Эпоха</text>')
    body.append(
        f'<text x="18" y="{height / 2}" text-anchor="middle" transform="rotate(-90 18 {height / 2})">{y_label}</text>'
    )
    for number, (name, line) in enumerate(series.items()):
        color = COLORS[number % len(COLORS)]
        points = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(line))
        body.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for i, v in enumerate(line):
            body.append(f'<circle cx="{px(i):.1f}" cy="{py(v):.1f}" r="3.5" fill="{color}"/>')
        body.append(f'<rect x="{left + 10 + number * 190}" y="34" width="12" height="12" fill="{color}"/>')
        body.append(f'<text x="{left + 28 + number * 190}" y="45">{name}</text>')
    return svg_wrap(width, height, "".join(body), title)


def bar_chart(title: str, groups: list[str], series: dict[str, list[float]], low: float = 0.0) -> str:
    """Рисует столбчатую диаграмму со сгруппированными столбцами.

    Args:
        title: Заголовок.
        groups: Названия групп по горизонтали.
        series: Название серии и ее значения для каждой группы, доли от нуля до единицы.
        low: Нижняя граница вертикальной оси.

    Returns:
        Текст svg файла.
    """
    width = max(640, 90 * len(groups) * len(series) // 2 + 120)
    height, left, right, top, bottom = 400, 60, 20, 60, 70
    plot_w, plot_h = width - left - right, height - top - bottom
    group_w = plot_w / len(groups)
    bar_w = group_w * 0.8 / len(series)
    body = []
    for step in range(6):
        value = low + (1 - low) * step / 5
        y = top + plot_h * (1 - (value - low) / (1 - low))
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#e2e8f0"/>')
        body.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">{value * 100:.0f}%</text>')
    for group, name in enumerate(groups):
        x0 = left + group * group_w + group_w * 0.1
        for number, values in enumerate(series.values()):
            value = max(values[group], low)
            bar_h = plot_h * (value - low) / (1 - low)
            x = x0 + number * bar_w
            color = COLORS[number % len(COLORS)]
            body.append(
                f'<rect x="{x:.1f}" y="{top + plot_h - bar_h:.1f}" width="{bar_w * 0.92:.1f}" '
                f'height="{bar_h:.1f}" fill="{color}"/>'
            )
            body.append(
                f'<text x="{x + bar_w * 0.46:.1f}" y="{top + plot_h - bar_h - 4:.1f}" '
                f'text-anchor="middle" font-size="10">{values[group] * 100:.1f}</text>'
            )
        body.append(f'<text x="{left + (group + 0.5) * group_w:.1f}" y="{height - bottom + 20}" text-anchor="middle">{name}</text>')
    for number, name in enumerate(series):
        color = COLORS[number % len(COLORS)]
        body.append(f'<rect x="{left + 10 + number * 200}" y="34" width="12" height="12" fill="{color}"/>')
        body.append(f'<text x="{left + 28 + number * 200}" y="45">{name}</text>')
    return svg_wrap(width, height, "".join(body), title)


def find_section(sections: dict[str, dict], method: str, mode_part: str) -> dict | None:
    """Находит раздел оценки по методу и части названия режима.

    Args:
        sections: Результат функции parse_evaluation.
        method: Название метода.
        mode_part: Фрагмент названия режима, например слово скрыт.

    Returns:
        Раздел или None, если такого нет.
    """
    for section in sections.values():
        if section["method"] == method and mode_part in section["mode"]:
            return section
    return None


def build_report() -> None:
    """Строит все таблицы и графики и сохраняет их в папку lab1/results.

    Args:
        Нет.

    Returns:
        Ничего не возвращает.
    """
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    tagger, lemma = parse_training(read_log("train.log"))
    sections = parse_evaluation(read_log("evaluate.log"))
    parts = ["# Результаты обучения и оценки", "", "Файл создан автоматически командой `python -m lab1 report`.", ""]

    parts += ["## Обучение модели разметки", ""]
    parts.append(markdown_table(
        ["Эпоха", "Потери", "Точность тегов на отладке"],
        [[str(e), f"{loss:.4f}", percent(acc)] for e, loss, acc in tagger],
    ))
    parts += ["", "## Обучение модели лемматизации неизвестных слов", ""]
    parts.append(markdown_table(
        ["Эпоха", "Потери", "Точность правил на контроле"],
        [[str(e), f"{loss:.4f}", percent(acc)] for e, loss, acc in lemma],
    ))
    if tagger:
        epochs = [row[0] for row in tagger]
        chart = line_chart("Обучение модели разметки: точность на отладке", epochs,
                           {"Точность тегов": [row[2] for row in tagger]}, "Точность")
        (RESULT_DIR / "tagger_accuracy.svg").write_text(chart, encoding="utf-8")
        chart = line_chart("Обучение модели разметки: потери", epochs,
                           {"Потери": [row[1] for row in tagger]}, "Потери")
        (RESULT_DIR / "tagger_loss.svg").write_text(chart, encoding="utf-8")
    if lemma:
        chart = line_chart("Обучение модели лемматизации", [row[0] for row in lemma],
                           {"Потери": [row[1] for row in lemma], "Точность правил": [row[2] for row in lemma]},
                           "Значение")
        (RESULT_DIR / "lemma_training.svg").write_text(chart, encoding="utf-8")

    parts += ["", "## Сравнение методов на тестовой части", ""]
    for mode_part, caption, category in (
        ("словарь доступен", "Обычный режим, словарь доступен", "Все слова"),
        ("скрыт", "Словарь скрыт у всех слов, все слова неизвестные", "Все слова"),
    ):
        base = find_section(sections, "Базовый метод", mode_part)
        neural = find_section(sections, "Нейросеть", mode_part)
        if not base or not neural:
            continue
        parts += [f"### {caption}", ""]
        rows = []
        for section in (base, neural):
            for name, count, tag, lem, both in section["categories"]:
                rows.append([section["method"], name, str(count), percent(tag), percent(lem), percent(both)])
        parts.append(markdown_table(["Метод", "Группа слов", "Слов", "Тег", "Лемма", "Тег и лемма"], rows))
        parts.append("")
        values = {}
        for section in (base, neural):
            row = next(item for item in section["categories"] if item[0] == category)
            values[section["method"]] = [row[2], row[3], row[4]]
        file_name = "compare_dictionary.svg" if "доступен" in mode_part else "compare_no_dictionary.svg"
        chart = bar_chart(f"{caption}", ["Тег", "Лемма", "Тег и лемма"], values, low=0.3)
        (RESULT_DIR / file_name).write_text(chart, encoding="utf-8")

    neural = find_section(sections, "Нейросеть", "словарь доступен")
    base = find_section(sections, "Базовый метод", "словарь доступен")
    if neural and base:
        parts += ["### Качество по тегам, обычный режим", ""]
        base_f1 = {row[0]: row[3] for row in base["tags"]}
        rows = [
            [tag, str(count), percent(p), percent(r), percent(f1), percent(base_f1.get(tag, 0.0))]
            for tag, p, r, f1, count in neural["tags"]
        ]
        parts.append(markdown_table(
            ["Тег", "Слов", "Точность", "Полнота", "F1 нейросети", "F1 базового метода"], rows))
        tags = [row[0] for row in neural["tags"]]
        chart = bar_chart("F1 по тегам", tags,
                          {"Базовый метод": [base_f1.get(t, 0.0) for t in tags],
                           "Нейросеть": [row[3] for row in neural["tags"]]}, low=0.3)
        (RESULT_DIR / "f1_by_tag.svg").write_text(chart, encoding="utf-8")

    parts += ["", "## Графики", ""]
    # графики сохраняются в формате png, svg остается только если нет браузера
    for file in sorted(RESULT_DIR.glob("*.svg")):
        svg_to_png(file)
    for file in sorted(RESULT_DIR.glob("*.png")) + sorted(RESULT_DIR.glob("*.svg")):
        parts += [f"![{file.stem}]({file.name})", ""]
    (RESULT_DIR / "tables.md").write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"Таблицы и графики сохранены в папку {RESULT_DIR}")
