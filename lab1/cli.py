"""Командная строка проекта.

Модуль связывает все этапы решения в команды: скачать данные, разобрать их,
обучить модели, оценить точность и обработать входной текст в формате задания.
Запуск выполняется командой python -m lab1 с названием команды.
"""

import argparse
import sys
from pathlib import Path

from .config import LOG_DIR, MODEL_PATH, SEED, LemmaConfig
from .eval.evaluation import evaluate_analyzer, format_report
from .prepare.data_download import ensure_raw_data
from .prepare.resources import load_data, load_dictionary
from .text_utils import configure_console_utf8, read_lines, write_lines


LOGGED_COMMANDS = {"train", "evaluate"}


class Tee:
    """Поток вывода, который дублирует текст в файл журнала."""

    def __init__(self, stream, log_file) -> None:
        """Запоминает основной поток и файл журнала.

        Args:
            stream: Исходный поток вывода, например консоль.
            log_file: Открытый файл журнала.

        Returns:
            Ничего не возвращает.
        """
        self.stream = stream
        self.log_file = log_file

    def write(self, text: str) -> int:
        """Записывает текст в консоль и в файл.

        Args:
            text: Записываемый текст.

        Returns:
            Число записанных символов.
        """
        self.log_file.write(text)
        self.log_file.flush()
        return self.stream.write(text)

    def flush(self) -> None:
        """Сбрасывает буферы консоли и файла.

        Args:
            Нет.

        Returns:
            Ничего не возвращает.
        """
        self.log_file.flush()
        self.stream.flush()


def command_download(args: argparse.Namespace) -> None:
    """Выполняет команду download: скачивает исходные данные.

    Args:
        args: Разобранные аргументы командной строки.

    Returns:
        Ничего не возвращает.
    """
    ensure_raw_data(force=args.force)


def command_prepare(args: argparse.Namespace) -> None:
    """Выполняет команду prepare: разбирает данные и сохраняет кэш.

    Args:
        args: Разобранные аргументы командной строки.

    Returns:
        Ничего не возвращает.
    """
    dictionary, _, _, _ = load_data(rebuild=True)
    print(f"Словоформ в словаре: {len(dictionary)}")


def command_train(args: argparse.Namespace) -> None:
    """Выполняет команду train: обучает и сохраняет модели.

    Args:
        args: Разобранные аргументы командной строки.

    Returns:
        Ничего не возвращает.
    """
    from .train import train_all

    dictionary, train, dev, _ = load_data()
    train_all(dictionary, train, dev)


def command_evaluate(args: argparse.Namespace) -> None:
    """Выполняет команду evaluate: сравнивает базовый метод и нейросеть на тесте.

    Args:
        args: Разобранные аргументы командной строки.

    Returns:
        Ничего не возвращает.
    """
    from .analyze.lemmatizer import load_lemmatizer
    from .eval.baseline import FrequencyBaseline
    from .train import select_device

    dictionary, train, _, test = load_data()
    # базовый метод без НС
    baseline = FrequencyBaseline.from_corpus(dictionary, train, LemmaConfig())
    # метод на основе НС
    neural = load_lemmatizer(MODEL_PATH, dictionary, select_device())
    for rate in args.oov_rates:
        suffix = "словарь доступен" if rate == 0 else f"словарь скрыт у {rate:.0%} слов"
        for name, analyzer in (("Базовый метод", baseline), ("Нейросеть", neural)):
            result = evaluate_analyzer(analyzer, dictionary, test, rate, SEED)
            print(format_report(f"{name}, {suffix}", result))
            print()


def command_run(args: argparse.Namespace) -> None:
    """Выполняет команду run: обрабатывает текст в формате задания.

    Args:
        args: Разобранные аргументы командной строки.

    Returns:
        Ничего не возвращает.
    """
    from .analyze.lemmatizer import load_lemmatizer, process_lines
    from .train import select_device

    dictionary = load_dictionary()
    analyzer = load_lemmatizer(MODEL_PATH, dictionary, select_device())
    lines = read_lines(Path(args.input)) if args.input else sys.stdin.read().splitlines()
    output = process_lines(analyzer, lines)
    if args.output:
        write_lines(Path(args.output), output)
        print(f"Результат записан в {args.output}")
    else:
        print("\n".join(output))


def command_report(args: argparse.Namespace) -> None:
    """Выполняет команду report: строит таблицы и графики по журналам.

    Args:
        args: Разобранные аргументы командной строки.

    Returns:
        Ничего не возвращает.
    """
    from .eval.report import build_report

    build_report()


def command_demo(args: argparse.Namespace) -> None:
    """Выполняет команду demo: обрабатывает примеры и сохраняет картинки с результатами.

    Args:
        args: Разобранные аргументы командной строки.

    Returns:
        Ничего не возвращает.
    """
    from .eval.demo import run_demo

    run_demo()


def command_all(args: argparse.Namespace) -> None:
    """Выполняет команду all: скачивание, подготовка, обучение и оценка.

    Args:
        args: Разобранные аргументы командной строки.

    Returns:
        Ничего не возвращает.
    """
    ensure_raw_data(force=args.force)
    command_prepare(args)
    command_train(args)
    command_evaluate(args)


def build_parser() -> argparse.ArgumentParser:
    """Создает разбор аргументов командной строки.

    Args:
        Нет.

    Returns:
        Настроенный разборщик аргументов.
    """
    parser = argparse.ArgumentParser(prog="lab1", description="Лемматизация и частеречная разметка")
    sub = parser.add_subparsers(dest="command", required=True)

    download = sub.add_parser("download", help="скачать корпус SynTagRus и словарь UniMorph")
    download.add_argument("--force", action="store_true", help="скачать заново")
    download.set_defaults(func=command_download)

    prepare = sub.add_parser("prepare", help="разобрать данные и сохранить кэш")
    prepare.set_defaults(func=command_prepare)

    train = sub.add_parser("train", help="обучить модели")
    train.set_defaults(func=command_train)

    evaluate = sub.add_parser("evaluate", help="оценить точность на тестовой части корпуса")
    evaluate.add_argument(
        "--oov-rates", type=float, nargs="+", default=[0.0, 1.0],
        help="доли слов со скрытым словарем, 0 это обычный режим, 1 все слова неизвестные",
    )
    evaluate.set_defaults(func=command_evaluate)

    run = sub.add_parser("run", help="обработать текст в формате задания")
    run.add_argument("--input", help="входной файл, по умолчанию стандартный ввод")
    run.add_argument("--output", help="выходной файл, по умолчанию стандартный вывод")
    run.set_defaults(func=command_run)

    report = sub.add_parser("report", help="построить таблицы и графики по журналам обучения")
    report.set_defaults(func=command_report)

    demo = sub.add_parser("demo", help="обработать примеры из папки lab1/examples и сохранить картинки")
    demo.set_defaults(func=command_demo)

    everything = sub.add_parser("all", help="скачать, подготовить, обучить и оценить")
    everything.add_argument("--oov-rates", type=float, nargs="+", default=[0.0, 1.0])
    everything.add_argument("--force", action="store_true")
    everything.set_defaults(func=command_all)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Запускает выбранную команду.

    Args:
        argv: Аргументы командной строки или None, тогда берутся аргументы процесса.

    Returns:
        Код завершения программы.
    """
    configure_console_utf8()
    args = build_parser().parse_args(argv)
    if args.command not in LOGGED_COMMANDS:
        args.func(args)
        return 0
    # вывод этих команд сохраняется в журнал, по нему report строит таблицы и графики
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    original = sys.stdout
    with open(LOG_DIR / f"{args.command}.log", "w", encoding="utf-8") as log_file:
        sys.stdout = Tee(original, log_file)
        try:
            args.func(args)
        finally:
            sys.stdout = original
    return 0
