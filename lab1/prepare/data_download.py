"""Загрузка исходных данных.

Модуль скачивает с github файлы корпуса SynTagRus (Universal Dependencies) и таблицу
форм UniMorph и сохраняет их в папку lab1/data/raw. Это первый шаг конвейера: без этих
файлов невозможно построить словарь и обучить модели. Если интернет недоступен,
файлы можно положить в эту папку вручную.
"""

import http.client
import urllib.request
from pathlib import Path

from ..config import (
    DEV_NAMES,
    RAW_DIR,
    SYNTAGRUS_URL,
    TEST_NAMES,
    TRAIN_NAMES,
    UNIMORPH_NAME,
    UNIMORPH_URL,
)

CHUNK_SIZE = 1024 * 1024


def download_file(url: str, target: Path) -> None:
    """Скачивает файл по адресу и показывает ход загрузки.

    Args:
        url: Адрес файла.
        target: Путь, по которому нужно сохранить файл.

    Returns:
        Ничего не возвращает.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 lab1-nlp"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, open(temp, "wb") as out:
            total = int(response.headers.get("Content-Length", 0))
            done = 0
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  скачано {done // CHUNK_SIZE} из {total // CHUNK_SIZE} МБ", end="")
                else:
                    print(f"\r  скачано {done // CHUNK_SIZE} МБ", end="")
        print()
    except (OSError, http.client.HTTPException) as error:
        raise RuntimeError(
            f"Не удалось скачать {url}: {error}. Скачайте файл вручную в браузере "
            f"и положите его в папку {RAW_DIR}."
        ) from error
    temp.replace(target)


def ensure_raw_data(force: bool = False) -> None:
    """Скачивает все исходные файлы, которых еще нет.

    Args:
        force: Если True, файлы скачиваются заново даже при наличии на диске.

    Returns:
        Ничего не возвращает.
    """
    jobs = [(SYNTAGRUS_URL + name, name) for name in TRAIN_NAMES + DEV_NAMES + TEST_NAMES]
    jobs.append((UNIMORPH_URL, UNIMORPH_NAME))
    for url, name in jobs:
        target = RAW_DIR / name
        if (target.exists() or target.with_name(name + ".gz").exists()) and not force:
            print(f"Файл уже есть: {name}")
            continue
        print(f"Загрузка {name}")
        download_file(url, target)
