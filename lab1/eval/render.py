"""Преобразование svg и html страниц в png картинки.

Модуль использует браузер Edge или Chrome, который уже есть на компьютере, в режиме
без окна. Благодаря этому для получения png не нужны сторонние библиотеки.
Если браузер не найден, функции сообщают об этом, а исходные файлы остаются на месте.
"""

import os
import re
import subprocess
import tempfile
from pathlib import Path
from shutil import which

BROWSER_PATHS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "chromium",
    "chromium-browser",
    "google-chrome",
]


def find_browser() -> str | None:
    """Ищет браузер для создания картинок.

    Args:
        Нет.

    Returns:
        Путь к программе браузера или None, если подходящего браузера нет.
    """
    local = os.environ.get("LOCALAPPDATA", "")
    candidates = list(BROWSER_PATHS)
    if local:
        candidates.append(str(Path(local) / "Google" / "Chrome" / "Application" / "chrome.exe"))
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    for name in ("chromium", "chromium-browser", "google-chrome"):
        found = which(name)
        if found:
            return found
    return None


def screenshot_page(page: Path, target: Path, width: int, height: int) -> bool:
    """Делает снимок страницы в браузере без окна.

    Args:
        page: Путь к html или svg файлу.
        target: Путь к создаваемой png картинке.
        width: Ширина снимка в пикселях.
        height: Высота снимка в пикселях.

    Returns:
        True, если картинка создана.
    """
    browser = find_browser()
    if browser is None:
        print("Браузер для создания png не найден")
        return False
    if target.exists():
        target.unlink()
    with tempfile.TemporaryDirectory() as profile:
        command = [
            browser, "--headless=new", "--disable-gpu", "--no-sandbox", "--hide-scrollbars",
            f"--user-data-dir={profile}", f"--screenshot={target}",
            f"--window-size={width},{height}", "--force-device-scale-factor=2",
            "--no-first-run", "--disable-background-networking", "--disable-sync", page.resolve().as_uri(),
        ]
        try:
            subprocess.run(command, check=False, timeout=90, capture_output=True)
        except (OSError, subprocess.TimeoutExpired) as error:
            print(f"Не удалось запустить браузер: {error}")
            return False
    return target.exists() and target.stat().st_size > 0


def svg_to_png(svg_path: Path, delete_svg: bool = True) -> Path:
    """Превращает svg график в png картинку.

    Args:
        svg_path: Путь к svg файлу.
        delete_svg: Если True, исходный svg удаляется после успешного преобразования.

    Returns:
        Путь к png файлу или к исходному svg, если преобразовать не удалось.
    """
    text = svg_path.read_text(encoding="utf-8")
    match = re.search(r'width="(\d+)" height="(\d+)"', text)
    width, height = (int(match.group(1)), int(match.group(2))) if match else (700, 450)
    page = svg_path.with_suffix(".html")
    page.write_text(
        '<!doctype html><meta charset="utf-8"><style>html,body{margin:0;background:white}'
        f"svg{{display:block}}</style>{text}",
        encoding="utf-8",
    )
    target = svg_path.with_suffix(".png")
    ok = screenshot_page(page, target, width, height)
    page.unlink()
    if not ok:
        return svg_path
    if delete_svg:
        svg_path.unlink()
    return target
