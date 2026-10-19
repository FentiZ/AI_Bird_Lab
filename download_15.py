"""Необов'язково: завантажити 15 фото з Wikipedia для шкільного тесту."""
import csv
import json
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "samples"
DEST.mkdir(exist_ok=True)
BIRDS = json.loads((ROOT / "birds.json").read_text(encoding="utf-8"))
SOURCES = []

session = requests.Session()
session.headers["User-Agent"] = "AI-Bird-Lab/1.0 (educational school project)"

for index, bird in enumerate(BIRDS, start=1):
    title = bird["english_name"]
    try:
        response = session.get("https://en.wikipedia.org/w/api.php", params={
            "action": "query", "format": "json", "redirects": "1",
            "prop": "pageimages", "pithumbsize": "900", "titles": title,
        }, timeout=20)
        response.raise_for_status()
        pages = response.json()["query"]["pages"]
        page = next(iter(pages.values()))
        picture_url = page.get("thumbnail", {}).get("source")
        if not picture_url:
            raise ValueError("На сторінці немає фотографії")
        photo_response = session.get(picture_url, timeout=25)
        photo_response.raise_for_status()
        if len(photo_response.content) > 10 * 1024 * 1024:
            raise ValueError("Фотографія завелика")
        with Image.open(BytesIO(photo_response.content)) as image:
            image.load()
            filename = f"{index:02d}_{title.replace(' ', '_')}.jpg"
            image.convert("RGB").save(DEST / filename, format="JPEG", quality=85)
        SOURCES.append({"filename": filename, "species": title,
                        "source": bird["source_url"], "photo_url": picture_url})
        print(f"OK {index:02d}/15: {title}")
    except (requests.RequestException, ValueError, OSError, KeyError) as exc:
        print(f"ПОМИЛКА {index:02d}/15: {title}: {exc}")

with (DEST / "sources.csv").open("w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(file, ["filename", "species", "source", "photo_url"])
    writer.writeheader()
    writer.writerows(SOURCES)

print(f"Готово: {len(SOURCES)}/15 фото. Відкрийте застосунок і завантажте їх через форму.")
if len(SOURCES) < 15:
    print("Якщо фото не завантажились, додайте відсутні з власних джерел і перевірте ліцензії.")
