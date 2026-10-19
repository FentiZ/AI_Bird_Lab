"""AI Bird Lab: простий застосунок для розпізнавання птахів за фото."""
import csv
import json
import re
import sqlite3
from datetime import datetime
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
DB_FILE = ROOT / "birds.db"
CSV_FILE = ROOT / "bird_analysis.csv"
PHOTOS = ROOT / "photos"
MODEL_ID = "dima806/bird_species_image_detection"
CSV_COLUMNS = ["image_name", "predicted_species", "confidence", "analysis_date", "description"]
Image.MAX_IMAGE_PIXELS = 25_000_000


def query(sql, params=(), fetch=False):
    """Виконати SQL та (за потреби) повернути рядки словниками."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    try:
        result = conn.execute(sql, params)
        rows = [dict(row) for row in result.fetchall()] if fetch else []
        conn.commit()
        return rows
    finally:
        conn.close()


def init_db():
    PHOTOS.mkdir(exist_ok=True)
    query("""CREATE TABLE IF NOT EXISTS analyses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_name TEXT NOT NULL,
        image_path TEXT NOT NULL,
        predicted_species TEXT NOT NULL,
        confidence REAL NOT NULL,
        analysis_date TEXT NOT NULL,
        description TEXT NOT NULL,
        extra TEXT NOT NULL,
        top3 TEXT NOT NULL
    )""")
    if not CSV_FILE.exists():
        export_csv()


def export_csv():
    records = query("SELECT image_name, predicted_species, confidence, "
                    "analysis_date, description FROM analyses ORDER BY id", fetch=True)
    with CSV_FILE.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(records)


def read_catalog():
    with (ROOT / "birds.json").open(encoding="utf-8") as file:
        return json.load(file)


def pretty_name(name):
    """Перетворити назву класу моделі на придатну для показу назву."""
    name = re.sub(r"^\s*\d+[.\-_ ]+", "", str(name))
    return name.replace("_", " ").strip().title()


def lookup_bird(name, catalog):
    for bird in catalog:
        if bird["english_name"].casefold() == name.casefold():
            return bird
    return None


def load_photo(upload):
    try:
        photo = Image.open(upload)
        photo.load()
        return photo.convert("RGB")
    except Exception:
        raise ValueError("Не удалось открыть изображение")


@st.cache_resource(show_spinner=False)
def get_model():
    # Завантажується тільки під час першого запуску аналізу і кешується.
    from transformers import pipeline
    return pipeline("image-classification", model=MODEL_ID, device=-1)


def analyze_one(upload, model, catalog):
    photo = load_photo(upload)
    result = model(photo, top_k=3)
    if not result:
        raise ValueError("Модель не повернула результат")
    top3 = [{"species": pretty_name(item["label"]),
             "confidence": round(float(item["score"]) * 100, 2)} for item in result]
    winner = top3[0]
    bird = lookup_bird(winner["species"], catalog)
    description = (bird["description_uk"] if bird
                   else "Інформації про цей вид у локальному довіднику немає.")
    date = datetime.now().astimezone().isoformat(timespec="seconds")

    # Фото зберігається під безпечною випадковою назвою.
    path = PHOTOS / f"{uuid4().hex}.jpg"
    photo.save(path, format="JPEG", quality=85)
    try:
        query("""INSERT INTO analyses
              (image_name, image_path, predicted_species, confidence,
               analysis_date, description, extra, top3)
              VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
              (Path(upload.name).name, str(path), winner["species"],
               winner["confidence"], date, description,
               json.dumps(bird or {}, ensure_ascii=False),
               json.dumps(top3, ensure_ascii=False)))
        export_csv()
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return {"image": photo, "filename": upload.name, "top3": top3, "bird": bird}


def delete_record(record):
    query("DELETE FROM analyses WHERE id = ?", (record["id"],))
    export_csv()
    path = Path(record["image_path"])
    if path.parent.resolve() == PHOTOS.resolve():
        path.unlink(missing_ok=True)


def show_info(bird):
    if not bird:
        st.info("Даних про цей вид немає в локальному довіднику. Не вигадуємо факти.")
        return
    st.write(f"**Українська назва:** {bird['ukrainian_name']}")
    st.write(f"**Наукова назва:** {bird['scientific_name']}")
    st.write(f"**Родина:** {bird['family']}")
    st.write(f"**Середовище:** {bird['habitat_uk']}")
    st.write(f"**Поширення:** {bird['distribution_uk']}")
    st.write(f"**Живлення:** {bird['diet_uk']}")
    st.write(f"**Розміри:** {bird['size_uk']}")
    st.write(f"**Цікавий факт:** {bird['interesting_fact_uk']}")
    st.write(f"**Природоохоронний статус:** {bird['conservation'] or 'Немає даних'}")
    st.markdown(f"[Джерело інформації]({bird['source_url']})")


def run_app():
    st.set_page_config(page_title="AI Bird Lab", page_icon="🐦", layout="wide")
    init_db()
    catalog = read_catalog()
    st.title("🐦 AI Bird Lab")
    st.caption("Розпізнавання птахів за фотографіями · Hugging Face · SQLite")

    tab_analyze, tab_history, tab_stats = st.tabs(["📷 Аналіз фото", "📚 Історія", "📊 Статистика"])

    with tab_analyze:
        files = st.file_uploader("Завантажте одну або декілька фотографій (JPG, JPEG, PNG)",
                                 type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        if st.button("Розпізнати", type="primary", disabled=not files):
            try:
                with st.spinner("Завантажуємо модель Hugging Face (перший запуск потребує інтернету)..."):
                    model = get_model()
            except Exception as exc:
                st.error(f"Не вдалося завантажити модель: {exc}")
                st.info("Перевірте інтернет та встановіть залежності: pip install -r requirements.txt")
            else:
                st.session_state["last_results"] = []
                bar = st.progress(0)
                for i, uploaded in enumerate(files):
                    try:
                        st.session_state["last_results"].append(analyze_one(uploaded, model, catalog))
                    except Exception as exc:
                        st.error(f"{uploaded.name}: {exc}")
                    bar.progress((i + 1) / len(files))
                st.success(f"Оброблено фотографій: {len(st.session_state['last_results'])} з {len(files)}")

        for item in st.session_state.get("last_results", []):
            st.divider()
            left, right = st.columns([1, 2])
            with left:
                st.image(item["image"], caption=item["filename"], use_container_width=True)
            with right:
                st.subheader(item["top3"][0]["species"])
                st.write("Три найімовірніші варіанти:")
                st.dataframe(pd.DataFrame(item["top3"]), hide_index=True,
                             use_container_width=True)
                show_info(item["bird"])
        st.caption("Упевненість — це оцінка моделі, а не гарантія правильності визначення.")

    with tab_history:
        name = st.text_input("Пошук за назвою птаха")
        records = query("SELECT * FROM analyses WHERE predicted_species LIKE ? "
                        "ORDER BY id DESC", (f"%{name}%",), fetch=True)
        st.write(f"Знайдено записів: **{len(records)}**")
        if records:
            df = pd.DataFrame(records)[["id", "image_name", "predicted_species", "confidence", "analysis_date"]]
            st.dataframe(df, hide_index=True, use_container_width=True)
            selected_id = st.selectbox("Переглянути запис", [r["id"] for r in records],
                                       format_func=lambda x: f"№ {x} · " + next(r["image_name"] for r in records if r["id"] == x))
            record = next(r for r in records if r["id"] == selected_id)
            if Path(record["image_path"]).exists():
                st.image(record["image_path"], width=350)
            st.write(f"**Визначено:** {record['predicted_species']} ({record['confidence']:.2f}%)")
            st.write(f"**Опис:** {record['description']}")
            st.dataframe(pd.DataFrame(json.loads(record["top3"])), hide_index=True)
            show_info(json.loads(record["extra"]) or None)
            if st.button("Видалити цей запис", type="secondary"):
                delete_record(record)
                st.rerun()
        if CSV_FILE.exists():
            st.download_button("⬇️ Завантажити bird_analysis.csv", CSV_FILE.read_bytes(),
                               file_name="bird_analysis.csv", mime="text/csv")

    with tab_stats:
        total = query("SELECT COUNT(*) AS total FROM analyses", fetch=True)[0]["total"]
        st.metric("Проаналізовано фото", total)
        if total:
            counts = query("SELECT predicted_species AS species, COUNT(*) AS total "
                           "FROM analyses GROUP BY predicted_species ORDER BY total DESC", fetch=True)
            st.bar_chart(pd.DataFrame(counts).set_index("species"))
            highest = query("SELECT * FROM analyses ORDER BY confidence DESC LIMIT 1", fetch=True)[0]
            lowest = query("SELECT * FROM analyses ORDER BY confidence ASC LIMIT 1", fetch=True)[0]
            st.write(f"**Найвища впевненість:** {highest['predicted_species']} — {highest['confidence']:.2f}%")
            st.write(f"**Найнижча впевненість:** {lowest['predicted_species']} — {lowest['confidence']:.2f}%")
        else:
            st.info("Спочатку проаналізуйте хоча б одну фотографію.")


if __name__ == "__main__":
    run_app()
