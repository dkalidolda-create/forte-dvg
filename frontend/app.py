import requests
import streamlit as st

st.set_page_config(page_title="Video Analysis", layout="wide")

st.title("Анализ видео")
st.write("Загрузите видео для поиска нарушений.")

try:
    response = requests.get(
        "http://127.0.0.1:8000/health",
        timeout=3,
    )
    response.raise_for_status()
    st.success("Backend подключён ✅")
except requests.RequestException:
    st.error("Backend недоступен. Проверьте его запуск.")

video = st.file_uploader(
    "Выберите видео",
    type=["mp4", "mov"],
)

if video is not None:
    st.write(f"Выбран файл: {video.name}")
    st.info("Обработку видео подключим следующим шагом.")