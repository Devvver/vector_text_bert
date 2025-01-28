import streamlit as st
from transformers import AutoTokenizer, AutoModel
import torch
from sentence_transformers.util import cos_sim
from newspaper import Article
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import io

@st.cache_resource
def load_labse_model():
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")
    model = AutoModel.from_pretrained("sentence-transformers/LaBSE")
    return tokenizer, model

def calculate_vector(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs).pooler_output
    normalized_vector = embeddings[0] / torch.norm(embeddings[0])
    return normalized_vector

def calculate_similarity(vector1, vector2):
    return cos_sim(vector1, vector2).item()

def extract_article_info(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        title = article.title.strip() or "Заголовок не найден"
        soup = BeautifulSoup(article.html, "html.parser")
        h1_tag = soup.find("h1")
        h1_text = h1_tag.get_text().strip() if h1_tag else None
        text = article.text or "Текст статьи не найден"
        return title, h1_text, text
    except Exception as e:
        return None, None, None

def get_color(value):
    if value is None or value == "None" or (isinstance(value, float) and np.isnan(value)):
        return ""
    red = 255
    green = 255
    blue = 255
    # Normalize the value to range [0, 1]
    norm_value = (value - 0) / (1 - 0)

    # Define color ranges
    if norm_value < 0.20:

        red, green, blue = 252, 108, 133  # Ultra Red
    elif norm_value < 0.3:

        red, green, blue = 252, 148, 161  # Salmon Pink
    elif norm_value < 0.4:
        red, green, blue = 254,237, 71
    elif norm_value < 0.6:
        red, green, blue = 205, 255, 204  # Tea Green
    elif norm_value < 0.8:
        red, green, blue = 176, 245, 171  # Menthol
    elif norm_value < 0.9:
        red, green, blue = 144, 239, 144  # Light Green
    elif norm_value > 0.9:
        red, green, blue = 46, 184, 46
    return f'background-color: rgb({red}, {green}, {blue})'

def analyze_text_mode(query_text, tokenizer, model):
    article_text = st.text_area("Введите текст статьи:", "")

    if st.button("Рассчитать", key="analyze_text_button"):
        if query_text and article_text:
            query_paragraphs = [p.strip() for p in query_text.split("\n") if p.strip()]
            article_paragraphs = [p.strip() for p in article_text.split("\n") if p.strip()]

            query_vectors = [calculate_vector(p, tokenizer, model).detach().numpy() for p in query_paragraphs]
            article_vectors = [calculate_vector(p, tokenizer, model).detach().numpy() for p in article_paragraphs]

            for i, query_vector in enumerate(query_vectors):

                st.markdown(
                    f'<p style="background-color: rgba(240, 240, 240, 0.5); padding: 10px;">'
                    f' Запрос {query_paragraphs[i]}</p>',
                    unsafe_allow_html=True,
                )

                for j, article_vector in enumerate(article_vectors):
                    similarity_score = calculate_similarity(query_vector, article_vector)
                    color = get_color(similarity_score)

                    st.markdown(
                        f'<div style="{color}; padding: 10px;">'
                        f'<p>{article_paragraphs[j]}</p>'
                        f'<p style="text-align: right;">Схожесть: {similarity_score:.4f}</p>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.error("Пожалуйста, введите текст в оба поля.")

def analyze_url_mode(query_text, tokenizer, model):
    urls = st.text_area("Введите URL (каждый с новой строки):", "")

    if st.button("Рассчитать", key="analyze_url_button"):
        if query_text and urls.strip():
            query_vector = calculate_vector(query_text, tokenizer, model).detach().numpy()
            results = []

            for url in urls.splitlines():
                title, h1, article_text = extract_article_info(url)
                if article_text:
                    article_vector = calculate_vector(article_text, tokenizer, model).detach().numpy()
                    similarity = calculate_similarity(query_vector, article_vector)
                else:
                    similarity = None
                title_similarity = (
                    calculate_similarity(query_vector, calculate_vector(title, tokenizer, model).detach().numpy())
                    if title
                    else None
                )
                h1_similarity = (
                    calculate_similarity(query_vector, calculate_vector(h1, tokenizer, model).detach().numpy())
                    if h1
                    else None
                )
                results.append({
                    "URL": url,
                    "Title": title or "Ошибка загрузки",
                    "H1": h1 or "Ошибка загрузки",
                    "Similarity (Article)": similarity,
                    "Similarity (Title)": title_similarity,
                    "Similarity (H1)": h1_similarity,
                })
            st.session_state.url_results = results

    if "url_results" in st.session_state:
        df = pd.DataFrame(st.session_state.url_results)
        styled_df = df.style.applymap(get_color, subset=["Similarity (Article)", "Similarity (Title)", "Similarity (H1)"])
        st.dataframe(styled_df)

        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()

        excel_buffer = io.BytesIO()
        df.to_excel(excel_buffer, index=False, engine='openpyxl')
        excel_buffer.seek(0)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="Скачать CSV",
                data=csv_data,
                file_name="url_results.csv",
                mime="text/csv"
            )
        with col2:
            st.download_button(
                label="Скачать Excel",
                data=excel_buffer,
                file_name="url_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

st.markdown("""
<style>
/* Увеличение шрифта по умолчанию для всего приложения */
html, body, [class*="css"]  {
    font-size: 20px !important;
}
</style>
""", unsafe_allow_html=True)

# Интерфейс Streamlit
st.markdown("""
<style>
.big-font {
    font-size:20px !important;
}
</style>
<h2 class="big-font">Вычисление схожести текстов с Language-agnostic BERT Sentence Embedding</h2>
""", unsafe_allow_html=True)

mode = st.radio("Выберите режим:", ["Анализ текста", "Анализ URL"])
query_text = st.text_input("Введите запрос:", "")

# Загрузка модели
tokenizer, model = load_labse_model()

if mode == "Анализ текста":
    st.session_state.pop("url_results", None)  # Удаляем данные URL-анализа
    analyze_text_mode(query_text, tokenizer, model)

if mode == "Анализ URL":
    st.session_state.pop("results", None)  # Удаляем данные текстового анализа
    analyze_url_mode(query_text, tokenizer, model)
