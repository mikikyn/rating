import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

# Настройка страницы
st.set_page_config(layout="wide", page_title="Прогноз оценок")

# Загрузка данных и модели
script_dir = os.path.dirname(__file__)
possible_data_paths = [
    os.path.join(script_dir, "dataset_study.csv"),
    os.path.normpath(os.path.join(script_dir, "..", "data", "dataset_study.csv")),
]

@st.cache_data
def load_data():
    data_path = next((p for p in possible_data_paths if os.path.exists(p)), possible_data_paths[0])
    return pd.read_csv(data_path)

try:
    df = load_data()    
    model = joblib.load(os.path.join(script_dir, 'study_model.v1'))
except Exception:
    st.error("Ошибка: Убедитесь, что файлы 'dataset_study.csv' и 'study_model.v1' находятся в правильной папке.")
    st.stop()

st.title("Прогноз академической успеваемости")
st.write("Тема: Линейная регрессия — зависимость оценки от времени обучения")
st.write("Название датасета: Student Study Hours and Grades")

col_left, col_right = st.columns(2)

with col_left:
    st.header("Визуализация модели")
    
    # Подготовка данных для графика
    features = ['study_hours']
    X = df[features]
    y = df['grade']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)

    # Построение графика
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, color='green', alpha=0.4, label="Предсказания")
    
    # Линия идеального прогноза
    ax.plot([y.min(), y.max()], [y.min(), y.max()], color='red', lw=2, label="Идеальная точность")

    ax.set_xlabel("Реальные оценки")
    ax.set_ylabel("Предсказанные оценки")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    st.pyplot(fig)

    st.header("Инструмент предсказания")
    st.write("Введите количество часов, чтобы узнать примерную оценку:")
    
    # Ввод данных пользователем
    val_hours = st.number_input("Часы обучения (study_hours)", min_value=0.0, max_value=24.0, value=5.0, step=0.5)

    if st.button("Рассчитать оценку"):
        input_row = pd.DataFrame([[val_hours]], columns=features)
        res = model.predict(input_row)
        
        # Ограничим результат разумными рамками (например, от 0 до 100)
        final_score = max(0, min(100, res[0]))
        
        st.write("Результат модели:")
        st.success(f"Предполагаемая оценка: {final_score:.2f}")

with col_right:
    st.header("Информация о данных")

    st.write(f"Общее количество записей в базе: **{len(df)}**")

    st.write("Первые 15 строк датасета:")
    st.dataframe(df.head(15), use_container_width=True)

    st.write("Все колонки в файле:")
    st.info(", ".join(df.columns))

    st.write("Входной параметр для модели:")
    st.code("study_hours")

    st.write("Целевой параметр (что предсказываем):")
    st.code("grade")

    # Примечание: точность лучше выводить динамически или брать из результатов train.py
    st.write("Описание:")
    st.write("Модель анализирует связь между временем, затраченным на учебу, и итоговым баллом. Чем выше плотность точек вдоль красной линии на графике, тем точнее работает алгоритм.")