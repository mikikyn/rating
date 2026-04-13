import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ПОИСК ФАЙЛА ПО ВСЕМ ПРАВИЛАМ
base_path = os.path.dirname(__file__)
possible_paths = [
    os.path.join(base_path, "dataset_study.csv"),
    os.path.normpath(os.path.join(base_path, "..", "data", "dataset_study.csv")),
]
file_path = next((p for p in possible_paths if os.path.exists(p)), possible_paths[0])

print(f"Ищу файл тут: {file_path}")

if not os.path.exists(file_path):
    print("ОШИБКА: Файл dataset_study.csv не найден ни в папке src, ни в папке data.")
    print("Переместите dataset_study.csv в папку src рядом со скриптом или в папку data на уровень выше.")
    exit()

# ЗАГРУЗКА
df = pd.read_csv(file_path)

# ОСТАЛЬНОЙ КОД
features = ['study_hours']
X = df[features]
y = df['grade']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = r2_score(y_test, y_pred)
print(f"РЕЗУЛЬТАТ")
print(f"Accuracy: {accuracy:.4f}")

# Сохраняем модель в ту же папку src
model_path = os.path.join(base_path, 'study_model.v1')
joblib.dump(model, model_path)
print(f"Модель сохранена как {model_path}")

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='green', alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=3)
plt.title(f"Accuracy: {accuracy:.4f}")
plt.show()