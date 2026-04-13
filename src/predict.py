import joblib
import pandas as pd
import os


script_dir = os.path.dirname(__file__)
model_path = os.path.join(script_dir, 'study_model.v1')
try:
    model = joblib.load(model_path)
except Exception:
    print(f"Ошибка: Сначала запустите train.py для создания модели '{model_path}'")
    exit()

print("СИСТЕМА ПРЕДСКАЗАНИЯ УЧЕБНЫХ РЕЗУЛЬТАТОВ")

try:

    study_hours = float(input("Введите количество часов обучения: "))

    
    input_data = pd.DataFrame(
        [[study_hours]],
        columns=['study_hours']
    )

   
    prediction = model.predict(input_data)
    
    print(f"\nРЕЗУЛЬТАТ ПРОВЕРКИ")
   
    predicted_grade = prediction[0]
    print(f"Предсказанная оценка: {predicted_grade:.2f}")

except Exception as e:
    print(f"Ошибка ввода введите только числа!")