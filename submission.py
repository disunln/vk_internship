import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Функция для предобработки данных
def prepare_data(df):
    X = np.array(df['values'])
    max_len = 97  # Приводим к длине 97
    X = np.array([np.pad(x, (0, max_len - len(x)), 'constant') for x in X])
    return X

# Основная функция
def make_submission(input_file, output_file, model_path):
    # Загрузка данных
    df = pd.read_parquet(input_file)
    
    # Предобработка данных
    X = prepare_data(df)
    X[np.isnan(X)] = 0  # Обработка NaN значений
    
    # Масштабирование данных
    scaler = StandardScaler()
    X = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    
    # Загрузка модели
    model = load_model(model_path)
    
    # Получение предсказаний
    predictions = model.predict(X)
    predicted_labels = np.argmax(predictions, axis=1)  # Предсказания классов
    
    # Формирование submission.csv
    submission_df = pd.DataFrame({
        'id': df['id'],  # Предполагается, что в df есть колонка с ID
        'label': predicted_labels
    })
    
    # Сохранение файла
    submission_df.to_csv(output_file, index=False)

# Пример вызова функции
make_submission(r'/content/train.parquet', 'submission.csv', r'/content/best_model.h5')
