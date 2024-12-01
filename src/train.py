from src.data_preparation import load_data, preprocess_data
from src.model import train_model, plot_feature_distributions
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def plot_confusion_matrix(y_test, y_pred):
    """
    Функция для визуализации матрицы ошибок.
    """
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()


def evaluate_model(model, X_test, y_test):
    """
    Функция для оценки модели и вывода метрик.
    """
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    logging.info(f"Model Accuracy: {accuracy:.2f}")
    logging.info(f"Model Precision: {precision:.2f}")
    logging.info(f"Model Recall: {recall:.2f}")
    logging.info(f"Model F1-Score: {f1:.2f}")

    # Визуализация матрицы ошибок
    plot_confusion_matrix(y_test, y_pred)


if __name__ == "__main__":
    logging.info("Начало процесса загрузки данных.")

    # Загрузка и предобработка данных
    df = load_data('hydraulic_system_processed.csv')  # Обновленный файл данных
    df = preprocess_data(df)
    df.to_csv('hydraulic_system_processed.csv', index=False)
    logging.info("Данные успешно предобработаны и сохранены.")

    # Визуализация распределений признаков
    logging.info("Начало визуализации распределений признаков.")
    plot_feature_distributions(df)

    # Разделение данных на признаки и целевую переменную
    X = df.drop('target', axis=1)  # Признаки (включая новые)
    y = df['target']  # Целевая переменная

    # Обучение модели
    logging.info("Начало обучения модели.")
    model, X_test, y_test = train_model(X, y)

    # Оценка модели
    logging.info("Оценка модели.")
    evaluate_model(model, X_test, y_test)

    # Сохранение обученной модели для последующего использования
    model_filename = 'hydraulic_system_model.pkl'
    joblib.dump(model, model_filename)
    logging.info(f"Модель сохранена в файл: {model_filename}")

    logging.info("Процесс завершен.")
