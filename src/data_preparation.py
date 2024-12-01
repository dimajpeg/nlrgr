import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging
import sys
import os

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_data(file_path):
    """
    Функция для загрузки данных из файла CSV.
    Проверяется наличие файла, если файл не найден — процесс завершится с ошибкой.
    """
    if not os.path.exists(file_path):
        logging.error(f"Файл {file_path} не найден.")
        sys.exit(1)
    logging.info(f"Загрузка данных из файла: {file_path}")
    return pd.read_csv(file_path)


def remove_duplicates(df):
    """
    Функция для удаления дубликатов в данных.
    """
    initial_len = len(df)
    df = df.drop_duplicates()
    final_len = len(df)
    if initial_len != final_len:
        logging.info(f"Удалено {initial_len - final_len} дубликатов.")
    return df


def remove_outliers(df, columns):
    """
    Функция для удаления выбросов на основе IQR (межквартильный размах).
    Проверяются только указанные колонки.
    """
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        initial_len = len(df)
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        final_len = len(df)
        if initial_len != final_len:
            logging.info(f"Удалено {initial_len - final_len} выбросов в колонке {col}.")
    return df


def normalize_data(df, columns):
    """
    Функция для нормализации данных с использованием StandardScaler.
    Применяется к указанным колонкам.
    """
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    logging.info(f"Нормализация данных в колонках: {', '.join(columns)}.")
    return df


def check_missing_values(df):
    """
    Функция для проверки наличия пропущенных значений.
    """
    missing_data = df.isnull().sum()
    if missing_data.any():
        logging.warning(f"Пропущенные значения в следующих колонках: \n{missing_data[missing_data > 0]}")
    else:
        logging.info("Пропущенные значения не найдены.")
    return df.dropna()  # Удаление строк с пропущенными значениями


def preprocess_data(df):
    """
    Функция для предобработки данных.
    Она включает удаление дубликатов, выбросов, нормализацию данных и удаление пропущенных значений.
    """
    logging.info("Начало предобработки данных.")

    # Удаляем дубликаты
    df = remove_duplicates(df)

    # Удаляем выбросы в указанных колонках
    df = remove_outliers(df, ['pressure', 'flow', 'temperature'])

    # Проверка и удаление пропущенных значений
    df = check_missing_values(df)

    # Нормализация данных
    df = normalize_data(df, ['pressure', 'flow', 'temperature'])

    logging.info("Предобработка данных завершена.")
    return df


if __name__ == "__main__":
    # Аргумент командной строки для пути к файлу
    file_path = sys.argv[1] if len(sys.argv) > 1 else 'hydraulic_system_processed.csv'

    # Загрузка данных
    df = load_data(file_path)

    # Предобработка данных
    df = preprocess_data(df)

    # Сохранение предобработанных данных в файл
    output_path = 'hydraulic_system_processed.csv'
    df.to_csv(output_path, index=False)
    logging.info(f"Предобработанные данные сохранены в файл: {output_path}")
