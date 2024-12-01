import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Папка для графиков
GRAPHICS_DIR = 'grafics'

# Проверим, существует ли папка для графиков, если нет — создадим
if not os.path.exists(GRAPHICS_DIR):
    os.makedirs(GRAPHICS_DIR)


def save_plot(fig, plot_name):
    """
    Функция для сохранения графика в папку grafics.
    """
    plot_path = os.path.join(GRAPHICS_DIR, plot_name)
    fig.savefig(plot_path)
    print(f"График сохранен как {plot_path}")


def plot_feature_distributions(df):
    """
    Функция для визуализации распределения признаков: давления, потока и температуры.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    # График распределения давления
    sns.histplot(df['pressure'], kde=True, color='blue', ax=axes[0])
    axes[0].set_title('Distribution of Pressure')

    # График распределения потока
    sns.histplot(df['flow'], kde=True, color='green', ax=axes[1])
    axes[1].set_title('Distribution of Flow')

    # График распределения температуры
    sns.histplot(df['temperature'], kde=True, color='red', ax=axes[2])
    axes[2].set_title('Distribution of Temperature')

    plt.tight_layout()
    save_plot(fig, 'feature_distributions.png')


def plot_feature_importance(model, X_train):
    """
    Функция для визуализации важности признаков.
    """
    feature_importance = model.feature_importances_
    feature_names = X_train.columns

    # Сортируем признаки по важности
    sorted_idx = np.argsort(feature_importance)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(feature_importance)), feature_importance[sorted_idx], align='center')
    ax.set_yticks(range(len(feature_importance)))
    ax.set_yticklabels(np.array(feature_names)[sorted_idx])
    ax.set_xlabel('Feature Importance')
    ax.set_title('Feature Importance in Random Forest Model')

    save_plot(fig, 'feature_importance.png')


def plot_confusion_matrix(y_test, y_pred):
    """
    Функция для построения матрицы путаницы.
    """
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 0', 'Class 1'])
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

    save_plot(fig, 'confusion_matrix.png')


def plot_correlation_matrix(df):
    """
    Функция для построения матрицы корреляции между признаками.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
    ax.set_title('Correlation Matrix')

    save_plot(fig, 'correlation_matrix.png')


def plot_boxplot(df):
    """
    Функция для построения диаграммы размаха (boxplot) для всех признаков.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(data=df[['pressure', 'flow', 'temperature']], ax=ax)
    ax.set_title('Boxplot for Pressure, Flow, and Temperature')

    save_plot(fig, 'boxplot.png')


def train_model(X, y):
    """
    Функция для обучения модели с использованием случайного леса.
    Возвращает обученную модель, а также данные для тестирования.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Параметры для GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }

    # Инициализация модели случайного леса
    model = RandomForestClassifier(random_state=42)

    # GridSearch для поиска лучших гиперпараметров
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    # Выводим лучшие параметры
    print("Лучшие параметры:", grid_search.best_params_)

    # Обучаем модель с оптимальными параметрами
    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)

    # Прогнозы на тестовых данных
    y_pred = best_model.predict(X_test)

    # Оценка модели
    accuracy = best_model.score(X_test, y_test)
    print(f"Точность модели: {accuracy:.2f}")

    # Выводим метрики классификации
    print("Метрики классификации:")
    print(classification_report(y_test, y_pred))

    # Строим матрицу путаницы
    plot_confusion_matrix(y_test, y_pred)

    # Визуализация важности признаков
    plot_feature_importance(best_model, X_train)

    return best_model, X_test, y_test


def cluster_data(X):
    """
    Функция для кластеризации данных методом K-Means.
    Возвращает кластеры и оценку качества кластеризации.
    """
    # Применяем KMeans для кластеризации
    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters = kmeans.fit_predict(X)

    # Оценка качества кластеризации с использованием silhouette score
    silhouette_avg = silhouette_score(X, clusters)
    print(f"Средний силуэтный коэффициент: {silhouette_avg:.2f}")

    # Визуализация кластеров
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=X['pressure'], y=X['flow'], hue=clusters, palette='viridis', ax=ax)
    ax.set_title('K-Means Clustering Results')
    ax.set_xlabel('Pressure')
    ax.set_ylabel('Flow')
    ax.legend(title='Cluster')

    save_plot(fig, 'kmeans_clustering.png')

    return clusters


if __name__ == "__main__":
    # Загрузка данных
    df = pd.read_csv('hydraulic_system_processed.csv')

    # Разделение данных на признаки и целевую переменную
    X = df.drop('target', axis=1)  # Признаки
    y = df['target']  # Целевая переменная

    # Графики распределений признаков
    plot_feature_distributions(df)

    # Кластеризация данных
    cluster_data(X)

    # Построение матрицы корреляции
    plot_correlation_matrix(df)

    # Построение диаграммы размаха
    plot_boxplot(df)

    # Обучение модели
    model, X_test, y_test = train_model(X, y)
