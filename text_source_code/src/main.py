import os
import re
import string
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)


def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def lemmatize_text(text, lemmatizer):
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(lemmatized_tokens)


def remove_stopwords(text, stop_words):
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)


def search_engine(query, vectorizer, tfidf_matrix, df, top_n=3):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    cleaned_query = clean_text(query)
    lemmed_query = lemmatize_text(cleaned_query, lemmatizer)
    filtered_query = remove_stopwords(lemmed_query, stop_words)

    query_vec = vectorizer.transform([filtered_query])

    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

    top_indices = similarities.argsort()[-top_n:][::-1]

    print(f"\nРезультаты поиска по запросу: '{query}'")
    print("-" * 50)
    for idx in top_indices:
        score = similarities[idx]
        if score > 0:
            print(f"Похожесть: {score:.4f} | Тип: {df.iloc[idx]['label_raw']}")
            print(f"Текст: {df.iloc[idx]['text']}\n")
        else:
            print("Ничего не найдено с совпадением выше нуля.")
            break


def main():
    # ЗАГРУЗКА И ПРЕДОБРАБОТКА ДАННЫХ
    print("Загрузка данных...")

    try:
        if os.path.exists('../data/data.csv'):
            df = pd.read_csv('../data/data.csv', encoding='latin-1')
        elif os.path.exists('data.csv'):
            df = pd.read_csv('data.csv', encoding='latin-1')
        else:
            raise FileNotFoundError
    except FileNotFoundError:
        print("Ошибка: Файл 'data.csv' не найден!")
        return

    df = df[['v1', 'v2']]
    df.columns = ['label_raw', 'text']
    df = df.dropna().reset_index(drop=True)
    df['label'] = df['label_raw'].map({'ham': 0, 'spam': 1})

    print(import_stats(df))
    print("\n" + "=" * 50 + "\nНАЧАЛО ОБРАБОТКИ\n" + "=" * 50)

    # ОЧИСТКА ТЕКСТА
    print("\n--- Шаг 1: Очистка текста ---")
    df['cleaned_text'] = df['text'].apply(clean_text)

    print("Примеры до и после очистки:")
    for i in range(2):
        print(f"До:    {df['text'].iloc[i]}")
        print(f"После: {df['cleaned_text'].iloc[i]}\n")

    # ЛЕММАТИЗАЦИЯ
    print("\n--- Шаг 2: Лемматизация ---")
    lemmatizer = WordNetLemmatizer()
    df['lemmatized_text'] = df['cleaned_text'].apply(lambda x: lemmatize_text(x, lemmatizer))

    print("Примеры до и после лемматизации:")
    for i in range(2):
        print(f"До:    {df['cleaned_text'].iloc[i]}")
        print(f"После: {df['lemmatized_text'].iloc[i]}\n")

    # ПОДСЧЁТ ЧАСТОТЫ СЛОВ (ДО УДАЛЕНИЯ СТОП-СЛОВ)
    print("\n--- Шаг 3: Подсчёт частоты слов (До удаления стоп-слов) ---")
    all_words_before = ' '.join(df['lemmatized_text']).split()
    freq_before = pd.Series(all_words_before).value_counts()
    print("Топ-10 самых частых слов:")
    print(freq_before.head(10))

    # Столбчатый график (До)
    plt.figure(figsize=(10, 5))
    freq_before.head(10).plot(kind='bar', color='skyblue')
    plt.title('Топ-10 слов до удаления стоп-слов')
    plt.ylabel('Частота')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Облако слов (До)
    wordcloud_before = WordCloud(width=800, height=400, background_color='white').generate(
        ' '.join(df['lemmatized_text']))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud_before, interpolation='bilinear')
    plt.axis('off')
    plt.title('Облако слов до удаления стоп-слов')
    plt.show()

    # УДАЛЕНИЕ СТОП-СЛОВ
    print("\n--- Шаг 4: Удаление стоп-слов ---")
    stop_words = set(stopwords.words('english'))
    df['final_text'] = df['lemmatized_text'].apply(lambda x: remove_stopwords(x, stop_words))

    print("Примеры текстов до и после удаления стоп-слов:")
    for i in range(2):
        print(f"До:    {df['lemmatized_text'].iloc[i]}")
        print(f"После: {df['final_text'].iloc[i]}\n")

    all_words_after = ' '.join(df['final_text']).split()
    freq_after = pd.Series(all_words_after).value_counts()
    print("Топ-10 самых частых слов после удаления стоп-слов:")
    print(freq_after.head(10))

    # Столбчатый график (После)
    plt.figure(figsize=(10, 5))
    freq_after.head(10).plot(kind='bar', color='salmon')
    plt.title('Топ-10 слов после удаления стоп-слов')
    plt.ylabel('Частота')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # TF-IDF ВЕКТОРИЗАЦИЯ И ИНФОРМАЦИОННЫЙ ПОИСК
    print("\n--- Шаг 5: TF-IDF векторизация ---")
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['final_text'])

    print(f"Размер словаря (количество признаков): {len(tfidf_vectorizer.vocabulary_)}")

    sample_vector = tfidf_matrix[0].toarray()
    print(f"Размерность вектора для одного текста: {sample_vector.shape}")
    print(f"Пример ненулевых значений вектора первого текста:\n",
          {word: sample_vector[0][idx] for word, idx in tfidf_vectorizer.vocabulary_.items() if
           sample_vector[0][idx] > 0})

    # ИНФОРМАЦИОННЫЙ ПОИСК
    print("\n--- Шаг 6: Информационный поиск ---")
    search_engine("Free entry cash prize", tfidf_vectorizer, tfidf_matrix, df, top_n=2)
    search_engine("Hey darling call me later", tfidf_vectorizer, tfidf_matrix, df, top_n=2)

    print("\n" + "=" * 50 + "\nПЕРЕХОД К ОБУЧЕНИЮ МОДЕЛИ ИЗ ИСХОДНОГО КОДА\n" + "=" * 50)

    # ОБУЧЕНИЕ МОДЕЛИ
    print("\nРазделение выборки на Train, Val и Test (80% / 10% / 10%)...")

    X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
        df['final_text'], df['label'],
        test_size=0.20,
        random_state=42,
        stratify=df['label']
    )

    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp_raw, y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    # === ВОЗВРАЩЕНО СОХРАНЕНИЕ ФАЙЛОВ НА ДИСК ===
    os.makedirs('data', exist_ok=True)
    pd.DataFrame({'text': X_train_raw, 'label': y_train}).to_csv('data/train.csv', index=False)
    pd.DataFrame({'text': X_val_raw, 'label': y_val}).to_csv('data/val.csv', index=False)
    pd.DataFrame({'text': X_test_raw, 'label': y_test}).to_csv('data/test.csv', index=False)
    print("Файлы train.csv, val.csv, test.csv успешно сохранены в папку 'data/'!")

    clf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train = clf_vectorizer.fit_transform(X_train_raw).toarray()
    X_val = clf_vectorizer.transform(X_val_raw).toarray()
    X_test = clf_vectorizer.transform(X_test_raw).toarray()

    print("\nОбучение модели Наивного Байеса (Multinomial Naive Bayes)...")
    model = MultinomialNB()
    model.fit(X_train, y_train)
    print("Модель успешно обучена!")

    print("\n=== РЕЗУЛЬТАТЫ НА ВАЛИДАЦИОННОЙ ВЫБОРКЕ ===")
    y_val_pred = model.predict(X_val)
    print(classification_report(y_val, y_val_pred, target_names=['Ham (0)', 'Spam (1)']))

    print("=== РЕЗУЛЬТАТЫ НА ТЕСТОВОЙ ВЫБОРКЕ ===")
    y_test_pred = model.predict(X_test)
    print(classification_report(y_test, y_test_pred, target_names=['Ham (0)', 'Spam (1)']))

    print("Матрица ошибок (Тест):")
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"True Negative (Верно определен полезный софт/текст): {cm[0][0]}")
    print(f"False Positive (Полезное письмо заблокировано как спам): {cm[0][1]}")
    print(f"False Negative (Спам пропущен в инбокс): {cm[1][0]}")
    print(f"True Positive (Спам успешно заблокирован): {cm[1][1]}")


def import_stats(df):
    total = len(df)
    ham_cnt = len(df[df['label'] == 0])
    spam_cnt = len(df[df['label'] == 1])
    return (f"Всего строк обработано: {total}\n"
            f"Класс 'ham' (не спам): {ham_cnt} ({ham_cnt / total * 100:.2f}%)\n"
            f"Класс 'spam' (спам): {spam_cnt} ({spam_cnt / total * 100:.2f}%)")


if __name__ == '__main__':
    main()