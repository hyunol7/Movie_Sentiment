"""
영화 리뷰 데이터 전처리 모듈
- IMDB 리뷰 데이터 정제 및 토큰화
- Netflix 메타데이터 처리
- 텍스트 임베딩 생성
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib
import os
from tqdm import tqdm

# NLTK 데이터 다운로드
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# punkt_tab 추가
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class MovieDataPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.tfidf_vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
        
    def clean_text(self, text):
        """
        텍스트 정제 함수
        - HTML 태그 제거
        - 특수문자 제거
        - 소문자 변환
        """
        if pd.isna(text):
            return ""
        
        # HTML 태그 제거
        text = re.sub(r'<.*?>', '', text)
        
        # 특수문자 제거 (문자, 숫자, 공백만 유지)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
        # 소문자 변환
        text = text.lower()
        
        # 여러 공백을 하나로 변환
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def tokenize_and_lemmatize(self, text):
        """
        토큰화 및 표제어 추출
        """
        if not text:
            return []
        
        # 토큰화
        tokens = word_tokenize(text)
        
        # 불용어 제거 및 표제어 추출
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]
        
        return tokens
    
    def preprocess_imdb_data(self, file_path):
        """
        IMDB 데이터 전처리
        """
        print("IMDB 데이터 로딩 중...")
        df = pd.read_csv(file_path)
        
        print(f"원본 데이터 크기: {df.shape}")
        print(f"라벨 분포:\n{df['sentiment'].value_counts()}")
        
        # 텍스트 정제
        print("텍스트 정제 중...")
        tqdm.pandas(desc="텍스트 정제")
        df['cleaned_review'] = df['review'].progress_apply(self.clean_text)
        
        # 토큰화
        print("토큰화 및 표제어 추출 중...")
        tqdm.pandas(desc="토큰화")
        df['tokens'] = df['cleaned_review'].progress_apply(self.tokenize_and_lemmatize)
        
        # 토큰을 다시 텍스트로 결합
        df['processed_review'] = df['tokens'].apply(lambda x: ' '.join(x))
        
        # 빈 리뷰 제거
        df = df[df['processed_review'].str.len() > 0]
        
        # 감성 라벨을 숫자로 변환
        df['sentiment_label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
        
        print(f"전처리 후 데이터 크기: {df.shape}")
        
        return df
    
    def preprocess_netflix_data(self, file_path):
        """
        Netflix 데이터 전처리
        """
        print("Netflix 데이터 로딩 중...")
        df = pd.read_csv(file_path)
        
        print(f"원본 데이터 크기: {df.shape}")
        
        # 결측값 처리
        df['description'] = df['description'].fillna('')
        df['director'] = df['director'].fillna('Unknown')
        df['cast'] = df['cast'].fillna('Unknown')
        df['country'] = df['country'].fillna('Unknown')
        
        # 설명 텍스트 정제
        print("설명 텍스트 정제 중...")
        tqdm.pandas(desc="설명 정제")
        df['cleaned_description'] = df['description'].progress_apply(self.clean_text)
        
        # 장르 데이터 정제
        df['genres'] = df['listed_in'].str.split(',').apply(
            lambda x: [genre.strip() for genre in x] if isinstance(x, list) else []
        )
        
        # 영화와 TV 쇼 분리
        movies_df = df[df['type'] == 'Movie'].copy()
        tv_shows_df = df[df['type'] == 'TV Show'].copy()
        
        print(f"영화 수: {len(movies_df)}")
        print(f"TV 쇼 수: {len(tv_shows_df)}")
        
        return df, movies_df, tv_shows_df
    
    def create_tfidf_features(self, texts, fit=True):
        """
        TF-IDF 특성 생성
        """
        if fit:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        else:
            tfidf_matrix = self.tfidf_vectorizer.transform(texts)
        
        return tfidf_matrix
    
    def split_data(self, df, test_size=0.2, random_state=42):
        """
        데이터 분할
        """
        X = df['processed_review']
        y = df['sentiment_label']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def save_preprocessed_data(self, data, file_path):
        """
        전처리된 데이터 저장
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        if isinstance(data, pd.DataFrame):
            data.to_csv(file_path, index=False)
        else:
            joblib.dump(data, file_path)
        
        print(f"데이터 저장 완료: {file_path}")

def main():
    """
    메인 실행 함수
    """
    preprocessor = MovieDataPreprocessor()
    
    # 데이터 경로
    imdb_path = "IMDB Dataset.csv"
    netflix_path = "netflix_titles.csv"
    
    # IMDB 데이터 전처리
    print("=" * 50)
    print("IMDB 데이터 전처리 시작")
    print("=" * 50)
    
    imdb_df = preprocessor.preprocess_imdb_data(imdb_path)
    
    # 데이터 분할
    X_train, X_test, y_train, y_test = preprocessor.split_data(imdb_df)
    
    # TF-IDF 특성 생성
    print("TF-IDF 특성 생성 중...")
    X_train_tfidf = preprocessor.create_tfidf_features(X_train, fit=True)
    X_test_tfidf = preprocessor.create_tfidf_features(X_test, fit=False)
    
    # 전처리된 데이터 저장
    preprocessor.save_preprocessed_data(imdb_df, "data/imdb_preprocessed.csv")
    
    # TF-IDF 데이터 저장
    joblib.dump(X_train_tfidf, "data/X_train_tfidf.pkl")
    joblib.dump(X_test_tfidf, "data/X_test_tfidf.pkl")
    joblib.dump(y_train, "data/y_train.pkl")
    joblib.dump(y_test, "data/y_test.pkl")
    joblib.dump(preprocessor.tfidf_vectorizer, "data/tfidf_vectorizer.pkl")
    
    # Netflix 데이터 전처리
    print("\n" + "=" * 50)
    print("Netflix 데이터 전처리 시작")
    print("=" * 50)
    
    netflix_df, movies_df, tv_shows_df = preprocessor.preprocess_netflix_data(netflix_path)
    
    # Netflix 데이터 저장
    preprocessor.save_preprocessed_data(netflix_df, "data/netflix_preprocessed.csv")
    preprocessor.save_preprocessed_data(movies_df, "data/netflix_movies.csv")
    preprocessor.save_preprocessed_data(tv_shows_df, "data/netflix_tv_shows.csv")
    
    print("\n" + "=" * 50)
    print("데이터 전처리 완료!")
    print("=" * 50)
    print(f"IMDB 학습 데이터: {X_train.shape[0]}개")
    print(f"IMDB 테스트 데이터: {X_test.shape[0]}개")
    print(f"TF-IDF 특성 수: {X_train_tfidf.shape[1]}개")
    print(f"Netflix 전체 데이터: {len(netflix_df)}개")
    print(f"Netflix 영화: {len(movies_df)}개")
    print(f"Netflix TV 쇼: {len(tv_shows_df)}개")

if __name__ == "__main__":
    main()
