"""
영화 추천 시스템 구현
- Content-based Filtering (내용 기반 필터링)
- Collaborative Filtering (협업 필터링)
- Hybrid Recommendation System (하이브리드 추천)
- 감성 분석 결과를 활용한 추천
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import joblib
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class MovieRecommendationSystem:
    def __init__(self):
        self.tfidf_vectorizer = None
        self.content_similarity_matrix = None
        self.movies_df = None
        self.sentiment_model = None
        self.user_profiles = {}
        
    def load_data(self):
        """
        데이터 로드
        """
        try:
            # Netflix 영화 데이터 로드
            self.movies_df = pd.read_csv("data/netflix_movies.csv")
            print(f"영화 데이터 로드 완료: {len(self.movies_df)}개")
            
            # 감성 분석 모델 로드 (최고 성능 모델)
            try:
                self.sentiment_model = joblib.load("data/models/best_model.pkl")
                self.tfidf_vectorizer = joblib.load("data/tfidf_vectorizer.pkl")
                print("감성 분석 모델 로드 완료")
            except:
                print("감성 분석 모델을 찾을 수 없습니다. 기본 추천만 제공됩니다.")
                
        except FileNotFoundError as e:
            print(f"데이터 파일을 찾을 수 없습니다: {e}")
            return False
        
        return True
    
    def preprocess_movies_data(self):
        """
        영화 데이터 전처리
        """
        # 결측값 처리
        self.movies_df['description'] = self.movies_df['description'].fillna('')
        self.movies_df['director'] = self.movies_df['director'].fillna('Unknown')
        self.movies_df['cast'] = self.movies_df['cast'].fillna('Unknown')
        self.movies_df['listed_in'] = self.movies_df['listed_in'].fillna('Unknown')
        
        # 장르 정리
        self.movies_df['genres'] = self.movies_df['listed_in'].apply(
            lambda x: [genre.strip() for genre in str(x).split(',')] if pd.notna(x) else []
        )
        
        # 연도 정리
        self.movies_df['release_year'] = pd.to_numeric(
            self.movies_df['release_year'], errors='coerce'
        ).fillna(2000).astype(int)
        
        # 컨텐츠 특성 결합
        self.movies_df['content_features'] = (
            self.movies_df['description'] + ' ' +
            self.movies_df['listed_in'] + ' ' +
            self.movies_df['director'] + ' ' +
            self.movies_df['cast']
        )
        
        print("영화 데이터 전처리 완료")
    
    def build_content_based_recommender(self):
        """
        내용 기반 추천 시스템 구축
        """
        print("내용 기반 추천 시스템 구축 중...")
        
        # TF-IDF 벡터화
        tfidf = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        tfidf_matrix = tfidf.fit_transform(self.movies_df['content_features'])
        
        # 코사인 유사도 계산
        self.content_similarity_matrix = cosine_similarity(tfidf_matrix)
        
        print("내용 기반 추천 시스템 구축 완료")
    
    def get_content_based_recommendations(self, movie_title, num_recommendations=10):
        """
        내용 기반 추천
        """
        try:
            # 영화 인덱스 찾기
            movie_indices = self.movies_df[
                self.movies_df['title'].str.contains(movie_title, case=False, na=False)
            ].index
            
            if len(movie_indices) == 0:
                return f"'{movie_title}' 영화를 찾을 수 없습니다."
            
            movie_idx = movie_indices[0]
            movie_info = self.movies_df.iloc[movie_idx]
            
            # 유사도 점수 계산
            similarity_scores = list(enumerate(self.content_similarity_matrix[movie_idx]))
            similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
            
            # 추천 영화 추출 (자기 자신 제외)
            recommended_indices = [i[0] for i in similarity_scores[1:num_recommendations+1]]
            
            recommendations = self.movies_df.iloc[recommended_indices][[
                'title', 'director', 'listed_in', 'release_year', 'description'
            ]].copy()
            
            recommendations['similarity_score'] = [
                similarity_scores[i+1][1] for i in range(num_recommendations)
            ]
            
            return {
                'input_movie': {
                    'title': movie_info['title'],
                    'director': movie_info['director'],
                    'genres': movie_info['listed_in'],
                    'year': movie_info['release_year'],
                    'description': movie_info['description'][:200] + '...'
                },
                'recommendations': recommendations
            }
            
        except Exception as e:
            return f"추천 중 오류가 발생했습니다: {str(e)}"
    
    def analyze_sentiment_for_movie(self, movie_description):
        """
        영화 설명에 대한 감성 분석
        """
        if self.sentiment_model is None or self.tfidf_vectorizer is None:
            return 0.5  # 기본값
        
        try:
            # 텍스트 전처리
            cleaned_text = self.clean_text(movie_description)
            
            # TF-IDF 변환
            text_vector = self.tfidf_vectorizer.transform([cleaned_text])
            
            # 감성 예측
            sentiment_prob = self.sentiment_model.predict_proba(text_vector)[0][1]
            
            return sentiment_prob
            
        except Exception as e:
            print(f"감성 분석 오류: {e}")
            return 0.5
    
    def clean_text(self, text):
        """
        텍스트 정제 (감성 분석용)
        """
        if pd.isna(text):
            return ""
        
        # HTML 태그 제거
        text = re.sub(r'<.*?>', '', text)
        
        # 특수문자 제거
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
        # 소문자 변환 및 공백 정리
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def get_sentiment_based_recommendations(self, preferred_sentiment='positive', 
                                          genres=None, num_recommendations=10):
        """
        감성 기반 추천
        """
        print(f"감성 기반 추천 생성 중... (선호 감성: {preferred_sentiment})")
        
        # 감성 점수 계산
        sentiment_threshold = 0.6 if preferred_sentiment == 'positive' else 0.4
        
        filtered_movies = self.movies_df.copy()
        
        # 장르 필터링
        if genres:
            genre_mask = filtered_movies['listed_in'].str.contains(
                '|'.join(genres), case=False, na=False
            )
            filtered_movies = filtered_movies[genre_mask]
        
        # 감성 점수 계산
        sentiment_scores = []
        for idx, row in filtered_movies.iterrows():
            sentiment_score = self.analyze_sentiment_for_movie(row['description'])
            sentiment_scores.append(sentiment_score)
        
        filtered_movies['sentiment_score'] = sentiment_scores
        
        # 선호 감성에 따른 필터링
        if preferred_sentiment == 'positive':
            filtered_movies = filtered_movies[
                filtered_movies['sentiment_score'] >= sentiment_threshold
            ]
            filtered_movies = filtered_movies.sort_values('sentiment_score', ascending=False)
        else:
            filtered_movies = filtered_movies[
                filtered_movies['sentiment_score'] <= sentiment_threshold
            ]
            filtered_movies = filtered_movies.sort_values('sentiment_score', ascending=True)
        
        # 상위 추천 영화 반환
        recommendations = filtered_movies.head(num_recommendations)[[
            'title', 'director', 'listed_in', 'release_year', 
            'description', 'sentiment_score'
        ]]
        
        return recommendations
    
    def get_genre_based_recommendations(self, preferred_genres, num_recommendations=10):
        """
        장르 기반 추천
        """
        print(f"장르 기반 추천 생성 중... (선호 장르: {preferred_genres})")
        
        # 장르 매칭 점수 계산
        genre_scores = []
        
        for idx, row in self.movies_df.iterrows():
            movie_genres = str(row['listed_in']).lower()
            score = sum(1 for genre in preferred_genres if genre.lower() in movie_genres)
            genre_scores.append(score / len(preferred_genres))
        
        self.movies_df['genre_match_score'] = genre_scores
        
        # 장르 매칭 점수가 높은 영화 추천
        filtered_movies = self.movies_df[self.movies_df['genre_match_score'] > 0]
        filtered_movies = filtered_movies.sort_values('genre_match_score', ascending=False)
        
        recommendations = filtered_movies.head(num_recommendations)[[
            'title', 'director', 'listed_in', 'release_year', 
            'description', 'genre_match_score'
        ]]
        
        return recommendations
    
    def get_hybrid_recommendations(self, movie_title=None, preferred_genres=None, 
                                 preferred_sentiment='positive', num_recommendations=10):
        """
        하이브리드 추천 시스템
        """
        print("하이브리드 추천 시스템 실행 중...")
        
        recommendations = []
        
        # 1. 내용 기반 추천 (영화 제목이 주어진 경우)
        if movie_title:
            content_recs = self.get_content_based_recommendations(
                movie_title, num_recommendations//2
            )
            if isinstance(content_recs, dict):
                recommendations.extend([
                    f"내용 기반 ('{movie_title}' 유사): {content_recs['recommendations']}"
                ])
        
        # 2. 감성 기반 추천
        sentiment_recs = self.get_sentiment_based_recommendations(
            preferred_sentiment, preferred_genres, num_recommendations//3
        )
        
        # 3. 장르 기반 추천
        if preferred_genres:
            genre_recs = self.get_genre_based_recommendations(
                preferred_genres, num_recommendations//3
            )
        
        return {
            'sentiment_based': sentiment_recs,
            'genre_based': genre_recs if preferred_genres else None,
            'content_based': content_recs if movie_title else None
        }
    
    def get_popular_movies_by_genre(self, genre, num_movies=10):
        """
        장르별 인기 영화
        """
        genre_movies = self.movies_df[
            self.movies_df['listed_in'].str.contains(genre, case=False, na=False)
        ]
        
        # 최신 영화 우선
        popular_movies = genre_movies.sort_values('release_year', ascending=False)
        
        return popular_movies.head(num_movies)[[
            'title', 'director', 'release_year', 'listed_in', 'description'
        ]]
    
    def save_recommendation_system(self, save_dir="data/models"):
        """
        추천 시스템 저장
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # 유사도 매트릭스 저장
        if self.content_similarity_matrix is not None:
            np.save(
                os.path.join(save_dir, "content_similarity_matrix.npy"),
                self.content_similarity_matrix
            )
        
        # 전처리된 영화 데이터 저장
        if self.movies_df is not None:
            self.movies_df.to_csv(
                os.path.join(save_dir, "processed_movies.csv"), 
                index=False
            )
        
        print("추천 시스템 저장 완료")

def main():
    """
    메인 실행 함수
    """
    print("영화 추천 시스템 구축 시작")
    print("=" * 50)
    
    # 추천 시스템 초기화
    recommender = MovieRecommendationSystem()
    
    # 데이터 로드
    if not recommender.load_data():
        return
    
    # 데이터 전처리
    recommender.preprocess_movies_data()
    
    # 내용 기반 추천 시스템 구축
    recommender.build_content_based_recommender()
    
    # 예시 추천 실행
    print("\n" + "=" * 50)
    print("추천 시스템 테스트")
    print("=" * 50)
    
    # 1. 내용 기반 추천 테스트
    test_movie = "The Crown"
    print(f"\n1. '{test_movie}' 기반 내용 추천:")
    content_recs = recommender.get_content_based_recommendations(test_movie, 5)
    if isinstance(content_recs, dict):
        print(f"입력 영화: {content_recs['input_movie']['title']}")
        print("추천 영화:")
        for idx, row in content_recs['recommendations'].iterrows():
            print(f"- {row['title']} ({row['release_year']}) - 유사도: {row['similarity_score']:.3f}")
    else:
        print(content_recs)
    
    # 2. 감성 기반 추천 테스트
    print(f"\n2. 긍정적 감성 기반 추천:")
    sentiment_recs = recommender.get_sentiment_based_recommendations('positive', ['Comedy'], 5)
    print("추천 영화:")
    for idx, row in sentiment_recs.iterrows():
        print(f"- {row['title']} ({row['release_year']}) - 감성 점수: {row['sentiment_score']:.3f}")
    
    # 3. 장르 기반 추천 테스트
    print(f"\n3. 액션 장르 기반 추천:")
    genre_recs = recommender.get_genre_based_recommendations(['Action'], 5)
    print("추천 영화:")
    for idx, row in genre_recs.iterrows():
        print(f"- {row['title']} ({row['release_year']}) - 장르 매칭: {row['genre_match_score']:.3f}")
    
    # 추천 시스템 저장
    recommender.save_recommendation_system()
    
    print("\n" + "=" * 50)
    print("추천 시스템 구축 완료!")
    print("=" * 50)

if __name__ == "__main__":
    main()

