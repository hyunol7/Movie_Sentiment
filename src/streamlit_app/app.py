"""
Streamlit 웹 대시보드
- 영화 리뷰 감성 분석
- 영화 추천 시스템
- 데이터 시각화
- 모델 성능 비교
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import sys
import os
from datetime import datetime

# 현재 디렉토리를 파이썬 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

try:
    from src.models.recommendation_system import MovieRecommendationSystem
    from src.preprocessing.data_preprocessor import MovieDataPreprocessor
except ImportError:
    st.error("모듈을 가져올 수 없습니다. 파일 경로를 확인해주세요.")

# 페이지 설정
st.set_page_config(
    page_title="영화 감성 분석 & 추천 시스템",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 사이드바 메뉴
st.sidebar.title("🎬 영화 분석 대시보드")
st.sidebar.markdown("---")

menu_options = [
    "🏠 홈",
    "📊 데이터 탐색",
    "🤖 감성 분석",
    "🎯 영화 추천",
    "📈 모델 성능",
    "ℹ️ 정보"
]

selected_menu = st.sidebar.selectbox("메뉴 선택", menu_options)

# 데이터 로딩 함수
@st.cache_data
def load_data():
    """데이터 로딩"""
    try:
        imdb_df = pd.read_csv("data/imdb_preprocessed.csv")
        netflix_df = pd.read_csv("data/netflix_preprocessed.csv")
        movies_df = netflix_df[netflix_df['type'] == 'Movie'].copy()
        
        return imdb_df, netflix_df, movies_df
    except FileNotFoundError:
        return None, None, None

# 모델 로딩 함수
@st.cache_resource
def load_models():
    """감성 분석 모델 로딩"""
    try:
        model = joblib.load("data/models/best_model.pkl")
        vectorizer = joblib.load("data/tfidf_vectorizer.pkl")
        return model, vectorizer
    except FileNotFoundError:
        return None, None

# 추천 시스템 로딩
@st.cache_resource
def load_recommendation_system():
    """추천 시스템 로딩"""
    try:
        recommender = MovieRecommendationSystem()
        if recommender.load_data():
            recommender.preprocess_movies_data()
            recommender.build_content_based_recommender()
            return recommender
    except:
        pass
    return None

# 메인 함수들
def show_home():
    """홈 페이지"""
    st.title("🎬 영화 리뷰 감성 분석 & 추천 시스템")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="📱 IMDB 리뷰 데이터",
            value="50,000개",
            delta="긍정/부정 라벨링"
        )
    
    with col2:
        st.metric(
            label="🎥 Netflix 영화",
            value="6,000+개",
            delta="메타데이터 포함"
        )
    
    with col3:
        st.metric(
            label="🤖 ML/DL 모델",
            value="7개",
            delta="성능 비교"
        )
    
    st.markdown("---")
    
    st.markdown("""
    ## 📖 프로젝트 개요
    
    이 프로젝트는 영화 리뷰에 대한 감성 분석과 개인화된 영화 추천 시스템을 구현합니다.
    
    ### 🎯 주요 기능
    
    1. **감성 분석**: IMDB 리뷰 데이터를 활용한 긍정/부정 감성 분류
    2. **영화 추천**: Netflix 메타데이터와 감성 분석 결과를 결합한 추천
    3. **모델 비교**: 머신러닝과 딥러닝 모델 성능 비교
    4. **시각화**: 데이터 탐색 및 분석 결과 시각화
    
    ### 🔧 사용 기술
    
    - **머신러닝**: Logistic Regression, Naive Bayes, SVM, Random Forest
    - **딥러닝**: LSTM, CNN, Hybrid Models
    - **추천 시스템**: Content-based, Collaborative Filtering
    - **웹 프레임워크**: Streamlit
    - **시각화**: Matplotlib, Seaborn, Plotly
    """)
    
    st.markdown("---")
    
    st.info("👈 사이드바에서 원하는 기능을 선택해주세요!")

def show_data_exploration():
    """데이터 탐색 페이지"""
    st.title("📊 데이터 탐색")
    
    # 데이터 로드
    imdb_df, netflix_df, movies_df = load_data()
    
    if imdb_df is None:
        st.error("데이터를 로드할 수 없습니다. 전처리를 먼저 실행해주세요.")
        return
    
    # 탭 생성
    tab1, tab2, tab3 = st.tabs(["📱 IMDB 데이터", "🎥 Netflix 데이터", "📈 통계"])
    
    with tab1:
        st.subheader("IMDB 리뷰 데이터")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("전체 리뷰 수", f"{len(imdb_df):,}")
            
            # 감성 분포
            sentiment_counts = imdb_df['sentiment'].value_counts()
            fig_pie = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="감성 분포",
                color_discrete_map={'positive': '#2E8B57', 'negative': '#DC143C'}
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # 리뷰 길이 분포
            imdb_df['review_length'] = imdb_df['review'].str.len()
            
            fig_hist = px.histogram(
                imdb_df, 
                x='review_length', 
                color='sentiment',
                title="리뷰 길이 분포",
                nbins=50,
                color_discrete_map={'positive': '#2E8B57', 'negative': '#DC143C'}
            )
            fig_hist.update_xaxes(range=[0, 5000])
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # 샘플 데이터 표시
        st.subheader("샘플 리뷰")
        sample_df = imdb_df.sample(5)[['review', 'sentiment']]
        st.dataframe(sample_df, use_container_width=True)
    
    with tab2:
        st.subheader("Netflix 영화 데이터")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("전체 영화 수", f"{len(movies_df):,}")
            
            # 연도별 영화 수
            year_counts = movies_df['release_year'].value_counts().sort_index()
            recent_years = year_counts[year_counts.index >= 2000]
            
            fig_line = px.line(
                x=recent_years.index,
                y=recent_years.values,
                title="연도별 영화 수 (2000년 이후)",
                labels={'x': '연도', 'y': '영화 수'}
            )
            st.plotly_chart(fig_line, use_container_width=True)
        
        with col2:
            # 상위 국가
            country_counts = movies_df['country'].value_counts().head(10)
            
            fig_bar = px.bar(
                x=country_counts.values,
                y=country_counts.index,
                orientation='h',
                title="상위 10개 국가",
                labels={'x': '영화 수', 'y': '국가'}
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # 장르 분석
        st.subheader("장르 분석")
        
        # 장르 추출
        all_genres = []
        for genres in movies_df['listed_in'].dropna():
            all_genres.extend([genre.strip() for genre in str(genres).split(',')])
        
        from collections import Counter
        genre_counts = Counter(all_genres)
        top_genres = dict(genre_counts.most_common(15))
        
        fig_genres = px.bar(
            x=list(top_genres.keys()),
            y=list(top_genres.values()),
            title="상위 15개 장르",
            labels={'x': '장르', 'y': '영화 수'}
        )
        fig_genres.update_xaxes(tickangle=45)
        st.plotly_chart(fig_genres, use_container_width=True)
    
    with tab3:
        st.subheader("통계 요약")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**IMDB 데이터 통계**")
            
            imdb_stats = {
                "전체 리뷰 수": f"{len(imdb_df):,}",
                "긍정 리뷰": f"{len(imdb_df[imdb_df['sentiment']=='positive']):,}",
                "부정 리뷰": f"{len(imdb_df[imdb_df['sentiment']=='negative']):,}",
                "평균 리뷰 길이": f"{imdb_df['review_length'].mean():.0f} 문자",
                "최대 리뷰 길이": f"{imdb_df['review_length'].max():,} 문자"
            }
            
            for key, value in imdb_stats.items():
                st.metric(key, value)
        
        with col2:
            st.write("**Netflix 데이터 통계**")
            
            netflix_stats = {
                "전체 영화 수": f"{len(movies_df):,}",
                "최신 영화 연도": f"{movies_df['release_year'].max()}",
                "가장 오래된 영화": f"{movies_df['release_year'].min()}",
                "고유 감독 수": f"{movies_df['director'].nunique():,}",
                "고유 국가 수": f"{movies_df['country'].nunique():,}"
            }
            
            for key, value in netflix_stats.items():
                st.metric(key, value)

def show_sentiment_analysis():
    """감성 분석 페이지"""
    st.title("🤖 영화 리뷰 감성 분석")
    
    # 모델 로드
    model, vectorizer = load_models()
    
    if model is None or vectorizer is None:
        st.error("감성 분석 모델을 로드할 수 없습니다. 모델을 먼저 학습해주세요.")
        return
    
    st.markdown("영화 리뷰를 입력하면 AI가 감성을 분석해드립니다!")
    
    # 사용자 입력
    user_review = st.text_area(
        "🎬 영화 리뷰를 입력하세요:",
        placeholder="예: This movie was absolutely amazing! The acting was superb and the plot was engaging...",
        height=150
    )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        analyze_button = st.button("🔍 감성 분석 실행", type="primary")
    
    if analyze_button and user_review.strip():
        with st.spinner("감성 분석 중..."):
            try:
                # 텍스트 전처리
                preprocessor = MovieDataPreprocessor()
                cleaned_text = preprocessor.clean_text(user_review)
                processed_text = ' '.join(preprocessor.tokenize_and_lemmatize(cleaned_text))
                
                # 예측
                text_vector = vectorizer.transform([processed_text])
                prediction = model.predict(text_vector)[0]
                probability = model.predict_proba(text_vector)[0]
                
                # 결과 표시
                st.markdown("---")
                st.subheader("📊 분석 결과")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    sentiment_label = "긍정적 😊" if prediction == 1 else "부정적 😞"
                    sentiment_color = "green" if prediction == 1 else "red"
                    st.markdown(f"**예측 감성**: :{sentiment_color}[{sentiment_label}]")
                
                with col2:
                    confidence = max(probability)
                    st.metric("신뢰도", f"{confidence:.1%}")
                
                with col3:
                    st.metric("처리된 단어 수", len(processed_text.split()))
                
                # 확률 시각화
                st.subheader("🎯 감성 확률")
                
                prob_df = pd.DataFrame({
                    'Sentiment': ['부정적', '긍정적'],
                    'Probability': probability,
                    'Color': ['#DC143C', '#2E8B57']
                })
                
                fig_prob = px.bar(
                    prob_df,
                    x='Sentiment',
                    y='Probability',
                    color='Color',
                    title="감성별 확률",
                    color_discrete_map={'#DC143C': '#DC143C', '#2E8B57': '#2E8B57'}
                )
                fig_prob.update_layout(showlegend=False)
                fig_prob.update_yaxes(range=[0, 1])
                
                st.plotly_chart(fig_prob, use_container_width=True)
                
                # 텍스트 전처리 결과
                with st.expander("🔧 텍스트 전처리 결과 보기"):
                    st.write("**원본 텍스트:**")
                    st.write(user_review)
                    st.write("**전처리된 텍스트:**")
                    st.write(processed_text)
                
            except Exception as e:
                st.error(f"분석 중 오류가 발생했습니다: {str(e)}")
    
    elif analyze_button:
        st.warning("리뷰를 입력해주세요!")
    
    # 예시 리뷰들
    st.markdown("---")
    st.subheader("📝 예시 리뷰로 테스트해보기")
    
    example_reviews = {
        "긍정적 예시 1": "This movie was absolutely fantastic! The cinematography was breathtaking and the acting was superb. I would definitely recommend it to everyone.",
        "긍정적 예시 2": "One of the best films I've ever seen. The story was compelling and the characters were well-developed. Amazing experience!",
        "부정적 예시 1": "This movie was a complete waste of time. The plot was confusing and the acting was terrible. I couldn't wait for it to end.",
        "부정적 예시 2": "Boring and predictable. The dialogue was awful and the special effects looked cheap. Not recommended at all."
    }
    
    selected_example = st.selectbox("예시 선택:", list(example_reviews.keys()))
    
    if st.button("예시 리뷰로 테스트"):
        st.text_area("선택된 예시 리뷰:", example_reviews[selected_example], height=100, disabled=True)

def show_movie_recommendation():
    """영화 추천 페이지"""
    st.title("🎯 영화 추천 시스템")
    
    # 추천 시스템 로드
    recommender = load_recommendation_system()
    
    if recommender is None:
        st.error("추천 시스템을 로드할 수 없습니다.")
        return
    
    # 추천 유형 선택
    recommendation_type = st.selectbox(
        "추천 방식을 선택하세요:",
        ["내용 기반 추천", "감성 기반 추천", "장르 기반 추천", "하이브리드 추천"]
    )
    
    if recommendation_type == "내용 기반 추천":
        st.subheader("🎬 비슷한 영화 찾기")
        st.markdown("좋아하는 영화를 입력하면 비슷한 영화를 추천해드립니다!")
        
        movie_title = st.text_input("영화 제목을 입력하세요:", placeholder="예: The Crown, Stranger Things")
        num_recs = st.slider("추천 받을 영화 수:", 1, 20, 10)
        
        if st.button("🔍 비슷한 영화 찾기") and movie_title:
            with st.spinner("비슷한 영화를 찾는 중..."):
                result = recommender.get_content_based_recommendations(movie_title, num_recs)
                
                if isinstance(result, dict):
                    st.success(f"'{result['input_movie']['title']}' 기반 추천 완료!")
                    
                    # 입력 영화 정보
                    with st.expander("📽️ 입력 영화 정보"):
                        input_movie = result['input_movie']
                        st.write(f"**제목**: {input_movie['title']}")
                        st.write(f"**감독**: {input_movie['director']}")
                        st.write(f"**장르**: {input_movie['genres']}")
                        st.write(f"**년도**: {input_movie['year']}")
                        st.write(f"**설명**: {input_movie['description']}")
                    
                    # 추천 영화들
                    st.subheader("📋 추천 영화 목록")
                    
                    for idx, (_, row) in enumerate(result['recommendations'].iterrows(), 1):
                        with st.container():
                            col1, col2 = st.columns([1, 4])
                            
                            with col1:
                                st.markdown(f"**#{idx}**")
                                st.markdown(f"⭐ {row['similarity_score']:.3f}")
                            
                            with col2:
                                st.markdown(f"**{row['title']}** ({row['release_year']})")
                                st.markdown(f"🎬 감독: {row['director']}")
                                st.markdown(f"🏷️ 장르: {row['listed_in']}")
                                st.markdown(f"📝 {row['description'][:200]}...")
                            
                            st.markdown("---")
                else:
                    st.error(result)
    
    elif recommendation_type == "감성 기반 추천":
        st.subheader("😊 감성 기반 영화 추천")
        st.markdown("원하는 감성과 장르를 선택하면 맞춤 영화를 추천해드립니다!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            preferred_sentiment = st.selectbox(
                "선호하는 감성:",
                ["positive", "negative"],
                format_func=lambda x: "긍정적 😊" if x == "positive" else "부정적 😞"
            )
        
        with col2:
            available_genres = ["Action", "Comedy", "Drama", "Horror", "Romance", "Thriller", "Sci-Fi", "Animation"]
            selected_genres = st.multiselect("선호 장르 (선택사항):", available_genres)
        
        num_recs = st.slider("추천 받을 영화 수:", 1, 20, 10, key="sentiment_recs")
        
        if st.button("🎯 감성 기반 추천 받기"):
            with st.spinner("감성 기반 추천 생성 중..."):
                recommendations = recommender.get_sentiment_based_recommendations(
                    preferred_sentiment, selected_genres if selected_genres else None, num_recs
                )
                
                if len(recommendations) > 0:
                    sentiment_text = "긍정적인" if preferred_sentiment == "positive" else "부정적인"
                    st.success(f"{sentiment_text} 감성의 영화 {len(recommendations)}개를 찾았습니다!")
                    
                    for idx, (_, row) in enumerate(recommendations.iterrows(), 1):
                        with st.container():
                            col1, col2 = st.columns([1, 4])
                            
                            with col1:
                                st.markdown(f"**#{idx}**")
                                sentiment_score = row['sentiment_score']
                                st.markdown(f"😊 {sentiment_score:.3f}" if sentiment_score > 0.5 else f"😞 {sentiment_score:.3f}")
                            
                            with col2:
                                st.markdown(f"**{row['title']}** ({row['release_year']})")
                                st.markdown(f"🎬 감독: {row['director']}")
                                st.markdown(f"🏷️ 장르: {row['listed_in']}")
                                st.markdown(f"📝 {row['description'][:200]}...")
                            
                            st.markdown("---")
                else:
                    st.warning("조건에 맞는 영화를 찾을 수 없습니다.")
    
    elif recommendation_type == "장르 기반 추천":
        st.subheader("🏷️ 장르 기반 영화 추천")
        
        available_genres = ["Action", "Comedy", "Drama", "Horror", "Romance", "Thriller", "Sci-Fi", "Animation", "Documentary"]
        selected_genres = st.multiselect("선호하는 장르를 선택하세요:", available_genres)
        num_recs = st.slider("추천 받을 영화 수:", 1, 20, 10, key="genre_recs")
        
        if st.button("🎬 장르 기반 추천 받기") and selected_genres:
            with st.spinner("장르 기반 추천 생성 중..."):
                recommendations = recommender.get_genre_based_recommendations(selected_genres, num_recs)
                
                if len(recommendations) > 0:
                    st.success(f"{', '.join(selected_genres)} 장르의 영화 {len(recommendations)}개를 찾았습니다!")
                    
                    for idx, (_, row) in enumerate(recommendations.iterrows(), 1):
                        with st.container():
                            col1, col2 = st.columns([1, 4])
                            
                            with col1:
                                st.markdown(f"**#{idx}**")
                                st.markdown(f"🎯 {row['genre_match_score']:.3f}")
                            
                            with col2:
                                st.markdown(f"**{row['title']}** ({row['release_year']})")
                                st.markdown(f"🎬 감독: {row['director']}")
                                st.markdown(f"🏷️ 장르: {row['listed_in']}")
                                st.markdown(f"📝 {row['description'][:200]}...")
                            
                            st.markdown("---")
                else:
                    st.warning("조건에 맞는 영화를 찾을 수 없습니다.")
        
        elif st.button("🎬 장르 기반 추천 받기"):
            st.warning("최소 하나의 장르를 선택해주세요!")

def show_model_performance():
    """모델 성능 페이지"""
    st.title("📈 모델 성능 비교")
    
    try:
        # 머신러닝 결과 로드
        ml_results = joblib.load("data/models/training_results.pkl")
        
        st.subheader("🤖 머신러닝 모델 성능")
        
        # 성능 데이터 준비
        ml_data = []
        for name, result in ml_results.items():
            metrics = result['metrics']
            ml_data.append({
                'Model': name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'ROC-AUC': metrics['roc_auc'],
                'CV Mean': metrics['cv_mean'],
                'CV Std': metrics['cv_std']
            })
        
        ml_df = pd.DataFrame(ml_data)
        ml_df = ml_df.sort_values('Accuracy', ascending=False)
        
        # 성능 테이블
        st.dataframe(ml_df.round(4), use_container_width=True)
        
        # 성능 시각화
        col1, col2 = st.columns(2)
        
        with col1:
            # 정확도 비교
            fig_acc = px.bar(
                ml_df, 
                x='Model', 
                y='Accuracy',
                title="모델별 정확도",
                color='Accuracy',
                color_continuous_scale='viridis'
            )
            fig_acc.update_xaxes(tickangle=45)
            st.plotly_chart(fig_acc, use_container_width=True)
        
        with col2:
            # F1-Score vs ROC-AUC
            fig_scatter = px.scatter(
                ml_df,
                x='F1-Score',
                y='ROC-AUC',
                text='Model',
                title="F1-Score vs ROC-AUC",
                size='Accuracy'
            )
            fig_scatter.update_traces(textposition="top center")
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # 종합 성능 레이더 차트
        st.subheader("🎯 종합 성능 비교 (레이더 차트)")
        
        metrics_for_radar = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        
        fig_radar = go.Figure()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, (_, row) in enumerate(ml_df.iterrows()):
            fig_radar.add_trace(go.Scatterpolar(
                r=[row[metric] for metric in metrics_for_radar],
                theta=metrics_for_radar,
                fill='toself',
                name=row['Model'],
                line_color=colors[i % len(colors)]
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="모델별 종합 성능 비교"
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # 최고 성능 모델
        best_model = ml_df.iloc[0]
        st.success(f"🏆 최고 성능 모델: **{best_model['Model']}** (정확도: {best_model['Accuracy']:.4f})")
        
    except FileNotFoundError:
        st.error("모델 결과 파일을 찾을 수 없습니다. 모델을 먼저 학습해주세요.")
    
    # 모델 학습 가이드
    st.markdown("---")
    st.subheader("🔧 모델 학습 가이드")
    
    st.markdown("""
    모델이 학습되지 않았다면 다음 단계를 따라해주세요:
    
    1. **데이터 전처리**: `python src/preprocessing/data_preprocessor.py`
    2. **머신러닝 모델 학습**: `python src/models/ml_models.py`
    3. **딥러닝 모델 학습**: `python src/models/dl_models.py`
    4. **추천 시스템 구축**: `python src/models/recommendation_system.py`
    """)

def show_info():
    """정보 페이지"""
    st.title("ℹ️ 프로젝트 정보")
    
    st.markdown("""
    ## 🎯 프로젝트 목표
    
    영화 리뷰에 대한 감성 분석과 개인화된 영화 추천 시스템을 구현하여 
    사용자에게 맞춤형 영화 추천 서비스를 제공합니다.
    
    ## 📊 데이터셋
    
    ### IMDB Movie Reviews
    - **크기**: 50,000개 리뷰
    - **라벨**: 긍정/부정 감성 라벨링
    - **용도**: 감성 분석 모델 학습
    
    ### Netflix Movies and TV Shows
    - **크기**: 8,000+ 작품 메타데이터
    - **포함 정보**: 제목, 감독, 배우, 장르, 설명, 출시년도
    - **용도**: 추천 시스템 구축
    
    ## 🤖 모델 아키텍처
    
    ### 머신러닝 모델
    - **Logistic Regression**: 선형 분류 모델
    - **Naive Bayes**: 확률 기반 분류
    - **SVM**: 서포트 벡터 머신
    - **Random Forest**: 앙상블 모델
    
    ### 딥러닝 모델
    - **LSTM**: 순환 신경망
    - **CNN**: 1D 합성곱 신경망
    - **Hybrid**: LSTM + CNN 결합 모델
    
    ### 추천 시스템
    - **Content-based**: 내용 기반 필터링
    - **Sentiment-based**: 감성 분석 기반 추천
    - **Genre-based**: 장르 기반 추천
    - **Hybrid**: 복합 추천 시스템
    
    ## 🛠️ 기술 스택
    
    - **언어**: Python 3.8+
    - **웹 프레임워크**: Streamlit
    - **머신러닝**: scikit-learn
    - **딥러닝**: TensorFlow/Keras
    - **자연어 처리**: NLTK, TF-IDF
    - **시각화**: Matplotlib, Seaborn, Plotly
    - **데이터 처리**: Pandas, NumPy
    
    ## 📁 프로젝트 구조
    
    ```
    movie_proj/
    ├── data/                    # 데이터 및 모델 저장
    ├── src/
    │   ├── preprocessing/       # 데이터 전처리
    │   ├── models/             # 모델 구현
    │   ├── utils/              # 유틸리티 함수
    │   └── streamlit_app/      # 웹 애플리케이션
    ├── static/                 # 정적 파일 (이미지, 그래프)
    ├── notebooks/              # Jupyter 노트북
    └── requirements.txt        # 의존성 패키지
    ```
    
    ## 🚀 실행 방법
    
    1. **환경 설정**
    ```bash
    pip install -r requirements.txt
    ```
    
    2. **데이터 전처리**
    ```bash
    python src/preprocessing/data_preprocessor.py
    ```
    
    3. **모델 학습**
    ```bash
    python src/models/ml_models.py
    python src/models/dl_models.py
    ```
    
    4. **웹 애플리케이션 실행**
    ```bash
    streamlit run src/streamlit_app/app.py
    ```
    
    ## 📈 성능 지표
    
    - **정확도 (Accuracy)**: 전체 예측 중 정확한 예측의 비율
    - **정밀도 (Precision)**: 긍정 예측 중 실제 긍정의 비율
    - **재현율 (Recall)**: 실제 긍정 중 정확히 예측한 비율
    - **F1-Score**: 정밀도와 재현율의 조화평균
    - **ROC-AUC**: ROC 곡선 아래 면적
    
    ## 👥 기여하기
    
    이 프로젝트는 교육 목적으로 제작되었습니다. 
    개선사항이나 버그를 발견하시면 이슈를 등록해주세요!
    """)

# 메인 실행
def main():
    """메인 함수"""
    
    # 메뉴에 따른 페이지 표시
    if selected_menu == "🏠 홈":
        show_home()
    elif selected_menu == "📊 데이터 탐색":
        show_data_exploration()
    elif selected_menu == "🤖 감성 분석":
        show_sentiment_analysis()
    elif selected_menu == "🎯 영화 추천":
        show_movie_recommendation()
    elif selected_menu == "📈 모델 성능":
        show_model_performance()
    elif selected_menu == "ℹ️ 정보":
        show_info()
    
    # 푸터
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "🎬 Movie Sentiment Analysis & Recommendation System | "
        f"© 2024 | 마지막 업데이트: {datetime.now().strftime('%Y-%m-%d')}"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

