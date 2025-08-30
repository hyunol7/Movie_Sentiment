"""
빠른 시작 스크립트
최소한의 모델만 학습하여 빠르게 시스템을 테스트할 수 있습니다.
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def run_command(command, description, timeout=None):
    """명령어 실행 함수"""
    print(f"\n🚀 {description}")
    print("-" * 40)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            capture_output=True, 
            text=True,
            timeout=timeout
        )
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"✅ 완료! (소요 시간: {elapsed_time:.2f}초)")
        return True
        
    except subprocess.TimeoutExpired:
        print(f"⏰ 시간 초과 ({timeout}초)")
        return False
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"❌ 실패! (소요 시간: {elapsed_time:.2f}초)")
        if e.stderr:
            print(f"오류: {e.stderr}")
        return False

def check_quick_requirements():
    """빠른 시작을 위한 최소 요구사항 확인"""
    print("🔍 빠른 시작 요구사항 확인 중...")
    
    # 필수 파일 확인
    required_files = ["IMDB Dataset.csv", "netflix_titles.csv"]
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ 다음 파일들이 필요합니다:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    
    # 디렉토리 생성
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/models", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    
    print("✅ 요구사항 확인 완료!")
    return True

def create_quick_preprocessor():
    """빠른 전처리 스크립트 생성"""
    quick_preprocessor = """
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib

# NLTK 데이터 다운로드
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

print("빠른 데이터 전처리 시작...")

# IMDB 데이터 로드 (샘플만)
df = pd.read_csv("IMDB Dataset.csv")
print(f"원본 데이터: {len(df)}개")

# 빠른 처리를 위해 샘플만 사용
df_sample = df.sample(n=10000, random_state=42)
print(f"샘플 데이터: {len(df_sample)}개")

# 간단한 텍스트 정제
def simple_clean(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\\s]', '', text)
    return text.lower().strip()

df_sample['cleaned_review'] = df_sample['review'].apply(simple_clean)
df_sample['sentiment_label'] = df_sample['sentiment'].map({'positive': 1, 'negative': 0})

# 데이터 분할
X = df_sample['cleaned_review']
y = df_sample['sentiment_label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF (간단 버전)
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# 저장
joblib.dump(X_train_tfidf, "data/X_train_tfidf_quick.pkl")
joblib.dump(X_test_tfidf, "data/X_test_tfidf_quick.pkl")
joblib.dump(y_train, "data/y_train_quick.pkl")
joblib.dump(y_test, "data/y_test_quick.pkl")
joblib.dump(tfidf, "data/tfidf_vectorizer_quick.pkl")

# Netflix 데이터 (간단 처리)
netflix_df = pd.read_csv("netflix_titles.csv")
movies_df = netflix_df[netflix_df['type'] == 'Movie'].copy()
movies_df['description'] = movies_df['description'].fillna('')
movies_df.to_csv("data/netflix_movies_quick.csv", index=False)

print("✅ 빠른 전처리 완료!")
"""
    
    with open("quick_preprocess.py", "w", encoding="utf-8") as f:
        f.write(quick_preprocessor)

def create_quick_model():
    """빠른 모델 학습 스크립트 생성"""
    quick_model = """
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

print("빠른 모델 학습 시작...")

# 데이터 로드
X_train = joblib.load("data/X_train_tfidf_quick.pkl")
X_test = joblib.load("data/X_test_tfidf_quick.pkl")
y_train = joblib.load("data/y_train_quick.pkl")
y_test = joblib.load("data/y_test_quick.pkl")

# 간단한 로지스틱 회귀 모델
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# 예측 및 평가
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"모델 정확도: {accuracy:.4f}")
print("\\n분류 리포트:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# 모델 저장
joblib.dump(model, "data/models/best_model.pkl")
print("✅ 빠른 모델 학습 완료!")
"""
    
    with open("quick_model.py", "w", encoding="utf-8") as f:
        f.write(quick_model)

def create_quick_recommendation():
    """빠른 추천 시스템 스크립트 생성"""
    quick_rec = """
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

print("빠른 추천 시스템 구축 시작...")

# 영화 데이터 로드
movies_df = pd.read_csv("data/netflix_movies_quick.csv")
print(f"영화 수: {len(movies_df)}")

# 간단한 콘텐츠 기반 추천 시스템
tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_df['description'])

# 유사도 계산 (샘플만)
sample_size = min(1000, len(movies_df))
sample_indices = np.random.choice(len(movies_df), sample_size, replace=False)
sample_matrix = tfidf_matrix[sample_indices]
similarity_matrix = cosine_similarity(sample_matrix)

# 저장
np.save("data/models/content_similarity_matrix.npy", similarity_matrix)
movies_df.iloc[sample_indices].to_csv("data/models/processed_movies.csv", index=False)

print("✅ 빠른 추천 시스템 완료!")
"""
    
    with open("quick_recommendation.py", "w", encoding="utf-8") as f:
        f.write(quick_rec)

def create_quick_streamlit():
    """빠른 Streamlit 앱 생성"""
    quick_app = """
import streamlit as st
import pandas as pd
import joblib
import re

st.set_page_config(page_title="영화 감성 분석 (빠른 버전)", page_icon="🎬")

st.title("🎬 영화 감성 분석 (빠른 버전)")
st.markdown("간단한 영화 리뷰 감성 분석을 체험해보세요!")

# 모델 로드
@st.cache_resource
def load_model():
    try:
        model = joblib.load("data/models/best_model.pkl")
        vectorizer = joblib.load("data/tfidf_vectorizer_quick.pkl")
        return model, vectorizer
    except:
        return None, None

model, vectorizer = load_model()

if model is None:
    st.error("모델을 로드할 수 없습니다. quick_start.py를 먼저 실행해주세요.")
else:
    # 감성 분석
    st.subheader("💬 리뷰 감성 분석")
    
    user_input = st.text_area("영화 리뷰를 입력하세요:", height=100)
    
    if st.button("분석하기") and user_input:
        # 간단한 전처리
        cleaned = re.sub(r'<.*?>', '', user_input)
        cleaned = re.sub(r'[^a-zA-Z\\s]', '', cleaned).lower().strip()
        
        # 예측
        X = vectorizer.transform([cleaned])
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0]
        
        # 결과 표시
        if prediction == 1:
            st.success(f"😊 긍정적 (신뢰도: {probability[1]:.1%})")
        else:
            st.error(f"😞 부정적 (신뢰도: {probability[0]:.1%})")
    
    # 간단한 통계
    st.subheader("📊 시스템 정보")
    st.info("이것은 빠른 테스트용 버전입니다. 전체 기능을 사용하려면 run_all.py를 실행하세요.")
"""
    
    with open("quick_app.py", "w", encoding="utf-8") as f:
        f.write(quick_app)

def cleanup_quick_files():
    """임시 파일들 정리"""
    quick_files = [
        "quick_preprocess.py",
        "quick_model.py", 
        "quick_recommendation.py",
        "quick_app.py"
    ]
    
    for file in quick_files:
        if os.path.exists(file):
            os.remove(file)

def main():
    """메인 함수"""
    print("🎬 영화 감성 분석 & 추천 시스템 - 빠른 시작")
    print("=" * 60)
    print("⚡ 빠른 테스트를 위해 샘플 데이터와 간단한 모델을 사용합니다.")
    print("📊 전체 기능을 사용하려면 run_all.py를 실행하세요.")
    
    # 요구사항 확인
    if not check_quick_requirements():
        return
    
    try:
        # 빠른 스크립트들 생성
        print("\n📝 빠른 실행 스크립트 생성 중...")
        create_quick_preprocessor()
        create_quick_model()
        create_quick_recommendation()
        create_quick_streamlit()
        
        # 단계별 실행
        steps = [
            ("python quick_preprocess.py", "빠른 데이터 전처리", 300),
            ("python quick_model.py", "빠른 모델 학습", 120),
            ("python quick_recommendation.py", "빠른 추천 시스템", 60)
        ]
        
        all_success = True
        for command, description, timeout in steps:
            success = run_command(command, description, timeout)
            if not success:
                all_success = False
                break
        
        if all_success:
            print("\n" + "=" * 60)
            print("✅ 빠른 시작 완료!")
            print("=" * 60)
            
            # 웹 앱 실행 제안
            response = input("\n웹 애플리케이션을 실행하시겠습니까? (y/N): ").lower()
            if response in ['y', 'yes']:
                print("\n🌐 웹 애플리케이션 실행 중...")
                print("브라우저에서 http://localhost:8501 로 접속하세요.")
                
                try:
                    subprocess.run("streamlit run quick_app.py", shell=True)
                except KeyboardInterrupt:
                    print("\n✅ 애플리케이션이 종료되었습니다.")
            
            print("\n📋 다음 단계:")
            print("  - 전체 기능: python run_all.py")
            print("  - 웹 대시보드: streamlit run src/streamlit_app/app.py")
        
        else:
            print("\n❌ 빠른 시작 중 오류가 발생했습니다.")
            print("setup.py를 먼저 실행하거나 run_all.py를 시도해보세요.")
    
    finally:
        # 임시 파일 정리
        cleanup_quick_files()

if __name__ == "__main__":
    main()





