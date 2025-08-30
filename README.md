# 🎬 영화 리뷰 감성 분석 & 추천 시스템

영화 리뷰에 대한 감성 분석과 개인화된 영화 추천 시스템을 구현한 머신러닝/딥러닝 프로젝트입니다.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25.0-red)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13.0-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-green)

## 📋 목차

- [프로젝트 개요](#-프로젝트-개요)
- [주요 기능](#-주요-기능)
- [데이터셋](#-데이터셋)
- [모델 아키텍처](#-모델-아키텍처)
- [설치 및 실행](#-설치-및-실행)
- [프로젝트 구조](#-프로젝트-구조)
- [사용법](#-사용법)
- [결과 예시](#-결과-예시)
- [성능 지표](#-성능-지표)
- [기술 스택](#-기술-스택)
- [라이선스](#-라이선스)

## 프로젝트 개요

이 프로젝트는 영화 리뷰 텍스트의 감성을 분석하고, 사용자의 선호도에 따라 개인화된 영화를 추천하는 시스템입니다. IMDB 리뷰 데이터를 활용하여 다양한 머신러닝과 딥러닝 모델을 비교하고, Netflix 메타데이터를 결합하여 종합적인 추천 서비스를 제공합니다.

### 프로젝트 특징

- **다중 모델 비교**: 7개의 서로 다른 ML/DL 모델 성능 비교
- **실시간 감성 분석**: 사용자 입력 리뷰에 대한 즉시 감성 분석
- **다양한 추천 방식**: 내용, 감성, 장르 기반 추천 시스템
- **인터랙티브 웹 인터페이스**: Streamlit 기반 사용자 친화적 대시보드
- **종합적 시각화**: 데이터 탐색 및 모델 성능 시각화

## 주요 기능

### 1. 감성 분석 (Sentiment Analysis)
- **실시간 분석**: 사용자가 입력한 리뷰의 긍정/부정 감성 분류
- **신뢰도 표시**: 예측 확률과 신뢰도 시각화
- **텍스트 전처리**: 자동 텍스트 정제 및 토큰화

### 2. 영화 추천 시스템 (Movie Recommendation)
- **내용 기반 필터링**: 영화 메타데이터 유사도 기반 추천
- **감성 기반 추천**: 사용자 선호 감성에 맞는 영화 추천
- **장르 기반 추천**: 선호 장르 기반 맞춤 추천
- **하이브리드 추천**: 여러 추천 방식을 결합한 종합 추천

### 3. 데이터 시각화
- **탐색적 데이터 분석**: 리뷰 데이터 및 영화 메타데이터 분석
- **WordCloud**: 감성별 주요 키워드 시각화
- **인터랙티브 차트**: Plotly 기반 동적 시각화
- **모델 성능 비교**: 다양한 모델의 성능 지표 비교

### 4. 웹 대시보드
- **직관적 인터페이스**: 사용자 친화적 Streamlit 웹 인터페이스
- **실시간 결과**: 즉시 확인 가능한 분석 및 추천 결과
- **종합 리포트**: 프로젝트 전반에 대한 상세 정보

## 데이터셋

### IMDB Movie Reviews Dataset
- **크기**: 50,000개 영화 리뷰
- **라벨**: 긍정(25,000) / 부정(25,000) 균등 분포
- **용도**: 감성 분석 모델 학습 및 평가
- **특징**: 다양한 길이와 스타일의 실제 사용자 리뷰

### Netflix Movies and TV Shows Dataset
- **크기**: 8,000+ 작품 메타데이터
- **포함 정보**: 
  - 제목, 감독, 주연 배우
  - 장르, 출시 년도, 국가
  - 줄거리 설명, 등급
- **용도**: 추천 시스템 구축 및 콘텐츠 기반 필터링

## 모델 아키텍처

### 머신러닝 모델
1. **Logistic Regression**: 선형 분류를 위한 기본 모델
2. **Naive Bayes**: 텍스트 분류에 효과적인 확률 기반 모델
3. **Support Vector Machine (SVM)**: 고차원 텍스트 데이터에 강력한 성능
4. **Random Forest**: 앙상블 기법을 활용한 robust한 분류기

### 딥러닝 모델
1. **LSTM (Long Short-Term Memory)**: 
   - 순차적 텍스트 패턴 학습
   - Bidirectional 구조로 성능 향상
   
2. **CNN (Convolutional Neural Network)**:
   - 1D Convolution으로 지역적 텍스트 패턴 추출
   - Multiple kernel sizes로 다양한 n-gram 패턴 학습
   
3. **Advanced CNN**:
   - 다중 커널 크기 (3, 4, 5)
   - Global Max Pooling으로 중요 특성 추출
   
4. **LSTM-CNN Hybrid**:
   - LSTM과 CNN의 장점 결합
   - 순차적 정보와 지역적 패턴 동시 학습

### 추천 시스템 아키텍처
```
사용자 입력
    ↓
감성 분석 모델 → 감성 점수
    ↓
콘텐츠 유사도 계산 → TF-IDF + Cosine Similarity
    ↓
하이브리드 점수 = α×감성점수 + β×콘텐츠유사도 + γ×장르매칭
    ↓
Top-N 추천 결과
```

## 설치 및 실행

### 1. 환경 설정

```bash
# 레포지토리 클론
git clone https://github.com/your-username/movie-sentiment-analysis.git
cd movie-sentiment-analysis

# 가상환경 생성 (권장)
python -m venv movie_env
source movie_env/bin/activate  # Linux/Mac
# 또는
movie_env\Scripts\activate     # Windows

# 패키지 설치
pip install -r requirements.txt
```

### 2. 데이터

- `IMDB Dataset.csv`: IMDB 리뷰 데이터
- `netflix_titles.csv`: Netflix 메타데이터

### 3. 단계별 실행

```bash
# 1단계: 데이터 전처리
python src/preprocessing/data_preprocessor.py

# 2단계: 머신러닝 모델 학습
python src/models/ml_models.py

# 3단계: 딥러닝 모델 학습 (선택사항)
python src/models/dl_models.py

# 4단계: 추천 시스템 구축
python src/models/recommendation_system.py

# 5단계: 시각화 생성
python src/utils/visualization.py

# 6단계: 웹 애플리케이션 실행
streamlit run src/streamlit_app/app.py
```

### 4. 웹 애플리케이션 접속

브라우저에서 `http://localhost:8501`로 접속하여 대시보드를 사용할 수 있습니다.


## 사용법

### 1. 홈 페이지
- 프로젝트 개요 및 주요 지표 확인
- 각 섹션으로의 빠른 네비게이션

### 2. 데이터 탐색
- **IMDB 탭**: 리뷰 데이터 분포 및 샘플 확인
- **Netflix 탭**: 영화 메타데이터 분석
- **통계 탭**: 종합 통계 정보

### 3. 감성 분석
- 텍스트 입력창에 영화 리뷰 입력
- 실시간 감성 분석 결과 및 신뢰도 확인
- 예시 리뷰로 빠른 테스트 가능

### 4. 영화 추천
- **내용 기반**: 좋아하는 영화명 입력으로 유사 영화 찾기
- **감성 기반**: 선호 감성(긍정/부정)과 장르 선택
- **장르 기반**: 여러 장르 조합으로 맞춤 추천
- **하이브리드**: 다양한 요소를 결합한 종합 추천

### 5. 모델 성능
- 각 모델의 성능 지표 비교
- 인터랙티브 차트를 통한 상세 분석
- 최고 성능 모델 확인

## 결과 예시

### 감성 분석 결과
```
입력: "This movie was absolutely fantastic! Amazing acting and great story."
결과: 긍정적 😊 (신뢰도: 89.3%)
```

### 추천 결과 예시
```
입력 영화: "The Crown"
추천 영화:
1. "Downton Abbey" (유사도: 0.847)
2. "Bridgerton" (유사도: 0.792)
3. "The Queen's Gambit" (유사도: 0.756)
...
```

## 성능 지표

### 머신러닝 모델 성능 (예시)
| 모델 | 정확도 | 정밀도 | 재현율 | F1-Score | ROC-AUC |
|------|--------|--------|--------|----------|---------|
| SVM | 0.891 | 0.889 | 0.894 | 0.891 | 0.891 |
| Logistic Regression | 0.884 | 0.882 | 0.887 | 0.884 | 0.884 |
| Random Forest | 0.876 | 0.874 | 0.879 | 0.876 | 0.876 |
| Naive Bayes | 0.859 | 0.857 | 0.862 | 0.859 | 0.859 |

### 딥러닝 모델 성능 (예시)
| 모델 | 테스트 정확도 | 테스트 손실 |
|------|---------------|-------------|
| LSTM-CNN Hybrid | 0.912 | 0.201 |
| Advanced CNN | 0.908 | 0.215 |
| LSTM | 0.895 | 0.234 |
| CNN | 0.887 | 0.256 |

## 기술 스택

### 핵심 기술
- **언어**: Python 3.8+
- **웹 프레임워크**: Streamlit
- **머신러닝**: scikit-learn
- **딥러닝**: TensorFlow/Keras
- **자연어 처리**: NLTK, TF-IDF

### 데이터 처리
- **데이터 조작**: Pandas, NumPy
- **시각화**: Matplotlib, Seaborn, Plotly
- **유틸리티**: tqdm, joblib

### 개발 도구
- **노트북**: Jupyter
- **패키지 관리**: pip
- **버전 관리**: Git

## 향후 개선 계획

- BERT 모델 추가 구현
- 사용자 피드백 시스템
- API 엔드포인트 제공
- 실시간 리뷰 크롤링






