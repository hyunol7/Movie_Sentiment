"""
데이터 시각화 및 분석 유틸리티
- 데이터 탐색적 분석 (EDA)
- 모델 성능 시각화
- WordCloud 생성
- 감성 분석 결과 시각화
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 없이 사용
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from collections import Counter
import joblib
import os

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class MovieDataVisualizer:
    def __init__(self, style='seaborn-v0_8'):
        plt.style.use('default')  # seaborn 스타일이 없을 경우 기본 스타일 사용
        sns.set_palette("husl")
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
    def load_data(self):
        """
        분석용 데이터 로드
        """
        try:
            # IMDB 데이터
            self.imdb_df = pd.read_csv("data/imdb_preprocessed.csv")
            print(f"IMDB 데이터 로드: {len(self.imdb_df)}개")
            
            # Netflix 데이터
            self.netflix_df = pd.read_csv("data/netflix_preprocessed.csv")
            print(f"Netflix 데이터 로드: {len(self.netflix_df)}개")
            
            # 영화만 필터링
            self.movies_df = self.netflix_df[self.netflix_df['type'] == 'Movie'].copy()
            print(f"Netflix 영화 데이터: {len(self.movies_df)}개")
            
            return True
            
        except FileNotFoundError as e:
            print(f"데이터 파일을 찾을 수 없습니다: {e}")
            return False
    
    def create_sentiment_distribution(self, save_path="static/sentiment_distribution.png"):
        """
        감성 분포 시각화
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 감성 분포 (카운트)
        sentiment_counts = self.imdb_df['sentiment'].value_counts()
        
        # 파이 차트
        axes[0].pie(sentiment_counts.values, labels=sentiment_counts.index, 
                   autopct='%1.1f%%', startangle=90, colors=['#ff7f7f', '#7fbf7f'])
        axes[0].set_title('감성 분포 (Sentiment Distribution)', fontsize=14, fontweight='bold')
        
        # 막대 그래프
        bars = axes[1].bar(sentiment_counts.index, sentiment_counts.values, 
                          color=['#ff7f7f', '#7fbf7f'])
        axes[1].set_title('감성별 리뷰 수', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('리뷰 수')
        axes[1].set_xlabel('감성')
        
        # 값 표시
        for bar in bars:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + 100,
                        f'{int(height):,}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # 저장
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"감성 분포 그래프 저장: {save_path}")
    
    def create_wordcloud(self, sentiment='positive', save_path=None):
        """
        WordCloud 생성
        """
        # 감성별 텍스트 결합
        if sentiment == 'positive':
            text_data = ' '.join(self.imdb_df[self.imdb_df['sentiment'] == 'positive']['processed_review'])
            title = '긍정 리뷰 WordCloud'
            colormap = 'Greens'
        else:
            text_data = ' '.join(self.imdb_df[self.imdb_df['sentiment'] == 'negative']['processed_review'])
            title = '부정 리뷰 WordCloud'
            colormap = 'Reds'
        
        # WordCloud 생성
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            colormap=colormap,
            max_words=100,
            relative_scaling=0.5,
            random_state=42
        ).generate(text_data)
        
        # 시각화
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(title, fontsize=16, fontweight='bold')
        plt.axis('off')
        
        # 저장
        if save_path is None:
            save_path = f"static/wordcloud_{sentiment}.png"
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"{sentiment} WordCloud 저장: {save_path}")
    
    def create_netflix_analysis(self, save_path="static/netflix_analysis.png"):
        """
        Netflix 데이터 분석 시각화
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Netflix 영화 데이터 분석', fontsize=16, fontweight='bold')
        
        # 1. 연도별 영화 수
        year_counts = self.movies_df['release_year'].value_counts().sort_index()
        recent_years = year_counts[year_counts.index >= 2000]
        
        axes[0, 0].plot(recent_years.index, recent_years.values, linewidth=2, marker='o')
        axes[0, 0].set_title('연도별 영화 수 (2000년 이후)', fontweight='bold')
        axes[0, 0].set_xlabel('연도')
        axes[0, 0].set_ylabel('영화 수')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 상위 10개 장르
        all_genres = []
        for genres in self.movies_df['listed_in'].dropna():
            all_genres.extend([genre.strip() for genre in str(genres).split(',')])
        
        genre_counts = Counter(all_genres)
        top_genres = dict(genre_counts.most_common(10))
        
        bars = axes[0, 1].barh(list(top_genres.keys()), list(top_genres.values()), color='skyblue')
        axes[0, 1].set_title('상위 10개 장르', fontweight='bold')
        axes[0, 1].set_xlabel('영화 수')
        
        # 값 표시
        for i, bar in enumerate(bars):
            width = bar.get_width()
            axes[0, 1].text(width + 5, bar.get_y() + bar.get_height()/2,
                           f'{int(width)}', ha='left', va='center')
        
        # 3. 상위 감독
        director_counts = self.movies_df['director'].value_counts().head(10)
        director_counts = director_counts[director_counts.index != 'Unknown']
        
        axes[1, 0].barh(director_counts.index, director_counts.values, color='lightgreen')
        axes[1, 0].set_title('상위 10명 감독', fontweight='bold')
        axes[1, 0].set_xlabel('영화 수')
        
        # 4. 국가별 영화 수
        country_counts = self.movies_df['country'].value_counts().head(10)
        country_counts = country_counts[country_counts.index != 'Unknown']
        
        axes[1, 1].pie(country_counts.values, labels=country_counts.index, 
                      autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('상위 10개 국가별 영화 비율', fontweight='bold')
        
        plt.tight_layout()
        
        # 저장
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Netflix 분석 그래프 저장: {save_path}")
    
    def create_text_length_analysis(self, save_path="static/text_length_analysis.png"):
        """
        텍스트 길이 분석
        """
        # 리뷰 길이 계산
        self.imdb_df['review_length'] = self.imdb_df['review'].str.len()
        self.imdb_df['processed_length'] = self.imdb_df['processed_review'].str.len()
        self.imdb_df['word_count'] = self.imdb_df['processed_review'].str.split().str.len()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('텍스트 길이 분석', fontsize=16, fontweight='bold')
        
        # 1. 감성별 리뷰 길이 분포
        for sentiment in ['positive', 'negative']:
            data = self.imdb_df[self.imdb_df['sentiment'] == sentiment]['review_length']
            axes[0, 0].hist(data, bins=50, alpha=0.7, label=sentiment, density=True)
        
        axes[0, 0].set_title('감성별 원본 리뷰 길이 분포', fontweight='bold')
        axes[0, 0].set_xlabel('리뷰 길이 (문자 수)')
        axes[0, 0].set_ylabel('밀도')
        axes[0, 0].legend()
        axes[0, 0].set_xlim(0, 5000)
        
        # 2. 감성별 단어 수 분포
        for sentiment in ['positive', 'negative']:
            data = self.imdb_df[self.imdb_df['sentiment'] == sentiment]['word_count']
            axes[0, 1].hist(data, bins=50, alpha=0.7, label=sentiment, density=True)
        
        axes[0, 1].set_title('감성별 단어 수 분포', fontweight='bold')
        axes[0, 1].set_xlabel('단어 수')
        axes[0, 1].set_ylabel('밀도')
        axes[0, 1].legend()
        axes[0, 1].set_xlim(0, 500)
        
        # 3. 박스플롯 - 감성별 리뷰 길이
        sentiment_data = [
            self.imdb_df[self.imdb_df['sentiment'] == 'positive']['review_length'],
            self.imdb_df[self.imdb_df['sentiment'] == 'negative']['review_length']
        ]
        
        axes[1, 0].boxplot(sentiment_data, labels=['Positive', 'Negative'])
        axes[1, 0].set_title('감성별 리뷰 길이 박스플롯', fontweight='bold')
        axes[1, 0].set_ylabel('리뷰 길이 (문자 수)')
        axes[1, 0].set_ylim(0, 5000)
        
        # 4. 산점도 - 리뷰 길이 vs 단어 수
        sample_df = self.imdb_df.sample(1000, random_state=42)
        
        for sentiment, color in zip(['positive', 'negative'], ['green', 'red']):
            data = sample_df[sample_df['sentiment'] == sentiment]
            axes[1, 1].scatter(data['review_length'], data['word_count'], 
                             alpha=0.6, label=sentiment, color=color)
        
        axes[1, 1].set_title('리뷰 길이 vs 단어 수', fontweight='bold')
        axes[1, 1].set_xlabel('리뷰 길이 (문자 수)')
        axes[1, 1].set_ylabel('단어 수')
        axes[1, 1].legend()
        axes[1, 1].set_xlim(0, 5000)
        axes[1, 1].set_ylim(0, 500)
        
        plt.tight_layout()
        
        # 저장
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"텍스트 길이 분석 그래프 저장: {save_path}")
    
    def create_interactive_dashboard(self, save_path="static/interactive_dashboard.html"):
        """
        Plotly를 사용한 인터랙티브 대시보드
        """
        # 서브플롯 생성
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('감성 분포', '연도별 Netflix 영화 수', '상위 장르', '국가별 영화 분포'),
            specs=[[{"type": "pie"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # 1. 감성 분포 파이 차트
        sentiment_counts = self.imdb_df['sentiment'].value_counts()
        fig.add_trace(
            go.Pie(labels=sentiment_counts.index, values=sentiment_counts.values,
                   name="Sentiment", marker_colors=['lightcoral', 'lightgreen']),
            row=1, col=1
        )
        
        # 2. 연도별 영화 수
        year_counts = self.movies_df['release_year'].value_counts().sort_index()
        recent_years = year_counts[year_counts.index >= 2000]
        
        fig.add_trace(
            go.Scatter(x=recent_years.index, y=recent_years.values,
                      mode='lines+markers', name='연도별 영화 수',
                      line=dict(color='blue', width=2)),
            row=1, col=2
        )
        
        # 3. 상위 장르
        all_genres = []
        for genres in self.movies_df['listed_in'].dropna():
            all_genres.extend([genre.strip() for genre in str(genres).split(',')])
        
        genre_counts = Counter(all_genres)
        top_genres = dict(genre_counts.most_common(10))
        
        fig.add_trace(
            go.Bar(x=list(top_genres.values()), y=list(top_genres.keys()),
                   orientation='h', name='장르별 영화 수',
                   marker_color='skyblue'),
            row=2, col=1
        )
        
        # 4. 국가별 영화 수
        country_counts = self.movies_df['country'].value_counts().head(10)
        country_counts = country_counts[country_counts.index != 'Unknown']
        
        fig.add_trace(
            go.Bar(x=country_counts.index, y=country_counts.values,
                   name='국가별 영화 수', marker_color='lightgreen'),
            row=2, col=2
        )
        
        # 레이아웃 업데이트
        fig.update_layout(
            title_text="영화 데이터 인터랙티브 대시보드",
            title_x=0.5,
            showlegend=False,
            height=800
        )
        
        # 저장
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_html(save_path)
        
        print(f"인터랙티브 대시보드 저장: {save_path}")
        
        return fig
    
    def create_model_performance_summary(self, save_path="static/model_summary.png"):
        """
        모델 성능 종합 요약
        """
        try:
            # 모델 결과 로드
            ml_results = joblib.load("data/models/training_results.pkl")
            
            # 성능 데이터 준비
            ml_data = []
            for name, result in ml_results.items():
                metrics = result['metrics']
                ml_data.append({
                    'Model': name,
                    'Type': 'Machine Learning',
                    'Accuracy': metrics['accuracy'],
                    'F1-Score': metrics['f1_score'],
                    'ROC-AUC': metrics['roc_auc']
                })
            
            ml_df = pd.DataFrame(ml_data)
            
            # 시각화
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle('모델 성능 종합 비교', fontsize=16, fontweight='bold')
            
            metrics = ['Accuracy', 'F1-Score', 'ROC-AUC']
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
            
            for i, metric in enumerate(metrics):
                bars = axes[i].bar(ml_df['Model'], ml_df[metric], color=colors[i], alpha=0.7)
                axes[i].set_title(f'{metric} 비교')
                axes[i].set_ylabel(metric)
                axes[i].set_ylim(0, 1)
                axes[i].tick_params(axis='x', rotation=45)
                
                # 값 표시
                for bar in bars:
                    height = bar.get_height()
                    axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            
            # 저장
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"모델 성능 요약 그래프 저장: {save_path}")
            
        except FileNotFoundError:
            print("모델 결과 파일을 찾을 수 없습니다.")
    
    def generate_comprehensive_report(self):
        """
        종합 분석 리포트 생성
        """
        print("종합 시각화 리포트 생성 중...")
        print("=" * 50)
        
        # 1. 감성 분포
        self.create_sentiment_distribution()
        
        # 2. WordCloud 생성
        self.create_wordcloud('positive')
        self.create_wordcloud('negative')
        
        # 3. Netflix 분석
        self.create_netflix_analysis()
        
        # 4. 텍스트 길이 분석
        self.create_text_length_analysis()
        
        # 5. 인터랙티브 대시보드
        self.create_interactive_dashboard()
        
        # 6. 모델 성능 요약
        self.create_model_performance_summary()
        
        print("\n" + "=" * 50)
        print("종합 시각화 리포트 생성 완료!")
        print("=" * 50)
        print("생성된 파일들:")
        print("- static/sentiment_distribution.png")
        print("- static/wordcloud_positive.png")
        print("- static/wordcloud_negative.png")
        print("- static/netflix_analysis.png")
        print("- static/text_length_analysis.png")
        print("- static/interactive_dashboard.html")
        print("- static/model_summary.png")

def main():
    """
    메인 실행 함수
    """
    print("데이터 시각화 및 분석 시작")
    print("=" * 50)
    
    # 시각화 객체 생성
    visualizer = MovieDataVisualizer()
    
    # 데이터 로드
    if not visualizer.load_data():
        return
    
    # 종합 리포트 생성
    visualizer.generate_comprehensive_report()

if __name__ == "__main__":
    main()

