"""
간단한 딥러닝 모델 (TensorFlow 없이)
sklearn의 MLPClassifier 사용
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 없이 사용
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os
from tqdm import tqdm

class SimpleDeepLearningModels:
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def initialize_models(self):
        """간단한 신경망 모델들 초기화"""
        self.models = {
            "Small MLP": MLPClassifier(
                hidden_layer_sizes=(128,),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                batch_size='auto',
                learning_rate='constant',
                learning_rate_init=0.001,
                max_iter=200,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10
            ),
            "Medium MLP": MLPClassifier(
                hidden_layer_sizes=(256, 128),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                batch_size='auto',
                learning_rate='constant',
                learning_rate_init=0.001,
                max_iter=200,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10
            ),
            "Large MLP": MLPClassifier(
                hidden_layer_sizes=(512, 256, 128),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                batch_size='auto',
                learning_rate='constant',
                learning_rate_init=0.001,
                max_iter=200,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10
            )
        }
        
        print("간단한 딥러닝 모델 초기화 완료:")
        for name in self.models.keys():
            print(f"- {name}")
    
    def train_models(self, X_train, y_train, X_test, y_test):
        """모델 학습 및 평가"""
        print("\n" + "=" * 50)
        print("간단한 딥러닝 모델 학습 시작")
        print("=" * 50)
        
        for name, model in self.models.items():
            print(f"\n{name} 학습 중...")
            
            # 모델 학습
            model.fit(X_train, y_train)
            
            # 예측
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # 평가 지표 계산
            accuracy = accuracy_score(y_test, y_pred)
            
            # 결과 저장
            self.results[name] = {
                'model': model,
                'accuracy': accuracy,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"{name} 완료 - 정확도: {accuracy:.4f}")
            
            # 분류 리포트
            print(f"\n{name} 분류 리포트:")
            print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    
    def compare_models(self):
        """모델 성능 비교"""
        if not self.results:
            print("학습된 모델이 없습니다.")
            return
        
        comparison_data = []
        for name, result in self.results.items():
            comparison_data.append({
                'Model': name,
                'Accuracy': result['accuracy'],
                'Hidden Layers': str(result['model'].hidden_layer_sizes),
                'Iterations': result['model'].n_iter_
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        
        print("\n" + "=" * 70)
        print("간단한 딥러닝 모델 성능 비교")
        print("=" * 70)
        print(comparison_df.round(4).to_string(index=False))
        
        best_model_name = comparison_df.iloc[0]['Model']
        print(f"\n최고 성능 모델: {best_model_name}")
        print(f"정확도: {comparison_df.iloc[0]['Accuracy']:.4f}")
        
        return comparison_df
    
    def plot_model_comparison(self, comparison_df, save_path="static/simple_dl_comparison.png"):
        """모델 성능 비교 시각화"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 정확도 비교
        bars = ax1.bar(comparison_df['Model'], comparison_df['Accuracy'], 
                      color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax1.set_title('간단한 딥러닝 모델 정확도 비교', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        
        # 값 표시
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 반복 횟수 비교
        bars2 = ax2.bar(comparison_df['Model'], comparison_df['Iterations'], 
                       color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax2.set_title('학습 반복 횟수', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Iterations')
        ax2.tick_params(axis='x', rotation=45)
        
        # 값 표시
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # 저장
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"간단한 딥러닝 모델 비교 그래프 저장: {save_path}")
    
    def plot_confusion_matrices(self, y_test, save_path="static/simple_dl_confusion.png"):
        """혼동 행렬 시각화"""
        n_models = len(self.results)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        fig.suptitle('간단한 딥러닝 모델 혼동 행렬', fontsize=16, fontweight='bold')
        
        for i, (name, result) in enumerate(self.results.items()):
            ax = axes[i]
            cm = confusion_matrix(y_test, result['y_pred'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Negative', 'Positive'],
                       yticklabels=['Negative', 'Positive'])
            ax.set_title(name)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        plt.tight_layout()
        
        # 저장
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"혼동 행렬 그래프 저장: {save_path}")
    
    def save_models(self, save_dir="data/models"):
        """학습된 모델 저장"""
        os.makedirs(save_dir, exist_ok=True)
        
        for name, result in self.results.items():
            model_path = os.path.join(save_dir, f"simple_{name.replace(' ', '_').lower()}_model.pkl")
            joblib.dump(result['model'], model_path)
            print(f"{name} 모델 저장: {model_path}")
        
        # 최고 성능 모델도 저장
        best_model = max(self.results.items(), key=lambda x: x[1]['accuracy'])
        best_model_path = os.path.join(save_dir, "simple_dl_best_model.pkl")
        joblib.dump(best_model[1]['model'], best_model_path)
        print(f"최고 성능 간단한 딥러닝 모델 저장: {best_model_path}")

def main():
    """메인 실행 함수"""
    print("간단한 딥러닝 모델 학습 시작")
    print("=" * 50)
    
    # 데이터 로드
    try:
        X_train = joblib.load("data/X_train_tfidf.pkl")
        X_test = joblib.load("data/X_test_tfidf.pkl")
        y_train = joblib.load("data/y_train.pkl")
        y_test = joblib.load("data/y_test.pkl")
        
        print(f"학습 데이터: {X_train.shape}")
        print(f"테스트 데이터: {X_test.shape}")
        
    except FileNotFoundError:
        print("전처리된 데이터를 찾을 수 없습니다.")
        print("먼저 data_preprocessor.py를 실행해주세요.")
        return
    
    # 간단한 딥러닝 모델 트레이너 초기화
    trainer = SimpleDeepLearningModels()
    trainer.initialize_models()
    
    # 모델 학습
    trainer.train_models(X_train, y_train, X_test, y_test)
    
    # 모델 비교
    comparison_df = trainer.compare_models()
    
    # 시각화
    trainer.plot_model_comparison(comparison_df)
    trainer.plot_confusion_matrices(y_test)
    
    # 모델 저장
    trainer.save_models()
    
    print("\n" + "=" * 50)
    print("간단한 딥러닝 모델 학습 완료!")
    print("=" * 50)

if __name__ == "__main__":
    main()
