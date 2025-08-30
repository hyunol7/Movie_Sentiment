"""
딥러닝 모델 구현
- LSTM (Long Short-Term Memory)
- CNN (Convolutional Neural Network) for text
- BERT (Bidirectional Encoder Representations from Transformers)
"""


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, LSTM, Embedding, Dropout, Conv1D, GlobalMaxPooling1D,
    Input, Bidirectional, Attention, GlobalAveragePooling1D
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # GUI 없이 사용
import matplotlib.pyplot as plt
import seaborn as sns
# from transformers import AutoTokenizer, TFAutoModel  # 선택적 import
import joblib
import os
from tqdm import tqdm

class DeepLearningModels:
    def __init__(self, max_features=10000, max_length=200):
        self.max_features = max_features
        self.max_length = max_length
        self.tokenizer = None
        self.models = {}
        self.histories = {}
        
    def prepare_data(self, texts, labels=None, fit_tokenizer=True):
        """
        텍스트 데이터를 딥러닝 모델용으로 준비
        """
        if fit_tokenizer:
            self.tokenizer = Tokenizer(num_words=self.max_features, oov_token="<OOV>")
            self.tokenizer.fit_on_texts(texts)
        
        # 텍스트를 시퀀스로 변환
        sequences = self.tokenizer.texts_to_sequences(texts)
        
        # 패딩
        X = pad_sequences(sequences, maxlen=self.max_length, padding='post', truncating='post')
        
        if labels is not None:
            y = np.array(labels)
            return X, y
        
        return X
    
    def build_lstm_model(self, embedding_dim=128, lstm_units=64):
        """
        LSTM 모델 구축
        """
        model = Sequential([
            Embedding(self.max_features, embedding_dim, input_length=self.max_length),
            Bidirectional(LSTM(lstm_units, dropout=0.3, recurrent_dropout=0.3)),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_cnn_model(self, embedding_dim=128, filters=64, kernel_size=5):
        """
        CNN 모델 구축
        """
        model = Sequential([
            Embedding(self.max_features, embedding_dim, input_length=self.max_length),
            
            # 첫 번째 CNN 블록
            Conv1D(filters, kernel_size, activation='relu'),
            GlobalMaxPooling1D(),
            
            # 완전연결층
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_advanced_cnn_model(self, embedding_dim=128):
        """
        다중 커널 크기를 가진 고급 CNN 모델
        """
        input_layer = Input(shape=(self.max_length,))
        embedding = Embedding(self.max_features, embedding_dim)(input_layer)
        
        # 다양한 커널 크기의 CNN
        conv_layers = []
        kernel_sizes = [3, 4, 5]
        
        for kernel_size in kernel_sizes:
            conv = Conv1D(64, kernel_size, activation='relu')(embedding)
            conv = GlobalMaxPooling1D()(conv)
            conv_layers.append(conv)
        
        # 결합
        merged = tf.keras.layers.concatenate(conv_layers)
        
        # 완전연결층
        dense = Dense(128, activation='relu')(merged)
        dense = Dropout(0.5)(dense)
        dense = Dense(64, activation='relu')(dense)
        dense = Dropout(0.3)(dense)
        output = Dense(1, activation='sigmoid')(dense)
        
        model = Model(inputs=input_layer, outputs=output)
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_lstm_cnn_hybrid(self, embedding_dim=128, lstm_units=64, filters=64):
        """
        LSTM + CNN 하이브리드 모델
        """
        input_layer = Input(shape=(self.max_length,))
        embedding = Embedding(self.max_features, embedding_dim)(input_layer)
        
        # LSTM 브랜치
        lstm_branch = Bidirectional(LSTM(lstm_units, return_sequences=True))(embedding)
        lstm_branch = GlobalAveragePooling1D()(lstm_branch)
        
        # CNN 브랜치
        cnn_branch = Conv1D(filters, 3, activation='relu')(embedding)
        cnn_branch = GlobalMaxPooling1D()(cnn_branch)
        
        # 결합
        merged = tf.keras.layers.concatenate([lstm_branch, cnn_branch])
        
        # 완전연결층
        dense = Dense(128, activation='relu')(merged)
        dense = Dropout(0.5)(dense)
        dense = Dense(64, activation='relu')(dense)
        dense = Dropout(0.3)(dense)
        output = Dense(1, activation='sigmoid')(dense)
        
        model = Model(inputs=input_layer, outputs=output)
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, model, X_train, y_train, X_val, y_val, 
                   model_name, epochs=20, batch_size=32):
        """
        모델 학습
        """
        print(f"\n{model_name} 학습 시작...")
        
        # 콜백 설정
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7),
            ModelCheckpoint(
                f'data/models/{model_name.replace(" ", "_").lower()}_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=0
            )
        ]
        
        # 모델 학습
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # 결과 저장
        self.models[model_name] = model
        self.histories[model_name] = history
        
        print(f"{model_name} 학습 완료!")
        
        return history
    
    def evaluate_models(self, X_test, y_test):
        """
        모델 평가
        """
        results = {}
        
        print("\n" + "=" * 50)
        print("딥러닝 모델 평가")
        print("=" * 50)
        
        for name, model in self.models.items():
            print(f"\n{name} 평가 중...")
            
            # 예측
            y_pred_proba = model.predict(X_test)
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # 평가 지표
            test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
            
            results[name] = {
                'model': model,
                'test_loss': test_loss,
                'test_accuracy': test_accuracy,
                'y_pred': y_pred.flatten(),
                'y_pred_proba': y_pred_proba.flatten()
            }
            
            print(f"테스트 정확도: {test_accuracy:.4f}")
            print(f"테스트 손실: {test_loss:.4f}")
            
            # 분류 리포트
            print(f"\n{name} 분류 리포트:")
            print(classification_report(y_test, y_pred.flatten(), 
                                      target_names=['Negative', 'Positive']))
        
        return results
    
    def plot_training_history(self, save_path="static/dl_training_history.png"):
        """
        학습 과정 시각화
        """
        if not self.histories:
            print("학습 기록이 없습니다.")
            return
        
        n_models = len(self.histories)
        fig, axes = plt.subplots(2, n_models, figsize=(5*n_models, 10))
        
        if n_models == 1:
            axes = axes.reshape(2, 1)
        
        fig.suptitle('딥러닝 모델 학습 과정', fontsize=16, fontweight='bold')
        
        for i, (name, history) in enumerate(self.histories.items()):
            # 정확도 그래프
            axes[0, i].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
            axes[0, i].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
            axes[0, i].set_title(f'{name} - Accuracy')
            axes[0, i].set_xlabel('Epoch')
            axes[0, i].set_ylabel('Accuracy')
            axes[0, i].legend()
            axes[0, i].grid(True, alpha=0.3)
            
            # 손실 그래프
            axes[1, i].plot(history.history['loss'], label='Training Loss', linewidth=2)
            axes[1, i].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
            axes[1, i].set_title(f'{name} - Loss')
            axes[1, i].set_xlabel('Epoch')
            axes[1, i].set_ylabel('Loss')
            axes[1, i].legend()
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 저장
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # GUI 창 대신 파일만 저장
        
        print(f"학습 과정 그래프 저장: {save_path}")
    
    def plot_model_comparison(self, results, save_path="static/dl_model_comparison.png"):
        """
        딥러닝 모델 성능 비교
        """
        model_names = list(results.keys())
        accuracies = [results[name]['test_accuracy'] for name in model_names]
        losses = [results[name]['test_loss'] for name in model_names]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 정확도 비교
        bars1 = ax1.bar(model_names, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax1.set_title('모델별 테스트 정확도', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        
        # 값 표시
        for bar, acc in zip(bars1, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax1.tick_params(axis='x', rotation=45)
        
        # 손실 비교
        bars2 = ax2.bar(model_names, losses, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax2.set_title('모델별 테스트 손실', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Loss')
        
        # 값 표시
        for bar, loss in zip(bars2, losses):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{loss:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # 저장
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # GUI 창 대신 파일만 저장
        
        print(f"딥러닝 모델 비교 그래프 저장: {save_path}")
    
    def save_models_and_tokenizer(self, save_dir="data/models"):
        """
        모델과 토크나이저 저장
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 토크나이저 저장
        if self.tokenizer:
            tokenizer_path = os.path.join(save_dir, "dl_tokenizer.pkl")
            joblib.dump(self.tokenizer, tokenizer_path)
            print(f"토크나이저 저장: {tokenizer_path}")
        
        # 모델 저장
        for name, model in self.models.items():
            model_path = os.path.join(save_dir, f"{name.replace(' ', '_').lower()}_model.h5")
            model.save(model_path)
            print(f"{name} 모델 저장: {model_path}")
        
        # 설정 저장
        config = {
            'max_features': self.max_features,
            'max_length': self.max_length
        }
        config_path = os.path.join(save_dir, "dl_config.pkl")
        joblib.dump(config, config_path)
        print(f"설정 저장: {config_path}")

def main():
    """
    메인 실행 함수
    """
    print("딥러닝 모델 학습 시작")
    print("=" * 50)
    
    # 전처리된 데이터 로드
    try:
        df = pd.read_csv("data/imdb_preprocessed.csv")
        print(f"데이터 로드 완료: {df.shape}")
        
    except FileNotFoundError:
        print("전처리된 데이터를 찾을 수 없습니다.")
        print("먼저 data_preprocessor.py를 실행해주세요.")
        return
    
    # 데이터 분할
    from sklearn.model_selection import train_test_split
    
    X = df['processed_review']
    y = df['sentiment_label']
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"학습 데이터: {len(X_train)}")
    print(f"검증 데이터: {len(X_val)}")
    print(f"테스트 데이터: {len(X_test)}")
    
    # 딥러닝 모델 초기화
    dl_models = DeepLearningModels(max_features=10000, max_length=200)
    
    # 데이터 준비
    print("\n데이터 준비 중...")
    X_train_seq, y_train_arr = dl_models.prepare_data(X_train, y_train, fit_tokenizer=True)
    X_val_seq, y_val_arr = dl_models.prepare_data(X_val, y_val, fit_tokenizer=False)
    X_test_seq, y_test_arr = dl_models.prepare_data(X_test, y_test, fit_tokenizer=False)
    
    print(f"시퀀스 데이터 형태: {X_train_seq.shape}")
    
    # 모델 구축 및 학습
    models_config = [
        ("LSTM", dl_models.build_lstm_model),
        ("CNN", dl_models.build_cnn_model),
        ("Advanced CNN", dl_models.build_advanced_cnn_model),
        ("LSTM-CNN Hybrid", dl_models.build_lstm_cnn_hybrid)
    ]
    
    for name, build_func in models_config:
        print(f"\n{name} 모델 구축 중...")
        model = build_func()
        
        print(f"{name} 모델 구조:")
        model.summary()
        
        # 모델 학습
        dl_models.train_model(
            model, X_train_seq, y_train_arr,
            X_val_seq, y_val_arr,
            name, epochs=15, batch_size=32
        )
    
    # 모델 평가
    results = dl_models.evaluate_models(X_test_seq, y_test_arr)
    
    # 시각화
    dl_models.plot_training_history()
    dl_models.plot_model_comparison(results)
    
    # 모델 저장
    dl_models.save_models_and_tokenizer()
    
    print("\n" + "=" * 50)
    print("딥러닝 모델 학습 완료!")
    print("=" * 50)
    
    # 최고 성능 모델 출력
    best_model = max(results.items(), key=lambda x: x[1]['test_accuracy'])
    print(f"최고 성능 모델: {best_model[0]}")
    print(f"테스트 정확도: {best_model[1]['test_accuracy']:.4f}")

if __name__ == "__main__":
    main()

