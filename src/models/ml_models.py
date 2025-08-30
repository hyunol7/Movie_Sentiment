"""
머신러닝 모델 구현 (속도 최적화 버전)
- Logistic Regression
- Naive Bayes
- Linear SVM (probability=False)
- Random Forest
- 모델 평가/비교 및 시각화
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # GUI 없이 사용
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)


class MLModelTrainer:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None

    def initialize_models(self):
        """머신러닝 모델 초기화 (빠른 학습을 위한 설정)"""
        self.models = {
            "Logistic Regression": LogisticRegression(
                random_state=42, max_iter=1000, solver="liblinear"
            ),
            "Naive Bayes": MultinomialNB(alpha=1.0),
            # LinearSVC로 변경 (SVC보다 훨씬 빠름)
            "SVM": LinearSVC(
                random_state=42,
                max_iter=1000,  # 조기 종료
                dual=False  # 특성 수 > 샘플 수일 때 더 빠름
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=50,  # 100→50 (빠르게)
                max_depth=20,     # 깊이 제한
                random_state=42, 
                n_jobs=-1
            ),
        }

        print("머신러닝 모델 초기화 완료:")
        for name in self.models:
            print(f"- {name}")

    @staticmethod
    def _get_proba(model, X):
        """
        모델의 양성 클래스 점수 반환 유틸
        - predict_proba가 있으면 그대로 사용
        - 없으면 decision_function을 0~1로 min-max 스케일링하여 의사 확률로 사용
        """
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X)
            s_min, s_max = scores.min(), scores.max()
            # 분모 0 방지
            return (scores - s_min) / (s_max - s_min + 1e-8)
        else:
            # 최후 수단: 예측 레이블(0/1)을 그대로 반환 (ROC 계산 제한적)
            return model.predict(X).astype(float)

    def train_models(self, X_train, y_train, X_test, y_test):
        """모델 학습 및 평가"""
        print("\n" + "=" * 50)
        print("모델 학습 시작")
        print("=" * 50)

        for name, model in self.models.items():
            print(f"\n{name} 학습 중...")

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_score = self._get_proba(model, X_test)

            metrics = self.calculate_metrics(y_test, y_pred, y_score)

            # 속도 위해 CV 5→3
            cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring="accuracy")
            metrics["cv_mean"] = cv_scores.mean()
            metrics["cv_std"] = cv_scores.std()

            self.results[name] = {
                "model": model,
                "metrics": metrics,
                "y_pred": y_pred,
                "y_pred_proba": y_score,
            }
            print(f"{name} 완료 - 정확도: {metrics['accuracy']:.4f}")

    @staticmethod
    def calculate_metrics(y_true, y_pred, y_score):
        """평가 지표 계산"""
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred),
            "roc_auc": roc_auc_score(y_true, y_score),
        }

    def hyperparameter_tuning(self, X_train, y_train):
        """(옵션) 하이퍼파라미터 튜닝 — 필요시만 실행 권장"""
        print("\n" + "=" * 50)
        print("하이퍼파라미터 튜닝 시작")
        print("=" * 50)

        param_grids = {
            "Logistic Regression": {
                "C": [0.1, 1, 10],
                "penalty": ["l1", "l2"],
                "solver": ["liblinear"],
            },
            "Naive Bayes": {"alpha": [0.1, 0.5, 1.0, 2.0]},
            "SVM": {"C": [0.1, 1, 10], "kernel": ["linear"]},  # rbf 제거로 속도↑
            "Random Forest": {
                "n_estimators": [100, 200],
                "max_depth": [None, 20],
                "min_samples_split": [2, 5],
            },
        }

        tuned = {}
        for name, model in self.models.items():
            if name not in param_grids:
                continue
            print(f"\n{name} 튜닝 중...")
            grid = GridSearchCV(
                model, param_grids[name], cv=3, scoring="accuracy", n_jobs=-1, verbose=0
            )
            grid.fit(X_train, y_train)
            tuned[name] = grid.best_estimator_
            print(f"최적 파라미터: {grid.best_params_}")
            print(f"최적 점수: {grid.best_score_:.4f}")
        return tuned

    def compare_models(self):
        """모델 성능 비교 테이블 생성"""
        if not self.results:
            print("학습된 모델이 없습니다.")
            return

        rows = []
        for name, res in self.results.items():
            m = res["metrics"]
            rows.append({
                "Model": name,
                "Accuracy": m["accuracy"],
                "Precision": m["precision"],
                "Recall": m["recall"],
                "F1-Score": m["f1_score"],
                "ROC-AUC": m["roc_auc"],
                "CV Mean": m["cv_mean"],
                "CV Std": m["cv_std"],
            })
        df = pd.DataFrame(rows).sort_values("Accuracy", ascending=False)

        print("\n" + "=" * 80)
        print("모델 성능 비교")
        print("=" * 80)
        print(df.round(4).to_string(index=False))

        best_name = df.iloc[0]["Model"]
        self.best_model = self.results[best_name]["model"]
        print(f"\n최고 성능 모델: {best_name}")
        print(f"정확도: {df.iloc[0]['Accuracy']:.4f}")
        return df

    def plot_model_comparison(self, comparison_df, save_path="static/model_comparison.png"):
        """모델 성능 비교 시각화"""
        metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("머신러닝 모델 성능 비교", fontsize=16, fontweight="bold")

        for i, metric in enumerate(metrics):
            r, c = divmod(i, 3)
            ax = axes[r, c]
            bars = ax.bar(comparison_df["Model"], comparison_df[metric],
                          color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
            ax.set_title(metric)
            ax.set_ylim(0, 1)
            ax.tick_params(axis="x", rotation=45)
            for b in bars:
                ax.text(b.get_x() + b.get_width()/2, b.get_height()+0.01,
                        f"{b.get_height():.3f}", ha="center")

        axes[1, 2].axis("off")
        table = axes[1, 2].table(
            cellText=comparison_df[["Model"] + metrics].round(3).values,
            colLabels=["Model"] + metrics,
            cellLoc="center", loc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()  # GUI 창 대신 파일만 저장
        print(f"모델 비교 그래프 저장: {save_path}")

    def plot_confusion_matrices(self, y_test, save_path="static/confusion_matrices.png"):
        """혼동행렬 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("혼동 행렬 (Confusion Matrix)", fontsize=16, fontweight="bold")

        for i, (name, res) in enumerate(self.results.items()):
            r, c = divmod(i, 2)
            ax = axes[r, c]
            cm = confusion_matrix(y_test, res["y_pred"])
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=["Negative", "Positive"],
                        yticklabels=["Negative", "Positive"])
            ax.set_title(name)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()  # GUI 창 대신 파일만 저장
        print(f"혼동 행렬 그래프 저장: {save_path}")

    def plot_roc_curves(self, y_test, save_path="static/roc_curves.png"):
        """ROC 곡선 시각화"""
        plt.figure(figsize=(10, 8))
        colors = ["blue", "red", "green", "orange"]

        for i, (name, res) in enumerate(self.results.items()):
            fpr, tpr, _ = roc_curve(y_test, res["y_pred_proba"])
            auc = res["metrics"]["roc_auc"]
            plt.plot(fpr, tpr, color=colors[i], lw=2, label=f"{name} (AUC={auc:.3f})")

        plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
        plt.xlim([0, 1]); plt.ylim([0, 1.05])
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title("ROC 곡선 비교"); plt.legend(loc="lower right"); plt.grid(True, alpha=0.3)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()  # GUI 창 대신 파일만 저장
        print(f"ROC 곡선 그래프 저장: {save_path}")

    def save_models(self, save_dir="data/models"):
        """학습된 모델 저장"""
        os.makedirs(save_dir, exist_ok=True)
        for name, res in self.results.items():
            path = os.path.join(save_dir, f"{name.replace(' ', '_').lower()}_model.pkl")
            joblib.dump(res["model"], path)
            print(f"{name} 모델 저장: {path}")

        if self.best_model:
            best_path = os.path.join(save_dir, "best_model.pkl")
            joblib.dump(self.best_model, best_path)
            print(f"최고 성능 모델 저장: {best_path}")

        joblib.dump(self.results, os.path.join(save_dir, "training_results.pkl"))


def main():
    print("머신러닝 모델 학습 시작")
    print("=" * 50)

    try:
        X_train = joblib.load("data/X_train_tfidf.pkl")
        X_test  = joblib.load("data/X_test_tfidf.pkl")
        y_train = joblib.load("data/y_train.pkl")
        y_test  = joblib.load("data/y_test.pkl")
        print(f"학습 데이터: {X_train.shape} / 테스트 데이터: {X_test.shape}")
    except FileNotFoundError:
        print("전처리된 데이터를 찾을 수 없습니다. 먼저 data_preprocessor.py를 실행하세요.")
        return

    trainer = MLModelTrainer()
    trainer.initialize_models()
    trainer.train_models(X_train, y_train, X_test, y_test)
    cmp_df = trainer.compare_models()
    trainer.plot_model_comparison(cmp_df)
    trainer.plot_confusion_matrices(y_test)
    trainer.plot_roc_curves(y_test)
    trainer.save_models()

    print("\n" + "=" * 50)
    print("머신러닝 모델 학습 완료!")
    print("=" * 50)


if __name__ == "__main__":
    main()
