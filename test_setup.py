
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import streamlit as st

print("✅ 모든 패키지가 정상적으로 설치되었습니다!")
print(f"  - pandas: {pd.__version__}")
print(f"  - numpy: {np.__version__}")
print(f"  - sklearn: {sklearn.__version__}")

# 데이터 파일 확인
try:
    df = pd.read_csv("IMDB Dataset.csv", nrows=5)
    print(f"  - IMDB 데이터: {len(df)}행 샘플 로드 성공")
except:
    print("  ❌ IMDB 데이터 로드 실패")

try:
    df = pd.read_csv("netflix_titles.csv", nrows=5)
    print(f"  - Netflix 데이터: {len(df)}행 샘플 로드 성공")
except:
    print("  ❌ Netflix 데이터 로드 실패")
