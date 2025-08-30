"""
빠른 문제 해결 스크립트
"""

import os
import subprocess
import sys

def check_python_version():
    """Python 버전 확인"""
    version = sys.version_info
    print(f"Python 버전: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 이상이 필요합니다.")
        return False
    
    print("✅ Python 버전 확인 완료")
    return True

def check_files():
    """필수 파일 확인"""
    required_files = ["IMDB Dataset.csv", "netflix_titles.csv"]
    
    print("\n📂 필수 파일 확인:")
    for file in required_files:
        if os.path.exists(file):
            size = os.path.getsize(file) / (1024*1024)  # MB
            print(f"  ✅ {file} ({size:.1f} MB)")
        else:
            print(f"  ❌ {file} - 파일이 없습니다!")
            return False
    
    return True

def install_basic_packages():
    """기본 패키지 설치"""
    basic_packages = [
        "pandas",
        "numpy", 
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "streamlit",
        "nltk"
    ]
    
    print("\n📦 기본 패키지 설치 중...")
    
    for package in basic_packages:
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"  ✅ {package}")
        except:
            print(f"  ❌ {package} - 설치 실패")

def create_requirements():
    """requirements.txt 생성"""
    requirements = """# 기본 패키지
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0

# 자연어 처리
nltk==3.8.1

# 시각화
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.15.0

# 웹 애플리케이션
streamlit==1.25.0

# 유틸리티
tqdm==4.65.0
joblib==1.3.1"""

    with open("requirements.txt", "w", encoding="utf-8") as f:
        f.write(requirements)
    
    print("✅ requirements.txt 생성 완료")

def create_simple_test():
    """간단한 테스트 스크립트 생성"""
    test_script = '''
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
'''
    
    with open("test_setup.py", "w", encoding="utf-8") as f:
        f.write(test_script)
    
    print("✅ 테스트 스크립트 생성 완료")

def main():
    print("🔧 영화 프로젝트 문제 해결 도구")
    print("=" * 40)
    
    # 1. Python 버전 확인
    if not check_python_version():
        return
    
    # 2. 파일 확인
    if not check_files():
        print("\n❌ 필수 데이터 파일이 없습니다.")
        print("IMDB Dataset.csv와 netflix_titles.csv를 다운로드해서")
        print("현재 폴더에 배치해주세요.")
        return
    
    # 3. requirements.txt 생성
    create_requirements()
    
    # 4. 기본 패키지 설치
    install_basic_packages()
    
    # 5. 테스트 스크립트 생성
    create_simple_test()
    
    print("\n" + "=" * 40)
    print("🎯 문제 해결 완료!")
    print("=" * 40)
    print("\n다음 명령어로 테스트해보세요:")
    print("python test_setup.py")
    
    print("\n문제가 지속되면 구체적인 에러 메시지를 알려주세요!")

if __name__ == "__main__":
    main()
