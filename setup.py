"""
프로젝트 초기 설정 스크립트
"""

import os
import subprocess
import sys

def install_requirements():
    """패키지 설치"""
    print("📦 필요한 패키지들을 설치하는 중...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ 패키지 설치 완료!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 패키지 설치 실패: {e}")
        return False

def download_nltk_data():
    """NLTK 데이터 다운로드"""
    print("📚 NLTK 데이터 다운로드 중...")
    
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True) 
        nltk.download('wordnet', quiet=True)
        print("✅ NLTK 데이터 다운로드 완료!")
        return True
    except Exception as e:
        print(f"❌ NLTK 데이터 다운로드 실패: {e}")
        return False

def create_directories():
    """필요한 디렉토리 생성"""
    print("📁 프로젝트 디렉토리 생성 중...")
    
    directories = [
        "data",
        "data/models",
        "static", 
        "notebooks",
        "templates"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  ✓ {directory}")
    
    print("✅ 디렉토리 생성 완료!")

def check_data_files():
    """데이터 파일 확인"""
    print("🔍 데이터 파일 확인 중...")
    
    required_files = [
        "IMDB Dataset.csv",
        "netflix_titles.csv"
    ]
    
    missing_files = []
    existing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            existing_files.append(file)
            print(f"  ✓ {file}")
        else:
            missing_files.append(file)
            print(f"  ❌ {file}")
    
    if missing_files:
        print(f"\n⚠️ 다음 파일들이 필요합니다:")
        for file in missing_files:
            print(f"  - {file}")
        print("\n📥 데이터 파일 다운로드 위치:")
        print("  - IMDB Dataset: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
        print("  - Netflix Titles: https://www.kaggle.com/datasets/shivamb/netflix-shows")
        print("\n파일들을 다운로드하여 프로젝트 루트 디렉토리에 배치해주세요.")
        return False
    else:
        print("✅ 모든 필수 데이터 파일이 존재합니다!")
        return True

def main():
    """메인 설정 함수"""
    print("🎬 영화 감성 분석 & 추천 시스템 - 초기 설정")
    print("=" * 60)
    
    # 1. 디렉토리 생성
    create_directories()
    
    # 2. 패키지 설치
    if not install_requirements():
        print("❌ 설정 실패: 패키지 설치에 실패했습니다.")
        return
    
    # 3. NLTK 데이터 다운로드
    if not download_nltk_data():
        print("⚠️ NLTK 데이터 다운로드에 실패했지만 계속 진행합니다.")
    
    # 4. 데이터 파일 확인
    data_ready = check_data_files()
    
    print("\n" + "=" * 60)
    print("🎯 설정 완료!")
    print("=" * 60)
    
    if data_ready:
        print("✅ 모든 설정이 완료되었습니다!")
        print("\n📋 다음 단계:")
        print("  1. python run_all.py - 전체 파이프라인 실행")
        print("  2. streamlit run src/streamlit_app/app.py - 웹 대시보드 실행")
    else:
        print("⚠️ 데이터 파일이 부족합니다.")
        print("필요한 데이터 파일들을 다운로드한 후 setup.py를 다시 실행하거나")
        print("run_all.py를 실행해주세요.")
    
    print("\n📖 자세한 사용법은 README.md를 참조해주세요!")

if __name__ == "__main__":
    main()





