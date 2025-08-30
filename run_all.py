"""
전체 프로젝트 실행 스크립트
모든 단계를 순차적으로 실행합니다.
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def run_command(command, description):
    """명령어 실행 함수"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"실행 명령어: {command}")
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"✅ 완료! (소요 시간: {elapsed_time:.2f}초)")
        
        if result.stdout:
            print("출력:")
            print(result.stdout)
        
        return True
        
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"❌ 실패! (소요 시간: {elapsed_time:.2f}초)")
        print(f"오류 메시지: {e}")
        
        if e.stderr:
            print("오류 상세:")
            print(e.stderr)
        
        return False

def check_files_exist():
    """필요한 파일들이 존재하는지 확인"""
    required_files = [
        "IMDB Dataset.csv",
        "netflix_titles.csv"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ 다음 파일들이 필요합니다:")
        for file in missing_files:
            print(f"  - {file}")
        print("\n파일들을 프로젝트 루트 디렉토리에 배치한 후 다시 실행해주세요.")
        return False
    
    print("✅ 모든 필수 파일이 존재합니다.")
    return True

def create_directories():
    """필요한 디렉토리 생성"""
    directories = [
        "data",
        "data/models", 
        "static",
        "notebooks"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ 필요한 디렉토리들을 생성했습니다.")

def main():
    """메인 실행 함수"""
    print("🎬 영화 감성 분석 & 추천 시스템 - 전체 실행")
    print("=" * 60)
    
    # 현재 디렉토리 확인
    current_dir = os.getcwd()
    print(f"현재 디렉토리: {current_dir}")
    
    # 필수 파일 확인
    if not check_files_exist():
        return
    
    # 디렉토리 생성
    create_directories()
    
    # 실행 단계들
    steps = [
        {
            "command": "python src/preprocessing/data_preprocessor.py",
            "description": "1단계: 데이터 전처리",
            "required": True
        },
        {
            "command": "python src/models/ml_models.py", 
            "description": "2단계: 머신러닝 모델 학습",
            "required": True
        },
        {
            "command": "python src/models/dl_models.py",
            "description": "3단계: 딥러닝 모델 학습 (시간이 오래 걸립니다)",
            "required": False
        },
        {
            "command": "python src/models/recommendation_system.py",
            "description": "4단계: 추천 시스템 구축", 
            "required": True
        },
        {
            "command": "python src/utils/visualization.py",
            "description": "5단계: 시각화 생성",
            "required": False
        }
    ]
    
    # 각 단계 실행
    failed_steps = []
    
    for i, step in enumerate(steps):
        if not step["required"]:
            response = input(f"\n{step['description']}를 실행하시겠습니까? (y/N): ").lower()
            if response not in ['y', 'yes']:
                print(f"⏭️ {step['description']} 건너뛰기")
                continue
        
        success = run_command(step["command"], step["description"])
        
        if not success:
            if step["required"]:
                failed_steps.append(step["description"])
                print(f"\n❌ 필수 단계인 '{step['description']}'가 실패했습니다.")
                
                continue_response = input("계속 진행하시겠습니까? (y/N): ").lower()
                if continue_response not in ['y', 'yes']:
                    break
            else:
                print(f"\n⚠️ 선택 단계인 '{step['description']}'가 실패했지만 계속 진행합니다.")
    
    # 결과 요약
    print(f"\n{'='*60}")
    print("🎯 실행 결과 요약")
    print(f"{'='*60}")
    
    if failed_steps:
        print("❌ 실패한 단계들:")
        for step in failed_steps:
            print(f"  - {step}")
        print("\n실패한 단계들을 개별적으로 다시 실행해보세요.")
    else:
        print("✅ 모든 단계가 성공적으로 완료되었습니다!")
    
    print("\n📊 웹 대시보드 실행 방법:")
    print("streamlit run src/streamlit_app/app.py")
    
    # 웹 앱 실행 여부 확인
    response = input("\n지금 웹 대시보드를 실행하시겠습니까? (y/N): ").lower()
    if response in ['y', 'yes']:
        print("\n🌐 웹 대시보드 실행 중...")
        print("브라우저에서 http://localhost:8501 로 접속하세요.")
        print("종료하려면 Ctrl+C를 누르세요.")
        
        try:
            subprocess.run("streamlit run src/streamlit_app/app.py", shell=True)
        except KeyboardInterrupt:
            print("\n✅ 웹 대시보드가 종료되었습니다.")

if __name__ == "__main__":
    main()






