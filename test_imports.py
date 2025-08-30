#!/usr/bin/env python3
"""Import 테스트"""

try:
    import nltk
    print("✅ NLTK import 성공")
except ImportError as e:
    print(f"❌ NLTK import 실패: {e}")

try:
    import pandas as pd
    print("✅ Pandas import 성공")
except ImportError as e:
    print(f"❌ Pandas import 실패: {e}")

try:
    import sklearn
    print("✅ Scikit-learn import 성공")
except ImportError as e:
    print(f"❌ Scikit-learn import 실패: {e}")






