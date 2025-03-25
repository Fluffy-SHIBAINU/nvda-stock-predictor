# 📈 NVDA 주가 예측기 (LSTM + PyTorch)

이 프로젝트는 PyTorch와 NVIDIA(NVDA)의 과거 주가 데이터를 활용해  
간단한 LSTM 기반 주가 예측 모델을 구축합니다.  
LSTM 신경망을 사용하여 과거 트렌드를 학습하고 미래 주가를 예측합니다.

## 🔧 주요 기능

- 📉 `yfinance`를 통해 NVDA 주가 데이터 다운로드
- 🧠 PyTorch 기반 LSTM 모델 학습
- 📊 실제 vs 예측 종가 시각화
- 🌐 Streamlit 기반 웹 데모 앱 제공

## 🚀 시작하기

### 1. 필요한 패키지 설치
```bash
pip install -r requirements.txt
```

### 2. 학습 스크립트 실행
```bash
python3 train.py
```

### 3. 웹 앱 실행
```bash
streamlit run app/app.py
```

## 🧪 모델 구성

- **구조**: 2층 LSTM + Fully Connected Layer
- **입력**: 과거 종가 시퀀스 (예: 30일)
- **출력**: 다음 날 종가 예측
- **손실함수**: MSELoss
- **옵티마이저**: Adam

## 🖼️ 예측 결과 예시

![Prediction Chart](screenshot.png)

## 📁 프로젝트 구조

```
nvda-stock-predictor/
├── app/                # Streamlit 웹 앱
│   └── app.py
├── model/              # LSTM 모델 정의
│   └── lstm_model.py
├── utils/              # 데이터 로딩 및 전처리
│   └── dataset.py
├── train.py            # 학습 스크립트
├── requirements.txt
└── README.md
```

## 📝 라이선스

MIT License
