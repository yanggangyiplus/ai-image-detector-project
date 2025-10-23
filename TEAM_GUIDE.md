# 🤝 팀원들을 위한 AI 이미지 분류기 사용 가이드

## 📋 프로젝트 개요
AI 생성 이미지와 실제 이미지를 구분하는 Vision Transformer 기반 분류기입니다.

## 🚀 빠른 시작

### 1. 저장소 클론
```bash
git clone https://github.com/yanggangyiplus/ai-image-detector-project.git
cd ai-image-detector-project
```

### 2. 환경 설정
```bash
# 가상환경 생성
python -m venv ai_detector_env

# 가상환경 활성화
# macOS/Linux:
source ai_detector_env/bin/activate
# Windows:
ai_detector_env\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

### 3. 모델 파일 다운로드
**중요**: 모델 파일이 GitHub에 없으므로 별도로 공유받아야 합니다.

모델 파일을 `ai_vs_real_image_detection/` 폴더에 넣어주세요:
```
ai_vs_real_image_detection/
├── config.json
├── model.safetensors
├── preprocessor_config.json
└── training_args.bin
```

### 4. 웹사이트 실행
```bash
python app.py
```

브라우저에서 `http://localhost:8080` 접속

## 🔧 주요 기능

### 웹 인터페이스
- 이미지 드래그 앤 드롭
- 파일 업로드
- 실시간 AI 분석
- 피드백 시스템

### 모델 평가
```bash
# 모델 성능 평가
python model_evaluation.py
```

### 모델 재학습
- 사용자 피드백 수집
- 자동 모델 재학습
- 성능 개선

## 📊 모델 성능

| 지표 | 값 |
|------|-----|
| 정확도 | 65.8% |
| ROC AUC | 90.8% |
| 실제 이미지 식별 | 100% |
| AI 생성 이미지 식별 | 0% |

## 🛠️ 개발 환경

### 필요한 패키지
- Python 3.11+
- PyTorch 2.6.0
- Transformers 4.45.0
- Flask 2.3.3
- Pillow, NumPy, Pandas 등

### 디렉토리 구조
```
ai-image-detector-project/
├── app.py                 # 메인 웹 애플리케이션
├── model_evaluation.py    # 모델 평가 스크립트
├── ai_vs_real_image_detection/  # 학습된 모델
├── templates/             # HTML 템플릿
├── static/               # CSS, JS 파일
├── data/                 # 피드백 데이터
└── results/              # 평가 결과
```

## 🚀 배포 옵션

### 1. Render.com (무료)
- GitHub 연동
- 자동 배포
- 무료 플랜 사용 가능

### 2. Railway.app (무료)
- 간단한 배포
- Docker 지원

### 3. Fly.io (무료)
- 글로벌 배포
- Docker 기반

## 📝 사용법

### 이미지 분석
1. 웹사이트 접속
2. 이미지 드래그 앤 드롭 또는 파일 선택
3. "분석하기" 버튼 클릭
4. 결과 확인 및 피드백 제공

### 모델 평가
```bash
# 테스트 데이터로 모델 평가
python model_evaluation.py
```

### 피드백 확인
```bash
# 피드백 데이터 확인
ls data/feedback/
```

## 🔍 문제 해결

### 모델 로드 실패
- 모델 파일이 올바른 위치에 있는지 확인
- `ai_vs_real_image_detection/` 폴더 구조 확인

### 패키지 설치 실패
```bash
# pip 업그레이드
pip install --upgrade pip

# 개별 패키지 설치
pip install torch torchvision transformers
```

### 메모리 부족
- 배치 크기 줄이기
- CPU 모드 사용

## 📞 지원

문제가 있으면 다음을 확인해주세요:
1. Python 버전 (3.11+)
2. 가상환경 활성화
3. 모델 파일 존재
4. 패키지 설치 완료

## 🎯 향후 계획

- [ ] 모델 성능 개선
- [ ] 더 많은 테스트 데이터
- [ ] 실시간 재학습 시스템
- [ ] API 서비스 제공
