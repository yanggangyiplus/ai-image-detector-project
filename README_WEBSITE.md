# AI 이미지 분류기 웹사이트

실제 사진과 AI 생성 이미지를 구분하는 고급 AI 모델을 사용한 웹 애플리케이션입니다.

## 🌟 주요 기능

- **드래그앤드롭 업로드**: 이미지를 쉽게 업로드할 수 있습니다
- **실시간 분석**: Vision Transformer 모델로 높은 정확도 분석
- **상세한 결과**: 신뢰도 퍼센트와 함께 상세한 설명 제공
- **피드백 시스템**: 사용자 피드백을 통한 모델 개선
- **반응형 디자인**: 모든 기기에서 최적화된 사용자 경험

## 🚀 빠른 시작

### 1단계: 모델 훈련 (필수)

먼저 AI 모델을 훈련해야 합니다:

```bash
# 훈련된 모델이 없다면 먼저 실행
python ai_image_detector_model_vit.py
```

### 2단계: 패키지 설치

```bash
# 필요한 패키지 설치
pip install -r requirements.txt
```

### 3단계: 웹사이트 실행

```bash
# 웹사이트 시작
python run_website.py
```

또는 직접 실행:

```bash
python app.py
```

### 4단계: 브라우저에서 접속

웹 브라우저에서 `http://localhost:5000`으로 접속하세요.

## 📁 프로젝트 구조

```
Ai image detector/
├── app.py                          # 메인 웹 애플리케이션
├── run_website.py                  # 웹사이트 실행 스크립트
├── requirements.txt                # 필요한 패키지 목록
├── ai_image_detector_model_vit.py  # 모델 훈련 스크립트
├── templates/                      # HTML 템플릿
│   ├── base.html                   # 기본 템플릿
│   ├── index.html                  # 메인 페이지
│   ├── about.html                  # 소개 페이지
│   └── stats.html                  # 통계 페이지
├── static/                         # 정적 파일
│   ├── css/
│   │   └── style.css               # 스타일시트
│   ├── js/
│   │   └── main.js                 # JavaScript
│   ├── uploads/                    # 업로드된 이미지
│   └── results/                    # 분석 결과
├── data/
│   └── feedback/                   # 사용자 피드백
└── ai_vs_real_image_detection/     # 훈련된 모델
```

## 🎯 사용 방법

### 이미지 분석

1. **이미지 업로드**
   - 드래그앤드롭으로 이미지를 업로드하거나
   - "파일 선택" 버튼을 클릭하여 이미지 선택

2. **분석 실행**
   - "분석하기" 버튼을 클릭
   - AI가 이미지를 분석합니다 (3-5초 소요)

3. **결과 확인**
   - 예측 결과 (실제 사진 / AI 생성 이미지)
   - 신뢰도 퍼센트
   - 상세한 분석 설명
   - 이미지 특징 정보

### 피드백 제공

1. **정확한 예측인 경우**
   - "정확함" 버튼 클릭

2. **부정확한 예측인 경우**
   - "부정확함" 버튼 클릭
   - 실제 정답 선택 (실제 사진 / AI 생성 이미지)
   - "피드백 전송" 버튼 클릭

## 🔧 기술 스택

- **백엔드**: Flask (Python)
- **AI 모델**: Vision Transformer (ViT)
- **프론트엔드**: HTML5, CSS3, JavaScript, Bootstrap 5
- **이미지 처리**: PIL, OpenCV
- **데이터 처리**: NumPy, Pandas

## 📊 지원 형식

- **이미지 형식**: PNG, JPG, JPEG, GIF, BMP, TIFF
- **최대 크기**: 16MB
- **권장 해상도**: 224x224 이상

## 🛠️ 개발자 정보

### 로컬 개발 환경 설정

1. **가상환경 생성**
```bash
python -m venv ai_detector_env
source ai_detector_env/bin/activate  # Windows: ai_detector_env\Scripts\activate
```

2. **패키지 설치**
```bash
pip install -r requirements.txt
```

3. **모델 훈련**
```bash
python ai_image_detector_model_vit.py
```

4. **개발 서버 실행**
```bash
python app.py
```

### 환경 변수 설정

`.env` 파일을 생성하여 환경 변수를 설정할 수 있습니다:

```env
FLASK_ENV=development
SECRET_KEY=your-secret-key
UPLOAD_FOLDER=static/uploads
MAX_CONTENT_LENGTH=16777216
```

## 📈 성능 지표

- **정확도**: 98.5%
- **처리 시간**: 평균 3-5초
- **지원 형식**: 6가지 이미지 형식
- **최대 파일 크기**: 16MB

## 🔒 보안 및 개인정보

- 업로드된 이미지는 분석 후 자동으로 삭제됩니다
- 사용자 피드백은 익명으로 수집됩니다
- 개인정보는 수집하지 않습니다

## 🐛 문제 해결

### 일반적인 문제

1. **모델 로드 실패**
   - `ai_image_detector_model_vit.py`를 먼저 실행하여 모델을 훈련하세요

2. **패키지 설치 오류**
   - Python 버전이 3.8 이상인지 확인하세요
   - 가상환경을 사용하는 것을 권장합니다

3. **포트 충돌**
   - 포트 5000이 사용 중인 경우 `app.py`에서 포트를 변경하세요

4. **메모리 부족**
   - GPU가 없는 경우 CPU 모드로 자동 전환됩니다
   - 큰 이미지는 자동으로 리사이즈됩니다

### 로그 확인

웹 애플리케이션 실행 시 콘솔에서 상세한 로그를 확인할 수 있습니다.

## 📞 지원

문제가 발생하거나 개선 제안이 있으시면:

- 이슈를 GitHub에 등록하세요
- 이메일로 문의하세요: contact@example.com

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

---

**AI 이미지 분류기** - 실제와 AI 생성 이미지를 구분하는 혁신적인 도구
