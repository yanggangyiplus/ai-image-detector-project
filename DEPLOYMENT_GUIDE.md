# AI 이미지 분류기 배포 가이드

이 문서는 AI 이미지 분류기 웹 애플리케이션을 다양한 플랫폼에 배포하는 방법을 설명합니다.

## 🚀 배포 옵션

### 1. GitHub Pages (정적 웹사이트)
- **장점**: 무료, 간단한 설정
- **단점**: AI 모델 실행 불가 (데모 버전만 가능)
- **용도**: 프로젝트 소개 및 데모

### 2. Heroku (완전한 웹 애플리케이션)
- **장점**: 무료 티어 제공, AI 모델 실행 가능
- **단점**: 30분 비활성 시 슬립 모드
- **용도**: 실제 AI 분석 서비스

### 3. 로컬 실행
- **장점**: 완전한 기능, 빠른 응답
- **단점**: 본인만 접근 가능
- **용도**: 개발 및 테스트

## 📋 사전 준비

### 1. 모델 훈련
```bash
# 훈련된 모델이 없다면 먼저 실행
python ai_image_detector_model_vit.py
```

### 2. 패키지 설치
```bash
pip install -r requirements.txt
```

## 🌐 GitHub Pages 배포

### 1단계: GitHub 저장소 생성
1. GitHub에서 새 저장소 생성
2. 저장소 이름: `ai-image-detector` (또는 원하는 이름)

### 2단계: 파일 업로드
```bash
# Git 초기화
git init
git add .
git commit -m "Initial commit: AI Image Detector"

# GitHub 저장소 연결
git remote add origin https://github.com/your-username/ai-image-detector.git
git branch -M main
git push -u origin main
```

### 3단계: GitHub Pages 활성화
1. GitHub 저장소 → Settings → Pages
2. Source: Deploy from a branch
3. Branch: main
4. Folder: / (root)
5. Save

### 4단계: 접속
- URL: `https://your-username.github.io/ai-image-detector`

## 🚀 Heroku 배포

### 1단계: Heroku CLI 설치
```bash
# macOS
brew install heroku/brew/heroku

# Windows
# https://devcenter.heroku.com/articles/heroku-cli 에서 다운로드
```

### 2단계: Heroku 로그인
```bash
heroku login
```

### 3단계: Heroku 앱 생성
```bash
heroku create your-app-name
```

### 4단계: 환경 변수 설정
```bash
heroku config:set FLASK_ENV=production
heroku config:set SECRET_KEY=your-secret-key-here
```

### 5단계: 모델 파일 업로드
```bash
# 모델을 Heroku에 업로드 (대용량 파일 처리)
# 방법 1: Git LFS 사용
git lfs install
git lfs track "ai_vs_real_image_detection/**"
git add .gitattributes
git add ai_vs_real_image_detection/
git commit -m "Add model files with LFS"
git push heroku main

# 방법 2: 모델을 외부 저장소에 업로드 후 다운로드
# app.py에서 모델 다운로드 코드 추가
```

### 6단계: 배포
```bash
git add .
git commit -m "Deploy to Heroku"
git push heroku main
```

### 7단계: 앱 실행
```bash
heroku open
```

## 🔧 로컬 실행 (문제 해결)

### HTTP 403 오류 해결
```bash
# 포트 8080으로 실행
python app.py

# 또는 다른 포트 사용
python -c "
from app import app
app.run(debug=True, host='127.0.0.1', port=8080)
"
```

### 접속 URL
- `http://localhost:8080` (포트 8080 사용)
- `http://127.0.0.1:8080`

## 📁 프로젝트 구조

```
Ai image detector/
├── app.py                          # Flask 웹 애플리케이션
├── index.html                      # GitHub Pages용 정적 페이지
├── run_website.py                  # 로컬 실행 스크립트
├── requirements.txt                # Python 패키지 목록
├── Procfile                        # Heroku 배포 설정
├── runtime.txt                     # Python 버전 지정
├── .gitignore                      # Git 무시 파일 목록
├── templates/                      # Flask 템플릿
│   ├── base.html
│   ├── index.html
│   ├── about.html
│   └── stats.html
├── static/                         # 정적 파일
│   ├── css/style.css
│   ├── js/main.js
│   └── uploads/                    # 업로드된 이미지
├── data/feedback/                  # 사용자 피드백
└── ai_vs_real_image_detection/     # 훈련된 모델
```

## 🛠️ 문제 해결

### 1. 모델 로드 실패
```python
# app.py에서 모델 경로 확인
model_path = './ai_vs_real_image_detection'
if not os.path.exists(model_path):
    print("❌ 모델을 찾을 수 없습니다. 먼저 훈련하세요.")
```

### 2. 메모리 부족
```python
# Heroku에서 메모리 제한 해결
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
```

### 3. 파일 업로드 오류
```python
# 파일 크기 제한 조정
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
```

### 4. 포트 충돌
```python
# app.py에서 포트 변경
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
```

## 📊 성능 최적화

### 1. 모델 최적화
```python
# 모델 로드 시 최적화
model = ViTForImageClassification.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # 메모리 절약
    device_map="auto"
)
```

### 2. 이미지 전처리 최적화
```python
# 이미지 크기 제한
def resize_image(image, max_size=1024):
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return image
```

### 3. 캐싱 추가
```python
from flask_caching import Cache
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

@cache.memoize(timeout=300)
def analyze_image_cached(image_path):
    return classifier(image_path)
```

## 🔒 보안 설정

### 1. 환경 변수 사용
```python
import os
from dotenv import load_dotenv

load_dotenv()
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key')
```

### 2. 파일 업로드 보안
```python
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def secure_filename_custom(filename):
    # 파일명 정리
    filename = secure_filename(filename)
    # 타임스탬프 추가
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{timestamp}_{filename}"
```

## 📈 모니터링

### 1. 로그 설정
```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/upload', methods=['POST'])
def upload_file():
    logger.info(f"File upload request from {request.remote_addr}")
    # ... 업로드 처리
```

### 2. 성능 모니터링
```python
import time

@app.before_request
def before_request():
    g.start_time = time.time()

@app.after_request
def after_request(response):
    diff = time.time() - g.start_time
    logger.info(f"Request took {diff:.2f} seconds")
    return response
```

## 🎯 배포 체크리스트

### GitHub Pages
- [ ] `index.html` 파일이 루트에 있는가?
- [ ] GitHub Pages가 활성화되었는가?
- [ ] 저장소가 public인가?

### Heroku
- [ ] `Procfile`이 있는가?
- [ ] `requirements.txt`가 있는가?
- [ ] 모델 파일이 업로드되었는가?
- [ ] 환경 변수가 설정되었는가?

### 로컬
- [ ] 모델이 훈련되었는가?
- [ ] 필요한 패키지가 설치되었는가?
- [ ] 포트가 사용 가능한가?

## 📞 지원

문제가 발생하거나 도움이 필요한 경우:

1. **GitHub Issues**: 저장소에 이슈 등록
2. **이메일**: contact@example.com
3. **문서**: README_WEBSITE.md 참조

---

**성공적인 배포를 위해 단계별로 차근차근 진행하세요!** 🚀
