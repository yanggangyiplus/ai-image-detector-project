# 🚀 Heroku 배포 가이드 - AI 이미지 분류기

이 가이드는 AI 이미지 분류기 웹 애플리케이션을 Heroku에 무료로 배포하는 방법을 설명합니다.

## 📋 사전 준비

### 1. 필수 요구사항
- GitHub 계정
- Heroku 계정 (무료)
- Git 설치
- Heroku CLI 설치

### 2. 모델 준비
```bash
# 훈련된 모델이 있는지 확인
ls -la ai_vs_real_image_detection/
```

## 🔧 1단계: Heroku CLI 설치 및 로그인

### macOS
```bash
brew tap heroku/brew && brew install heroku
```

### Windows
[Heroku CLI 다운로드](https://devcenter.heroku.com/articles/heroku-cli)

### 로그인
```bash
heroku login
```

## 📁 2단계: Git 저장소 설정

```bash
# Git 초기화 (이미 되어있다면 생략)
git init

# 모든 파일 추가
git add .

# 첫 커밋
git commit -m "Initial commit: AI Image Detector with retraining"

# GitHub 저장소 연결 (선택사항)
git remote add origin https://github.com/your-username/ai-image-detector.git
git push -u origin main
```

## 🌐 3단계: Heroku 앱 생성

```bash
# Heroku 앱 생성
heroku create your-app-name

# 예시: heroku create ai-image-detector-2024
```

## ⚙️ 4단계: 환경 변수 설정

```bash
# Flask 환경 설정
heroku config:set FLASK_ENV=production

# 시크릿 키 설정
heroku config:set SECRET_KEY=your-secret-key-here

# 메모리 최적화
heroku config:set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

## 📦 5단계: 모델 파일 업로드

### 방법 1: Git LFS 사용 (권장)
```bash
# Git LFS 설치
git lfs install

# 모델 파일을 LFS로 추적
git lfs track "ai_vs_real_image_detection/**"
git lfs track "*.safetensors"
git lfs track "*.bin"

# .gitattributes 파일 추가
git add .gitattributes
git commit -m "Add LFS tracking for model files"

# Heroku에 배포
git push heroku main
```

### 방법 2: 외부 저장소 사용
```bash
# 모델을 Google Drive나 Dropbox에 업로드
# app.py에서 모델 다운로드 코드 추가
```

## 🚀 6단계: 배포 실행

```bash
# Heroku에 배포
git push heroku main

# 로그 확인
heroku logs --tail

# 앱 열기
heroku open
```

## 🔍 7단계: 배포 확인

### 1. 앱 상태 확인
```bash
heroku ps
```

### 2. 로그 모니터링
```bash
heroku logs --tail
```

### 3. 웹사이트 접속
- Heroku에서 제공하는 URL로 접속
- 예: `https://your-app-name.herokuapp.com`

## 🛠️ 문제 해결

### 1. 메모리 부족 오류
```bash
# Heroku 메모리 제한 확인
heroku config:set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64

# 모델 최적화
# - 모델 크기 줄이기
# - 배치 크기 줄이기
```

### 2. 빌드 실패
```bash
# 빌드 로그 확인
heroku logs --tail

# requirements.txt 확인
pip freeze > requirements.txt
```

### 3. 앱 크래시
```bash
# 프로세스 상태 확인
heroku ps

# 로그 확인
heroku logs --tail

# 앱 재시작
heroku restart
```

## 📊 8단계: 성능 모니터링

### 1. Heroku 메트릭 확인
```bash
# 앱 메트릭 보기
heroku addons:create newrelic:wayne
```

### 2. 로그 분석
```bash
# 에러 로그만 보기
heroku logs --tail --source app

# 특정 시간대 로그
heroku logs --tail --since="1 hour ago"
```

## 🔄 9단계: 지속적 배포 설정

### GitHub Actions 사용 (선택사항)
```yaml
# .github/workflows/deploy.yml
name: Deploy to Heroku

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - uses: akhileshns/heroku-deploy@v3.12.12
      with:
        heroku_api_key: ${{secrets.HEROKU_API_KEY}}
        heroku_app_name: "your-app-name"
        heroku_email: "your-email@example.com"
```

## 💡 10단계: 최적화 팁

### 1. 메모리 최적화
```python
# app.py에서
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# 모델 로드 시
model = ViTForImageClassification.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # 메모리 절약
    device_map="auto"
)
```

### 2. 응답 시간 최적화
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

## 📈 11단계: 피드백 시스템 활용

### 1. 피드백 수집 모니터링
- 통계 페이지에서 피드백 현황 확인
- `/stats` 페이지에서 재학습 가능 여부 확인

### 2. 자동 재학습
- 피드백이 50개 이상 모이면 재학습 가능
- 통계 페이지에서 "모델 재학습 시작" 버튼 클릭

### 3. 성능 개선 확인
- 재학습 후 성능 지표 확인
- 사용자 만족도 모니터링

## 🎯 배포 체크리스트

### 배포 전 확인사항
- [ ] 훈련된 모델 파일 존재
- [ ] requirements.txt 최신화
- [ ] Procfile 설정 완료
- [ ] 환경 변수 설정
- [ ] .gitignore 설정
- [ ] Git LFS 설정 (모델 파일용)

### 배포 후 확인사항
- [ ] 웹사이트 정상 접속
- [ ] 이미지 업로드 기능 동작
- [ ] AI 분석 결과 표시
- [ ] 피드백 시스템 동작
- [ ] 통계 페이지 정상 표시
- [ ] 재학습 기능 동작

## 🆘 지원 및 도움

### 문제 발생 시
1. **로그 확인**: `heroku logs --tail`
2. **앱 재시작**: `heroku restart`
3. **환경 변수 확인**: `heroku config`
4. **프로세스 상태**: `heroku ps`

### 추가 도움
- [Heroku 공식 문서](https://devcenter.heroku.com/)
- [Flask 배포 가이드](https://flask.palletsprojects.com/en/2.0.x/deploying/)
- [PyTorch 메모리 최적화](https://pytorch.org/docs/stable/notes/cuda.html)

## 🎉 완료!

배포가 완료되면 전 세계 어디서나 AI 이미지 분류기를 사용할 수 있습니다!

**배포된 URL**: `https://your-app-name.herokuapp.com`

---

**성공적인 배포를 위해 단계별로 차근차근 진행하세요!** 🚀
