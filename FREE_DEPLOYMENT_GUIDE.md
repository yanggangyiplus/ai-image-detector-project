# 🆓 무료 배포 가이드 - AI 이미지 분류기

Heroku가 유료화되면서 무료 배포 옵션들을 정리한 종합 가이드입니다.

## 🎯 추천 순위

### 1. **Render.com** ⭐⭐⭐⭐⭐ (가장 추천!)
- **무료 플랜**: 750시간/월
- **장점**: Heroku와 유사한 사용법, 안정적
- **단점**: 15분 비활성 시 슬립 모드
- **가이드**: [RENDER_DEPLOYMENT.md](./RENDER_DEPLOYMENT.md)

### 2. **Railway.app** ⭐⭐⭐⭐
- **무료 플랜**: $5 크레딧/월
- **장점**: 간단한 배포, 빠른 속도
- **단점**: 크레딧 제한
- **가이드**: [RAILWAY_DEPLOYMENT.md](./RAILWAY_DEPLOYMENT.md)

### 3. **Fly.io** ⭐⭐⭐
- **무료 플랜**: 3개 앱, 256MB RAM
- **장점**: 전 세계 CDN, Docker 지원
- **단점**: CLI 필요, 복잡한 설정
- **가이드**: [FLY_DEPLOYMENT.md](./FLY_DEPLOYMENT.md)

## 📊 비교표

| 서비스 | 무료 플랜 | RAM | 슬립 모드 | 설정 난이도 | 추천도 |
|--------|-----------|-----|-----------|-------------|--------|
| Render.com | 750시간/월 | 512MB | 15분 | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| Railway.app | $5/월 | 512MB | 5분 | ⭐ | ⭐⭐⭐⭐ |
| Fly.io | 3개 앱 | 256MB | 5분 | ⭐⭐⭐ | ⭐⭐⭐ |

## 🚀 빠른 시작

### Render.com (추천)
```bash
# 1. GitHub 저장소 생성
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/your-username/ai-image-detector.git
git push -u origin main

# 2. Render.com에서 웹 서비스 생성
# - GitHub 저장소 연결
# - render.yaml 자동 인식
# - Deploy 클릭
```

### Railway.app
```bash
# 1. GitHub 저장소 생성 (위와 동일)

# 2. Railway.app에서 프로젝트 생성
# - GitHub 저장소 연결
# - railway.json 자동 인식
# - Deploy 클릭
```

### Fly.io
```bash
# 1. Fly CLI 설치
brew install flyctl  # macOS

# 2. 로그인 및 앱 생성
fly auth login
fly launch

# 3. 배포
fly deploy
```

## 🔧 공통 설정

### 1. 모델 파일 업로드
```bash
# Git LFS 설정 (모든 서비스 공통)
git lfs install
git lfs track "ai_vs_real_image_detection/**"
git lfs track "*.safetensors"
git lfs track "*.bin"
git add .gitattributes
git commit -m "Add LFS tracking"
git push origin main
```

### 2. 환경 변수 설정
```bash
# 모든 서비스에서 설정해야 할 환경 변수
FLASK_ENV=production
SECRET_KEY=your-secret-key-here
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

### 3. 메모리 최적화
```python
# app.py에서 메모리 최적화
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# 모델 로드 시 최적화
model = ViTForImageClassification.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # 메모리 절약
    device_map="auto"
)
```

## 🛠️ 문제 해결

### 공통 문제들

#### 1. 빌드 실패
```bash
# requirements.txt 확인
pip freeze > requirements.txt

# 로그에서 구체적인 오류 확인
# 각 서비스의 로그 탭에서 확인
```

#### 2. 메모리 부족
```bash
# 환경 변수 추가
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64

# 모델 최적화
# - 모델 크기 줄이기
# - 배치 크기 줄이기
```

#### 3. 슬립 모드 문제
- **Render.com**: 15분 비활성 시 슬립, 첫 요청 시 30초 지연
- **Railway.app**: 5분 비활성 시 슬립, 첫 요청 시 10초 지연
- **Fly.io**: 5분 비활성 시 슬립, 첫 요청 시 5초 지연

#### 4. 서비스 크래시
```bash
# 로그 확인
# 각 서비스의 로그 탭에서 확인

# 서비스 재시작
# 각 서비스의 재시작 기능 사용
```

## 📈 성능 최적화

### 1. 이미지 처리 최적화
```python
def resize_image(image, max_size=1024):
    """이미지 크기 최적화"""
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return image
```

### 2. 모델 최적화
```python
# 모델 로드 시 최적화
model = ViTForImageClassification.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # 메모리 절약
    device_map="auto"
)

# 추론 시 최적화
with torch.no_grad():
    outputs = model(**inputs)
```

### 3. 캐싱 추가
```python
from flask_caching import Cache
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

@cache.memoize(timeout=300)
def analyze_image_cached(image_path):
    return classifier(image_path)
```

## 🔄 지속적 배포

### GitHub Actions 설정
```yaml
# .github/workflows/deploy.yml
name: Deploy to Multiple Platforms

on:
  push:
    branches: [ main ]

jobs:
  deploy-render:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Deploy to Render
      run: |
        # Render 자동 배포 (GitHub 연동)
        echo "Render will auto-deploy from GitHub"
  
  deploy-railway:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - uses: bervProject/railway-deploy@v1.0.7
      with:
        railway_token: ${{ secrets.RAILWAY_TOKEN }}
        service: ai-image-detector
  
  deploy-fly:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - uses: superfly/flyctl-actions/setup-flyctl@master
    - run: flyctl deploy --remote-only
      env:
        FLY_API_TOKEN: ${{ secrets.FLY_API_TOKEN }}
```

## 🎯 배포 체크리스트

### 배포 전 확인사항
- [ ] 훈련된 모델 파일 존재
- [ ] requirements.txt 최신화
- [ ] 배포 설정 파일 준비 (render.yaml, railway.json, fly.toml)
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
1. **로그 확인**: 각 서비스의 로그 탭에서 확인
2. **서비스 재시작**: 각 서비스의 재시작 기능 사용
3. **환경 변수 확인**: 각 서비스의 환경 변수 탭에서 확인
4. **서비스 상태**: 각 서비스의 대시보드에서 확인

### 추가 도움
- [Render 공식 문서](https://render.com/docs)
- [Railway 공식 문서](https://docs.railway.app/)
- [Fly.io 공식 문서](https://fly.io/docs/)
- [Flask 배포 가이드](https://flask.palletsprojects.com/en/2.0.x/deploying/)

## 🎉 완료!

배포가 완료되면 전 세계 어디서나 AI 이미지 분류기를 사용할 수 있습니다!

**배포된 URL 예시**:
- Render.com: `https://ai-image-detector.onrender.com`
- Railway.app: `https://ai-image-detector-production.up.railway.app`
- Fly.io: `https://ai-image-detector.fly.dev`

---

**무료 배포로 성공적인 AI 서비스를 만들어보세요!** 🚀
