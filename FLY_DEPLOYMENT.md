# 🚀 Fly.io 무료 배포 가이드 - AI 이미지 분류기

Fly.io는 3개 앱, 256MB RAM을 제공하는 무료 대안으로 Python Flask 앱을 배포할 수 있는 서비스입니다.

## 📋 사전 준비

### 1. 필수 요구사항
- GitHub 계정
- Fly.io 계정 (무료)
- Git 설치
- Fly CLI 설치

### 2. 모델 준비
```bash
# 훈련된 모델이 있는지 확인
ls -la ai_vs_real_image_detection/
```

## 🔧 1단계: Fly.io 계정 생성 및 CLI 설치

### 계정 생성
1. [Fly.io](https://fly.io) 접속
2. "Sign Up" 클릭
3. GitHub 계정으로 로그인
4. 이메일 인증 완료

### CLI 설치
```bash
# macOS
brew install flyctl

# Windows (PowerShell)
iwr https://fly.io/install.ps1 -useb | iex

# Linux
curl -L https://fly.io/install.sh | sh
```

### 로그인
```bash
fly auth login
```

## 📁 2단계: GitHub 저장소 설정

```bash
# Git 초기화 (이미 되어있다면 생략)
git init

# 모든 파일 추가
git add .

# 첫 커밋
git commit -m "Initial commit: AI Image Detector with retraining"

# GitHub 저장소 생성 및 연결
git remote add origin https://github.com/your-username/ai-image-detector.git
git push -u origin main
```

## 🌐 3단계: Fly.io 앱 생성

```bash
# 앱 생성
fly launch

# 앱 이름 입력 (예: ai-image-detector)
# 지역 선택 (예: nrt - Tokyo)
# Dockerfile 사용 여부: Yes
# PostgreSQL 필요 여부: No
```

## ⚙️ 4단계: 모델 파일 업로드

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

# GitHub에 푸시
git push origin main
```

### 방법 2: 외부 저장소 사용
```bash
# 모델을 Google Drive나 Dropbox에 업로드
# app.py에서 모델 다운로드 코드 추가
```

## 🚀 5단계: 배포 실행

```bash
# 배포 실행
fly deploy

# 배포 상태 확인
fly status

# 로그 확인
fly logs
```

## 🔍 6단계: 배포 확인

### 1. 서비스 상태 확인
```bash
fly status
```

### 2. 로그 모니터링
```bash
fly logs
```

### 3. 웹사이트 접속
```bash
fly open
# 또는 브라우저에서 https://ai-image-detector.fly.dev 접속
```

## 🛠️ 문제 해결

### 1. 빌드 실패
```bash
# 로컬에서 빌드 테스트
fly deploy --local-only

# 로그에서 구체적인 오류 확인
fly logs
```

### 2. 메모리 부족
```bash
# fly.toml에서 메모리 설정 확인
# memory_mb = 256

# 환경 변수 추가
fly secrets set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
```

### 3. 서비스 크래시
```bash
# 로그 확인
fly logs

# 서비스 재시작
fly restart
```

## 📊 7단계: 성능 모니터링

### 1. Fly.io 메트릭 확인
```bash
# 메트릭 확인
fly metrics

# 앱 상태 확인
fly status
```

### 2. 사용자 피드백 모니터링
- 웹사이트에서 피드백 수집 현황 확인
- `/stats` 페이지에서 재학습 가능 여부 확인

## 🔄 8단계: 자동 배포 설정

### GitHub Actions 사용
```yaml
# .github/workflows/deploy.yml
name: Deploy to Fly.io

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - uses: superfly/flyctl-actions/setup-flyctl@master
    - run: flyctl deploy --remote-only
      env:
        FLY_API_TOKEN: ${{ secrets.FLY_API_TOKEN }}
```

### 환경 변수 관리
```bash
# 시크릿 설정
fly secrets set SECRET_KEY=your-secret-key-here

# 환경 변수 확인
fly secrets list
```

## 💡 9단계: 최적화 팁

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

### 3. 무료 플랜 제한 고려
- **3개 앱** 제한
- **256MB RAM** 제한
- **슬립 모드**: 5분 비활성 시 슬립

## 🎯 배포 체크리스트

### 배포 전 확인사항
- [ ] 훈련된 모델 파일 존재
- [ ] requirements.txt 최신화
- [ ] Dockerfile 설정 완료
- [ ] fly.toml 설정 완료
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
1. **로그 확인**: `fly logs`
2. **서비스 재시작**: `fly restart`
3. **환경 변수 확인**: `fly secrets list`
4. **서비스 상태**: `fly status`

### 추가 도움
- [Fly.io 공식 문서](https://fly.io/docs/)
- [Flask 배포 가이드](https://flask.palletsprojects.com/en/2.0.x/deploying/)
- [PyTorch 메모리 최적화](https://pytorch.org/docs/stable/notes/cuda.html)

## 🎉 완료!

배포가 완료되면 전 세계 어디서나 AI 이미지 분류기를 사용할 수 있습니다!

**배포된 URL**: `https://ai-image-detector.fly.dev`

---

**Fly.io로 성공적인 무료 배포를 위해 단계별로 차근차근 진행하세요!** 🚀
