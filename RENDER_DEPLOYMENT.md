# 🚀 Render.com 무료 배포 가이드 - AI 이미지 분류기

Render.com은 Heroku의 무료 대안으로 Python Flask 앱을 무료로 배포할 수 있는 서비스입니다.

## 📋 사전 준비

### 1. 필수 요구사항
- GitHub 계정
- Render.com 계정 (무료)
- Git 설치

### 2. 모델 준비
```bash
# 훈련된 모델이 있는지 확인
ls -la ai_vs_real_image_detection/
```

## 🔧 1단계: Render.com 계정 생성

1. [Render.com](https://render.com) 접속
2. "Get Started for Free" 클릭
3. GitHub 계정으로 로그인
4. 이메일 인증 완료

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

## 🌐 3단계: Render.com에서 웹 서비스 생성

### 방법 1: 자동 배포 (render.yaml 사용)

1. **Render Dashboard** → **New** → **Web Service**
2. **Build and deploy from a Git repository** 선택
3. GitHub 저장소 연결
4. **render.yaml** 파일이 자동으로 인식됨
5. **Create Web Service** 클릭

### 방법 2: 수동 설정

1. **Render Dashboard** → **New** → **Web Service**
2. **Build and deploy from a Git repository** 선택
3. GitHub 저장소 연결
4. 설정 입력:
   - **Name**: `ai-image-detector`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`
5. **Advanced** → **Environment Variables**:
   - `FLASK_ENV`: `production`
   - `SECRET_KEY`: `your-secret-key-here`
   - `PYTORCH_CUDA_ALLOC_CONF`: `max_split_size_mb:128`
6. **Create Web Service** 클릭

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

1. **Render Dashboard**에서 생성한 서비스 클릭
2. **Manual Deploy** → **Deploy latest commit** 클릭
3. 빌드 로그 확인
4. 배포 완료 후 URL 확인

## 🔍 6단계: 배포 확인

### 1. 서비스 상태 확인
- Render Dashboard에서 서비스 상태 확인
- "Live" 상태가 되면 배포 완료

### 2. 로그 모니터링
- **Logs** 탭에서 실시간 로그 확인
- 에러 발생 시 로그에서 원인 파악

### 3. 웹사이트 접속
- Render에서 제공하는 URL로 접속
- 예: `https://ai-image-detector.onrender.com`

## 🛠️ 문제 해결

### 1. 빌드 실패
```bash
# requirements.txt 확인
pip freeze > requirements.txt

# 로그에서 구체적인 오류 확인
# Render Dashboard → Logs 탭
```

### 2. 메모리 부족
```bash
# 환경 변수 추가
PYTORCH_CUDA_ALLOC_CONF: max_split_size_mb:64

# 모델 최적화
# - 모델 크기 줄이기
# - 배치 크기 줄이기
```

### 3. 서비스 크래시
```bash
# 로그 확인
# Render Dashboard → Logs 탭

# 서비스 재시작
# Render Dashboard → Manual Deploy
```

## 📊 7단계: 성능 모니터링

### 1. Render 메트릭 확인
- **Metrics** 탭에서 CPU, 메모리 사용량 확인
- **Logs** 탭에서 응답 시간 확인

### 2. 사용자 피드백 모니터링
- 웹사이트에서 피드백 수집 현황 확인
- `/stats` 페이지에서 재학습 가능 여부 확인

## 🔄 8단계: 자동 배포 설정

### GitHub 연동
1. GitHub 저장소에 코드 푸시
2. Render에서 자동으로 배포 시작
3. 빌드 및 배포 완료 후 자동으로 새 버전 적용

### 환경 변수 관리
- **Environment** 탭에서 환경 변수 추가/수정
- 민감한 정보는 **Secret**으로 설정

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
- **750시간/월** 제한
- **512MB RAM** 제한
- **슬립 모드**: 15분 비활성 시 슬립

## 🎯 배포 체크리스트

### 배포 전 확인사항
- [ ] 훈련된 모델 파일 존재
- [ ] requirements.txt 최신화
- [ ] render.yaml 설정 완료
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
1. **로그 확인**: Render Dashboard → Logs 탭
2. **서비스 재시작**: Manual Deploy
3. **환경 변수 확인**: Environment 탭
4. **서비스 상태**: Dashboard에서 확인

### 추가 도움
- [Render 공식 문서](https://render.com/docs)
- [Flask 배포 가이드](https://flask.palletsprojects.com/en/2.0.x/deploying/)
- [PyTorch 메모리 최적화](https://pytorch.org/docs/stable/notes/cuda.html)

## 🎉 완료!

배포가 완료되면 전 세계 어디서나 AI 이미지 분류기를 사용할 수 있습니다!

**배포된 URL**: `https://your-app-name.onrender.com`

---

**Render.com으로 성공적인 무료 배포를 위해 단계별로 차근차근 진행하세요!** 🚀
