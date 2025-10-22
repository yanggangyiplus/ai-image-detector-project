/**
 * AI 이미지 분류기 - 메인 JavaScript
 * 웹 애플리케이션의 주요 기능을 담당하는 JavaScript 파일
 */

// 전역 변수
let currentFile = null;
let currentResult = null;
let correctLabel = null;

// DOM이 로드되면 실행
document.addEventListener('DOMContentLoaded', function () {
    initializeApp();
});

/**
 * 애플리케이션 초기화
 */
function initializeApp() {
    console.log('🚀 AI 이미지 분류기 초기화 중...');

    // 이벤트 리스너 등록
    setupEventListeners();

    // 애니메이션 효과 추가
    addAnimationEffects();

    console.log('✅ 애플리케이션 초기화 완료');
}

/**
 * 이벤트 리스너 설정
 */
function setupEventListeners() {
    // 파일 입력 이벤트
    const fileInput = document.getElementById('file-input');
    if (fileInput) {
        fileInput.addEventListener('change', handleFileSelect);
    }

    // 파일 선택 버튼 이벤트
    const fileSelectBtn = document.getElementById('file-select-btn');
    if (fileSelectBtn) {
        fileSelectBtn.addEventListener('click', () => {
            if (fileInput) fileInput.click();
        });
    }

    // 업로드 영역 이벤트 (드래그앤드롭만)
    const uploadArea = document.getElementById('upload-area');
    if (uploadArea) {
        uploadArea.addEventListener('dragover', handleDragOver);
        uploadArea.addEventListener('dragleave', handleDragLeave);
        uploadArea.addEventListener('drop', handleDrop);

        // 업로드 영역 클릭 이벤트 제거 (파일 선택 버튼으로만 파일 선택)
        // 이제 업로드 영역을 클릭해도 파일 선택 다이얼로그가 열리지 않음
    }

    // 분석 버튼 이벤트
    const analyzeBtn = document.getElementById('analyze-btn');
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', analyzeImage);
    }

    // 취소 버튼 이벤트
    const cancelBtn = document.getElementById('cancel-btn');
    if (cancelBtn) {
        cancelBtn.addEventListener('click', cancelUpload);
    }
}

/**
 * 드래그 오버 이벤트 처리
 */
function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.classList.add('drag-over');
}

/**
 * 드래그 리브 이벤트 처리
 */
function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.classList.remove('drag-over');
}

/**
 * 드롭 이벤트 처리
 */
function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.classList.remove('drag-over');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

/**
 * 파일 선택 이벤트 처리
 */
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

/**
 * 파일 처리
 * @param {File} file - 업로드된 파일
 */
function handleFile(file) {
    console.log('📁 파일 처리 시작:', file.name);

    // 파일 유효성 검사
    if (!validateFile(file)) {
        return;
    }

    currentFile = file;

    // 파일 미리보기 표시
    showFilePreview(file);

    console.log('✅ 파일 처리 완료');
}

/**
 * 파일 유효성 검사
 * @param {File} file - 검사할 파일
 * @returns {boolean} - 유효성 여부
 */
function validateFile(file) {
    // 파일 타입 검사
    const allowedTypes = [
        'image/png',
        'image/jpeg',
        'image/jpg',
        'image/gif',
        'image/bmp',
        'image/tiff'
    ];

    if (!allowedTypes.includes(file.type)) {
        showAlert('지원되지 않는 파일 형식입니다.', 'error');
        return false;
    }

    // 파일 크기 검사 (16MB)
    const maxSize = 16 * 1024 * 1024;
    if (file.size > maxSize) {
        showAlert('파일 크기가 너무 큽니다. 16MB 이하의 파일만 업로드 가능합니다.', 'error');
        return false;
    }

    return true;
}

/**
 * 파일 미리보기 표시
 * @param {File} file - 미리보기할 파일
 */
function showFilePreview(file) {
    const reader = new FileReader();

    reader.onload = function (e) {
        const previewImage = document.getElementById('preview-image');
        const uploadContent = document.getElementById('upload-content');
        const filePreview = document.getElementById('file-preview');

        if (previewImage) {
            previewImage.src = e.target.result;
        }

        if (uploadContent) {
            uploadContent.classList.add('d-none');
        }

        if (filePreview) {
            filePreview.classList.remove('d-none');
        }

        // 애니메이션 효과
        if (filePreview) {
            filePreview.classList.add('fade-in');
        }
    };

    reader.readAsDataURL(file);
}

/**
 * 업로드 취소
 */
function cancelUpload() {
    console.log('❌ 업로드 취소');

    currentFile = null;
    correctLabel = null;

    // 파일 입력 초기화
    const fileInput = document.getElementById('file-input');
    if (fileInput) {
        fileInput.value = '';
    }

    // UI 초기화
    const uploadContent = document.getElementById('upload-content');
    const filePreview = document.getElementById('file-preview');
    const resultsSection = document.getElementById('results-section');

    if (uploadContent) {
        uploadContent.classList.remove('d-none');
    }

    if (filePreview) {
        filePreview.classList.add('d-none');
    }

    if (resultsSection) {
        resultsSection.classList.add('d-none');
    }

    // 피드백 폼 초기화
    resetFeedbackForm();
}

/**
 * 이미지 분석
 */
async function analyzeImage() {
    if (!currentFile) {
        showAlert('파일을 선택해주세요.', 'error');
        return;
    }

    console.log('🔍 이미지 분석 시작');

    // 로딩 상태 표시
    showLoading(true);

    try {
        // FormData 생성
        const formData = new FormData();
        formData.append('file', currentFile);

        // 서버에 요청 전송
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            currentResult = data.result;
            displayResults(data.result);
            console.log('✅ 이미지 분석 완료');
        } else {
            throw new Error(data.error || '분석 중 오류가 발생했습니다.');
        }

    } catch (error) {
        console.error('❌ 이미지 분석 오류:', error);
        showAlert('분석 중 오류가 발생했습니다: ' + error.message, 'error');
        showFilePreview(currentFile);
    } finally {
        showLoading(false);
    }
}

/**
 * 로딩 상태 표시/숨김
 * @param {boolean} show - 로딩 표시 여부
 */
function showLoading(show) {
    const filePreview = document.getElementById('file-preview');
    const loading = document.getElementById('loading');

    if (show) {
        if (filePreview) {
            filePreview.classList.add('d-none');
        }
        if (loading) {
            loading.classList.remove('d-none');
        }
    } else {
        if (loading) {
            loading.classList.add('d-none');
        }
    }
}

/**
 * 분석 결과 표시
 * @param {Object} result - 분석 결과
 */
function displayResults(result) {
    console.log('📊 결과 표시:', result);

    // 이미지 표시
    const resultImage = document.getElementById('result-image');
    if (resultImage) {
        resultImage.src = '/static/uploads/' + result.filename;
    }

    // 예측 결과 표시
    const predictionText = result.prediction === 'REAL' ? '실제 사진' : 'AI 생성 이미지';
    const confidencePercent = (result.confidence * 100).toFixed(1);

    const predictionElement = document.getElementById('prediction-text');
    const confidenceBadge = document.getElementById('confidence-badge');

    if (predictionElement) {
        predictionElement.textContent = predictionText;
    }

    if (confidenceBadge) {
        confidenceBadge.textContent = confidencePercent + '%';
    }

    // 설명 표시
    const explanationElement = document.getElementById('explanation-text');
    if (explanationElement) {
        explanationElement.textContent = result.explanation;
    }

    // 특징 표시
    if (result.features) {
        displayFeatures(result.features);
    }

    // 결과 섹션 표시
    const resultsSection = document.getElementById('results-section');
    if (resultsSection) {
        resultsSection.classList.remove('d-none');
        resultsSection.classList.add('fade-in');
    }

    // 피드백 폼 초기화
    resetFeedbackForm();
}

/**
 * 이미지 특징 표시
 * @param {Object} features - 이미지 특징
 */
function displayFeatures(features) {
    const featuresList = document.getElementById('features-list');
    if (!featuresList) return;

    featuresList.innerHTML = `
        <div class="col-6">
            <small class="text-muted">
                <i class="fas fa-expand-arrows-alt me-1"></i>
                크기: ${features.size}
            </small>
        </div>
        <div class="col-6">
            <small class="text-muted">
                <i class="fas fa-ratio me-1"></i>
                종횡비: ${features.aspect_ratio}
            </small>
        </div>
        <div class="col-6">
            <small class="text-muted">
                <i class="fas fa-sun me-1"></i>
                밝기: ${features.brightness}
            </small>
        </div>
        <div class="col-6">
            <small class="text-muted">
                <i class="fas fa-adjust me-1"></i>
                대비: ${features.contrast}
            </small>
        </div>
    `;
}

/**
 * 피드백 제출
 * @param {string} type - 피드백 타입 ('correct' 또는 'incorrect')
 */
function submitFeedback(type) {
    console.log('💬 피드백 제출:', type);

    if (type === 'correct') {
        sendFeedbackData('correct', currentResult.prediction);
    } else {
        showIncorrectFeedbackForm();
    }
}

/**
 * 부정확한 피드백 폼 표시
 */
function showIncorrectFeedbackForm() {
    const incorrectFeedback = document.getElementById('incorrect-feedback');
    if (incorrectFeedback) {
        incorrectFeedback.classList.remove('d-none');
        incorrectFeedback.classList.add('fade-in');
    }
}

/**
 * 정답 라벨 설정
 * @param {string} label - 정답 라벨 ('REAL' 또는 'FAKE')
 */
function setCorrectLabel(label) {
    correctLabel = label;
    console.log('🏷️ 정답 라벨 설정:', label);

    // 버튼 스타일 업데이트
    const buttons = document.querySelectorAll('#incorrect-feedback button');
    buttons.forEach(btn => {
        btn.classList.remove('btn-primary');
        btn.classList.add('btn-outline-primary', 'btn-outline-warning');
    });

    const selectedBtn = event.target;
    selectedBtn.classList.remove('btn-outline-primary', 'btn-outline-warning');
    selectedBtn.classList.add('btn-primary');
}

/**
 * 피드백 전송
 */
function sendFeedback() {
    if (!correctLabel) {
        showAlert('실제 정답을 선택해주세요.', 'warning');
        return;
    }

    sendFeedbackData('incorrect', correctLabel);
}

/**
 * 피드백 데이터 전송
 * @param {string} userFeedback - 사용자 피드백
 * @param {string} correctLabel - 정답 라벨
 */
async function sendFeedbackData(userFeedback, correctLabel) {
    if (!currentResult) {
        showAlert('분석 결과가 없습니다.', 'error');
        return;
    }

    const feedbackData = {
        image_path: currentResult.filename,
        prediction: currentResult.prediction,
        confidence: currentResult.confidence,
        user_feedback: userFeedback,
        correct_label: correctLabel
    };

    console.log('📤 피드백 전송:', feedbackData);

    try {
        const response = await fetch('/feedback', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(feedbackData)
        });

        const data = await response.json();

        if (data.success) {
            showFeedbackSuccess();
            console.log('✅ 피드백 전송 완료');
        } else {
            throw new Error(data.error || '피드백 전송 중 오류가 발생했습니다.');
        }

    } catch (error) {
        console.error('❌ 피드백 전송 오류:', error);
        showAlert('피드백 전송 중 오류가 발생했습니다: ' + error.message, 'error');
    }
}

/**
 * 피드백 성공 메시지 표시
 */
function showFeedbackSuccess() {
    const incorrectFeedback = document.getElementById('incorrect-feedback');
    const feedbackSuccess = document.getElementById('feedback-success');

    if (incorrectFeedback) {
        incorrectFeedback.classList.add('d-none');
    }

    if (feedbackSuccess) {
        feedbackSuccess.classList.remove('d-none');
        feedbackSuccess.classList.add('fade-in');
    }
}

/**
 * 피드백 폼 초기화
 */
function resetFeedbackForm() {
    const incorrectFeedback = document.getElementById('incorrect-feedback');
    const feedbackSuccess = document.getElementById('feedback-success');

    if (incorrectFeedback) {
        incorrectFeedback.classList.add('d-none');
    }

    if (feedbackSuccess) {
        feedbackSuccess.classList.add('d-none');
    }

    correctLabel = null;
}

/**
 * 알림 메시지 표시
 * @param {string} message - 메시지 내용
 * @param {string} type - 메시지 타입 ('success', 'error', 'warning', 'info')
 */
function showAlert(message, type = 'info') {
    // 기존 알림 제거
    const existingAlert = document.querySelector('.custom-alert');
    if (existingAlert) {
        existingAlert.remove();
    }

    // 알림 요소 생성
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type === 'error' ? 'danger' : type} custom-alert`;
    alertDiv.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 9999;
        min-width: 300px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    `;

    alertDiv.innerHTML = `
        <i class="fas fa-${getAlertIcon(type)} me-2"></i>
        ${message}
        <button type="button" class="btn-close" onclick="this.parentElement.remove()"></button>
    `;

    // 알림 표시
    document.body.appendChild(alertDiv);

    // 자동 제거 (5초 후)
    setTimeout(() => {
        if (alertDiv.parentElement) {
            alertDiv.remove();
        }
    }, 5000);
}

/**
 * 알림 아이콘 가져오기
 * @param {string} type - 알림 타입
 * @returns {string} - 아이콘 클래스
 */
function getAlertIcon(type) {
    const icons = {
        'success': 'check-circle',
        'error': 'exclamation-circle',
        'warning': 'exclamation-triangle',
        'info': 'info-circle'
    };
    return icons[type] || 'info-circle';
}

/**
 * 애니메이션 효과 추가
 */
function addAnimationEffects() {
    // 스크롤 애니메이션
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
            }
        });
    }, observerOptions);

    // 애니메이션 대상 요소들 관찰
    const animatedElements = document.querySelectorAll('.card, .feature-card, .stat-card');
    animatedElements.forEach(el => {
        observer.observe(el);
    });
}

/**
 * 유틸리티 함수들
 */

// 파일 크기 포맷팅
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// 시간 포맷팅
function formatTime(timestamp) {
    const date = new Date(timestamp);
    return date.toLocaleString('ko-KR');
}

// 숫자 포맷팅
function formatNumber(num) {
    return num.toLocaleString('ko-KR');
}

// 전역 함수로 등록
window.submitFeedback = submitFeedback;
window.setCorrectLabel = setCorrectLabel;
window.sendFeedback = sendFeedback;

console.log('📱 AI 이미지 분류기 JavaScript 로드 완료');
