# -*- coding: utf-8 -*-
"""
AI 이미지 분류 웹 애플리케이션
사용자가 이미지를 업로드하여 AI 생성 이미지인지 실제 이미지인지 판별하는 웹사이트
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import uuid
from datetime import datetime
import json
from werkzeug.utils import secure_filename
import torch
from transformers import pipeline, ViTForImageClassification, ViTImageProcessor
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import threading
import time
from pathlib import Path
from datetime import timedelta
from model_retrain import ModelRetrainer

# Flask 앱 초기화
app = Flask(__name__)
app.config['SECRET_KEY'] = 'ai-image-detector-2024'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB 최대 파일 크기

# 허용된 파일 확장자
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

# 업로드 폴더 생성
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/results', exist_ok=True)
os.makedirs('data/feedback', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# 전역 변수
retrainer = None
retraining_status = {'status': 'idle', 'progress': 0, 'message': ''}

# AI 모델 로드
print("🤖 AI 모델 로딩 중...")
try:
    # 메모리 최적화 설정
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    # 훈련된 모델 로드
    model_path = './ai_vs_real_image_detection'
    device = 0 if torch.cuda.is_available() else -1
    
    # 파이프라인 생성 (메모리 최적화)
    classifier = pipeline(
        'image-classification',
        model=model_path,
        device=device,
        torch_dtype=torch.float16 if device == 0 else torch.float32
    )
    
    # 개별 모델과 프로세서도 로드 (상세 분석용)
    model = ViTForImageClassification.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == 0 else torch.float32
    )
    processor = ViTImageProcessor.from_pretrained(model_path)
    
    print(f"✅ AI 모델 로드 완료! (디바이스: {'GPU' if device == 0 else 'CPU'})")
    
except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")
    classifier = None
    model = None
    processor = None

def allowed_file(filename):
    """허용된 파일 확장자인지 확인"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_feedback(image_path, prediction, confidence, user_feedback, correct_label):
    """사용자 피드백을 저장"""
    feedback_data = {
        'timestamp': datetime.now().isoformat(),
        'image_path': image_path,
        'prediction': prediction,
        'confidence': confidence,
        'user_feedback': user_feedback,
        'correct_label': correct_label
    }
    
    feedback_file = f"data/feedback/feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.json"
    
    with open(feedback_file, 'w', encoding='utf-8') as f:
        json.dump(feedback_data, f, ensure_ascii=False, indent=2)
    
    return feedback_file

def analyze_image_features(image_path):
    """이미지의 특징을 분석하여 설명 생성"""
    try:
        image = Image.open(image_path).convert('RGB')
        
        # 이미지 기본 정보
        width, height = image.size
        aspect_ratio = width / height
        
        # 이미지를 numpy 배열로 변환
        img_array = np.array(image)
        
        # 기본 통계
        mean_brightness = np.mean(img_array)
        std_brightness = np.std(img_array)
        
        # 색상 분석
        r_mean, g_mean, b_mean = np.mean(img_array, axis=(0, 1))
        
        # 특징 기반 설명 생성
        features = {
            'size': f"{width}x{height}",
            'aspect_ratio': round(aspect_ratio, 2),
            'brightness': round(mean_brightness, 1),
            'contrast': round(std_brightness, 1),
            'dominant_colors': {
                'red': round(r_mean, 1),
                'green': round(g_mean, 1),
                'blue': round(b_mean, 1)
            }
        }
        
        return features
        
    except Exception as e:
        print(f"이미지 분석 오류: {e}")
        return None

def generate_explanation(prediction, confidence, features):
    """예측 결과에 대한 설명 생성"""
    if prediction == 'REAL':
        explanation = f"이 이미지는 실제 사진으로 판단됩니다 (신뢰도: {confidence:.1%}). "
        
        if features:
            if features['contrast'] > 50:
                explanation += "높은 대비와 자연스러운 색상 분포가 실제 사진의 특징을 보여줍니다. "
            if features['brightness'] > 100:
                explanation += "적절한 밝기와 자연스러운 조명이 실제 환경에서 촬영된 것으로 보입니다. "
            
            explanation += f"이미지 크기는 {features['size']}이며, 종횡비는 {features['aspect_ratio']}입니다."
    else:
        explanation = f"이 이미지는 AI가 생성한 것으로 판단됩니다 (신뢰도: {confidence:.1%}). "
        
        if features:
            if features['contrast'] < 30:
                explanation += "낮은 대비와 부자연스러운 색상 분포가 AI 생성 이미지의 특징을 보여줍니다. "
            if features['brightness'] < 80 or features['brightness'] > 200:
                explanation += "부자연스러운 밝기나 조명이 AI 생성 이미지의 특징입니다. "
            
            explanation += f"이미지 크기는 {features['size']}이며, 종횡비는 {features['aspect_ratio']}입니다."
    
    return explanation

@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """이미지 업로드 및 분석"""
    if 'file' not in request.files:
        return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': '지원되지 않는 파일 형식입니다. (PNG, JPG, JPEG, GIF, BMP, TIFF만 지원)'}), 400
    
    if classifier is None:
        return jsonify({'error': 'AI 모델이 로드되지 않았습니다.'}), 500
    
    try:
        # 파일 저장
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # 이미지 분석
        result = classifier(filepath)
        
        # 결과 처리
        prediction = result[0]['label']
        confidence = result[0]['score']
        
        # 이미지 특징 분석
        features = analyze_image_features(filepath)
        
        # 설명 생성
        explanation = generate_explanation(prediction, confidence, features)
        
        # 결과 저장
        result_data = {
            'filename': unique_filename,
            'original_filename': filename,
            'prediction': prediction,
            'confidence': confidence,
            'explanation': explanation,
            'features': features,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify({
            'success': True,
            'result': result_data
        })
        
    except Exception as e:
        return jsonify({'error': f'분석 중 오류가 발생했습니다: {str(e)}'}), 500

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    """사용자 피드백 수집"""
    try:
        data = request.json
        image_path = data.get('image_path')
        prediction = data.get('prediction')
        confidence = data.get('confidence')
        user_feedback = data.get('user_feedback')  # 'correct' 또는 'incorrect'
        correct_label = data.get('correct_label')  # 'REAL' 또는 'FAKE'
        
        if not all([image_path, prediction, confidence, user_feedback, correct_label]):
            return jsonify({'error': '필수 정보가 누락되었습니다.'}), 400
        
        # 피드백 저장
        feedback_file = save_feedback(image_path, prediction, confidence, user_feedback, correct_label)
        
        return jsonify({
            'success': True,
            'message': '피드백이 저장되었습니다. 감사합니다!',
            'feedback_file': feedback_file
        })
        
    except Exception as e:
        return jsonify({'error': f'피드백 저장 중 오류가 발생했습니다: {str(e)}'}), 500

@app.route('/about')
def about():
    """소개 페이지"""
    return render_template('about.html')

@app.route('/stats')
def stats():
    """통계 페이지"""
    # 피드백 데이터 수집
    feedback_files = []
    feedback_dir = 'data/feedback'
    
    if os.path.exists(feedback_dir):
        for file in os.listdir(feedback_dir):
            if file.endswith('.json'):
                feedback_files.append(os.path.join(feedback_dir, file))
    
    # 통계 계산
    total_feedback = len(feedback_files)
    correct_predictions = 0
    incorrect_predictions = 0
    
    for file_path in feedback_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if data.get('user_feedback') == 'correct':
                    correct_predictions += 1
                else:
                    incorrect_predictions += 1
        except:
            continue
    
    accuracy = (correct_predictions / total_feedback * 100) if total_feedback > 0 else 0
    
    stats_data = {
        'total_feedback': total_feedback,
        'correct_predictions': correct_predictions,
        'incorrect_predictions': incorrect_predictions,
        'accuracy': round(accuracy, 1)
    }
    
    return render_template('stats.html', stats=stats_data)

@app.route('/retrain', methods=['POST'])
def start_retraining():
    """모델 재학습 시작"""
    global retrainer, retraining_status
    
    try:
        if retraining_status['status'] == 'running':
            return jsonify({'error': '재학습이 이미 진행 중입니다.'}), 400
        
        # 피드백 데이터 확인
        feedback_files = list(Path('data/feedback').glob('*.json'))
        if len(feedback_files) < 50:
            return jsonify({
                'error': f'피드백 데이터가 부족합니다. (현재: {len(feedback_files)}/50)',
                'required': 50,
                'current': len(feedback_files)
            }), 400
        
        # 재학습 상태 초기화
        retraining_status = {
            'status': 'running',
            'progress': 0,
            'message': '재학습을 시작합니다...'
        }
        
        # 백그라운드에서 재학습 실행
        def run_retraining():
            global retraining_status
            try:
                retraining_status['message'] = '피드백 데이터를 수집하고 있습니다...'
                retraining_status['progress'] = 10
                
                retrainer = ModelRetrainer()
                
                retraining_status['message'] = '훈련 데이터를 준비하고 있습니다...'
                retraining_status['progress'] = 20
                
                feedback_data = retrainer.collect_feedback_data()
                if feedback_data is None:
                    retraining_status = {
                        'status': 'failed',
                        'progress': 0,
                        'message': '피드백 데이터가 부족합니다.'
                    }
                    return
                
                retraining_status['message'] = '모델을 재학습하고 있습니다...'
                retraining_status['progress'] = 40
                
                training_data = retrainer.prepare_training_data(feedback_data)
                if training_data is None:
                    retraining_status = {
                        'status': 'failed',
                        'progress': 0,
                        'message': '훈련 데이터 준비에 실패했습니다.'
                    }
                    return
                
                retraining_status['progress'] = 60
                
                if not retrainer.retrain_model(training_data):
                    retraining_status = {
                        'status': 'failed',
                        'progress': 0,
                        'message': '모델 재학습에 실패했습니다.'
                    }
                    return
                
                retraining_status['message'] = '모델 성능을 평가하고 있습니다...'
                retraining_status['progress'] = 80
                
                performance = retrainer.evaluate_model()
                
                retraining_status['message'] = '모델을 업데이트하고 있습니다...'
                retraining_status['progress'] = 90
                
                if retrainer.update_model():
                    retraining_status = {
                        'status': 'completed',
                        'progress': 100,
                        'message': '모델 재학습이 완료되었습니다!',
                        'performance': performance
                    }
                else:
                    retraining_status = {
                        'status': 'failed',
                        'progress': 0,
                        'message': '모델 업데이트에 실패했습니다.'
                    }
                    
            except Exception as e:
                retraining_status = {
                    'status': 'failed',
                    'progress': 0,
                    'message': f'재학습 중 오류가 발생했습니다: {str(e)}'
                }
        
        # 백그라운드 스레드에서 재학습 실행
        thread = threading.Thread(target=run_retraining)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'message': '재학습을 시작했습니다.',
            'status': retraining_status
        })
        
    except Exception as e:
        return jsonify({'error': f'재학습 시작 중 오류가 발생했습니다: {str(e)}'}), 500

@app.route('/retrain/status')
def get_retraining_status():
    """재학습 상태 확인"""
    global retraining_status
    return jsonify(retraining_status)

@app.route('/feedback/stats')
def get_feedback_stats():
    """피드백 통계 정보"""
    try:
        feedback_files = list(Path('data/feedback').glob('*.json'))
        
        total_feedback = len(feedback_files)
        incorrect_feedback = 0
        recent_feedback = 0
        
        # 최근 7일간의 피드백
        cutoff_date = datetime.now() - timedelta(days=7)
        
        for file_path in feedback_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                if data.get('user_feedback') == 'incorrect':
                    incorrect_feedback += 1
                
                feedback_date = datetime.fromisoformat(data['timestamp'])
                if feedback_date >= cutoff_date:
                    recent_feedback += 1
                    
            except:
                continue
        
        stats = {
            'total_feedback': total_feedback,
            'incorrect_feedback': incorrect_feedback,
            'recent_feedback': recent_feedback,
            'can_retrain': total_feedback >= 50 and incorrect_feedback >= 10
        }
        
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({'error': f'통계 조회 중 오류가 발생했습니다: {str(e)}'}), 500

if __name__ == '__main__':
    print("🚀 AI 이미지 분류 웹사이트 시작 중...")
    
    # Heroku 배포를 위한 포트 설정
    port = int(os.environ.get('PORT', 8080))
    debug_mode = os.environ.get('FLASK_ENV') != 'production'
    
    if debug_mode:
        print("📱 브라우저에서 http://localhost:8080 으로 접속하세요!")
        app.run(debug=True, host='127.0.0.1', port=port)
    else:
        print(f"🌐 Heroku에서 포트 {port}로 실행 중...")
        app.run(debug=False, host='0.0.0.0', port=port)
