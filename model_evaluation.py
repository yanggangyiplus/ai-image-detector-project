#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI 이미지 분류기 모델 평가 스크립트
- 모델 성능 평가
- 혼동 행렬 생성
- 분류 리포트 생성
- ROC 곡선 및 AUC 계산
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from transformers import pipeline, ViTForImageClassification, ViTImageProcessor
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """AI 이미지 분류기 모델 평가 클래스"""
    
    def __init__(self, model_path='./ai_vs_real_image_detection'):
        """
        모델 평가기 초기화
        
        Args:
            model_path (str): 훈련된 모델 경로
        """
        self.model_path = model_path
        self.device = 0 if torch.cuda.is_available() else -1
        
        # 모델 로드
        print("🤖 AI 모델 로딩 중...")
        try:
            self.classifier = pipeline(
                'image-classification',
                model=model_path,
                device=self.device
            )
            self.model = ViTForImageClassification.from_pretrained(model_path)
            self.processor = ViTImageProcessor.from_pretrained(model_path)
            print(f"✅ 모델 로드 완료! (디바이스: {'GPU' if self.device == 0 else 'CPU'})")
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            raise
    
    def load_test_data(self, test_data_path='test data'):
        """
        테스트 데이터 로드
        
        Args:
            test_data_path (str): 테스트 데이터 경로
            
        Returns:
            tuple: (images, labels) 리스트
        """
        print("📁 테스트 데이터 로딩 중...")
        
        images = []
        labels = []
        
        # AI 생성 이미지 (label=1) - fake 폴더
        ai_path = Path(test_data_path) / 'fake'
        if ai_path.exists():
            for img_file in ai_path.glob('*'):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                    try:
                        image = Image.open(img_file).convert('RGB')
                        images.append(image)
                        labels.append(1)  # AI 생성
                        print(f"   AI 이미지 로드: {img_file.name}")
                    except Exception as e:
                        print(f"⚠️ 이미지 로드 실패: {img_file} - {e}")
        
        # 실제 이미지 (label=0) - real 폴더
        real_path = Path(test_data_path) / 'real'
        if real_path.exists():
            for img_file in real_path.glob('*'):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                    try:
                        image = Image.open(img_file).convert('RGB')
                        images.append(image)
                        labels.append(0)  # 실제
                        print(f"   실제 이미지 로드: {img_file.name}")
                    except Exception as e:
                        print(f"⚠️ 이미지 로드 실패: {img_file} - {e}")
        
        print(f"✅ 테스트 데이터 로드 완료: {len(images)}개 이미지")
        print(f"   - AI 생성: {labels.count(1)}개")
        print(f"   - 실제: {labels.count(0)}개")
        
        return images, labels
    
    def predict_batch(self, images, batch_size=32):
        """
        배치 단위로 예측 수행
        
        Args:
            images (list): 이미지 리스트
            batch_size (int): 배치 크기
            
        Returns:
            tuple: (predictions, probabilities)
        """
        print("🔮 모델 예측 수행 중...")
        
        all_predictions = []
        all_probabilities = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            print(f"   배치 {i//batch_size + 1}/{(len(images)-1)//batch_size + 1} 처리 중...")
            
            for image in batch_images:
                try:
                    # 예측 수행
                    result = self.classifier(image)
                    
                    # 결과 파싱
                    prediction = 1 if result[0]['label'] == 'AI_GENERATED' else 0
                    probability = result[0]['score']
                    
                    all_predictions.append(prediction)
                    all_probabilities.append(probability)
                    
                except Exception as e:
                    print(f"⚠️ 예측 실패: {e}")
                    all_predictions.append(0)  # 기본값
                    all_probabilities.append(0.5)
        
        print(f"✅ 예측 완료: {len(all_predictions)}개")
        return all_predictions, all_probabilities
    
    def calculate_metrics(self, y_true, y_pred, y_prob):
        """
        성능 지표 계산
        
        Args:
            y_true (list): 실제 레이블
            y_pred (list): 예측 레이블
            y_prob (list): 예측 확률
            
        Returns:
            dict: 성능 지표 딕셔너리
        """
        print("📊 성능 지표 계산 중...")
        
        # 기본 지표
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # ROC AUC
        try:
            roc_auc = roc_auc_score(y_true, y_prob)
        except:
            roc_auc = 0.0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'total_samples': len(y_true),
            'ai_generated_correct': sum(1 for i, (true, pred) in enumerate(zip(y_true, y_pred)) if true == 1 and pred == 1),
            'real_correct': sum(1 for i, (true, pred) in enumerate(zip(y_true, y_pred)) if true == 0 and pred == 0),
            'ai_generated_total': sum(y_true),
            'real_total': len(y_true) - sum(y_true)
        }
        
        return metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path='results/confusion_matrix.png'):
        """
        혼동 행렬 시각화
        
        Args:
            y_true (list): 실제 레이블
            y_pred (list): 예측 레이블
            save_path (str): 저장 경로
        """
        print("📈 혼동 행렬 생성 중...")
        
        # 혼동 행렬 계산
        cm = confusion_matrix(y_true, y_pred)
        
        # 시각화
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['실제', 'AI 생성'],
                   yticklabels=['실제', 'AI 생성'])
        plt.title('혼동 행렬 (Confusion Matrix)')
        plt.xlabel('예측 레이블')
        plt.ylabel('실제 레이블')
        
        # 저장
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 혼동 행렬 저장: {save_path}")
    
    def plot_roc_curve(self, y_true, y_prob, save_path='results/roc_curve.png'):
        """
        ROC 곡선 시각화
        
        Args:
            y_true (list): 실제 레이블
            y_prob (list): 예측 확률
            save_path (str): 저장 경로
        """
        print("📈 ROC 곡선 생성 중...")
        
        try:
            # ROC 곡선 계산
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = roc_auc_score(y_true, y_prob)
            
            # 시각화
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC 곡선 (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                    label='무작위 분류기')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC 곡선 (Receiver Operating Characteristic)')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            
            # 저장
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ ROC 곡선 저장: {save_path}")
            
        except Exception as e:
            print(f"⚠️ ROC 곡선 생성 실패: {e}")
    
    def generate_classification_report(self, y_true, y_pred, save_path='results/classification_report.txt'):
        """
        분류 리포트 생성
        
        Args:
            y_true (list): 실제 레이블
            y_pred (list): 예측 레이블
            save_path (str): 저장 경로
        """
        print("📋 분류 리포트 생성 중...")
        
        # 분류 리포트 생성
        report = classification_report(y_true, y_pred, 
                                    target_names=['실제', 'AI 생성'],
                                    output_dict=True)
        
        # 텍스트 리포트
        text_report = classification_report(y_true, y_pred, 
                                          target_names=['실제', 'AI 생성'])
        
        # 저장
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("AI 이미지 분류기 모델 평가 리포트\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"평가 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"총 샘플 수: {len(y_true)}\n\n")
            f.write(text_report)
        
        print(f"✅ 분류 리포트 저장: {save_path}")
        return report
    
    def save_evaluation_results(self, metrics, save_path='results/evaluation_results.json'):
        """
        평가 결과 저장
        
        Args:
            metrics (dict): 성능 지표
            save_path (str): 저장 경로
        """
        print("💾 평가 결과 저장 중...")
        
        # 결과에 메타데이터 추가
        results = {
            'evaluation_time': datetime.now().isoformat(),
            'model_path': self.model_path,
            'device': 'GPU' if self.device == 0 else 'CPU',
            'metrics': metrics
        }
        
        # 저장
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 평가 결과 저장: {save_path}")
    
    def print_summary(self, metrics):
        """
        평가 결과 요약 출력
        
        Args:
            metrics (dict): 성능 지표
        """
        print("\n" + "="*60)
        print("🎯 AI 이미지 분류기 모델 평가 결과")
        print("="*60)
        
        print(f"📊 전체 성능:")
        print(f"   정확도 (Accuracy): {metrics['accuracy']:.3f}")
        print(f"   정밀도 (Precision): {metrics['precision']:.3f}")
        print(f"   재현율 (Recall): {metrics['recall']:.3f}")
        print(f"   F1 점수: {metrics['f1_score']:.3f}")
        print(f"   ROC AUC: {metrics['roc_auc']:.3f}")
        
        print(f"\n📈 클래스별 성능:")
        print(f"   AI 생성 이미지:")
        print(f"     - 정확도: {metrics['ai_generated_correct']}/{metrics['ai_generated_total']} ({metrics['ai_generated_correct']/metrics['ai_generated_total']:.3f})")
        print(f"   실제 이미지:")
        print(f"     - 정확도: {metrics['real_correct']}/{metrics['real_total']} ({metrics['real_correct']/metrics['real_total']:.3f})")
        
        print(f"\n📋 총 샘플 수: {metrics['total_samples']}")
        print("="*60)
    
    def evaluate_model(self, test_data_path='data/test', output_dir='results'):
        """
        전체 모델 평가 수행
        
        Args:
            test_data_path (str): 테스트 데이터 경로
            output_dir (str): 결과 저장 디렉토리
        """
        print("🚀 AI 이미지 분류기 모델 평가 시작!")
        print("="*60)
        
        try:
            # 1. 테스트 데이터 로드
            images, labels = self.load_test_data(test_data_path)
            
            if len(images) == 0:
                print("❌ 테스트 데이터가 없습니다.")
                return
            
            # 2. 예측 수행
            predictions, probabilities = self.predict_batch(images)
            
            # 3. 성능 지표 계산
            metrics = self.calculate_metrics(labels, predictions, probabilities)
            
            # 4. 시각화 생성
            self.plot_confusion_matrix(labels, predictions, 
                                    f'{output_dir}/confusion_matrix.png')
            self.plot_roc_curve(labels, probabilities, 
                              f'{output_dir}/roc_curve.png')
            
            # 5. 리포트 생성
            self.generate_classification_report(labels, predictions, 
                                              f'{output_dir}/classification_report.txt')
            
            # 6. 결과 저장
            self.save_evaluation_results(metrics, f'{output_dir}/evaluation_results.json')
            
            # 7. 결과 출력
            self.print_summary(metrics)
            
            print(f"\n🎉 모델 평가 완료! 결과는 '{output_dir}' 폴더에 저장되었습니다.")
            
        except Exception as e:
            print(f"❌ 모델 평가 중 오류 발생: {e}")
            raise

def main():
    """메인 함수"""
    print("🤖 AI 이미지 분류기 모델 평가 스크립트")
    print("="*60)
    
    # 모델 평가기 초기화
    evaluator = ModelEvaluator()
    
    # 모델 평가 수행
    evaluator.evaluate_model(
        test_data_path='test data',
        output_dir='results'
    )

if __name__ == "__main__":
    main()
