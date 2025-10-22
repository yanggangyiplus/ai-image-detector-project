# -*- coding: utf-8 -*-
"""
피드백 기반 모델 재학습 시스템
사용자 피드백을 수집하여 모델을 지속적으로 개선하는 모듈
"""

import os
import json
import torch
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from transformers import (
    ViTForImageClassification, 
    ViTImageProcessor, 
    TrainingArguments, 
    Trainer,
    DefaultDataCollator
)
from datasets import Dataset, Image, ClassLabel
from PIL import Image as PILImage
from sklearn.metrics import accuracy_score, f1_score
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelRetrainer:
    """모델 재학습 클래스"""
    
    def __init__(self, model_path='./ai_vs_real_image_detection', feedback_dir='data/feedback'):
        self.model_path = model_path
        self.feedback_dir = feedback_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 모델과 프로세서 로드
        self.load_model()
        
    def load_model(self):
        """기존 모델 로드"""
        try:
            self.model = ViTForImageClassification.from_pretrained(self.model_path)
            self.processor = ViTImageProcessor.from_pretrained(self.model_path)
            logger.info(f"✅ 모델 로드 완료: {self.model_path}")
        except Exception as e:
            logger.error(f"❌ 모델 로드 실패: {e}")
            raise
    
    def collect_feedback_data(self, min_feedback_count=50):
        """피드백 데이터 수집"""
        feedback_files = list(Path(self.feedback_dir).glob('*.json'))
        
        if len(feedback_files) < min_feedback_count:
            logger.info(f"피드백 데이터 부족: {len(feedback_files)}/{min_feedback_count}")
            return None
        
        # 최근 30일간의 피드백만 수집
        cutoff_date = datetime.now() - timedelta(days=30)
        
        feedback_data = []
        for file_path in feedback_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # 날짜 확인
                feedback_date = datetime.fromisoformat(data['timestamp'])
                if feedback_date < cutoff_date:
                    continue
                
                # 부정확한 피드백만 수집 (재학습 대상)
                if data.get('user_feedback') == 'incorrect':
                    feedback_data.append(data)
                    
            except Exception as e:
                logger.warning(f"피드백 파일 읽기 실패: {file_path}, {e}")
                continue
        
        logger.info(f"수집된 피드백 데이터: {len(feedback_data)}개")
        return feedback_data if len(feedback_data) >= 10 else None
    
    def prepare_training_data(self, feedback_data):
        """재학습용 데이터 준비"""
        images = []
        labels = []
        
        for data in feedback_data:
            try:
                # 이미지 경로에서 실제 이미지 로드
                image_path = data['image_path']
                if not os.path.exists(image_path):
                    continue
                
                # 이미지 로드 및 전처리
                image = PILImage.open(image_path).convert('RGB')
                images.append(image)
                
                # 정답 라벨 설정
                correct_label = data['correct_label']
                label = 0 if correct_label == 'REAL' else 1
                labels.append(label)
                
            except Exception as e:
                logger.warning(f"이미지 처리 실패: {image_path}, {e}")
                continue
        
        if len(images) < 10:
            logger.warning("재학습용 데이터가 부족합니다.")
            return None
        
        # 데이터셋 생성
        dataset = Dataset.from_dict({
            'image': images,
            'label': labels
        })
        
        # 클래스 라벨 설정
        dataset = dataset.cast_column('image', Image())
        dataset = dataset.cast_column('label', ClassLabel(names=['REAL', 'FAKE']))
        
        return dataset
    
    def retrain_model(self, training_data, num_epochs=3):
        """모델 재학습"""
        try:
            logger.info("🔄 모델 재학습 시작...")
            
            # 훈련 인수 설정
            training_args = TrainingArguments(
                output_dir='./retrained_model',
                num_train_epochs=num_epochs,
                per_device_train_batch_size=4,
                per_device_eval_batch_size=4,
                warmup_steps=100,
                weight_decay=0.01,
                logging_dir='./logs',
                logging_steps=10,
                save_steps=500,
                evaluation_strategy="no",
                save_total_limit=2,
                load_best_model_at_end=False,
                report_to=None,  # wandb 비활성화
            )
            
            # 데이터 콜레이터
            data_collator = DefaultDataCollator()
            
            # 트레이너 생성
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=training_data,
                data_collator=data_collator,
                tokenizer=self.processor,
            )
            
            # 재학습 실행
            trainer.train()
            
            # 모델 저장
            trainer.save_model()
            self.processor.save_pretrained('./retrained_model')
            
            logger.info("✅ 모델 재학습 완료!")
            return True
            
        except Exception as e:
            logger.error(f"❌ 모델 재학습 실패: {e}")
            return False
    
    def evaluate_model(self, test_data=None):
        """모델 성능 평가"""
        try:
            if test_data is None:
                # 기본 테스트 데이터 사용
                test_data = self.prepare_test_data()
            
            if test_data is None:
                logger.warning("평가용 데이터가 없습니다.")
                return None
            
            # 예측 수행
            predictions = []
            true_labels = []
            
            for item in test_data:
                try:
                    # 이미지 전처리
                    inputs = self.processor(item['image'], return_tensors="pt")
                    
                    # 예측
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        prediction = torch.argmax(outputs.logits, dim=-1).item()
                        predictions.append(prediction)
                        true_labels.append(item['label'])
                        
                except Exception as e:
                    logger.warning(f"예측 실패: {e}")
                    continue
            
            if len(predictions) == 0:
                return None
            
            # 성능 지표 계산
            accuracy = accuracy_score(true_labels, predictions)
            f1 = f1_score(true_labels, predictions, average='weighted')
            
            results = {
                'accuracy': accuracy,
                'f1_score': f1,
                'total_samples': len(predictions)
            }
            
            logger.info(f"모델 성능 - 정확도: {accuracy:.3f}, F1: {f1:.3f}")
            return results
            
        except Exception as e:
            logger.error(f"모델 평가 실패: {e}")
            return None
    
    def prepare_test_data(self):
        """테스트 데이터 준비 (기존 훈련 데이터 일부 사용)"""
        # 실제 구현에서는 별도의 테스트 데이터를 사용하는 것이 좋습니다
        # 여기서는 간단한 예시만 제공
        return None
    
    def backup_original_model(self):
        """원본 모델 백업"""
        try:
            backup_dir = f"./backup_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.system(f"cp -r {self.model_path} {backup_dir}")
            logger.info(f"원본 모델 백업 완료: {backup_dir}")
            return backup_dir
        except Exception as e:
            logger.error(f"모델 백업 실패: {e}")
            return None
    
    def update_model(self):
        """재학습된 모델로 업데이트"""
        try:
            if os.path.exists('./retrained_model'):
                # 원본 모델 백업
                self.backup_original_model()
                
                # 재학습된 모델로 교체
                os.system(f"rm -rf {self.model_path}")
                os.system(f"mv ./retrained_model {self.model_path}")
                
                # 모델 재로드
                self.load_model()
                
                logger.info("✅ 모델 업데이트 완료!")
                return True
            else:
                logger.warning("재학습된 모델이 없습니다.")
                return False
                
        except Exception as e:
            logger.error(f"모델 업데이트 실패: {e}")
            return False
    
    def run_retraining_pipeline(self):
        """전체 재학습 파이프라인 실행"""
        logger.info("🚀 모델 재학습 파이프라인 시작...")
        
        # 1. 피드백 데이터 수집
        feedback_data = self.collect_feedback_data()
        if feedback_data is None:
            logger.info("재학습할 피드백 데이터가 부족합니다.")
            return False
        
        # 2. 훈련 데이터 준비
        training_data = self.prepare_training_data(feedback_data)
        if training_data is None:
            logger.info("훈련 데이터 준비 실패")
            return False
        
        # 3. 모델 재학습
        if not self.retrain_model(training_data):
            logger.error("모델 재학습 실패")
            return False
        
        # 4. 모델 성능 평가
        performance = self.evaluate_model()
        if performance is None:
            logger.warning("모델 성능 평가 실패")
        
        # 5. 모델 업데이트
        if self.update_model():
            logger.info("🎉 모델 재학습 파이프라인 완료!")
            return True
        else:
            logger.error("모델 업데이트 실패")
            return False

def main():
    """메인 실행 함수"""
    retrainer = ModelRetrainer()
    
    # 재학습 파이프라인 실행
    success = retrainer.run_retraining_pipeline()
    
    if success:
        print("✅ 모델 재학습이 성공적으로 완료되었습니다!")
    else:
        print("❌ 모델 재학습에 실패했습니다.")

if __name__ == '__main__':
    main()
