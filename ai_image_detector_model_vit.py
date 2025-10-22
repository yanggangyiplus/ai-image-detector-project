# -*- coding: utf-8 -*-
"""AI Image Detector Model VIT

아나콘다 파이썬3 환경에서 실행 가능하도록 수정된 버전
로컬 데이터 경로를 사용하여 AI 생성 이미지와 실제 이미지를 분류하는 ViT 모델
"""

"""
=== AI Cursor / 아나콘다 파이썬3 환경에서 실행하기 위한 가이드 ===

🚀 AI Cursor에서 실행하는 방법:
1. Cursor 터미널에서 직접 실행:
   - Ctrl+` (백틱)으로 터미널 열기
   - python ai_image_detector_model_vit.py

2. Cursor의 Python 인터프리터에서 실행:
   - Shift+Enter로 셀 단위 실행
   - 또는 전체 파일 선택 후 Ctrl+Enter

📦 라이브러리 설치 (Cursor 터미널에서):
1. PyTorch 설치 (GPU 사용 시):
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

2. PyTorch 설치 (CPU만 사용 시):
   conda install pytorch torchvision torchaudio cpuonly -c pytorch

3. 기타 필요한 라이브러리 설치:
   pip install transformers datasets evaluate accelerate imbalanced-learn
   pip install huggingface_hub tqdm matplotlib scikit-learn pandas numpy

✅ 실행 전 확인사항:
   - 데이터 경로가 올바른지 확인: '/Users/yanggangyi/fastcampus/Hugging Face Project/Ai image detector/data1/train'
   - FAKE와 REAL 폴더에 이미지 파일이 있는지 확인
   - 충분한 디스크 공간이 있는지 확인 (모델 저장용)
   - Python 인터프리터가 올바르게 설정되었는지 확인

🎯 실행 방법:
   - Cursor 터미널: python ai_image_detector_model_vit.py
   - 또는 Cursor에서 파일을 열고 Shift+Enter로 셀 단위 실행
"""

# =============================================================================
# 셀 1: 라이브러리 import 및 기본 설정
# =============================================================================

# 필요한 라이브러리 import
import warnings  # 경고 메시지 처리
warnings.filterwarnings("ignore")  # 실행 중 경고 무시

import gc  # 가비지 컬렉션
import numpy as np  # 수치 연산
import pandas as pd  # 데이터 조작
import itertools  # 반복자 및 루핑
from collections import Counter  # 요소 카운팅
import matplotlib.pyplot as plt  # 데이터 시각화
from sklearn.metrics import (  # scikit-learn 메트릭
    accuracy_score,  # 정확도 계산
    roc_auc_score,  # ROC AUC 점수
    confusion_matrix,  # 혼동 행렬
    classification_report,  # 분류 보고서
    f1_score  # F1 점수
)

# 커스텀 모듈 및 클래스 import
from imblearn.over_sampling import RandomOverSampler  # 랜덤 오버샘플링
import accelerate  # 가속화 모듈
import evaluate  # 평가 모듈
from datasets import Dataset, Image, ClassLabel  # 데이터셋, 이미지, 클래스 라벨
from transformers import (  # Transformers 라이브러리 모듈들
    TrainingArguments,  # 훈련 인수
    Trainer,  # 모델 훈련
    ViTImageProcessor,  # ViT 모델용 이미지 처리
    ViTForImageClassification,  # 이미지 분류용 ViT 모델
    DefaultDataCollator  # 기본 데이터 콜레이터
)
import torch  # PyTorch 딥러닝
from torch.utils.data import DataLoader  # 데이터 로더 생성
from torchvision.transforms import (  # 이미지 변환 함수들
    CenterCrop,  # 이미지 중앙 크롭
    Compose,  # 여러 이미지 변환 조합
    Normalize,  # 이미지 픽셀 값 정규화
    RandomRotation,  # 랜덤 회전 적용
    RandomResizedCrop,  # 랜덤 크롭 및 리사이즈
    RandomHorizontalFlip,  # 랜덤 수평 뒤집기
    RandomAdjustSharpness,  # 랜덤 선명도 조정
    Resize,  # 이미지 리사이즈
    ToTensor  # 이미지를 PyTorch 텐서로 변환
)

# PIL 라이브러리에서 필요한 모듈 import
from PIL import ImageFile

# 잘린 이미지 로드 옵션 활성화
# 이 설정은 손상되거나 불완전한 이미지도 로드 시도하도록 허용
ImageFile.LOAD_TRUNCATED_IMAGES = True

print("✅ 라이브러리 import 완료!")

# =============================================================================
# 셀 2: 데이터 로딩 및 전처리
# =============================================================================

# 데이터 로딩 및 전처리
from pathlib import Path
from tqdm import tqdm
import os

# 로컬 데이터 경로 설정
data_path = '/Users/yanggangyi/fastcampus/Hugging Face Project/Ai image detector/data1/train'

# 파일명과 라벨을 저장할 리스트 초기화
file_names = []
labels = []

print(f"데이터 로딩 시작: {data_path}")
print("지원되는 이미지 확장자: .jpg, .jpeg, .png, .bmp, .tiff")

# 지정된 디렉토리의 모든 이미지 파일을 순회
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
for ext in image_extensions:
    for file in tqdm(sorted(Path(data_path).glob(f'*/*{ext}')), desc=f"로딩 중 {ext}"):
        label = str(file).split('/')[-2]  # 파일 경로에서 라벨 추출
        labels.append(label)  # 라벨을 리스트에 추가
        file_names.append(str(file))  # 파일 경로를 리스트에 추가

# 파일명과 라벨의 총 개수 출력
print(f"총 파일 수: {len(file_names)}")
print(f"총 라벨 수: {len(labels)}")
print(f"고유 라벨: {set(labels)}")
print("✅ 데이터 로딩 완료!")

# =============================================================================
# 셀 3: 데이터프레임 생성 및 오버샘플링
# =============================================================================

# 수집된 파일명과 라벨로부터 pandas DataFrame 생성
df = pd.DataFrame.from_dict({"image": file_names, "label": labels})
print(f"데이터프레임 크기: {df.shape}")

# 데이터프레임의 첫 5행 출력
print("\n데이터프레임 샘플:")
print(df.head())

# 고유 라벨 확인
print(f"\n고유 라벨: {df['label'].unique()}")

# 클래스별 데이터 분포 확인
print("\n클래스별 데이터 분포:")
print(df['label'].value_counts())

# 소수 클래스 랜덤 오버샘플링
# 'y'는 예측하고자 하는 타겟 변수(라벨)를 포함
y = df[['label']]

# 피처와 타겟 변수를 분리하기 위해 DataFrame 'df'에서 'label' 컬럼 제거
df = df.drop(['label'], axis=1)

# 지정된 랜덤 시드(random_state=83)로 RandomOverSampler 객체 생성
ros = RandomOverSampler(random_state=83)

# RandomOverSampler를 사용하여 소수 클래스를 오버샘플링하여 데이터셋 재샘플링
# 'df'는 피처 데이터를 포함하고, 'y_resampled'는 재샘플링된 타겟 변수를 포함
df, y_resampled = ros.fit_resample(df, y)

# 더 이상 필요하지 않은 원본 'y' 변수를 메모리 절약을 위해 삭제
del y

# 재샘플링된 타겟 변수 'y_resampled'를 DataFrame 'df'의 새로운 'label' 컬럼으로 추가
df['label'] = y_resampled

# 메모리 절약을 위해 더 이상 필요하지 않은 'y_resampled' 변수 삭제
del y_resampled

# 삭제된 변수들로 사용된 메모리를 해제하기 위해 가비지 컬렉션 수행
gc.collect()

print(f"\n오버샘플링 후 데이터프레임 크기: {df.shape}")
print("오버샘플링 후 클래스별 분포:")
print(df['label'].value_counts())
print("✅ 데이터 전처리 완료!")

# =============================================================================
# 셀 4: 데이터셋 생성 및 라벨 매핑
# =============================================================================

# Pandas DataFrame으로부터 데이터셋 생성
dataset = Dataset.from_pandas(df).cast_column("image", Image())

# 데이터셋의 첫 번째 이미지 표시
print("데이터셋의 첫 번째 이미지:")
print(dataset[0]["image"])

# 고유 라벨 리스트 생성
labels_list = ['REAL', 'FAKE']

# 라벨과 ID 간의 매핑을 위한 빈 딕셔너리 초기화
label2id, id2label = dict(), dict()

# 고유 라벨을 순회하며 각 라벨에 ID를 할당하고, 그 반대도 수행
for i, label in enumerate(labels_list):
    label2id[label] = i  # 라벨을 해당 ID에 매핑
    id2label[i] = label  # ID를 해당 라벨에 매핑

# 참조를 위한 결과 딕셔너리 출력
print("ID to Label 매핑:", id2label)
print("Label to ID 매핑:", label2id)

# 라벨을 ID와 매칭하기 위한 ClassLabel 생성
ClassLabels = ClassLabel(num_classes=len(labels_list), names=labels_list)

# 라벨을 ID로 매핑하는 함수
def map_label2id(example):
    example['label'] = ClassLabels.str2int(example['label'])
    return example

# 데이터셋에 라벨 매핑 적용
dataset = dataset.map(map_label2id, batched=True)

# 라벨 컬럼을 ClassLabel 객체로 캐스팅
dataset = dataset.cast_column('label', ClassLabels)

# 60-40 분할 비율을 사용하여 데이터셋을 훈련 및 테스트 세트로 분할
dataset = dataset.train_test_split(test_size=0.4, shuffle=True, stratify_by_column="label")

# 분할된 데이터셋에서 훈련 데이터 추출
train_data = dataset['train']

# 분할된 데이터셋에서 테스트 데이터 추출
test_data = dataset['test']

print(f"훈련 데이터 크기: {len(train_data)}")
print(f"테스트 데이터 크기: {len(test_data)}")
print("✅ 데이터셋 생성 및 분할 완료!")

# =============================================================================
# 셀 5: 모델 및 전처리기 설정
# =============================================================================

# 사전 훈련된 ViT 모델 문자열 정의
model_str = "dima806/ai_vs_real_image_detection"

# 사전 훈련된 모델로부터 ViT 모델 입력용 프로세서 생성
processor = ViTImageProcessor.from_pretrained(model_str)

# 정규화에 사용되는 이미지 평균과 표준편차 검색
image_mean, image_std = processor.image_mean, processor.image_std

# ViT 모델의 입력 이미지 크기(높이) 가져오기
size = processor.size["height"]
print(f"이미지 크기: {size}")

# 입력 이미지에 대한 정규화 변환 정의
normalize = Normalize(mean=image_mean, std=image_std)

# 훈련 데이터에 대한 변환 세트 정의
_train_transforms = Compose(
    [
        Resize((size, size)),             # 이미지를 ViT 모델의 입력 크기로 리사이즈
        RandomRotation(90),               # 랜덤 회전 적용
        RandomAdjustSharpness(2),         # 랜덤 선명도 조정
        ToTensor(),                       # 이미지를 텐서로 변환
        normalize                         # 평균과 표준편차를 사용하여 이미지 정규화
    ]
)

# 검증 데이터에 대한 변환 세트 정의
_val_transforms = Compose(
    [
        Resize((size, size)),             # 이미지를 ViT 모델의 입력 크기로 리사이즈
        ToTensor(),                       # 이미지를 텐서로 변환
        normalize                         # 평균과 표준편차를 사용하여 이미지 정규화
    ]
)

# 예제 배치에 훈련 변환을 적용하는 함수 정의
def train_transforms(examples):
    examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['image']]
    return examples

# 예제 배치에 검증 변환을 적용하는 함수 정의
def val_transforms(examples):
    examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['image']]
    return examples

# 훈련 데이터에 변환 설정
train_data.set_transform(train_transforms)

# 테스트/검증 데이터에 변환 설정
test_data.set_transform(val_transforms)

# 모델 훈련을 위해 배치 데이터를 준비하는 콜레이트 함수 정의
def collate_fn(examples):
    # 개별 예제의 픽셀 값을 단일 텐서로 스택
    pixel_values = torch.stack([example["pixel_values"] for example in examples])

    # 예제의 라벨 문자열을 label2id 딕셔너리를 사용하여 해당 숫자 ID로 변환
    labels = torch.tensor([example['label'] for example in examples])

    # 배치된 픽셀 값과 라벨을 포함하는 딕셔너리 반환
    return {"pixel_values": pixel_values, "labels": labels}

print("✅ 모델 및 전처리기 설정 완료!")

# =============================================================================
# 셀 6: 모델 로드 및 훈련 설정
# =============================================================================

# 지정된 출력 라벨 수로 사전 훈련된 체크포인트에서 ViTForImageClassification 모델 생성
model = ViTForImageClassification.from_pretrained(model_str, num_labels=len(labels_list))

# 나중에 참조할 수 있도록 클래스 라벨을 해당 인덱스에 매핑하는 설정 구성
model.config.id2label = id2label
model.config.label2id = label2id

# 모델의 훈련 가능한 매개변수 수를 백만 단위로 계산하고 출력
print(f"훈련 가능한 매개변수 수: {model.num_parameters(only_trainable=True) / 1e6:.2f}M")

# 'evaluate' 모듈에서 정확도 메트릭 로드
accuracy = evaluate.load("accuracy")

# 평가 메트릭을 계산하는 'compute_metrics' 함수 정의
def compute_metrics(eval_pred):
    # 평가 예측 객체에서 모델 예측 추출
    predictions = eval_pred.predictions

    # 평가 예측 객체에서 실제 라벨 추출
    label_ids = eval_pred.label_ids

    # 로드된 정확도 메트릭을 사용하여 정확도 계산
    # 가장 높은 확률을 가진 클래스를 선택하여 모델 예측을 클래스 라벨로 변환 (argmax)
    predicted_labels = predictions.argmax(axis=1)

    # 예측된 라벨을 실제 라벨과 비교하여 정확도 점수 계산
    acc_score = accuracy.compute(predictions=predicted_labels, references=label_ids)['accuracy']

    # "accuracy" 키를 가진 딕셔너리로 계산된 정확도 반환
    return {
        "accuracy": acc_score
    }

# 훈련 및 평가 중에 사용될 평가 메트릭의 이름 정의
metric_name = "accuracy"

# 모델 체크포인트와 출력을 저장할 디렉토리를 생성하는 데 사용될 모델 이름 정의
model_name = "ai_vs_real_image_detection"
model_save_path = f"./{model_name}"  # 로컬 저장 경로

# GPU 사용 가능 여부 확인 및 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 중인 디바이스: {device}")

# GPU 메모리 정보 출력 (GPU 사용 시)
if torch.cuda.is_available():
    print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
    print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# 모델 훈련을 위한 에포크 수 정의
num_train_epochs = 2

# GPU/CPU에 따른 배치 크기 조정
if torch.cuda.is_available():
    train_batch_size = 32  # GPU 사용 시
    eval_batch_size = 16
    print("GPU 사용 - 배치 크기: 32 (훈련), 16 (평가)")
else:
    train_batch_size = 8   # CPU 사용 시
    eval_batch_size = 4
    print("CPU 사용 - 배치 크기: 8 (훈련), 4 (평가)")

# 훈련 설정을 구성하기 위한 TrainingArguments 인스턴스 생성
args = TrainingArguments(
    # 모델 체크포인트와 출력이 저장될 디렉토리 지정
    output_dir=model_name,

    # 훈련 로그가 저장될 디렉토리 지정
    logging_dir='./logs',

    # 각 에포크 끝에서 수행되는 평가 전략 정의
    eval_strategy="epoch",

    # 옵티마이저를 위한 학습률 설정
    learning_rate=1e-6,

    # 각 디바이스에서 훈련을 위한 배치 크기 정의
    per_device_train_batch_size=train_batch_size,

    # 각 디바이스에서 평가를 위한 배치 크기 정의
    per_device_eval_batch_size=eval_batch_size,

    # 총 훈련 에포크 수 지정
    num_train_epochs=num_train_epochs,

    # 과적합 방지를 위한 가중치 감쇠 적용
    weight_decay=0.02,

    # 학습률 스케줄러를 위한 워밍업 스텝 수 설정
    warmup_steps=50,

    # 데이터셋에서 사용되지 않는 컬럼 제거 비활성화
    remove_unused_columns=False,

    # 모델 체크포인트 저장 전략 정의 (이 경우 에포크당)
    save_strategy='epoch',

    # 훈련 끝에 최고 모델 로드
    load_best_model_at_end=True,

    # 공간 절약을 위해 저장되는 총 체크포인트 수 제한
    save_total_limit=1,

    # 훈련 진행 상황을 보고하지 않도록 지정
    report_to="none"  # 로그 없음
)

print("✅ 모델 로드 및 훈련 설정 완료!")

# =============================================================================
# 셀 7: 훈련 및 평가
# =============================================================================

# 언어 모델 파인튜닝을 위한 Trainer 인스턴스 생성
# - `model`: 파인튜닝할 사전 훈련된 언어 모델
# - `args`: 훈련을 위한 구성 설정 및 하이퍼파라미터
# - `train_dataset`: 모델 훈련에 사용되는 데이터셋
# - `eval_dataset`: 훈련 중 모델 평가에 사용되는 데이터셋
# - `data_collator`: 데이터 배치가 어떻게 콜레이트되고 처리되는지 정의하는 함수
# - `compute_metrics`: 사용자 정의 평가 메트릭을 계산하는 함수
# - `tokenizer`: 텍스트 데이터 처리에 사용되는 토크나이저

trainer = Trainer(
    model,
    args,
    train_dataset=train_data,
    eval_dataset=test_data,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    tokenizer=processor,
)

print("🚀 훈련 시작 전 사전 훈련 모델 성능 평가...")
# 테스트 데이터셋에서 사전 훈련 모델의 성능을 평가
# 이 함수는 정확도, 손실 등 다양한 메트릭을 계산하여
# 모델이 보지 못한 데이터에서 얼마나 잘 수행하는지 평가
pre_training_results = trainer.evaluate()
print(f"사전 훈련 모델 정확도: {pre_training_results['eval_accuracy']:.4f}")

print("\n🎯 모델 훈련 시작...")
# trainer 객체를 사용하여 모델 훈련 시작
trainer.train()

print("\n📊 훈련 후 모델 성능 평가...")
# 검증 또는 테스트 데이터셋에서 훈련 후 모델의 성능을 평가
# 이 함수는 정확도, 손실 등 다양한 평가 메트릭을 계산하고
# 모델이 얼마나 잘 수행하는지에 대한 통찰을 제공
post_training_results = trainer.evaluate()
print(f"훈련 후 모델 정확도: {post_training_results['eval_accuracy']:.4f}")

print("✅ 훈련 및 평가 완료!")

# =============================================================================
# 셀 8: 예측 및 결과 분석
# =============================================================================

print("🔍 테스트 데이터에 대한 예측 수행...")
# 훈련된 'trainer'를 사용하여 'test_data'에 대한 예측 수행
outputs = trainer.predict(test_data)

# 예측 출력에서 얻은 메트릭 출력
print("예측 결과 메트릭:")
print(outputs.metrics)

# 모델 출력에서 실제 라벨 추출
y_true = outputs.label_ids

# 가장 높은 확률을 가진 클래스를 선택하여 라벨 예측
y_pred = outputs.predictions.argmax(1)

# 혼동 행렬을 그리는 함수 정의
def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues, figsize=(10, 8)):
    """
    이 함수는 혼동 행렬을 그립니다.

    매개변수:
        cm (array-like): sklearn.metrics.confusion_matrix에서 반환된 혼동 행렬
        classes (list): 클래스 이름 목록, 예: ['Class 0', 'Class 1']
        title (str): 플롯의 제목
        cmap (matplotlib colormap): 플롯의 컬러맵
    """
    # 지정된 크기로 그림 생성
    plt.figure(figsize=figsize)

    # 컬러맵을 사용하여 혼동 행렬을 이미지로 표시
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    # 축의 클래스에 대한 틱 마크와 라벨 정의
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.0f'
    # 셀의 값을 나타내는 텍스트 주석을 플롯에 추가
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    # 축에 라벨 지정
    plt.ylabel('실제 라벨')
    plt.xlabel('예측 라벨')

    # 플롯 레이아웃을 타이트하게 설정
    plt.tight_layout()
    # 플롯 표시
    plt.show()

# 정확도와 F1 점수 계산
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='macro')

# 정확도와 F1 점수 표시
print(f"\n📊 최종 성능 지표:")
print(f"정확도: {accuracy:.4f}")
print(f"F1 점수: {f1:.4f}")

# 라벨 수가 적은 경우 혼동 행렬 가져오기
if len(labels_list) <= 150:
    print("\n📈 혼동 행렬 생성 중...")
    # 혼동 행렬 계산
    cm = confusion_matrix(y_true, y_pred)

    # 정의된 함수를 사용하여 혼동 행렬 플롯
    plot_confusion_matrix(cm, labels_list, figsize=(8, 6))

# 마지막으로 분류 보고서 표시
print("\n📋 분류 보고서:")
print(classification_report(y_true, y_pred, target_names=labels_list, digits=4))

print("✅ 예측 및 결과 분석 완료!")

# =============================================================================
# 셀 9: 모델 저장 및 테스트
# =============================================================================

print("💾 훈련된 모델 저장 중...")
# 훈련된 모델 저장: 이 코드 라인은 trainer 객체를 사용하여 훈련된 모델을 저장하는 역할을 합니다.
# 모델과 관련 가중치를 직렬화하여 나중에 재로드하고 사용할 수 있게 합니다.
# 재훈련 없이 모델을 사용할 수 있게 됩니다.
trainer.save_model()

# 'transformers' 라이브러리에서 'pipeline' 함수 import
from transformers import pipeline

# 이미지 분류 작업을 위한 파이프라인 생성
# 추론에 사용할 'model_name'과 'device'를 지정해야 합니다.
# - 'model_name': 이미지 분류에 사용할 사전 훈련된 모델의 이름
# - 'device': 모델을 실행할 디바이스 지정 (0은 GPU, -1은 CPU)
# GPU 사용 가능 여부에 따라 자동으로 설정
device = 0 if torch.cuda.is_available() else -1
pipe = pipeline('image-classification', model=model_name, device=device)

print("🧪 모델 테스트 중...")
# 'test_data' 데이터셋에서 인덱스 1을 사용하여 이미지에 접근
image = test_data[1]["image"]

# 'image' 변수 표시
print("테스트 이미지:")
print(image)

# 'image' 변수를 처리하기 위해 'pipe' 함수 적용
print("\n예측 결과:")
result = pipe(image)
print(result)

# 이 코드 라인은 test_data 리스트의 특정 요소에서 "label" 속성에 접근합니다.
# 테스트 데이터 포인트와 관련된 실제 라벨을 검색하는 데 사용됩니다.
print(f"\n실제 라벨: {id2label[test_data[1]['label']]}")

print("✅ 모델 저장 및 테스트 완료!")

# =============================================================================
# 셀 10: 모델 사용법 안내
# =============================================================================

"""# 모델 저장 및 로컬 사용"""

# 모델이 로컬에 저장되었습니다.
# 저장된 모델은 './ai_vs_real_image_detection' 디렉토리에 있습니다.
print(f"모델이 '{model_save_path}' 디렉토리에 저장되었습니다.")

# 로컬에서 저장된 모델을 사용하는 방법:
print("\n=== 저장된 모델 사용 방법 ===")
print("1. 새로운 Python 스크립트에서:")
print("   from transformers import pipeline")
print(f"   pipe = pipeline('image-classification', model='{model_save_path}', device={device})")
print("   result = pipe('이미지_경로')")
print("\n2. 또는 직접 모델 로드:")
print("   from transformers import ViTForImageClassification, ViTImageProcessor")
print(f"   model = ViTForImageClassification.from_pretrained('{model_save_path}')")
print(f"   processor = ViTImageProcessor.from_pretrained('{model_save_path}')")

print("\n" + "="*60)
print("🎉 AI 이미지 분류 모델 훈련이 완료되었습니다!")
print("="*60)
print(f"✅ 훈련된 모델이 '{model_save_path}' 디렉토리에 저장되었습니다.")
print("✅ 이제 새로운 이미지에 대해 AI 생성 여부를 분류할 수 있습니다.")
print("="*60)
