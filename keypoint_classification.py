#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
keypoint classifier 학습 스크립트
실행: conda run -n tf-mac-metal python keypoint_classification.py
"""
import os
os.environ['TF_DISABLE_METAL'] = '1'  # Metal 비결정성 차단 → CPU 학습

import shutil
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
dataset            = 'model/keypoint_classifier/keypoint.csv'
checkpoint_path    = 'model/keypoint_classifier/keypoint_classifier_ckpt.keras'  # 학습 중 best 저장
model_save_path    = 'model/keypoint_classifier/keypoint_classifier.keras'        # 최종 배포용
tflite_save_path   = 'model/keypoint_classifier/keypoint_classifier.tflite'

# ── 분류 수 설정 ──────────────────────────────────────────────────────────────
NUM_CLASSES = 10         # 정적 제스처 10개
EXCLUDE_CLASSES = []     # 학습에서 제외할 클래스 ID (빈 리스트 [] 이면 전체 사용)

# ── 학습 데이터 로드 ──────────────────────────────────────────────────────────
X_dataset = np.loadtxt(dataset, delimiter=',', dtype='float32',
                       usecols=list(range(1, (21 * 2) + 1)))
y_dataset = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=(0))

if EXCLUDE_CLASSES:
    mask = ~np.isin(y_dataset, EXCLUDE_CLASSES)
    X_dataset, y_dataset = X_dataset[mask], y_dataset[mask]
    print(f'제외 클래스: {EXCLUDE_CLASSES} → 남은 샘플: {len(y_dataset)}')

X_train, X_test, y_train, y_test = train_test_split(
    X_dataset, y_dataset, train_size=0.75, random_state=RANDOM_SEED)

# ── 클래스 가중치 계산 (불균형 보정) ─────────────────────────────────────────
total = len(y_train)
class_weight = {
    cls: total / (NUM_CLASSES * count)
    for cls, count in zip(*np.unique(y_train, return_counts=True))
}
print('class_weight:', {k: f'{v:.2f}' for k, v in sorted(class_weight.items())})

# ── 모델 구성 ─────────────────────────────────────────────────────────────────
model = tf.keras.models.Sequential([
    tf.keras.layers.Input((21 * 2, )),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])
model.summary()

# ── 콜백 설정 ─────────────────────────────────────────────────────────────────
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=False, save_best_only=True)
es_callback = tf.keras.callbacks.EarlyStopping(
    patience=30, verbose=1, restore_best_weights=True)
lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=10, min_lr=1e-5, verbose=1)

# ── 모델 컴파일 ───────────────────────────────────────────────────────────────
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ── 모델 학습 ─────────────────────────────────────────────────────────────────
model.fit(
    X_train, y_train,
    epochs=1000,
    batch_size=128,
    validation_data=(X_test, y_test),
    callbacks=[cp_callback, es_callback, lr_callback],
    class_weight=class_weight
)

# ── 모델 평가 ─────────────────────────────────────────────────────────────────
val_loss, val_acc = model.evaluate(X_test, y_test, batch_size=128)
print(f'val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f}')

# ── 클래스별 성능 분석 ────────────────────────────────────────────────────────
ALL_LABELS = ['pinch', 'ok_sign', 'thumb_left', 'open_palm',
              'one_finger_left', 'one_finger_right', 'one_finger_up', 'one_finger_down',
              'two_finger_left', 'two_finger_right']
LABELS = ALL_LABELS[:NUM_CLASSES]

y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
cm = confusion_matrix(y_test, y_pred, labels=list(range(NUM_CLASSES)))

print()
print('─' * 60)
print('  클래스별 성능 (Recall 기준)')
print('─' * 60)
print(f'  {"ID":<3} {"레이블":<18} {"Recall":>7} {"Precision":>10} {"F1":>7} {"샘플":>5}')
print('  ' + '-' * 50)
weak_classes = []
for i in range(NUM_CLASSES):
    tp = cm[i, i]
    fn = cm[i, :].sum() - tp
    fp = cm[:, i].sum() - tp
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    support = int(cm[i, :].sum())
    flag = '  ⚠' if recall < 0.7 else ''
    print(f'  {i:<3} {LABELS[i]:<18} {recall*100:>6.1f}%  {precision*100:>8.1f}%  {f1*100:>6.1f}%  {support:>4}{flag}')
    if recall < 0.7:
        weak_classes.append((i, LABELS[i], recall, f1))

print()
if weak_classes:
    print('  ⚠ 학습 부진 클래스 (Recall < 70%):')
    for i, label, recall, f1 in weak_classes:
        top_confused = [(j, cm[i, j]) for j in range(NUM_CLASSES) if j != i and cm[i, j] > 0]
        top_confused.sort(key=lambda x: -x[1])
        confused_str = ', '.join(f'{LABELS[j]}({n})' for j, n in top_confused[:3])
        print(f'    class {i} {label}: recall {recall*100:.1f}%  → 주로 오분류: {confused_str}')
else:
    print('  모든 클래스 Recall >= 70%')
print('─' * 60)

# ── 체크포인트에서 best 모델 로드 후 배포용 경로에 저장 ───────────────────────
model = tf.keras.models.load_model(checkpoint_path)
model.save(model_save_path)
print(f'배포용 모델 저장 완료: {model_save_path}')

# ── 추론 테스트 ───────────────────────────────────────────────────────────────
predict_result = model.predict(np.array([X_test[0]]))
print(np.squeeze(predict_result))
print(np.argmax(np.squeeze(predict_result)))

# ── TFLite 변환 (from_saved_model — Apple Silicon MLIR 버그 회피) ─────────────
saved_model_dir = 'model/keypoint_classifier/keypoint_saved_model'
shutil.rmtree(saved_model_dir, ignore_errors=True)
model.export(saved_model_dir)

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_quantized_model = converter.convert()

with open(tflite_save_path, 'wb') as f:
    f.write(tflite_quantized_model)
print(f'TFLite 저장 완료: {tflite_save_path}')

# ── TFLite 추론 테스트 ────────────────────────────────────────────────────────
interpreter = tf.lite.Interpreter(model_path=tflite_save_path)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], np.array([X_test[0]]))
interpreter.invoke()
tflite_results = interpreter.get_tensor(output_details[0]['index'])
print(np.squeeze(tflite_results))
print(np.argmax(np.squeeze(tflite_results)))
