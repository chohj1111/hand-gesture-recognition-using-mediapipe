#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
point_history classifier 학습 스크립트
실행: conda run -n tf-mac-metal python point_history_classification.py
"""
import shutil
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
dataset          = 'model/point_history_classifier/point_history.csv'
model_save_path  = 'model/point_history_classifier/point_history_classifier.keras'
tflite_save_path = 'model/point_history_classifier/point_history_classifier.tflite'

# ── 분류 수 / 입력 설정 ───────────────────────────────────────────────────────
NUM_CLASSES = 4
TIME_STEPS  = 16
DIMENSION   = 2

# ── 학습 데이터 로드 ──────────────────────────────────────────────────────────
X_dataset = np.loadtxt(dataset, delimiter=',', dtype='float32',
                       usecols=list(range(1, (TIME_STEPS * DIMENSION) + 1)))
y_dataset = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=(0))

X_train, X_test, y_train, y_test = train_test_split(
    X_dataset, y_dataset, train_size=0.75, random_state=RANDOM_SEED)

# ── 모델 구성 ─────────────────────────────────────────────────────────────────
use_lstm = False

if use_lstm:
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(TIME_STEPS * DIMENSION,)),
        tf.keras.layers.Reshape((TIME_STEPS, DIMENSION)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(16),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
else:
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(TIME_STEPS * DIMENSION,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])

model.summary()

# ── 콜백 설정 ─────────────────────────────────────────────────────────────────
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    model_save_path, verbose=1, save_weights_only=False, save_best_only=True)
es_callback = tf.keras.callbacks.EarlyStopping(
    patience=30, verbose=1, restore_best_weights=True)
lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=10, min_lr=1e-5, verbose=1)

# ── 모델 컴파일 ───────────────────────────────────────────────────────────────
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ── 모델 학습 ─────────────────────────────────────────────────────────────────
model.fit(
    X_train, y_train,
    epochs=1000,
    batch_size=128,
    validation_data=(X_test, y_test),
    callbacks=[cp_callback, es_callback, lr_callback]
)

# ── 모델 평가 ─────────────────────────────────────────────────────────────────
val_loss, val_acc = model.evaluate(X_test, y_test, batch_size=128)
print(f'val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f}')

# ── 저장된 모델 로드 ──────────────────────────────────────────────────────────
model = tf.keras.models.load_model(model_save_path)

# ── 추론 테스트 ───────────────────────────────────────────────────────────────
predict_result = model.predict(np.array([X_test[0]]))
print(np.squeeze(predict_result))
print(np.argmax(np.squeeze(predict_result)))

# ── TFLite 변환 (from_saved_model — Apple Silicon MLIR 버그 회피) ─────────────
saved_model_dir = 'model/point_history_classifier/point_history_saved_model'
shutil.rmtree(saved_model_dir, ignore_errors=True)
model.export(saved_model_dir)

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()

with open(tflite_save_path, 'wb') as f:
    f.write(tflite_model)
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
