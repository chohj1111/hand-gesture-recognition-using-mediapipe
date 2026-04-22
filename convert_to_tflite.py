"""
Keras 모델 → TFLite 변환 스크립트
실행: conda run -n tf-mac-metal python convert_to_tflite.py
"""
import os
import numpy as np
import tensorflow as tf

KEYPOINT_KERAS   = 'model/keypoint_classifier/keypoint_classifier.keras'
KEYPOINT_TFLITE  = 'model/keypoint_classifier/keypoint_classifier.tflite'

POINT_HISTORY_KERAS  = 'model/point_history_classifier/point_history_classifier.keras'
POINT_HISTORY_TFLITE = 'model/point_history_classifier/point_history_classifier.tflite'


def convert(keras_path, tflite_path):
    if not os.path.exists(keras_path):
        print(f'  ⚠ 건너뜀: {keras_path} 없음')
        return False

    print(f'  로드: {keras_path}')
    model = tf.keras.models.load_model(keras_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    size_kb = os.path.getsize(tflite_path) / 1024
    print(f'  저장: {tflite_path}  ({size_kb:.1f} KB)')
    return True


def verify(tflite_path, input_size):
    """TFLite 모델 추론 동작 확인"""
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    in_idx  = interp.get_input_details()[0]['index']
    out_idx = interp.get_output_details()[0]['index']

    dummy = np.zeros((1, input_size), dtype=np.float32)
    interp.set_tensor(in_idx, dummy)
    interp.invoke()
    result = np.squeeze(interp.get_tensor(out_idx))
    print(f'  검증: 출력 shape={result.shape}, argmax={np.argmax(result)}  ✓')


print('=' * 50)
print('  keypoint_classifier 변환')
print('=' * 50)
if convert(KEYPOINT_KERAS, KEYPOINT_TFLITE):
    verify(KEYPOINT_TFLITE, 21 * 2)

print()
print('=' * 50)
print('  point_history_classifier 변환')
print('=' * 50)
if convert(POINT_HISTORY_KERAS, POINT_HISTORY_TFLITE):
    verify(POINT_HISTORY_TFLITE, 16 * 2)

print('\n완료')
