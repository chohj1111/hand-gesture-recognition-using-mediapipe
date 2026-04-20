import tensorflow as tf
from tensorflow import keras

# 1. 기존 h5 모델 로드
model = keras.models.load_model('/Users/chohj/Library/CloudStorage/OneDrive-개인/Projects/LGE/SW_BootCamp_260326-260424/hand-gesture-recognition-using-mediapipe/model/keypoint_classifier/keypoint_classifier.hdf5')

# 2. .keras 포맷으로 저장
model.save('/Users/chohj/Library/CloudStorage/OneDrive-개인/Projects/LGE/SW_BootCamp_260326-260424/hand-gesture-recognition-using-mediapipe/model/keypoint_classifier/keypoint_classifier.keras')
