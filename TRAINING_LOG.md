# Keypoint Classifier 학습 기록

---

## Run 1 — 2026-04-20

### 데이터셋

| 클래스 | 레이블 | 학습 샘플 |
|---|---|---|
| 0 | pinch | 1,007 |
| 1 | ok_sign | 1,026 |
| 2 | thumb_left | 1,006 |
| 3 | fist | 1,110 |
| 4 | one_finger_point | 1,024 |
| 5 | two_finger_point | 1,018 |
| 6 | none | 1,107 |
| **합계** | | **7,298** |

### 모델 구성

```
Input(42) → Dropout(0.2) → Dense(20, relu) → Dropout(0.4) → Dense(10, relu) → Dense(7, softmax)
총 파라미터: 1,147
```

### 학습 설정

| 항목 | 값 |
|---|---|
| Optimizer | Adam (lr=1e-3) |
| Loss | sparse_categorical_crossentropy |
| EarlyStopping patience | 50 (restore_best_weights=True) |
| ReduceLROnPlateau | factor=0.5, patience=10, min_lr=1e-5 |
| Batch size | 128 |
| Train/Val split | 75 / 25 |
| class_weight | 적용 (불균형 보정) |

### 클래스별 성능

| ID | 레이블 | Recall | Precision | F1 | 샘플 |
|---|---|---|---|---|---|
| 0 | pinch | 84.9% | 86.2% | 85.6% | 272 |
| 1 | ok_sign | 95.1% | 89.7% | 92.3% | 265 |
| 2 | thumb_left | 95.0% | 92.2% | 93.6% | 260 |
| 3 | fist | 95.1% | 95.1% | 95.1% | 266 |
| 4 | one_finger_point | 95.6% | 89.2% | 92.3% | 251 |
| 5 | two_finger_point | 81.3% | 85.4% | 83.3% | 252 |
| 6 | none | 85.7% | 95.3% | 90.2% | 259 |

### 요약

- **전체 val_acc**: 약 90%
- **개선 필요**: pinch (84.9%), two_finger_point (81.3%)
- **전처리 수정 이력**: `pre_process_landmark`의 `bade_x` 타이포 수정 후 첫 재수집 데이터로 학습
