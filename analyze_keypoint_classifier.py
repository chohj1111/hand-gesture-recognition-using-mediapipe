"""
keypoint classifier 성능 분석 스크립트
실행: python analyze_keypoint_classifier.py

- TFLite 모델 기준 전체 데이터 추론
- 클래스별 Precision / Recall / F1
- 혼동 행렬 (Confusion Matrix)
- 클래스별 신뢰도 분포
"""
import csv
import numpy as np
import tensorflow as tf

CSV_PATH   = 'model/keypoint_classifier/keypoint.csv'
MODEL_PATH = 'model/keypoint_classifier/keypoint_classifier.tflite'

LABELS = {
    0: 'pinch',
    1: 'ok_sign',
    2: 'thumb_left',
    3: 'fist',
    4: 'one_finger_left',
    5: 'one_finger_right',
    6: 'one_finger_up',
    7: 'one_finger_down',
    8: 'two_finger_left',
    9: 'two_finger_right',
}

# ── 데이터 로드 ──────────────────────────────────────────────────────────────

def load_data():
    X, y = [], []
    with open(CSV_PATH) as f:
        for row in csv.reader(f):
            if not row:
                continue
            y.append(int(row[0]))
            X.append([float(v) for v in row[1:]])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# ── TFLite 배치 추론 ─────────────────────────────────────────────────────────

def predict_all(X):
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    in_idx  = interpreter.get_input_details()[0]['index']
    out_idx = interpreter.get_output_details()[0]['index']

    preds   = []
    confs   = []
    for x in X:
        interpreter.set_tensor(in_idx, np.array([x], dtype=np.float32))
        interpreter.invoke()
        prob = np.squeeze(interpreter.get_tensor(out_idx))
        preds.append(int(np.argmax(prob)))
        confs.append(prob)

    return np.array(preds), np.array(confs)


# ── 통계 계산 ────────────────────────────────────────────────────────────────

def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t < n_classes and p < n_classes:
            cm[t][p] += 1
    return cm


def per_class_metrics(cm):
    metrics = {}
    for i in range(len(cm)):
        tp = cm[i][i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        support = cm[i, :].sum()
        metrics[i] = dict(tp=tp, fp=fp, fn=fn,
                          precision=precision, recall=recall,
                          f1=f1, support=int(support))
    return metrics


# ── 출력 헬퍼 ────────────────────────────────────────────────────────────────

def bar(value, width=20):
    filled = int(value * width)
    return '█' * filled + '░' * (width - filled)


def print_section(title):
    print()
    print('─' * 68)
    print(f'  {title}')
    print('─' * 68)


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    print('=' * 68)
    print('  keypoint classifier 성능 분석')
    print('=' * 68)

    # 데이터 로드
    print(f'\n  데이터 로드 중: {CSV_PATH}')
    X, y_true = load_data()

    # TFLite 모델 출력 클래스 수 확인 (LABELS 딕셔너리와 독립적으로 검출)
    _interp = tf.lite.Interpreter(model_path=MODEL_PATH)
    _interp.allocate_tensors()
    n_classes = int(_interp.get_output_details()[0]['shape'][-1])
    del _interp
    print(f'  TFLite 출력 클래스 수: {n_classes}')
    known_mask = y_true < n_classes
    X_k, y_k = X[known_mask], y_true[known_mask]
    print(f'  총 샘플: {len(y_true)}  (알 수 없는 클래스 제외 후: {len(y_k)})')

    unknown = set(y_true[~known_mask])
    if unknown:
        print(f'  ⚠ 알 수 없는 클래스 ID: {unknown} — 분석에서 제외')

    # 추론
    print(f'\n  모델 추론 중: {MODEL_PATH}')
    y_pred, confs = predict_all(X_k)
    accuracy = (y_pred == y_k).mean()
    print(f'  전체 정확도: {accuracy*100:.2f}%  ({(y_pred==y_k).sum()} / {len(y_k)})')

    # 클래스별 지표
    cm = confusion_matrix(y_k, y_pred, n_classes)
    metrics = per_class_metrics(cm)

    print_section('클래스별 성능 (Precision / Recall / F1)')
    print(f'  {"ID":<4} {"레이블":<20} {"Prec":>6} {"Recall":>7} {"F1":>6} {"샘플":>6}')
    print('  ' + '-' * 52)

    for cls in range(n_classes):
        label = LABELS.get(cls, str(cls))
        m = metrics[cls]
        if m['support'] == 0:
            print(f'  {cls:<4} {label:<20} {"(데이터 없음)":>22}')
            continue
        print(
            f'  {cls:<4} {label:<20} '
            f'{m["precision"]*100:>5.1f}%  '
            f'{m["recall"]*100:>6.1f}%  '
            f'{m["f1"]*100:>5.1f}%  '
            f'{m["support"]:>5}'
        )

    # 혼동 행렬
    print_section('혼동 행렬 (행: 실제 / 열: 예측)')
    active = [c for c in range(n_classes) if metrics[c]['support'] > 0]

    # 헤더
    header = '  {:>20}'.format('실제↓ 예측→')
    for c in active:
        header += f'  {LABELS[c][:8]:>8}'
    print(header)
    print('  ' + '-' * (22 + 10 * len(active)))

    for r in active:
        row_str = f'  {LABELS[r]:>20}'
        for c in active:
            val = cm[r][c]
            marker = '*' if r == c else ' '
            row_str += f'  {val:>7}{marker}'
        recall = metrics[r]['recall']
        row_str += f'   recall {recall*100:.1f}%'
        print(row_str)

    # 신뢰도 분포
    print_section('클래스별 예측 신뢰도 (정답 샘플만)')
    print(f'  {"ID":<4} {"레이블":<20} {"평균":>6} {"최소":>6} {"최대":>6}  분포')
    print('  ' + '-' * 60)

    for cls in range(n_classes):
        mask = (y_k == cls) & (y_pred == cls)
        if mask.sum() == 0:
            if metrics[cls]['support'] > 0:
                print(f'  {cls:<4} {LABELS.get(cls, str(cls)):<20} {"(정답 없음)"}')
            continue
        c_vals = confs[mask, cls]
        mean_c = c_vals.mean()
        min_c  = c_vals.min()
        max_c  = c_vals.max()
        print(
            f'  {cls:<4} {LABELS.get(cls, str(cls)):<20} '
            f'{mean_c*100:>5.1f}%  '
            f'{min_c*100:>5.1f}%  '
            f'{max_c*100:>5.1f}%  '
            f'[{bar(mean_c, 16)}]'
        )

    # 오분류 상세
    print_section('주요 오분류 (10개 이상)')
    found = False
    for r in active:
        for c in active:
            if r != c and cm[r][c] >= 10:
                print(f'  {LABELS[r]:>20} → {LABELS[c]:<20}  {cm[r][c]}회')
                found = True
    if not found:
        print('  없음 (10회 이상 오분류 없음)')

    print()
    print('=' * 68)


if __name__ == '__main__':
    main()
