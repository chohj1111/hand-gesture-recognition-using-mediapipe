"""
point_history classifier 성능 분석 스크립트
실행: conda run -n tf-mac-metal python analyze_point_history_classifier.py

- TFLite 모델 기준 전체 데이터 추론
- 클래스별 Precision / Recall / F1
- 혼동 행렬
- 궤적 특성 분석 (클래스별 평균 net_x / net_y / 이동거리)
- 클래스별 신뢰도 분포
"""
import csv
import math
import numpy as np
import tensorflow as tf

CSV_PATH   = 'model/point_history_classifier/point_history.csv'
MODEL_PATH = 'model/point_history_classifier/point_history_classifier.tflite'
TARGET     = 1500

LABELS = {
    0: 'swipe_left',
    1: 'swipe_right',
    2: 'swipe_up',
    3: 'swipe_down',
}

# 각 클래스의 기대 변위 방향
EXPECTED = {
    0: 'net_x ↓음수',
    1: 'net_x ↑양수',
    2: 'net_y ↓음수',
    3: 'net_y ↑양수',
}


# ── 데이터 로드 ──────────────────────────────────────────────────────────────

def load_data():
    X, y = [], []
    with open(CSV_PATH) as f:
        for row in csv.reader(f):
            if not row or not row[0].strip():
                continue
            vals = [float(v) for v in row]
            if len(vals) < 33:
                continue
            y.append(int(vals[0]))
            X.append(vals[1:33])   # 32 features
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# ── 궤적 통계 ────────────────────────────────────────────────────────────────

def trajectory_stats(X, y, n_classes):
    stats = {c: {'net_x': [], 'net_y': [], 'mag': [], 'linearity': []}
             for c in range(n_classes)}

    for feat, cls in zip(X, y):
        if cls >= n_classes:
            continue
        xs = feat[0::2]   # x 값 16개
        ys = feat[1::2]   # y 값 16개

        net_x = float(xs[-1] - xs[0])
        net_y = float(ys[-1] - ys[0])
        mag   = sum(
            math.sqrt((xs[i+1]-xs[i])**2 + (ys[i+1]-ys[i])**2)
            for i in range(len(xs)-1)
        )
        straight = math.sqrt(net_x**2 + net_y**2)
        linearity = straight / mag if mag > 1e-6 else 0.0

        stats[cls]['net_x'].append(net_x)
        stats[cls]['net_y'].append(net_y)
        stats[cls]['mag'].append(mag)
        stats[cls]['linearity'].append(linearity)

    return stats


# ── TFLite 배치 추론 ─────────────────────────────────────────────────────────

def predict_all(X):
    interp = tf.lite.Interpreter(model_path=MODEL_PATH)
    interp.allocate_tensors()
    in_idx  = interp.get_input_details()[0]['index']
    out_idx = interp.get_output_details()[0]['index']

    preds, confs = [], []
    for x in X:
        interp.set_tensor(in_idx, np.array([x], dtype=np.float32))
        interp.invoke()
        prob = np.squeeze(interp.get_tensor(out_idx))
        preds.append(int(np.argmax(prob)))
        confs.append(prob)

    return np.array(preds), np.array(confs)


# ── 혼동 행렬 / 지표 ─────────────────────────────────────────────────────────

def confusion_matrix(y_true, y_pred, n):
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t < n and p < n:
            cm[t][p] += 1
    return cm


def per_class_metrics(cm):
    metrics = {}
    for i in range(len(cm)):
        tp = cm[i][i]
        fp = int(cm[:, i].sum()) - tp
        fn = int(cm[i, :].sum()) - tp
        prec   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2*prec*recall / (prec+recall) if (prec+recall) > 0 else 0.0
        metrics[i] = dict(tp=tp, fp=fp, fn=fn,
                          precision=prec, recall=recall,
                          f1=f1, support=int(cm[i, :].sum()))
    return metrics


# ── 출력 헬퍼 ────────────────────────────────────────────────────────────────

def bar(value, width=16):
    filled = int(min(value, 1.0) * width)
    return '█' * filled + '░' * (width - filled)


def mean(lst):
    return sum(lst) / len(lst) if lst else 0.0


def section(title):
    print()
    print('─' * 72)
    print(f'  {title}')
    print('─' * 72)


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    print('=' * 72)
    print('  point_history classifier 성능 분석')
    print('=' * 72)

    # 데이터 확인
    import os
    if not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0:
        print(f'\n  ⚠ 데이터 없음: {CSV_PATH}')
        print('  app.py 에서 h 키로 동적 제스처 데이터를 수집하세요.')
        return

    print(f'\n  데이터 로드 중: {CSV_PATH}')
    X, y_true = load_data()
    n_classes  = len(LABELS)
    known_mask = y_true < n_classes
    X_k, y_k   = X[known_mask], y_true[known_mask]

    # ── 수집 현황 ────────────────────────────────────────────────────────────
    section('수집 현황')
    from collections import Counter
    counts = Counter(y_true.tolist())
    total  = sum(counts.values())
    print(f'  총 샘플: {total}')
    print()
    print(f'  {"ID":<4} {"레이블":<14} {"진행도":<32} {"수집량":>6}')
    print('  ' + '-' * 60)
    for cls in sorted(LABELS.keys()):
        cnt  = counts.get(cls, 0)
        done = ' ✓' if cnt >= TARGET else '  '
        filled = int(min(cnt / TARGET, 1.0) * 20)
        pb = '[' + '█'*filled + '░'*(20-filled) + ']'
        print(f'  {cls:<4} {LABELS[cls]:<14} {pb} {cnt:>5}/{TARGET}{done}')

    unknown = {k: v for k, v in counts.items() if k not in LABELS}
    if unknown:
        print()
        for cls, cnt in sorted(unknown.items()):
            print(f'  ⚠ unknown class {cls}: {cnt}개')

    if len(y_k) == 0:
        print('\n  데이터가 부족하여 분석을 진행할 수 없습니다.')
        return

    # ── 궤적 특성 분석 ───────────────────────────────────────────────────────
    section('궤적 특성 분석 (클래스별 평균)')
    t_stats = trajectory_stats(X_k, y_k, n_classes)
    print(f'  {"ID":<4} {"레이블":<14} {"기대방향":<12} {"avg net_x":>10} {"avg net_y":>10} {"avg 거리":>9} {"직선도":>7}')
    print('  ' + '-' * 70)
    for cls in sorted(LABELS.keys()):
        s = t_stats[cls]
        if not s['net_x']:
            print(f'  {cls:<4} {LABELS[cls]:<14} (데이터 없음)')
            continue
        nx  = mean(s['net_x'])
        ny  = mean(s['net_y'])
        mg  = mean(s['mag'])
        lin = mean(s['linearity'])
        print(
            f'  {cls:<4} {LABELS[cls]:<14} {EXPECTED[cls]:<12} '
            f'{nx:>+9.4f}  {ny:>+9.4f}  {mg:>8.4f}  {lin:>6.1%}'
        )

    # ── 모델 성능 분석 ───────────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        print(f'\n  ⚠ 모델 없음: {MODEL_PATH} — 학습 후 재실행하세요.')
        return

    section('모델 성능 (전체 데이터 기준)')
    print(f'  모델: {MODEL_PATH}')
    y_pred, confs = predict_all(X_k)
    accuracy = (y_pred == y_k).mean()
    print(f'  전체 정확도: {accuracy*100:.2f}%  ({(y_pred==y_k).sum()} / {len(y_k)})')

    cm      = confusion_matrix(y_k, y_pred, n_classes)
    metrics = per_class_metrics(cm)

    section('클래스별 성능 (Precision / Recall / F1)')
    print(f'  {"ID":<4} {"레이블":<14} {"Prec":>7} {"Recall":>7} {"F1":>7} {"샘플":>6}')
    print('  ' + '-' * 50)
    for cls in sorted(LABELS.keys()):
        m = metrics[cls]
        if m['support'] == 0:
            print(f'  {cls:<4} {LABELS[cls]:<14} {"(데이터 없음)":>24}')
            continue
        print(
            f'  {cls:<4} {LABELS[cls]:<14} '
            f'{m["precision"]*100:>6.1f}%  '
            f'{m["recall"]*100:>6.1f}%  '
            f'{m["f1"]*100:>6.1f}%  '
            f'{m["support"]:>5}'
        )

    section('혼동 행렬 (행: 실제 / 열: 예측)')
    active = [c for c in sorted(LABELS.keys()) if metrics[c]['support'] > 0]
    header = '  {:>16}'.format('실제↓ 예측→')
    for c in active:
        header += f'  {LABELS[c][:11]:>11}'
    print(header)
    print('  ' + '-' * (18 + 13 * len(active)))
    for r in active:
        row_str = f'  {LABELS[r]:>16}'
        for c in active:
            marker  = '*' if r == c else ' '
            row_str += f'  {cm[r][c]:>10}{marker}'
        row_str += f'   recall {metrics[r]["recall"]*100:.1f}%'
        print(row_str)

    section('클래스별 신뢰도 (정답 샘플)')
    print(f'  {"ID":<4} {"레이블":<14} {"평균":>7} {"최소":>7} {"최대":>7}  분포')
    print('  ' + '-' * 56)
    for cls in sorted(LABELS.keys()):
        mask = (y_k == cls) & (y_pred == cls)
        if mask.sum() == 0:
            if metrics[cls]['support'] > 0:
                print(f'  {cls:<4} {LABELS[cls]:<14} (정답 없음)')
            continue
        c_vals = confs[mask, cls]
        print(
            f'  {cls:<4} {LABELS[cls]:<14} '
            f'{c_vals.mean()*100:>6.1f}%  '
            f'{c_vals.min()*100:>6.1f}%  '
            f'{c_vals.max()*100:>6.1f}%  '
            f'[{bar(c_vals.mean())}]'
        )

    section('주요 오분류 (10회 이상)')
    found = False
    for r in active:
        for c in active:
            if r != c and cm[r][c] >= 10:
                print(f'  {LABELS[r]:>14} → {LABELS[c]:<14}  {cm[r][c]}회')
                found = True
    if not found:
        print('  없음')

    print()
    print('=' * 72)


if __name__ == '__main__':
    main()
