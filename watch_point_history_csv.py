"""
point_history.csv 실시간 통계 추적기
실행: python watch_point_history_csv.py
"""
import csv
import os
import time
from collections import defaultdict
import math

CSV_PATH = 'model/point_history_classifier/point_history.csv'
TARGET   = 1500

LABELS = {
    0: 'swipe_left',
    1: 'swipe_right',
    2: 'swipe_up',
    3: 'swipe_down',
}

BAR_WIDTH = 20
# 각 클래스의 기대 방향 (net_x 부호, net_y 부호)
EXPECTED_DIR = {
    0: ('←', 'net_x < 0'),
    1: ('→', 'net_x > 0'),
    2: ('↑', 'net_y < 0'),
    3: ('↓', 'net_y > 0'),
}


def read_stats():
    class_counts  = defaultdict(int)
    class_net_x   = defaultdict(list)   # 마지막 x 위치 (궤적 전체 변위)
    class_net_y   = defaultdict(list)
    class_mag     = defaultdict(list)   # 궤적 총 이동 거리

    with open(CSV_PATH) as f:
        for row in csv.reader(f):
            if not row or not row[0].strip():
                continue
            vals = [float(v) for v in row]
            cls  = int(vals[0])
            pts  = vals[1:]   # x0,y0, x1,y1, ..., x15,y15  (32개)

            if len(pts) < 32:
                continue

            xs = pts[0::2]   # x 값 16개
            ys = pts[1::2]   # y 값 16개

            net_x = xs[-1] - xs[0]
            net_y = ys[-1] - ys[0]
            mag   = sum(
                math.sqrt((xs[i+1]-xs[i])**2 + (ys[i+1]-ys[i])**2)
                for i in range(len(xs)-1)
            )

            class_counts[cls] += 1
            class_net_x[cls].append(net_x)
            class_net_y[cls].append(net_y)
            class_mag[cls].append(mag)

    return class_counts, class_net_x, class_net_y, class_mag


def progress_bar(count, target, width=BAR_WIDTH):
    ratio = min(count / target, 1.0)
    filled = int(ratio * width)
    return f'[{"█"*filled}{"░"*(width-filled)}] {count:>5}/{target}'


def mean(lst):
    return sum(lst) / len(lst) if lst else 0.0


def render(class_counts, class_net_x, class_net_y, class_mag):
    total = sum(class_counts.values())
    lines = []
    lines.append('=' * 76)
    lines.append(f'  point_history.csv 실시간 통계    총 샘플: {total}')
    lines.append('=' * 76)

    # 수집 현황
    lines.append(f'  {"ID":<4} {"레이블":<14} {"진행도":<30} {"평균net_x":>9} {"평균net_y":>9} {"평균거리":>8}')
    lines.append('  ' + '-' * 72)

    for cls in sorted(LABELS.keys()):
        label    = LABELS[cls]
        count    = class_counts.get(cls, 0)
        bar      = progress_bar(count, TARGET)
        done     = ' ✓' if count >= TARGET else '  '
        arrow, _ = EXPECTED_DIR[cls]

        if count > 0:
            nx  = mean(class_net_x[cls])
            ny  = mean(class_net_y[cls])
            mg  = mean(class_mag[cls])
            lines.append(
                f'  {cls:<4} {label:<14} {bar}  {nx:>+8.3f}  {ny:>+8.3f}  {mg:>7.3f}{done}'
            )
        else:
            lines.append(f'  {cls:<4} {label:<14} {bar}{done}')

    # 알 수 없는 클래스 경고
    unknown = {k: v for k, v in class_counts.items() if k not in LABELS}
    if unknown:
        lines.append('  ' + '-' * 72)
        for cls, cnt in sorted(unknown.items()):
            lines.append(f'  ⚠ unknown class {cls}: {cnt}개 — 삭제 권장')

    lines.append('=' * 76)
    lines.append(f'  갱신: {time.strftime("%H:%M:%S")}  |  Ctrl+C 로 종료')
    return lines


def clear_lines(n):
    for _ in range(n):
        print('\033[F\033[K', end='')


def main():
    last_mtime      = None
    last_line_count = 0
    print(f'모니터링 중: {CSV_PATH}\n')

    try:
        while True:
            if not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0:
                msg = f'\r  대기 중... {time.strftime("%H:%M:%S")} (데이터 없음)'
                print(msg, end='', flush=True)
                time.sleep(1)
                continue

            mtime = os.path.getmtime(CSV_PATH)
            if mtime != last_mtime:
                last_mtime = mtime
                class_counts, class_net_x, class_net_y, class_mag = read_stats()
                lines = render(class_counts, class_net_x, class_net_y, class_mag)
                clear_lines(last_line_count)
                print('\n'.join(lines))
                last_line_count = len(lines)

            time.sleep(0.5)

    except KeyboardInterrupt:
        print('\n종료')


if __name__ == '__main__':
    main()
