"""
keypoint.csv 실시간 통계 추적기
실행: python watch_keypoint_csv.py
"""
import csv
import os
import time
from collections import defaultdict

CSV_PATH = 'model/keypoint_classifier/keypoint.csv'
TARGET = 1500

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

BAR_WIDTH = 20


def read_stats():
    class_counts = defaultdict(int)
    hand_counts = defaultdict(lambda: {'R': 0, 'L': 0})

    with open(CSV_PATH) as f:
        for row in csv.reader(f):
            if not row:
                continue
            cls = int(row[0])
            thumb_x = float(row[3])   # landmark 1 x (thumb CMC)
            pinky_x = float(row[35])  # landmark 17 x (pinky MCP)
            class_counts[cls] += 1
            if thumb_x > pinky_x:
                hand_counts[cls]['R'] += 1
            else:
                hand_counts[cls]['L'] += 1

    return class_counts, hand_counts


def progress_bar(count, target, width=BAR_WIDTH):
    ratio = min(count / target, 1.0)
    filled = int(ratio * width)
    bar = '█' * filled + '░' * (width - filled)
    return f'[{bar}] {count:>5}/{target}'


def render(class_counts, hand_counts):
    total = sum(class_counts.values())

    lines = []
    lines.append('=' * 72)
    lines.append(f'  keypoint.csv 실시간 통계    총 샘플: {total}')
    lines.append('=' * 72)
    lines.append(f'  {"ID":<4} {"레이블":<20} {"진행도":<30} {"전체":>5}  {"R":>5}  {"L":>5}')
    lines.append('-' * 72)

    for cls in sorted(LABELS.keys()):
        label = LABELS[cls]
        count = class_counts.get(cls, 0)
        r = hand_counts[cls]['R']
        l = hand_counts[cls]['L']
        bar = progress_bar(count, TARGET)

        done = ' ✓' if count >= TARGET else '  '
        lines.append(f'  {cls:<4} {label:<20} {bar}  {r:>5}  {l:>5}{done}')

    # 알 수 없는 클래스 경고
    unknown = {k: v for k, v in class_counts.items() if k not in LABELS}
    if unknown:
        lines.append('-' * 72)
        for cls, cnt in sorted(unknown.items()):
            lines.append(f'  ⚠ unknown class {cls}: {cnt}개 — 삭제 권장')

    lines.append('=' * 72)
    lines.append(f'  갱신: {time.strftime("%H:%M:%S")}  |  Ctrl+C 로 종료')
    return lines


def clear_lines(n):
    for _ in range(n):
        print('\033[F\033[K', end='')


def main():
    last_mtime = None
    last_line_count = 0

    print(f'모니터링 중: {CSV_PATH}\n')

    try:
        while True:
            mtime = os.path.getmtime(CSV_PATH)
            if mtime != last_mtime:
                last_mtime = mtime
                class_counts, hand_counts = read_stats()
                lines = render(class_counts, hand_counts)

                clear_lines(last_line_count)
                print('\n'.join(lines))
                last_line_count = len(lines)

            time.sleep(0.5)

    except KeyboardInterrupt:
        print('\n종료')


if __name__ == '__main__':
    main()
