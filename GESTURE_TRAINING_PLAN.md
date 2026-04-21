# 커스텀 제스처 학습 계획 (10-class static only)

## 목표 제스처 — keypoint classifier (10클래스)

| 클래스 ID | 제스처 | 매핑 |
| --- | --- | --- |
| 0 | pinch | 메뉴 선택 |
| 1 | ok_sign | 결제 화면 이동 |
| 2 | thumb_left | 이전 화면 이동 |
| 3 | open_palm | 음성 인식 활성화 |
| 4 | one_finger_left | 포커스 좌 이동 |
| 5 | one_finger_right | 포커스 우 이동 |
| 6 | one_finger_up | 포커스 상 이동 |
| 7 | one_finger_down | 포커스 하 이동 |
| 8 | two_finger_left | 이전 페이지 |
| 9 | two_finger_right | 다음 페이지 |

> point_history classifier 불필요 — 방향 정보를 정적 포즈에 포함

---

## 아키텍처

- **단일 파이프라인**: MediaPipe → keypoint 전처리 → keypoint classifier → 액션
- point_history, sliding window, active_trigger_id 모두 제거됨
- `app.py`: keypoint_classifier만 사용, h키 / point_history 로직 없음

---

## 데이터 수집 현황

| 클래스 | 현재 샘플 | 목표 |
| --- | --- | --- |
| 0 pinch | 1,000 | 1,000 ✓ |
| 1 ok_sign | 1,000 | 1,000 ✓ |
| 2 thumb_left | 1,000 | 1,000 ✓ |
| 3 open_palm | 1,000 | 1,000 ✓ |
| 4 one_finger_left | 0 | 1,000 |
| 5 one_finger_right | 0 | 1,000 |
| 6 one_finger_up | 0 | 1,000 |
| 7 one_finger_down | 0 | 1,000 |
| 8 two_finger_left | 0 | 1,000 |
| 9 two_finger_right | 0 | 1,000 |

---

## Step 1: 데이터 수집 (클래스 4~9)

`python app.py` → `k` 키 → HUD: "MODE: Logging Key Point"

| 클래스 | 키 | 자세 |
| --- | --- | --- |
| 4 one_finger_left | `4` | 검지 하나, 왼쪽을 향해 가리키기 |
| 5 one_finger_right | `5` | 검지 하나, 오른쪽을 향해 가리키기 |
| 6 one_finger_up | `6` | 검지 하나, 위를 향해 가리키기 |
| 7 one_finger_down | `7` | 검지 하나, 아래를 향해 가리키기 |
| 8 two_finger_left | `8` | 검지+중지, 왼쪽 방향 |
| 9 two_finger_right | `9` | 검지+중지, 오른쪽 방향 |

**데이터 품질 팁:**
- 방향이 명확하게 유지된 정적 포즈로 수집
- 거리 변화: 30cm / 50cm / 80cm
- 각도 변화: ±10도 회전

---

## Step 2: keypoint classifier 재학습

```bash
conda run -n tf-mac-metal python keypoint_classification.py
```

- `NUM_CLASSES = 10` 확인
- 출력: `model/keypoint_classifier/keypoint_classifier.tflite` 갱신

---

## Step 3: 성능 검증

```bash
conda run -n tf-mac-metal python analyze_keypoint_classifier.py
```

---

## 검증 체크리스트

- [ ] pinch(0) / ok_sign(1) / thumb_left(2) / open_palm(3) 각각 인식됨
- [ ] one_finger_left(4) / right(5) / up(6) / down(7) 방향별 인식됨
- [ ] two_finger_left(8) / right(9) 인식됨
- [ ] 모든 클래스 Recall ≥ 85%
