import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from ultralytics import YOLO

# === CONFIG ===
video_path = 'test.mp4'
model = YOLO('best.pt')

# === THRESHOLDS ===
JUMP_VELOCITY_THRESHOLD = 0.012
STANCE_HIP_Y_MIN = 0.5
RELEASE_WRIST_HEAD_DIFF = 0.03
BALL_RELEASE_DISTANCE_THRESHOLD = 40
LANDING_FOOT_Y_THRESHOLD = 0.80
LANDING_HIP_VELOCITY_THRESHOLD = 0.002
APEX_HIP_VELOCITY_THRESHOLD = 0.001
cooldown_after_landing = 0.2

# === INIT VIDEO ===
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ Could not open video.")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
width, height = int(cap.get(3)), int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('end_moo12.mp4', fourcc, fps, (width, height))

# === INIT POSE ===
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# === STATE ===
phase = "Unknown"
frame_counter = 0
phase_start_frame = 0
phase_log = deque(maxlen=10)

jumping = False
released = False
apex_reached = False
in_air = False
trajectory_started = False
ball_scored = False
shot_counted = False
waiting_for_result = False

total_shots = 0
made_shots = 0

prev_hip_y = None
prev_foot_y = None
release_frame = None
release_time = None
landing_time = None
last_release_frame = None
ball_positions = deque(maxlen=100)
ball_missing_frames = 0
max_wait_frames = 30
wait_counter = 0

def boxes_overlap(boxA, boxB):
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    return not (min(ax2, bx2) < max(ax1, bx1) or min(ay2, by2) < max(ay1, by1))

def draw_text_with_background(img, text, pos, text_color=(0,0,0), bg_color=(255,255,255), font_scale=1, thickness=2):
    font = cv2.FONT_HERSHEY_TRIPLEX
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size
    x, y = pos
    cv2.rectangle(img, (x, y - text_h - 5), (x + text_w + 5, y + 5), bg_color, -1)
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness)

with mp_pose.Pose(static_image_mode=False, model_complexity=1) as pose:
    print("✅ Processing video...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("⚠️ End of video.")
            break

        frame_counter += 1
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        yolo_results = model(frame)[0]
        detections = yolo_results.boxes
        ball_box, person_box, rim_box = None, None, None

        for box in detections:
            cls = int(box.cls.item())
            label = model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf.item()
            color = (255, 0, 0) if label == 'person' else (0, 255, 0) if label == 'rim' else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            draw_text_with_background(frame, f'{label} {conf:.2f}', (x1, y1 - 10))

            if label.lower() == 'person':
                person_box = (x1, y1, x2, y2)
            elif label.lower() == 'ball':
                ball_box = (x1, y1, x2, y2)
            elif label.lower() == 'rim':
                rim_box = (x1, y1, x2, y2)

        if ball_box is None and released and len(ball_positions) >= 2:
            (x1, y1), (x2, y2) = ball_positions[-2], ball_positions[-1]
            dx, dy = x2 - x1, y2 - y1
            ghost_cx, ghost_cy = x2 + dx, y2 + dy
            ball_positions.append((ghost_cx, ghost_cy))
            cv2.circle(frame, (int(ghost_cx), int(ghost_cy)), 8, (200, 200, 200), -1)
            ball_missing_frames += 1
        elif ball_box is not None:
            ball_missing_frames = 0

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )

            landmarks = results.pose_landmarks.landmark
            hip_y = np.mean([landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
                             landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y])
            foot_y = np.mean([landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y,
                              landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y])
            wrist_y = np.mean([landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
                               landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y])
            head_y = landmarks[mp_pose.PoseLandmark.NOSE.value].y

            if prev_hip_y is not None and prev_foot_y is not None:
                hip_vel = prev_hip_y - hip_y

                if not jumping and abs(hip_vel) < 0.005 and hip_y > STANCE_HIP_Y_MIN:
                    new_phase = "Stance"
                    released = False
                    apex_reached = False

                elif not jumping and hip_vel > JUMP_VELOCITY_THRESHOLD:
                    new_phase = "Jump"
                    jumping = True
                    released = False
                    apex_reached = False

                elif jumping and not released and wrist_y < head_y - RELEASE_WRIST_HEAD_DIFF:
                    new_phase = "Release"
                    released = True
                    release_frame = frame_counter
                    release_time = release_frame / fps
                    last_release_frame = frame_counter
                    in_air = True
                    trajectory_started = True
                    ball_positions.clear()
                    waiting_for_result = True
                    wait_counter = 0

                elif jumping and released and person_box is not None:
                    if len(ball_positions) > 0:
                        ball_cx, ball_cy = ball_positions[-1]
                        px1, py1, px2, py2 = person_box
                        pcx, pcy = (px1 + px2) // 2, (py1 + py2) // 2
                        ph = py2 - py1
                        dist = np.linalg.norm([ball_cx - pcx, ball_cy - pcy])
                        if dist > ph * 0.6:
                            new_phase = "Landing"
                            in_air = False
                            landing_time = frame_counter / fps
                        else:
                            new_phase = phase

                elif not in_air and jumping and last_release_frame is not None:
                    time_since_release = (frame_counter - last_release_frame) / fps
                    if time_since_release >= cooldown_after_landing and hip_y > STANCE_HIP_Y_MIN:
                        new_phase = "Stance"
                        jumping = False
                        released = False
                        apex_reached = False
                        trajectory_started = False
                        ball_scored = False
                        shot_counted = False
                        waiting_for_result = False
                        landing_time = None
                    else:
                        new_phase = phase
                else:
                    new_phase = phase

                if new_phase != phase:
                    duration = (frame_counter - phase_start_frame) / fps
                    if phase != "Unknown":
                        phase_log.appendleft((phase, duration))
                    phase_start_frame = frame_counter
                    phase = new_phase

            prev_hip_y = hip_y
            prev_foot_y = foot_y

        if ball_box is not None:
            cx = (ball_box[0] + ball_box[2]) // 2
            cy = (ball_box[1] + ball_box[3]) // 2
            if trajectory_started:
                ball_positions.append((cx, cy))

        if released and not shot_counted and waiting_for_result:
            if rim_box and ball_box:
                if boxes_overlap(ball_box, rim_box):
                    ball_scored = True
                    waiting_for_result = False
                else:
                    wait_counter += 1
                    if wait_counter >= max_wait_frames:
                        ball_scored = False
                        waiting_for_result = False

            if not waiting_for_result:
                total_shots += 1
                made_shots += int(ball_scored)
                shot_label = "✅ SCORED" if ball_scored else "❌ MISS"
                shot_color = (0, 255, 0) if ball_scored else (0, 0, 255)
                if rim_box:
                    draw_text_with_background(frame, shot_label, (rim_box[0], rim_box[1] - 30), shot_color)

                if trajectory_started and len(ball_positions) > 1:
                    for i in range(1, len(ball_positions)):
                        cv2.line(frame, ball_positions[i - 1], ball_positions[i], shot_color, 2)

                shot_counted = True

        if release_time is not None:
            draw_text_with_background(frame, f'Release at: {release_time:.2f}s', (30, 100))

        if landing_time is not None:
            draw_text_with_background(frame, f'Landing at: {landing_time:.2f}s', (30, 140))

        draw_text_with_background(frame, f'Phase: {phase}', (30, 60), font_scale=1.2)
        draw_text_with_background(frame, f'Shots: {made_shots}/{total_shots}', (30, 180))

        y_offset = 40
        for i, (ph, dur) in enumerate(phase_log):
            draw_text_with_background(frame, f'{ph}: {dur:.2f}s', (width - 300, y_offset + i * 30), font_scale=0.8)

        out.write(frame)
        cv2.imshow('Jump Shot Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 Interrupted by user.")
            break

if phase != "Unknown":
    duration = (frame_counter - phase_start_frame) / fps
    phase_log.appendleft((phase, duration))

cap.release()
out.release()
cv2.destroyAllWindows()

with open("release_info.txt", "w", encoding="utf-8") as f:
    if release_time is not None:
        f.write(f"Release detected at frame {release_frame} ({release_time:.2f} seconds)\n")
        f.write("Shot made (ball entered rim)\n" if ball_scored else "Shot missed\n")
    else:
        f.write("No release detected.\n")

print("✅ Done. Annotated video saved as 'end_moo1.mp4'")
#