"""
Rock Paper Scissors with OpenCV + MediaPipe (Tasks API)

Install:  pip install opencv-python mediapipe
Run:      python rps_game.py

Controls:
  SPACE - start a round (3, 2, 1, SHOOT!)
  R     - reset score
  Q/ESC - quit
"""
import math
import os
import random
import time
import urllib.request
from collections import Counter, deque

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

MOVES = ("Rock", "Paper", "Scissors")
BEATS = {"Rock": "Scissors", "Paper": "Rock", "Scissors": "Paper"}

STEP_SECONDS = 0.8      # length of each "3", "2", "1" beat
CAPTURE_SECONDS = 0.5   # how long after SHOOT the game reads your hand
RESULT_SECONDS = 3.0
SHUFFLE_SECONDS = 0.12  # how fast the computer panel cycles while "thinking"

# (tip, pip) landmark ids for index, middle, ring, pinky
FINGERS = [(8, 6), (12, 10), (16, 14), (20, 18)]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmark model (one time)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


class Vote:
    """Sliding-window majority vote over recent gesture detections."""

    def __init__(self, window=0.4):
        self.window = window
        self.samples = deque()

    def add(self, t, gesture):
        if gesture:
            self.samples.append((t, gesture))
        while self.samples and t - self.samples[0][0] > self.window:
            self.samples.popleft()

    def best(self):
        if not self.samples:
            return None
        return Counter(g for _, g in self.samples).most_common(1)[0][0]

    def clear(self):
        self.samples.clear()


def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def classify(lm):
    """Return 'Rock', 'Paper', 'Scissors' or None from a list of 21 landmarks."""
    wrist = lm[0]
    # Finger extended if its tip is farther from the wrist than its middle joint.
    ext = [dist(wrist, lm[tip]) > dist(wrist, lm[pip]) for tip, pip in FINGERS]
    n = sum(ext)

    if n == 0:
        return "Rock"
    if n == 4:
        return "Paper"
    if ext[0] and ext[1] and not ext[2] and not ext[3]:
        return "Scissors"
    return None  # e.g. one finger or three fingers: not a valid throw


def decide(player, computer):
    if player == computer:
        return "Draw"
    return "You win!" if BEATS[player] == computer else "Computer wins!"


# ---------- drawing helpers ----------

def draw_hand(img, lm):
    h, w = img.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in lm]
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, pts[a], pts[b], (0, 255, 0), 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, p, 4, (0, 0, 255), -1, cv2.LINE_AA)


def put_text(img, text, org, scale=1.0, color=(255, 255, 255), thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def put_text_centered(img, text, y, scale, color, thick, cx=None):
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cx = img.shape[1] // 2 if cx is None else cx
    put_text(img, text, (cx - tw // 2, y), scale, color, thick)


def draw_icon(img, move, cx, cy, s):
    """Simple icons: rock = circle, paper = sheet, scissors = crossed blades."""
    if move == "Rock":
        cv2.circle(img, (cx, cy), s, (130, 130, 140), -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), s, (60, 60, 70), 3, cv2.LINE_AA)
    elif move == "Paper":
        x0, y0, x1, y1 = int(cx - 0.8 * s), cy - s, int(cx + 0.8 * s), cy + s
        cv2.rectangle(img, (x0, y0), (x1, y1), (245, 245, 245), -1)
        cv2.rectangle(img, (x0, y0), (x1, y1), (90, 90, 90), 3)
        for i in range(1, 5):
            yy = y0 + i * (y1 - y0) // 5
            cv2.line(img, (x0 + 8, yy), (x1 - 8, yy), (170, 170, 170), 2)
    elif move == "Scissors":
        tl, tr = (int(cx - 0.7 * s), cy - s), (int(cx + 0.7 * s), cy - s)
        bl, br = (int(cx - 0.5 * s), int(cy + 0.9 * s)), (int(cx + 0.5 * s), int(cy + 0.9 * s))
        cv2.line(img, tl, br, (220, 220, 220), 6, cv2.LINE_AA)
        cv2.line(img, tr, bl, (220, 220, 220), 6, cv2.LINE_AA)
        cv2.circle(img, bl, int(0.25 * s), (60, 60, 255), 4, cv2.LINE_AA)
        cv2.circle(img, br, int(0.25 * s), (60, 60, 255), 4, cv2.LINE_AA)
    else:  # unknown / hidden
        put_text_centered(img, "?", cy + int(0.5 * s), s / 25, (200, 200, 200), 5, cx=cx)


def draw_computer_panel(frame, move, caption, highlight=None):
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = w - 260, 20, w - 20, 260
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    border = highlight or (200, 200, 200)
    cv2.rectangle(frame, (x0, y0), (x1, y1), border, 3)
    cx = (x0 + x1) // 2
    put_text_centered(frame, "COMPUTER", y0 + 35, 0.8, (255, 255, 255), 2, cx=cx)
    draw_icon(frame, move, cx, y0 + 125, 55)
    put_text_centered(frame, caption, y1 - 15, 0.65, border, 2, cx=cx)


# ---------- game loop ----------

def run_game(cap, landmarker):
    state = "IDLE"  # IDLE -> COUNTDOWN -> SHOOT -> RESULT -> IDLE
    state_start = 0.0
    vote = Vote(window=0.4)
    computer_pick = player_move = outcome = None
    score = {"You": 0, "Computer": 0, "Draws": 0}
    t0 = time.time()
    last_ts = -1

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # mirror view
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = max(int((time.time() - t0) * 1000), last_ts + 1)  # must strictly increase
        last_ts = ts
        result = landmarker.detect_for_video(mp_image, ts)

        live = None
        if result.hand_landmarks:
            lm = result.hand_landmarks[0]
            draw_hand(frame, lm)
            live = classify(lm)

        now = time.time()
        vote.add(now, live)
        smooth = vote.best()

        if state == "IDLE":
            put_text(frame, "Press SPACE to play", (20, 50), 1.1, (0, 255, 255))
            draw_computer_panel(frame, None, "waiting...")

        elif state == "COUNTDOWN":
            elapsed = now - state_start
            if elapsed < 3 * STEP_SECONDS:
                number = 3 - int(elapsed / STEP_SECONDS)
                put_text_centered(frame, str(number), h // 2 + 40, 5, (0, 200, 255), 8, cx=w // 2 - 120)
                put_text(frame, "Get ready...", (20, 50), 1.1, (0, 255, 255))
                shown = MOVES[int(elapsed / SHUFFLE_SECONDS) % 3]  # shuffle animation
                draw_computer_panel(frame, shown, "thinking...", (0, 200, 255))
            else:
                state, state_start = "SHOOT", now
                vote.clear()

        elif state == "SHOOT":
            put_text_centered(frame, "SHOOT!", h // 2 + 30, 3, (0, 255, 0), 7, cx=w // 2 - 120)
            draw_computer_panel(frame, computer_pick, "locked in!", (0, 255, 0))
            if now - state_start >= CAPTURE_SECONDS:
                player_move = smooth
                if player_move is None:
                    outcome = "No clear throw - try again"
                else:
                    outcome = decide(player_move, computer_pick)
                    if outcome == "You win!":
                        score["You"] += 1
                    elif outcome == "Computer wins!":
                        score["Computer"] += 1
                    else:
                        score["Draws"] += 1
                state, state_start = "RESULT", now

        elif state == "RESULT":
            color = (
                (0, 255, 0) if outcome == "You win!"
                else (0, 0, 255) if outcome == "Computer wins!"
                else (0, 255, 255)
            )
            put_text(frame, f"You: {player_move or '-'}", (20, 50), 1.1)
            put_text(frame, f"Computer: {computer_pick}", (20, 95), 1.1)
            put_text(frame, outcome, (20, 150), 1.2, color, 3)
            draw_computer_panel(frame, computer_pick, computer_pick, color)
            if now - state_start > RESULT_SECONDS:
                state = "IDLE"

        put_text(frame, f"Detected: {smooth or '-'}", (20, h - 60), 0.8, (200, 255, 200))
        put_text(
            frame,
            f"You {score['You']}  Computer {score['Computer']}  Draws {score['Draws']}",
            (20, h - 20),
            0.8,
        )

        cv2.imshow("Rock Paper Scissors", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" ") and state in ("IDLE", "RESULT"):
            computer_pick = random.choice(MOVES)  # chosen now, revealed at SHOOT
            state, state_start = "COUNTDOWN", time.time()
            vote.clear()
        if key == ord("r"):
            score = {"You": 0, "Computer": 0, "Draws": 0}


def open_camera():
    for index in (0, 1, 2):
        for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF):
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                return cap
            cap.release()
    raise SystemExit("Could not open webcam.")


def main():
    ensure_model()
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.6,
    )
    cap = open_camera()
    landmarker = vision.HandLandmarker.create_from_options(options)
    try:
        run_game(cap, landmarker)
    finally:
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()