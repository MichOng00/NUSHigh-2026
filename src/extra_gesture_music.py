import cv2
import mediapipe as mp
import time
import numpy as np
import pygame

# ==============================
# AUDIO SETUP (stereo-safe)
# ==============================
pygame.mixer.pre_init(44100, -16, 2, 512)
pygame.init()
pygame.mixer.init()
pygame.mixer.set_num_channels(32)

SR = 44100

def make_piano_note(freq, duration=1.6, sr=SR, vol=0.35, sustain=False):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    harmonics = [(1,1.00),(2,0.35),(3,0.22),(4,0.14),(5,0.10),(6,0.06)]
    detune = 0.003
    wave = np.zeros_like(t, dtype=np.float32)

    for k, a in harmonics:
        wave += a * np.sin(2*np.pi*(freq*k)*t)
        wave += (a*0.25) * np.sin(2*np.pi*(freq*k*(1+detune))*t)

    attack = 0.01
    aN = int(sr * attack)
    env = np.ones_like(t, dtype=np.float32)
    if aN > 0:
        env[:aN] = np.linspace(0, 1, aN, endpoint=False)

    decay_rate = 1.8 if sustain else 3.5
    env *= np.exp(-decay_rate * t)

    wave *= env
    wave = np.tanh(1.2 * wave)

    wave /= (np.max(np.abs(wave)) + 1e-9)
    wave_i16 = (wave * (32767 * vol)).astype(np.int16)

    if pygame.mixer.get_init()[2] == 2:
        wave_i16 = np.column_stack((wave_i16, wave_i16))

    return pygame.sndarray.make_sound(wave_i16)

# ==============================
# NOTES + OCTAVES
# ==============================
BASE_FREQ = {
    "C": 261.63, "D": 293.66, "E": 329.63,
    "F": 349.23, "G": 392.00, "A": 440.00, "B": 493.88,
}
OCTAVES = [-1, 0, 1]

tones = {n: {} for n in BASE_FREQ}
tones_sus = {n: {} for n in BASE_FREQ}
for note, base in BASE_FREQ.items():
    for o in OCTAVES:
        f = base * (2 ** o)
        tones[note][o] = make_piano_note(f, duration=1.2, sustain=False)
        tones_sus[note][o] = make_piano_note(f, duration=2.4, sustain=True)

# ==============================
# MEDIAPIPE
# ==============================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def fingers_up(lm, label):
    f = [0]*5
    if label == "Right":
        f[0] = 1 if lm[4].x < lm[3].x else 0
    else:
        f[0] = 1 if lm[4].x > lm[3].x else 0
    for i, (t, p) in enumerate([(8,6),(12,10),(16,14),(20,18)], start=1):
        f[i] = 1 if lm[t].y < lm[p].y else 0
    return f

# ==============================
# MAPPING
# ==============================
FINGER_NOTE = {
    1: "C",  # left index
    0: "D",  # left thumb
    5: "E",  # right thumb
    6: "F",  # right index
    7: "G",  # right middle
    8: "A",  # right ring
    9: "B",  # right pinky
}
NOTE_TO_BIT = {v: k for k, v in FINGER_NOTE.items()}

SUSTAIN_BIT = 2
OCTAVE_BIT  = 3

# ==============================
# UI: KEYBOARD LAYOUT
# ==============================
UI_KEYS = [
    ("SUS", 2),
    ("OCT", 3),
    ("C", 1),
    ("D", 0),
    ("E", 5),
    ("F", 6),
    ("G", 7),
    ("A", 8),
    ("B", 9),
]

def keyboard_geometry(frame):
    h, w = frame.shape[:2]
    kb_h = int(h * 0.22)
    y0 = h - kb_h - 10
    x0 = 10
    pad = 6
    n = len(UI_KEYS)
    key_w = int((w - 2*x0 - (n-1)*pad) / n)
    key_h = kb_h
    return x0, y0, key_w, key_h, pad

def bit_to_key_center_x(frame, bit_idx):
    x0, y0, key_w, key_h, pad = keyboard_geometry(frame)
    for i, (_, b) in enumerate(UI_KEYS):
        if b == bit_idx:
            x1 = x0 + i * (key_w + pad)
            x2 = x1 + key_w
            return (x1 + x2) / 2
    return frame.shape[1] / 2

def draw_keyboard(frame, cur_bits, flash_until, octave, sustain_on):
    h, w = frame.shape[:2]
    x0, y0, key_w, key_h, pad = keyboard_geometry(frame)

    now = time.time()
    cv2.rectangle(frame, (0, y0-60), (w, h), (15, 15, 15), -1)

    for i, (name, bit_idx) in enumerate(UI_KEYS):
        x1 = x0 + i * (key_w + pad)
        y1 = y0
        x2 = x1 + key_w
        y2 = y1 + key_h

        pressed = (cur_bits[bit_idx] == 0)
        glowing = (flash_until.get(bit_idx, 0) > now)

        base = (40, 40, 40) if name in ("SUS", "OCT") else (230, 230, 230)
        if glowing:
            fill = (80, 220, 255)
        elif pressed:
            fill = (180, 180, 180)
        else:
            fill = base

        cv2.rectangle(frame, (x1, y1), (x2, y2), fill, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)

        txt_color = (0, 0, 0) if name not in ("SUS", "OCT") else (220, 220, 220)
        cv2.putText(frame, name, (x1 + 10, y2 - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, txt_color, 2)

    oct_txt = { -1: "LOW", 0: "MID", 1: "HIGH" }[octave]
    sus_txt = "ON" if sustain_on else "OFF"
    cv2.putText(frame, f"Sustain: {sus_txt}   Octave: {oct_txt} ({octave})   (G: song on/off, R: restart)",
                (10, y0 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)

# ==============================
# SONG CHART: Twinkle Twinkle
# Notes are in your available set: C D E F G A B
# Each tuple = (note, beats)
# ==============================
SONG_NAME = "Twinkle Twinkle Little Star"
BPM = 90  # change this for speed
BEAT = 60.0 / BPM

SONG = [
    ("C",1),("C",1),("G",1),("G",1),("A",1),("A",1),("G",2),
    ("F",1),("F",1),("E",1),("E",1),("D",1),("D",1),("C",2),
    ("G",1),("G",1),("F",1),("F",1),("E",1),("E",1),("D",2),
    ("G",1),("G",1),("F",1),("F",1),("E",1),("E",1),("D",2),
    ("C",1),("C",1),("G",1),("G",1),("A",1),("A",1),("G",2),
    ("F",1),("F",1),("E",1),("E",1),("D",1),("D",1),("C",2),
]

# ==============================
# MINI GAME: falling notes
# ==============================
class FallingNote:
    def __init__(self, note, x, y=-40, speed=240):
        self.note = note
        self.x = x
        self.y = y
        self.speed = speed
        self.hit = False
        self.missed = False

def draw_song_game(frame, falling, hit_y, score, combo, song_on, idx, total):
    h, w = frame.shape[:2]
    cv2.line(frame, (0, int(hit_y)), (w, int(hit_y)), (255, 255, 255), 2)

    status = "ON" if song_on else "OFF"
    cv2.putText(frame, f"Song Mode: {status}  {SONG_NAME}  ({idx}/{total})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)
    cv2.putText(frame, f"Score: {score}   Combo: {combo}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255,255,255), 2)

    for n in falling:
        if n.hit:
            col = (80, 220, 255)
        elif n.missed:
            col = (80, 80, 255)
        else:
            col = (255, 255, 255)

        cv2.rectangle(frame, (int(n.x - 28), int(n.y - 18)), (int(n.x + 28), int(n.y + 18)), col, -1)
        cv2.rectangle(frame, (int(n.x - 28), int(n.y - 18)), (int(n.x + 28), int(n.y + 18)), (0,0,0), 2)
        cv2.putText(frame, n.note, (int(n.x - 10), int(n.y + 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)

# ==============================
# MAIN
# ==============================
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return

    prev = [0]*10
    octave = 0
    sustain_on = False
    ringing = {}

    cooldown = 0.05
    last_play = 0.0

    flash_until = {}
    flash_time = 0.18

    # Song game state
    song_on = False
    score = 0
    combo = 0
    falling = []

    song_index = 0
    next_spawn_time = None

    last_frame_t = time.time()

    def reset_song():
        nonlocal score, combo, falling, song_index, next_spawn_time
        score = 0
        combo = 0
        falling.clear()
        song_index = 0
        next_spawn_time = time.time() + 0.7

    with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.5) as hands:
        print("🎹 Air Piano + Song Mode | Q quit | G toggle song | R restart song")

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            L = [0]*5
            R = [0]*5

            if res.multi_hand_landmarks and res.multi_handedness:
                for lm, hnd in zip(res.multi_hand_landmarks, res.multi_handedness):
                    mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
                    label = hnd.classification[0].label
                    if label == "Left":
                        L = fingers_up(lm.landmark, "Left")
                    else:
                        R = fingers_up(lm.landmark, "Right")

            cur = L + R
            fallen_edges = [i for i in range(10) if prev[i] == 1 and cur[i] == 0]

            now = time.time()
            dt = now - last_frame_t
            last_frame_t = now

            # flash keys
            for i in fallen_edges:
                flash_until[i] = now + flash_time

            sustain_now = (cur[SUSTAIN_BIT] == 0)

            if sustain_on and not sustain_now:
                for ch in list(ringing.values()):
                    if ch:
                        ch.fadeout(250)
                ringing.clear()

            sustain_on = sustain_now

            # octave toggle
            if OCTAVE_BIT in fallen_edges:
                octave += 1
                if octave > 1:
                    octave = -1
                flash_until[OCTAVE_BIT] = now + 0.25

            # play notes normally
            if fallen_edges and (now - last_play) > cooldown:
                for i in fallen_edges:
                    note = FINGER_NOTE.get(i)
                    if not note:
                        continue

                    snd = tones_sus[note][octave] if sustain_on else tones[note][octave]
                    old = ringing.get(note)
                    if old:
                        old.fadeout(80)

                    ch = pygame.mixer.find_channel(True)
                    ch.play(snd)

                    if sustain_on:
                        ringing[note] = ch

                last_play = now

            # song game geometry
            x0, y0, key_w, key_h, pad = keyboard_geometry(frame)
            hit_y = y0 - 65
            hit_window = 50

            # spawn scheduled song notes
            if song_on:
                if next_spawn_time is None:
                    next_spawn_time = now + 0.7

                while song_index < len(SONG) and now >= next_spawn_time:
                    note, beats = SONG[song_index]
                    bit = NOTE_TO_BIT[note]
                    center_x = bit_to_key_center_x(frame, bit)

                    # choose speed so it reaches hit line in ~1.4s
                    travel_time = 1.4
                    start_y = -40
                    dist = (hit_y - start_y)
                    speed = dist / travel_time

                    falling.append(FallingNote(note=note, x=center_x, y=start_y, speed=speed))

                    next_spawn_time += beats * BEAT
                    song_index += 1

            # move falling notes
            for n in falling:
                if not n.hit and not n.missed:
                    n.y += n.speed * dt
                    if n.y > hit_y + 90:
                        n.missed = True
                        combo = 0

            # hit detection (player presses correct note near hit line)
            if song_on and fallen_edges:
                pressed_notes = []
                for b in fallen_edges:
                    nm = FINGER_NOTE.get(b)
                    if nm in BASE_FREQ:
                        pressed_notes.append(nm)

                for pn in pressed_notes:
                    for n in falling:
                        if n.hit or n.missed:
                            continue
                        if n.note != pn:
                            continue
                        if abs(n.y - hit_y) <= hit_window:
                            n.hit = True
                            combo += 1
                            score += 120 + combo * 6
                            flash_until[NOTE_TO_BIT[pn]] = now + 0.25
                            break

            # cleanup
            falling = [n for n in falling if n.y < frame.shape[0] + 120 and not (n.hit and n.y > hit_y + 120)]

            # draw overlays
            draw_song_game(frame, falling, hit_y, score, combo, song_on, song_index, len(SONG))
            draw_keyboard(frame, cur, flash_until, octave, sustain_on)

            # if song ended, auto stop (optional)
            if song_on and song_index >= len(SONG) and len(falling) == 0:
                song_on = False

            cv2.imshow("Air Piano + Song Mode", frame)
            prev = cur[:]

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("g"):
                song_on = not song_on
                if song_on:
                    reset_song()
                else:
                    falling.clear()
            elif key == ord("r"):
                reset_song()
                song_on = True

    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()

if __name__ == "__main__":
    main()
