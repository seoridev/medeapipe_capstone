from collections import deque
import tkinter as tk
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image, ImageTk

import detailed_holistic_tracker as core
from emotion_recognizer import EmotionRecognizer
from project_version import __version__


DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 540
AIR_DRAWING_COLORS = {
    "left": (0, 0, 255),
    "right": (255, 0, 0),
}


LABEL_CAMERA = "\uce74\uba54\ub77c"
LABEL_REFRESH = "\uce74\uba54\ub77c \uc0c8\ub85c\uace0\uce68"
LABEL_START = "\uc2dc\uc791"
LABEL_MODE = "\ubaa8\ub4dc"
LABEL_RPS = "\uac00\uc704\ubc14\uc704\ubcf4"
LABEL_CHAM = "\ucc38\ucc38\ucc38"
LABEL_AIR = "\uc5d0\uc5b4\ub4dc\ub85c\uc789"
LABEL_WAVE = "\uc190 \ud754\ub4e4\uae30"
LABEL_HAND_GESTURE = "\uc5c4\uc9c0\ucc99/\ud558\ud2b8/OK"
LABEL_HEAD_GESTURE = "\uace0\uac1c \ub044\ub355/\uc813\uae30"
LABEL_ATTENTION = "\uc751\uc2dc/\uc790\ub9ac\ube44\uc6c0"
LABEL_TOGGLES = "\ud1a0\uae00"
LABEL_TRACKING = "\ud2b8\ub798\ud0b9 \ud45c\uc2dc"
LABEL_MARKER_ONLY = "\uac80\uc740\ud654\uba74 \ub9c8\ucee4\ub9cc"
LABEL_MIRROR = "\uc88c\uc6b0 \ubc18\uc804"
LABEL_INFO_OVERLAY = "\uc88c\uce21 \uc0c1\ub2e8 \uc815\ubcf4"
LABEL_EMOTION = "\uac10\uc815 \uc778\uc2dd"
LABEL_STATUS = "\uc0c1\ud0dc"

MODE_LABELS = {
    "rps": LABEL_RPS,
    "cham": LABEL_CHAM,
    "air": LABEL_AIR,
    "wave": LABEL_WAVE,
    "gesture": LABEL_HAND_GESTURE,
    "head": LABEL_HEAD_GESTURE,
    "attention": LABEL_ATTENTION,
}


class HolisticGuiApp:
    def __init__(self, root):
        self.root = root
        self.root.title(f"Holistic Tracking GUI v{__version__}")
        self.root.geometry("1420x860")
        self.root.configure(bg="#101418")

        self.capture_settings = SimpleNamespace(
            camera_backend="auto",
            camera_width=1280,
            camera_height=720,
        )

        self.cap = None
        self.capture_fps = 30.0
        self.pending_frame = None
        self.current_photo = None
        self.frame_index = 0
        self.camera_candidates = []
        self.active_mode = "rps"
        self.air_paths = {"left": [], "right": []}
        self.air_max_points = 180
        self.wave_histories = {"left": deque(maxlen=30), "right": deque(maxlen=30)}
        self.head_history = deque(maxlen=36)
        self.away_frame_count = 0
        self.emotion_result = None

        self.camera_var = tk.StringVar()
        self.source_var = tk.StringVar(
            value="\uce74\uba54\ub77c\ub97c \uc120\ud0dd\ud55c \ub4a4 \uc2dc\uc791\ud558\uc138\uc694."
        )
        self.mode_var = tk.StringVar(
            value="\ud604\uc7ac \ubaa8\ub4dc: " + LABEL_RPS
        )
        self.result_var = tk.StringVar(
            value="\uc778\uc2dd \uacb0\uacfc \ub300\uae30 \uc911"
        )
        self.status_var = tk.StringVar(
            value="\ub300\uae30 \uc911"
        )

        self.tracking_var = tk.BooleanVar(value=True)
        self.marker_only_var = tk.BooleanVar(value=False)
        self.mirror_var = tk.BooleanVar(value=False)
        self.info_overlay_var = tk.BooleanVar(value=True)
        self.emotion_var = tk.BooleanVar(value=False)

        self.holistic = core.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.emotion_recognizer = None
        self.load_emotion_model()

        self.build_ui()
        self.refresh_cameras()
        self.show_placeholder(
            "\uce74\uba54\ub77c\ub97c \uc2dc\uc791\ud558\uba74 \uc778\uc2dd \ud654\uba74\uc774 \uc5ec\uae30\uc5d0 \ud45c\uc2dc\ub429\ub2c8\ub2e4."
        )
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(0, self.update_frame)

    def build_ui(self):
        container = tk.Frame(self.root, bg="#101418")
        container.pack(fill="both", expand=True, padx=16, pady=16)

        control_panel = tk.Frame(container, bg="#161c22", width=340)
        control_panel.pack(side="left", fill="y")
        control_panel.pack_propagate(False)

        video_panel = tk.Frame(container, bg="#0b0f13")
        video_panel.pack(side="right", fill="both", expand=True, padx=(16, 0))

        title = tk.Label(
            control_panel,
            text="Holistic Control",
            bg="#161c22",
            fg="#f4f7fb",
            font=("Malgun Gothic", 20, "bold"),
            anchor="w",
        )
        title.pack(fill="x", padx=18, pady=(18, 8))

        subtitle = tk.Label(
            control_panel,
            text=(
                "\uc5bc\uad74, \uc0c1\uccb4, \uc190 \ud2b8\ub798\ud0b9\uacfc "
                "\uc81c\uc2a4\ucc98 \ubaa8\ub4dc\ub97c \ud55c \ud654\uba74\uc5d0\uc11c \uc81c\uc5b4\ud569\ub2c8\ub2e4."
            ),
            bg="#161c22",
            fg="#b8c2cc",
            font=("Malgun Gothic", 10),
            justify="left",
            wraplength=290,
            anchor="w",
        )
        subtitle.pack(fill="x", padx=18, pady=(0, 18))

        self.add_section_label(control_panel, LABEL_CAMERA)

        self.camera_menu = tk.OptionMenu(control_panel, self.camera_var, "")
        self.camera_menu.config(
            bg="#1f2730",
            fg="#f4f7fb",
            activebackground="#27313d",
            activeforeground="#f4f7fb",
            highlightthickness=0,
            relief="flat",
            font=("Consolas", 10),
        )
        self.camera_menu["menu"].config(
            bg="#1f2730",
            fg="#f4f7fb",
            activebackground="#314051",
            activeforeground="#ffffff",
        )
        self.camera_menu.pack(fill="x", padx=18)

        camera_row = tk.Frame(control_panel, bg="#161c22")
        camera_row.pack(fill="x", padx=18, pady=(10, 18))

        tk.Button(
            camera_row,
            text=LABEL_REFRESH,
            command=self.refresh_cameras,
            bg="#24303d",
            fg="#f4f7fb",
            relief="flat",
            activebackground="#314051",
            activeforeground="#ffffff",
            font=("Malgun Gothic", 10, "bold"),
        ).pack(side="left", fill="x", expand=True)

        tk.Button(
            camera_row,
            text=LABEL_START,
            command=self.start_selected_camera,
            bg="#2d6a4f",
            fg="#ffffff",
            relief="flat",
            activebackground="#3b8b66",
            activeforeground="#ffffff",
            font=("Malgun Gothic", 10, "bold"),
        ).pack(side="left", fill="x", expand=True, padx=(10, 0))

        self.add_section_label(control_panel, LABEL_MODE)
        mode_grid = tk.Frame(control_panel, bg="#161c22")
        mode_grid.pack(fill="x", padx=18)
        mode_grid.columnconfigure(0, weight=1)
        mode_grid.columnconfigure(1, weight=1)

        self.mode_buttons = {}
        for index, (mode_key, label) in enumerate((
            ("rps", LABEL_RPS),
            ("cham", LABEL_CHAM),
            ("air", LABEL_AIR),
            ("wave", LABEL_WAVE),
            ("gesture", LABEL_HAND_GESTURE),
            ("head", LABEL_HEAD_GESTURE),
            ("attention", LABEL_ATTENTION),
        )):
            button = tk.Button(
                mode_grid,
                text=label,
                command=lambda value=mode_key: self.set_mode(value),
                bg="#24303d",
                fg="#f4f7fb",
                relief="flat",
                activebackground="#314051",
                activeforeground="#ffffff",
                font=("Malgun Gothic", 10, "bold"),
                width=12,
                wraplength=132,
            )
            button.grid(
                row=index // 2,
                column=index % 2,
                sticky="ew",
                padx=(0, 8 if index % 2 == 0 else 0),
                pady=(0, 8),
            )
            self.mode_buttons[mode_key] = button

        self.add_section_label(control_panel, LABEL_TOGGLES)
        toggle_frame = tk.Frame(control_panel, bg="#161c22")
        toggle_frame.pack(fill="x", padx=18)

        self.make_toggle(toggle_frame, LABEL_TRACKING, self.tracking_var).pack(fill="x", pady=(0, 8))
        self.make_toggle(toggle_frame, LABEL_MARKER_ONLY, self.marker_only_var).pack(fill="x", pady=(0, 8))
        self.make_toggle(toggle_frame, LABEL_MIRROR, self.mirror_var).pack(fill="x", pady=(0, 8))
        self.make_toggle(toggle_frame, LABEL_INFO_OVERLAY, self.info_overlay_var).pack(fill="x", pady=(0, 8))
        self.make_toggle(toggle_frame, LABEL_EMOTION, self.emotion_var).pack(fill="x")

        self.add_section_label(control_panel, LABEL_STATUS)
        self.make_info_label(control_panel, self.source_var).pack(fill="x", padx=18, pady=(0, 10))
        self.make_info_label(control_panel, self.mode_var).pack(fill="x", padx=18, pady=(0, 10))
        self.make_info_label(control_panel, self.result_var).pack(fill="x", padx=18, pady=(0, 10))
        self.make_info_label(control_panel, self.status_var).pack(fill="x", padx=18)

        video_title = tk.Label(
            video_panel,
            text="Recognition Screen",
            bg="#0b0f13",
            fg="#f4f7fb",
            font=("Malgun Gothic", 18, "bold"),
            anchor="w",
        )
        video_title.pack(fill="x", pady=(0, 12))

        self.video_label = tk.Label(
            video_panel,
            bg="#05080c",
            bd=0,
        )
        self.video_label.pack(fill="both", expand=True)

        self.update_mode_buttons()

    def add_section_label(self, parent, text):
        label = tk.Label(
            parent,
            text=text,
            bg="#161c22",
            fg="#7fd1b9",
            font=("Malgun Gothic", 11, "bold"),
            anchor="w",
        )
        label.pack(fill="x", padx=18, pady=(0, 8))

    def make_toggle(self, parent, text, variable):
        return tk.Checkbutton(
            parent,
            text=text,
            variable=variable,
            bg="#161c22",
            fg="#f4f7fb",
            selectcolor="#24303d",
            activebackground="#161c22",
            activeforeground="#ffffff",
            font=("Malgun Gothic", 10),
            anchor="w",
            relief="flat",
            highlightthickness=0,
        )

    def make_info_label(self, parent, variable):
        return tk.Label(
            parent,
            textvariable=variable,
            bg="#1b232c",
            fg="#d6dde5",
            font=("Malgun Gothic", 10),
            justify="left",
            wraplength=290,
            anchor="w",
            padx=12,
            pady=10,
        )

    def show_placeholder(self, text):
        frame = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
        cv2.putText(
            frame,
            "Holistic Preview",
            (70, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (240, 245, 250),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            text,
            (70, 290),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (180, 190, 200),
            2,
            cv2.LINE_AA,
        )
        self.render_frame(frame)

    def refresh_cameras(self):
        self.camera_candidates = core.discover_webcams(self.capture_settings)
        menu = self.camera_menu["menu"]
        menu.delete(0, "end")

        if not self.camera_candidates:
            label = "\uc0ac\uc6a9 \uac00\ub2a5\ud55c \uc6f9\ucea0 \uc5c6\uc74c"
            self.camera_var.set(label)
            menu.add_command(label=label, command=lambda value=label: self.camera_var.set(value))
            self.status_var.set("\uc6f9\ucea0\uc744 \ucc3e\uc9c0 \ubabb\ud588\uc2b5\ub2c8\ub2e4.")
            return

        for candidate in self.camera_candidates:
            label = f"index {candidate['index']} / {candidate['backend_label']}"
            menu.add_command(label=label, command=lambda value=label: self.camera_var.set(value))

        default_label = f"index {self.camera_candidates[0]['index']} / {self.camera_candidates[0]['backend_label']}"
        self.camera_var.set(default_label)
        self.status_var.set(f"\uc6f9\ucea0 {len(self.camera_candidates)}\uac1c \uac10\uc9c0")

    def load_emotion_model(self):
        try:
            self.emotion_recognizer = EmotionRecognizer("epoch72_best_acc_0.8664.pth")
        except Exception as exc:
            self.emotion_recognizer = None
            self.status_var.set(f"\uac10\uc815 \ubaa8\ub378 \ub85c\ub4dc \uc2e4\ud328: {exc}")

    def start_selected_camera(self):
        if not self.camera_candidates:
            self.refresh_cameras()
            if not self.camera_candidates:
                return

        selected_label = self.camera_var.get()
        selected_candidate = self.camera_candidates[0]
        for candidate in self.camera_candidates:
            candidate_label = f"index {candidate['index']} / {candidate['backend_label']}"
            if candidate_label == selected_label:
                selected_candidate = candidate
                break

        self.release_camera()
        result = core.try_open_webcam(
            selected_candidate["index"],
            selected_candidate["backend"],
            selected_candidate["backend_label"],
            self.capture_settings.camera_width,
            self.capture_settings.camera_height,
        )

        if result is None:
            self.status_var.set("\uc120\ud0dd\ud55c \uce74\uba54\ub77c\ub97c \uc5f4\uc9c0 \ubabb\ud588\uc2b5\ub2c8\ub2e4.")
            return

        self.cap = result["cap"]
        self.pending_frame = result["frame"]
        self.capture_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        if self.capture_fps <= 0:
            self.capture_fps = 30.0
        self.frame_index = 0
        self.clear_air_paths()
        self.reset_motion_states()
        self.source_var.set(
            f"\uc18c\uc2a4: webcam index {selected_candidate['index']} via {selected_candidate['backend_label']}"
        )
        self.status_var.set("\uce74\uba54\ub77c \uc2e4\ud589 \uc911")

    def set_mode(self, mode_key):
        if self.active_mode == mode_key:
            self.active_mode = None
        else:
            self.active_mode = mode_key

        self.clear_air_paths()
        self.reset_motion_states()
        self.update_mode_buttons()
        if self.active_mode in MODE_LABELS:
            self.mode_var.set("\ud604\uc7ac \ubaa8\ub4dc: " + MODE_LABELS[self.active_mode])
            self.result_var.set("\uc778\uc2dd \uacb0\uacfc \ub300\uae30 \uc911")
        else:
            self.mode_var.set("\ud604\uc7ac \ubaa8\ub4dc: \uc5c6\uc74c")
            self.result_var.set("\ubaa8\ub4dc \uaebc\uc9d0")

    def update_mode_buttons(self):
        for mode_key, button in self.mode_buttons.items():
            if mode_key == self.active_mode:
                button.configure(bg="#7fd1b9", fg="#0d141a")
            else:
                button.configure(bg="#24303d", fg="#f4f7fb")

    def clear_air_paths(self):
        self.air_paths = {"left": [], "right": []}

    def reset_motion_states(self):
        self.wave_histories = {"left": deque(maxlen=30), "right": deque(maxlen=30)}
        self.head_history = deque(maxlen=36)
        self.away_frame_count = 0

    def release_camera(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.pending_frame = None

    def update_frame(self):
        if self.cap is None:
            self.root.after(30, self.update_frame)
            return

        if self.pending_frame is not None:
            has_frame = True
            frame_bgr = self.pending_frame
            self.pending_frame = None
        else:
            has_frame, frame_bgr = self.cap.read()

        if not has_frame or frame_bgr is None:
            self.status_var.set("\ud504\ub808\uc784\uc744 \uc77d\uc9c0 \ubabb\ud588\uc2b5\ub2c8\ub2e4.")
            self.release_camera()
            self.root.after(30, self.update_frame)
            return

        if self.mirror_var.get():
            frame_bgr = cv2.flip(frame_bgr, 1)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.holistic.process(frame_rgb)
        frame_rgb.flags.writeable = True

        height, width = frame_bgr.shape[:2]
        timestamp_ms = int((self.frame_index / self.capture_fps) * 1000)
        tracking_enabled = self.tracking_var.get()
        rps_enabled = self.active_mode == "rps"

        frame_record = core.build_frame_record(
            self.frame_index,
            timestamp_ms,
            width,
            height,
            results,
            tracking_enabled,
            rps_enabled,
        )
        self.emotion_result = self.predict_emotion(frame_bgr, frame_record)

        if self.marker_only_var.get():
            base_frame = np.zeros_like(frame_bgr)
        else:
            base_frame = frame_bgr

        annotated = core.annotate_frame(
            base_frame,
            frame_record,
            results,
            tracking_enabled,
            rps_enabled,
            draw_labels=False,
            show_info_overlay=self.info_overlay_var.get(),
            hand_label_side_map=self.get_hand_label_side_map(),
        )

        self.apply_mode_overlay(annotated, frame_record)
        self.apply_emotion_overlay(annotated)
        self.render_frame(annotated)

        self.frame_index += 1
        self.root.after(15, self.update_frame)

    def apply_mode_overlay(self, frame_bgr, frame_record):
        if self.active_mode == "rps":
            self.apply_rps_overlay(frame_bgr, frame_record)
        elif self.active_mode == "cham":
            self.apply_cham_overlay(frame_bgr, frame_record)
        elif self.active_mode == "air":
            self.apply_air_overlay(frame_bgr, frame_record)
        elif self.active_mode == "wave":
            self.apply_wave_overlay(frame_bgr, frame_record)
        elif self.active_mode == "gesture":
            self.apply_gesture_overlay(frame_bgr, frame_record)
        elif self.active_mode == "head":
            self.apply_head_overlay(frame_bgr, frame_record)
        elif self.active_mode == "attention":
            self.apply_attention_overlay(frame_bgr, frame_record)
        else:
            self.result_var.set("\ubaa8\ub4dc \uaebc\uc9d0")

    def predict_emotion(self, frame_bgr, frame_record):
        if not self.emotion_var.get() or self.emotion_recognizer is None:
            return None

        face_crop = self.extract_face_crop(frame_bgr, frame_record)
        if face_crop is None:
            return None

        try:
            return self.emotion_recognizer.predict(face_crop)
        except Exception:
            return None

    def extract_face_crop(self, frame_bgr, frame_record):
        face_landmarks = frame_record["face_landmarks"]
        if not face_landmarks:
            return None

        height, width = frame_bgr.shape[:2]
        xs = [point["pixel_x"] for point in face_landmarks]
        ys = [point["pixel_y"] for point in face_landmarks]
        if not xs or not ys:
            return None

        x1, x2 = max(0, min(xs)), min(width, max(xs))
        y1, y2 = max(0, min(ys)), min(height, max(ys))

        box_width = x2 - x1
        box_height = y2 - y1
        if box_width <= 0 or box_height <= 0:
            return None

        pad_x = int(box_width * 0.18)
        pad_y = int(box_height * 0.22)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(width, x2 + pad_x)
        y2 = min(height, y2 + pad_y)

        if x2 <= x1 or y2 <= y1:
            return None

        return frame_bgr[y1:y2, x1:x2]

    def apply_emotion_overlay(self, frame_bgr):
        if not self.emotion_var.get():
            return

        if self.emotion_recognizer is None:
            text = "Emotion: MODEL ERROR"
            confidence_text = ""
        elif self.emotion_result is None:
            text = "Emotion: NONE"
            confidence_text = ""
        else:
            label = self.emotion_result["label"]
            confidence = self.emotion_result["confidence"] * 100.0
            text = f"Emotion: {label}"
            confidence_text = f"{confidence:.1f}%"

        box_width = 280
        box_height = 86 if confidence_text else 58
        x1 = frame_bgr.shape[1] - box_width - 20
        y1 = 20
        x2 = frame_bgr.shape[1] - 20
        y2 = y1 + box_height

        overlay = frame_bgr.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (15, 20, 26), -1)
        cv2.addWeighted(overlay, 0.72, frame_bgr, 0.28, 0, frame_bgr)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (127, 209, 185), 2)
        cv2.putText(
            frame_bgr,
            text,
            (x1 + 14, y1 + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if confidence_text:
            cv2.putText(
                frame_bgr,
                confidence_text,
                (x1 + 14, y1 + 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (210, 226, 235),
                2,
                cv2.LINE_AA,
            )

    def apply_rps_overlay(self, frame_bgr, frame_record):
        left_state, right_state = self.get_display_hand_states(frame_record)
        self.result_var.set(f"{LABEL_RPS}: L={left_state}, R={right_state}")
        text = f"RPS  L:{left_state}  R:{right_state}"

        cv2.putText(
            frame_bgr,
            text,
            (40, frame_bgr.shape[0] - 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def apply_cham_overlay(self, frame_bgr, frame_record):
        direction = self.detect_face_direction(frame_record["upper_body_landmarks"])
        self.result_var.set(f"{LABEL_CHAM}: {direction}")
        cv2.putText(
            frame_bgr,
            f"Cham Cham Cham: {direction}",
            (40, frame_bgr.shape[0] - 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def apply_air_overlay(self, frame_bgr, frame_record):
        self.update_air_paths(frame_record)
        self.draw_air_paths(frame_bgr)
        self.result_var.set("\uc5d0\uc5b4\ub4dc\ub85c\uc789: \uc190\uac00\ub77d \uacbd\ub85c \ud45c\uc2dc \uc911")
        self.draw_mode_text(frame_bgr, "Air Drawing Active")

    def apply_wave_overlay(self, frame_bgr, frame_record):
        self.update_wave_histories(frame_record)
        wave_states = {
            side: self.detect_wave(side, frame_bgr.shape[1])
            for side in ("left", "right")
        }
        left_state, right_state = self.get_display_side_values(wave_states)
        result_text = f"{LABEL_WAVE}: L={left_state}, R={right_state}"
        self.result_var.set(result_text)
        self.draw_hand_state_labels(frame_bgr, frame_record, wave_states)
        self.draw_mode_text(frame_bgr, f"Wave  L:{left_state}  R:{right_state}")

    def apply_gesture_overlay(self, frame_bgr, frame_record):
        gesture_states = {
            "left": self.detect_static_hand_gesture(frame_record["left_hand_landmarks"]),
            "right": self.detect_static_hand_gesture(frame_record["right_hand_landmarks"]),
        }
        if self.detect_two_hand_heart(frame_record):
            gesture_states["left"] = "HEART"
            gesture_states["right"] = "HEART"

        left_state, right_state = self.get_display_side_values(gesture_states)
        self.result_var.set(f"{LABEL_HAND_GESTURE}: L={left_state}, R={right_state}")
        self.draw_hand_state_labels(frame_bgr, frame_record, gesture_states)
        self.draw_mode_text(frame_bgr, f"Gesture  L:{left_state}  R:{right_state}")

    def apply_head_overlay(self, frame_bgr, frame_record):
        self.update_head_history(frame_record)
        head_state = self.detect_head_motion()
        if head_state == "AGREE":
            result_state = "\ub3d9\uc758"
            overlay_state = "AGREE"
        elif head_state == "NEGATIVE":
            result_state = "\ubd80\uc815"
            overlay_state = "NEGATIVE"
        else:
            result_state = "\uc778\uc2dd \ub300\uae30"
            overlay_state = "WAITING"

        self.result_var.set(f"{LABEL_HEAD_GESTURE}: {result_state}")
        self.draw_mode_text(frame_bgr, f"Head Motion: {overlay_state}")

    def apply_attention_overlay(self, frame_bgr, frame_record):
        attention_state = self.detect_attention_state(frame_record)
        if attention_state == "SCREEN":
            result_state = "\ud654\uba74 \uc751\uc2dc"
            overlay_state = "LOOKING AT SCREEN"
        elif attention_state == "LOOKING_AWAY":
            result_state = "\ud654\uba74 \ubc16 \uc751\uc2dc"
            overlay_state = "LOOKING AWAY"
        elif attention_state == "AWAY":
            result_state = "\uc790\ub9ac \ube44\uc6c0"
            overlay_state = "AWAY"
        else:
            result_state = "\uc5bc\uad74 \ucc3e\ub294 \uc911"
            overlay_state = "SEARCHING FACE"

        self.result_var.set(f"{LABEL_ATTENTION}: {result_state}")
        self.draw_mode_text(frame_bgr, f"Attention: {overlay_state}")

    def draw_mode_text(self, frame_bgr, text, color=(255, 255, 255)):
        cv2.putText(
            frame_bgr,
            text,
            (40, frame_bgr.shape[0] - 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
            cv2.LINE_AA,
        )

    def draw_hand_state_labels(self, frame_bgr, frame_record, side_states):
        for side, state in side_states.items():
            records = frame_record[f"{side}_hand_landmarks"]
            anchor = core.hand_anchor_point(records)
            if anchor is None:
                continue
            display_side = self.get_display_side_name(side)
            cv2.putText(
                frame_bgr,
                f"{display_side.title()}: {state}",
                (anchor[0] + 10, max(30, anchor[1] - 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (127, 209, 185),
                2,
                cv2.LINE_AA,
            )

    def get_display_side_values(self, side_values):
        if self.mirror_var.get():
            return side_values["right"], side_values["left"]
        return side_values["left"], side_values["right"]

    def get_display_side_name(self, side):
        if not self.mirror_var.get():
            return side
        return "right" if side == "left" else "left"

    def update_wave_histories(self, frame_record):
        for side in ("left", "right"):
            anchor = core.hand_anchor_point(frame_record[f"{side}_hand_landmarks"])
            self.wave_histories[side].append(anchor)

    def detect_wave(self, side, frame_width):
        points = [point for point in self.wave_histories[side] if point is not None]
        if len(points) < 12:
            return "WAIT"

        xs = [point[0] for point in points]
        amplitude = max(xs) - min(xs)
        min_amplitude = max(42, frame_width * 0.055)
        direction_changes = self.count_direction_changes(xs, min_delta=8)
        if amplitude >= min_amplitude and direction_changes >= 2:
            return "HELLO"
        return "NONE"

    def detect_static_hand_gesture(self, hand_records):
        pose = self.analyze_hand_pose(hand_records)
        if pose is None:
            return "NONE"

        extended = pose["extended"]
        folded_fingers = not any(
            extended[finger] for finger in ("index", "middle", "ring", "pinky")
        )
        if pose["thumb_up"] and folded_fingers:
            return "THUMBS_UP"
        if pose["pinch"] and extended["middle"] and extended["ring"] and extended["pinky"]:
            return "OK"
        if pose["pinch"] and not extended["middle"] and not extended["ring"] and not extended["pinky"]:
            return "HEART"
        return "NONE"

    def detect_two_hand_heart(self, frame_record):
        left_map = core.records_by_name(frame_record["left_hand_landmarks"])
        right_map = core.records_by_name(frame_record["right_hand_landmarks"])
        if not left_map or not right_map:
            return False

        left_scale = self.hand_scale(left_map)
        right_scale = self.hand_scale(right_map)
        scale = max(40, (left_scale + right_scale) / 2)
        threshold = scale * 0.75

        left_thumb = core.point_from_record(left_map.get("THUMB_TIP"))
        right_thumb = core.point_from_record(right_map.get("THUMB_TIP"))
        left_index = core.point_from_record(left_map.get("INDEX_FINGER_TIP"))
        right_index = core.point_from_record(right_map.get("INDEX_FINGER_TIP"))
        left_wrist = core.point_from_record(left_map.get("WRIST"))
        right_wrist = core.point_from_record(right_map.get("WRIST"))

        thumb_distance = core.distance_between_points(left_thumb, right_thumb)
        index_distance = core.distance_between_points(left_index, right_index)
        wrist_distance = core.distance_between_points(left_wrist, right_wrist)
        if thumb_distance is None or index_distance is None or wrist_distance is None:
            return False
        return thumb_distance < threshold and index_distance < threshold and wrist_distance > scale

    def analyze_hand_pose(self, hand_records):
        if not hand_records:
            return None

        hand_map = core.records_by_name(hand_records)
        scale = self.hand_scale(hand_map)
        thumb_tip = core.point_from_record(hand_map.get("THUMB_TIP"))
        thumb_ip = core.point_from_record(hand_map.get("THUMB_IP"))
        thumb_mcp = core.point_from_record(hand_map.get("THUMB_MCP"))
        wrist = core.point_from_record(hand_map.get("WRIST"))
        index_tip = core.point_from_record(hand_map.get("INDEX_FINGER_TIP"))

        thumb_angle = core.angle_between_points(thumb_mcp, thumb_ip, thumb_tip)
        thumb_up = (
            thumb_tip is not None
            and thumb_ip is not None
            and thumb_mcp is not None
            and wrist is not None
            and thumb_angle is not None
            and thumb_angle >= 145
            and thumb_tip[1] < thumb_ip[1] < thumb_mcp[1]
            and wrist[1] - thumb_tip[1] > scale * 0.28
        )

        pinch_distance = core.distance_between_points(thumb_tip, index_tip)
        pinch = pinch_distance is not None and pinch_distance <= scale * 0.42

        extended = {
            "thumb": thumb_up,
            "index": core.finger_is_extended(
                hand_map,
                "INDEX_FINGER_MCP",
                "INDEX_FINGER_PIP",
                "INDEX_FINGER_TIP",
            ),
            "middle": core.finger_is_extended(
                hand_map,
                "MIDDLE_FINGER_MCP",
                "MIDDLE_FINGER_PIP",
                "MIDDLE_FINGER_TIP",
            ),
            "ring": core.finger_is_extended(
                hand_map,
                "RING_FINGER_MCP",
                "RING_FINGER_PIP",
                "RING_FINGER_TIP",
            ),
            "pinky": core.finger_is_extended(
                hand_map,
                "PINKY_MCP",
                "PINKY_PIP",
                "PINKY_TIP",
            ),
        }

        return {
            "extended": extended,
            "pinch": pinch,
            "thumb_up": thumb_up,
        }

    def hand_scale(self, hand_map):
        wrist = core.point_from_record(hand_map.get("WRIST"))
        middle_mcp = core.point_from_record(hand_map.get("MIDDLE_FINGER_MCP"))
        index_mcp = core.point_from_record(hand_map.get("INDEX_FINGER_MCP"))
        pinky_mcp = core.point_from_record(hand_map.get("PINKY_MCP"))

        scale = core.distance_between_points(wrist, middle_mcp)
        if scale is None or scale <= 0:
            scale = core.distance_between_points(index_mcp, pinky_mcp)
        if scale is None or scale <= 0:
            scale = 80
        return scale

    def update_head_history(self, frame_record):
        offset = self.get_head_offset(frame_record)
        self.head_history.append(offset)

    def get_head_offset(self, frame_record):
        upper_body = frame_record["upper_body_landmarks"]
        nose = upper_body.get("NOSE")
        left_eye = upper_body.get("LEFT_EYE")
        right_eye = upper_body.get("RIGHT_EYE")
        if nose is None or left_eye is None or right_eye is None:
            return None

        left_eye_point = core.point_from_record(left_eye)
        right_eye_point = core.point_from_record(right_eye)
        nose_point = core.point_from_record(nose)
        eye_distance = core.distance_between_points(left_eye_point, right_eye_point)
        if eye_distance is None or eye_distance <= 0:
            return None

        center_x = (left_eye_point[0] + right_eye_point[0]) / 2
        center_y = (left_eye_point[1] + right_eye_point[1]) / 2
        return (
            (nose_point[0] - center_x) / eye_distance,
            (nose_point[1] - center_y) / eye_distance,
        )

    def detect_head_motion(self):
        offsets = [offset for offset in self.head_history if offset is not None]
        if len(offsets) < 12:
            return "WAIT"

        xs = [offset[0] for offset in offsets]
        ys = [offset[1] for offset in offsets]
        x_amplitude = max(xs) - min(xs)
        y_amplitude = max(ys) - min(ys)
        x_changes = self.count_direction_changes(xs, min_delta=0.035)
        y_changes = self.count_direction_changes(ys, min_delta=0.035)

        if y_amplitude >= 0.23 and y_changes >= 2 and y_amplitude > x_amplitude * 1.12:
            return "AGREE"
        if x_amplitude >= 0.2 and x_changes >= 2 and x_amplitude > y_amplitude * 1.12:
            return "NEGATIVE"
        return "WAIT"

    def detect_attention_state(self, frame_record):
        has_face = bool(frame_record["face_landmarks"])
        has_nose = frame_record["upper_body_landmarks"].get("NOSE") is not None
        if not has_face and not has_nose:
            self.away_frame_count += 1
        else:
            self.away_frame_count = 0

        if self.away_frame_count >= 12:
            return "AWAY"
        if not has_face:
            return "SEARCHING"

        direction = self.detect_face_direction(frame_record["upper_body_landmarks"])
        if direction == "CENTER":
            return "SCREEN"
        if direction in ("LEFT", "RIGHT"):
            return "LOOKING_AWAY"
        return "SEARCHING"

    def count_direction_changes(self, values, min_delta):
        signs = []
        for previous, current in zip(values, values[1:]):
            delta = current - previous
            if abs(delta) < min_delta:
                continue
            signs.append(1 if delta > 0 else -1)

        changes = 0
        previous_sign = None
        for sign in signs:
            if previous_sign is not None and sign != previous_sign:
                changes += 1
            previous_sign = sign
        return changes

    def get_display_hand_states(self, frame_record):
        left_state = frame_record["hand_gestures"]["left"]["state"]
        right_state = frame_record["hand_gestures"]["right"]["state"]
        if self.mirror_var.get():
            return right_state, left_state
        return left_state, right_state

    def get_hand_label_side_map(self):
        if self.mirror_var.get():
            return {"left": "right", "right": "left"}
        return None

    def detect_face_direction(self, upper_body_landmarks):
        nose = upper_body_landmarks.get("NOSE")
        left_eye = upper_body_landmarks.get("LEFT_EYE")
        right_eye = upper_body_landmarks.get("RIGHT_EYE")

        if nose is None or left_eye is None or right_eye is None:
            return "NONE"

        center_x = (left_eye["pixel_x"] + right_eye["pixel_x"]) / 2
        eye_distance = abs(right_eye["pixel_x"] - left_eye["pixel_x"])
        if eye_distance <= 0:
            return "CENTER"

        delta = nose["pixel_x"] - center_x
        threshold = eye_distance * 0.16
        if delta <= -threshold:
            return "LEFT"
        if delta >= threshold:
            return "RIGHT"
        return "CENTER"

    def update_air_paths(self, frame_record):
        for side in ("left", "right"):
            hand_records = frame_record[f"{side}_hand_landmarks"]
            if not hand_records:
                self.air_paths[side].append(None)
                self.trim_air_path(side)
                continue

            hand_map = core.records_by_name(hand_records)
            index_tip = hand_map.get("INDEX_FINGER_TIP")
            if index_tip is None:
                self.air_paths[side].append(None)
            else:
                self.air_paths[side].append((index_tip["pixel_x"], index_tip["pixel_y"]))
            self.trim_air_path(side)

    def trim_air_path(self, side):
        if len(self.air_paths[side]) > self.air_max_points:
            self.air_paths[side] = self.air_paths[side][-self.air_max_points :]

    def draw_air_paths(self, frame_bgr):
        for side, points in self.air_paths.items():
            previous = None
            for point in points:
                if point is None:
                    previous = None
                    continue
                if previous is not None:
                    cv2.line(frame_bgr, previous, point, AIR_DRAWING_COLORS[side], 3, cv2.LINE_AA)
                previous = point

    def render_frame(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = self.fit_frame(frame_rgb, DISPLAY_WIDTH, DISPLAY_HEIGHT)
        image = Image.fromarray(resized)
        self.current_photo = ImageTk.PhotoImage(image=image)
        self.video_label.configure(image=self.current_photo)

    def fit_frame(self, frame_rgb, max_width, max_height):
        height, width = frame_rgb.shape[:2]
        scale = min(max_width / width, max_height / height)
        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))
        return cv2.resize(frame_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)

    def on_close(self):
        self.release_camera()
        self.holistic.close()
        self.root.destroy()


def main():
    root = tk.Tk()
    HolisticGuiApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
