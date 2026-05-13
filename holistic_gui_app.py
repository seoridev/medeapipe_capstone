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
LABEL_TOGGLES = "\ud1a0\uae00"
LABEL_TRACKING = "\ud2b8\ub798\ud0b9 \ud45c\uc2dc"
LABEL_MARKER_ONLY = "\uac80\uc740\ud654\uba74 \ub9c8\ucee4\ub9cc"
LABEL_MIRROR = "\uc88c\uc6b0 \ubc18\uc804"
LABEL_INFO_OVERLAY = "\uc88c\uce21 \uc0c1\ub2e8 \uc815\ubcf4"
LABEL_EMOTION = "\uac10\uc815 \uc778\uc2dd"
LABEL_STATUS = "\uc0c1\ud0dc"


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
        mode_row = tk.Frame(control_panel, bg="#161c22")
        mode_row.pack(fill="x", padx=18)

        self.mode_buttons = {}
        for mode_key, label in (
            ("rps", LABEL_RPS),
            ("cham", LABEL_CHAM),
            ("air", LABEL_AIR),
        ):
            button = tk.Button(
                mode_row,
                text=label,
                command=lambda value=mode_key: self.set_mode(value),
                bg="#24303d",
                fg="#f4f7fb",
                relief="flat",
                activebackground="#314051",
                activeforeground="#ffffff",
                font=("Malgun Gothic", 10, "bold"),
                width=10,
            )
            button.pack(side="left", expand=True, fill="x", padx=(0, 8))
            self.mode_buttons[mode_key] = button
        self.mode_buttons["air"].pack_configure(padx=(0, 0))

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
        self.source_var.set(
            f"\uc18c\uc2a4: webcam index {selected_candidate['index']} via {selected_candidate['backend_label']}"
        )
        self.status_var.set("\uce74\uba54\ub77c \uc2e4\ud589 \uc911")

    def set_mode(self, mode_key):
        if self.active_mode == mode_key:
            self.active_mode = None
        else:
            self.active_mode = mode_key

        self.update_mode_buttons()
        if self.active_mode == "rps":
            self.mode_var.set("\ud604\uc7ac \ubaa8\ub4dc: " + LABEL_RPS)
        elif self.active_mode == "cham":
            self.mode_var.set("\ud604\uc7ac \ubaa8\ub4dc: " + LABEL_CHAM)
        elif self.active_mode == "air":
            self.mode_var.set("\ud604\uc7ac \ubaa8\ub4dc: " + LABEL_AIR)
            self.clear_air_paths()
        else:
            self.mode_var.set("\ud604\uc7ac \ubaa8\ub4dc: \uc5c6\uc74c")
            self.result_var.set("\ubaa8\ub4dc \uaebc\uc9d0")
            self.clear_air_paths()

    def update_mode_buttons(self):
        for mode_key, button in self.mode_buttons.items():
            if mode_key == self.active_mode:
                button.configure(bg="#7fd1b9", fg="#0d141a")
            else:
                button.configure(bg="#24303d", fg="#f4f7fb")

    def clear_air_paths(self):
        self.air_paths = {"left": [], "right": []}

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
        cv2.putText(
            frame_bgr,
            "Air Drawing Active",
            (40, frame_bgr.shape[0] - 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

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
