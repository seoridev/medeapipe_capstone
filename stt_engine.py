from __future__ import annotations

import queue
import threading
from dataclasses import dataclass

try:
    import numpy as np
except ImportError as exc:
    np = None
    NUMPY_IMPORT_ERROR = exc
else:
    NUMPY_IMPORT_ERROR = None

try:
    import sounddevice as sd
except ImportError as exc:
    sd = None
    SOUNDDEVICE_IMPORT_ERROR = exc
else:
    SOUNDDEVICE_IMPORT_ERROR = None

try:
    from faster_whisper import WhisperModel
except ImportError as exc:
    WhisperModel = None
    WHISPER_IMPORT_ERROR = exc
else:
    WHISPER_IMPORT_ERROR = None


SAMPLE_RATE = 16_000
CHANNELS = 1
CHUNK_SECONDS = 5

DEVICE_OPTIONS = ("auto", "cpu", "cuda")
MODEL_OPTIONS = ("tiny", "base", "small", "medium", "large-v3", "turbo")
LANGUAGE_OPTIONS = ("auto", "ko", "en", "ja", "zh")


@dataclass
class AudioDevice:
    index: int | None
    label: str


class RealtimeSTT:
    def __init__(self):
        self.audio_devices: list[AudioDevice] = []
        self.audio_queue: queue.Queue = queue.Queue()
        self.ui_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self.stop_event = threading.Event()
        self.worker_thread: threading.Thread | None = None
        self.stream = None
        self.model = None
        self.model_key: tuple[str, str] | None = None
        self.is_running = False

        self.active_device = "auto"
        self.active_model = "tiny"
        self.active_mic_label = ""
        self.active_language = "ko"
        self.active_timestamps = True

    def dependency_error(self) -> str | None:
        missing = []
        if NUMPY_IMPORT_ERROR is not None:
            missing.append("numpy")
        if SOUNDDEVICE_IMPORT_ERROR is not None:
            missing.append("sounddevice")
        if WHISPER_IMPORT_ERROR is not None:
            missing.append("faster-whisper")
        if not missing:
            return None
        return "STT dependency missing: " + ", ".join(missing)

    def refresh_microphones(self) -> list[str]:
        dependency_error = self.dependency_error()
        if dependency_error is not None:
            raise RuntimeError(dependency_error)

        devices = sd.query_devices()
        self.audio_devices = []
        labels: list[str] = []
        for index, device in enumerate(devices):
            if int(device.get("max_input_channels", 0)) <= 0:
                continue
            label = f"index {index} / {device['name']}"
            self.audio_devices.append(AudioDevice(index=index, label=label))
            labels.append(label)
        return labels

    def start(
        self,
        mic_label: str,
        device: str,
        model_name: str,
        language: str,
        timestamps: bool,
    ) -> None:
        if self.is_running:
            return

        dependency_error = self.dependency_error()
        if dependency_error is not None:
            raise RuntimeError(dependency_error)
        if not mic_label:
            raise RuntimeError("No microphone selected.")

        self.active_mic_label = mic_label
        self.active_device = device
        self.active_model = model_name
        self.active_language = language
        self.active_timestamps = timestamps

        self.stop_event.clear()
        self.audio_queue = queue.Queue()
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._run_stt, daemon=True)
        self.worker_thread.start()

    def stop(self) -> None:
        if not self.is_running:
            return
        self.stop_event.set()
        self._stop_stream()
        self.ui_queue.put(("status", "STT stopping..."))

    def drain_events(self) -> list[tuple[str, str]]:
        events: list[tuple[str, str]] = []
        while True:
            try:
                events.append(self.ui_queue.get_nowait())
            except queue.Empty:
                return events

    def _run_stt(self) -> None:
        try:
            model = self._load_model()
            self.ui_queue.put(("status", "STT recording"))
            self.ui_queue.put(("text", "\n[STT started]\n"))
            self._start_stream()

            pending = []
            min_samples = SAMPLE_RATE * CHUNK_SECONDS
            audio_offset = 0.0

            while not self.stop_event.is_set():
                try:
                    chunk = self.audio_queue.get(timeout=0.2)
                    pending.append(chunk)
                except queue.Empty:
                    continue

                sample_count = sum(len(item) for item in pending)
                if sample_count >= min_samples:
                    audio = np.concatenate(pending)
                    pending.clear()
                    self._transcribe_audio(model, audio, audio_offset)
                    audio_offset += len(audio) / SAMPLE_RATE

            if pending:
                audio = np.concatenate(pending)
                self._transcribe_audio(model, audio, audio_offset)

        except Exception as exc:
            self.ui_queue.put(("error", str(exc)))
        finally:
            self._stop_stream()
            self.is_running = False
            self.ui_queue.put(("status", "STT stopped"))
            self.ui_queue.put(("running", "false"))
            self.ui_queue.put(("text", "[STT stopped]\n"))

    def _load_model(self):
        model_name = self.active_model
        device = self.active_device
        key = (model_name, device)
        if self.model is not None and self.model_key == key:
            return self.model

        self.ui_queue.put(("status", f"Loading STT model: {model_name} ({device})"))
        self.model = WhisperModel(model_name, device=device, compute_type="auto")
        self.model_key = key
        return self.model

    def _start_stream(self) -> None:
        selected = self.active_mic_label
        device_index = next(
            (item.index for item in self.audio_devices if item.label == selected),
            None,
        )

        def callback(indata, frames, time_info, status) -> None:
            if status:
                self.ui_queue.put(("status", f"Audio warning: {status}"))
            mono = indata[:, 0].astype(np.float32, copy=True)
            self.audio_queue.put(mono)

        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            device=device_index,
            callback=callback,
        )
        self.stream.start()

    def _stop_stream(self) -> None:
        if self.stream is None:
            return
        try:
            self.stream.stop()
            self.stream.close()
        finally:
            self.stream = None

    def _transcribe_audio(self, model, audio, offset: float) -> None:
        if audio.size == 0:
            return

        rms = float(np.sqrt(np.mean(np.square(audio))))
        if rms < 0.003:
            self.ui_queue.put(("status", "STT silence detected"))
            return

        self.ui_queue.put(("status", "STT transcribing"))
        language = self.active_language
        kwargs = {
            "beam_size": 5,
            "vad_filter": True,
            "condition_on_previous_text": False,
        }
        if language != "auto":
            kwargs["language"] = language

        segments, info = model.transcribe(audio, **kwargs)
        lines: list[str] = []
        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue
            if self.active_timestamps:
                start = segment.start + offset
                end = segment.end + offset
                lines.append(f"[{start:05.2f} - {end:05.2f}] {text}")
            else:
                lines.append(text)

        if lines:
            self.ui_queue.put(("text", "\n".join(lines) + "\n"))
            self.ui_queue.put(("status", f"STT transcribed ({info.language})"))
        else:
            self.ui_queue.put(("status", "STT no speech"))
