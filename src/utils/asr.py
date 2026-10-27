import queue
import collections
import time
import os
import ctranslate2
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel


class ASR:
    # ---------------- Taskora Config ----------------
    TASKORA_PROMPT = (
        "Taskora, VS Code, Chrome, Brave, GitHub, terminal, folder, volume, "
        "brightness, Spotify, Python, screenshot, file, open, close, "
        "band kar do, khol do, chalu karo, kam karo, badhao."
    )

    ASR_MODEL = "base"

    SAMPLE_RATE = 16000
    FRAME_MS = 30

    SILENCE_TIMEOUT = 1.0
    NO_SPEECH_TIMEOUT = 8.0
    MAX_RECORD = 30.0
    MIN_SPEECH = 0.3
    PRE_ROLL = 0.3
    CALIBRATION = 0.5
    MIN_THRESHOLD = 0.01

    def __init__(self):
        self.frame_size = int(
            self.SAMPLE_RATE * self.FRAME_MS / 1000
        )

        self.device, self.compute_type, self.cpu_threads = (
            self.get_optimal_hardware_config()
        )

        print(
            f"⚡ Auto-configured ASR: "
            f"device={self.device}, "
            f"compute_type={self.compute_type}, "
            f"threads={self.cpu_threads}"
        )

        self.model = WhisperModel(
            self.ASR_MODEL,
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=self.cpu_threads,
        )

        print("✅ Whisper Ready!")

    # ---------------- Hardware Configuration ----------------
    @staticmethod
    def get_optimal_hardware_config() -> tuple[str, str, int]:
        device = (
            "cuda"
            if ctranslate2.get_cuda_device_count() > 0
            else "cpu"
        )

        supported = ctranslate2.get_supported_compute_types(device)

        if device == "cuda":
            preference_order = [
                "int8_float16",
                "float16",
                "int8",
                "float32",
            ]
        else:
            preference_order = [
                "int8",
                "int8_float32",
                "float32",
            ]

        compute_type = next(
            (t for t in preference_order if t in supported),
            "default",
        )

        # Limit CPU threads to avoid contention on hybrid CPUs.
        logical_cores = os.cpu_count() or 4
        cpu_threads = min(4, max(1, logical_cores // 2))

        return device, compute_type, cpu_threads

    # ---------------- Speech Energy Detection ----------------
    @staticmethod
    def speech_energy(frame: np.ndarray) -> float:
        """
        Apply pre-emphasis to reduce low-frequency rumble
        and calculate the filtered signal's RMS energy.
        """
        filtered = np.append(
            frame[0],
            frame[1:] - 0.97 * frame[:-1],
        )

        return float(np.sqrt(np.mean(filtered ** 2)))

    # ---------------- Audio Recording ----------------
    def record_until_silence(self) -> np.ndarray:
        q: queue.Queue = queue.Queue()

        def callback(indata, frames, time_info, status):
            if status:
                print(status)

            q.put(indata[:, 0].copy())

        with sd.InputStream(
            samplerate=self.SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=self.frame_size,
            callback=callback,
        ):
            print(
                "Calibrating background noise... "
                "stay quiet for a moment."
            )

            noise = [
                self.speech_energy(q.get())
                for _ in range(
                    int(
                        self.CALIBRATION
                        * 1000
                        / self.FRAME_MS
                    )
                )
            ]

            noise_mean = float(np.mean(noise))
            noise_std = float(np.std(noise))

            threshold = max(
                self.MIN_THRESHOLD,
                noise_mean + (2.5 * noise_std) + 0.005,
            )

            print(
                f"📊 Noise Mean: {noise_mean:.4f} | "
                f"Std: {noise_std:.4f} | "
                f"Threshold: {threshold:.4f}"
            )

            print("🎤 Listening... start speaking.")

            pre_roll = collections.deque(
                maxlen=int(
                    self.PRE_ROLL * 1000 / self.FRAME_MS
                )
            )

            recorded = []
            speech_started = False
            speech_frames = 0
            silence_frames = 0
            total_frames = 0

            silence_limit = int(
                self.SILENCE_TIMEOUT * 1000 / self.FRAME_MS
            )
            no_speech_limit = int(
                self.NO_SPEECH_TIMEOUT * 1000 / self.FRAME_MS
            )
            max_frames = int(
                self.MAX_RECORD * 1000 / self.FRAME_MS
            )
            min_speech_frames = int(
                self.MIN_SPEECH * 1000 / self.FRAME_MS
            )

            while total_frames < max_frames:
                frame = q.get()
                total_frames += 1

                is_speech = (
                    self.speech_energy(frame) > threshold
                )

                if not speech_started:
                    pre_roll.append(frame)

                    if is_speech:
                        speech_started = True
                        speech_frames = 1
                        recorded.extend(pre_roll)

                        print("🗣️ Speech detected...")

                    elif total_frames >= no_speech_limit:
                        print("No speech detected.")
                        return np.array([], dtype="float32")

                else:
                    recorded.append(frame)

                    if is_speech:
                        speech_frames += 1
                        silence_frames = 0
                    else:
                        silence_frames += 1

                        if silence_frames >= silence_limit:
                            break

        if speech_frames < min_speech_frames:
            return np.array([], dtype="float32")

        print("🛑 Stopped listening.")

        return np.concatenate(recorded)

    # ---------------- Audio Transcription ----------------
    def transcribe_audio(self, audio: np.ndarray) -> str:
        segments, _ = self.model.transcribe(
            audio,
            beam_size=1,
            language="en",
            without_timestamps=True,
            temperature=0.0,
            initial_prompt=self.TASKORA_PROMPT,
            condition_on_previous_text=False,
        )

        return " ".join(
            segment.text for segment in segments
        ).strip()

    # ---------------- Main Microphone Interface ----------------
    def get_text_from_mic(self) -> str:
        audio = self.record_until_silence()

        if audio.size == 0:
            print("Nothing to transcribe.")
            return ""

        print("Transcribing...")

        start_time = time.perf_counter()
        text = self.transcribe_audio(audio)

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        print(f"\n--- Transcription ({latency_ms:.0f} ms) ---")
        print(text)
        print("-----------------------------------")

        return text


# ---------------- Standalone Testing ----------------
if __name__ == "__main__":
    asr = ASR()
    asr.get_text_from_mic()
