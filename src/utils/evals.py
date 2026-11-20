import time
from typing import Optional, Dict, Any


def calculate_metrics(
    start_time: float,
    first_token_time: Optional[float],
    end_time: float,
    token_count: int,
    speech_stopped_time: Optional[float] = None,
    transcription_time_s: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Calculate TTFT (Time to First Token), token generation speed, and pipeline metrics.

    Args:
        start_time: Timestamp when generation request was initiated.
        first_token_time: Timestamp when the first content token was produced.
        end_time: Timestamp when generation concluded.
        token_count: Total number of tokens generated.
        speech_stopped_time: Timestamp when the user stopped speaking (from ASR).
        transcription_time_s: Total time taken to transcribe audio in seconds (from ASR).

    Returns:
        Dict containing ttft_ms, tokens_per_sec, total_time_s, token_count,
        stopped_speaking_to_ttft_ms, transcription_time_ms, etc.
    """
    total_time_s = max(0.0, end_time - start_time)

    if first_token_time is not None:
        ttft_ms = max(0.0, (first_token_time - start_time) * 1000)
        generation_time_s = max(0.0, end_time - first_token_time)
        stopped_speaking_to_ttft_ms = (
            max(0.0, (first_token_time - speech_stopped_time) * 1000)
            if speech_stopped_time is not None
            else None
        )
    else:
        ttft_ms = 0.0
        generation_time_s = 0.0
        stopped_speaking_to_ttft_ms = None

    transcription_time_ms = (
        (transcription_time_s * 1000)
        if transcription_time_s is not None
        else None
    )

    # Tokens/sec during decode phase (excluding the prompt evaluation / TTFT phase)
    if generation_time_s > 0 and token_count > 1:
        tokens_per_sec = (token_count - 1) / generation_time_s
    elif total_time_s > 0 and token_count > 0:
        tokens_per_sec = token_count / total_time_s
    else:
        tokens_per_sec = 0.0

    # Overall throughput including prompt processing
    overall_tokens_per_sec = (
        (token_count / total_time_s) if total_time_s > 0 else 0.0
    )

    end_to_end_latency_s = (
        max(0.0, end_time - speech_stopped_time)
        if speech_stopped_time is not None
        else None
    )

    return {
        "ttft_ms": round(ttft_ms, 2),
        "model_ttft_ms": round(ttft_ms, 2),
        "tokens_per_sec": round(tokens_per_sec, 2),
        "overall_tokens_per_sec": round(overall_tokens_per_sec, 2),
        "token_count": token_count,
        "total_time_s": round(total_time_s, 2),
        "model_generation_time_s": round(total_time_s, 2),
        "generation_time_s": round(generation_time_s, 2),
        "transcription_time_ms": (
            round(transcription_time_ms, 2)
            if transcription_time_ms is not None
            else None
        ),
        "transcription_time_s": (
            round(transcription_time_s, 4)
            if transcription_time_s is not None
            else None
        ),
        "stopped_speaking_to_ttft_ms": (
            round(stopped_speaking_to_ttft_ms, 2)
            if stopped_speaking_to_ttft_ms is not None
            else None
        ),
        "end_to_end_latency_s": (
            round(end_to_end_latency_s, 2)
            if end_to_end_latency_s is not None
            else None
        ),
        "end_to_end_latency_ms": (
            round(end_to_end_latency_s * 1000.0, 2)
            if end_to_end_latency_s is not None
            else None
        ),
    }


def print_metrics(metrics: Dict[str, Any]) -> None:
    """
    Print formatted generation performance metrics with clear labels and explanations
    so anyone reading the logs knows exactly what each measurement represents.
    """
    ttft_ms = metrics.get("ttft_ms", 0.0)
    tps = metrics.get("tokens_per_sec", 0.0)
    token_count = metrics.get("token_count", 0)
    gen_time_s = metrics.get("total_time_s", 0.0)
    transcription_ms = metrics.get("transcription_time_ms")
    stopped_to_ttft_ms = metrics.get("stopped_speaking_to_ttft_ms")
    e2e_time_s = metrics.get("end_to_end_latency_s")

    if stopped_to_ttft_ms is not None and transcription_ms is not None and e2e_time_s is not None:
        print("--- End-to-End Performance Metrics (Voice Pipeline) ---")
        print(f"• ASR Transcription Time   : {transcription_ms:.0f} ms   [Audio speech -> text]")
        print(f"• Model TTFT               : {ttft_ms:.1f} ms [Prompt fed to LLM -> 1st token]")
        print(f"• Stopped Speaking -> TTFT : {stopped_to_ttft_ms:.1f} ms [User stopped speaking -> 1st token]")
        print(f"• Model Generation Time    : {gen_time_s:.2f} s   [Prompt fed to LLM -> final token]")
        print(f"• Full End-to-End Latency  : {e2e_time_s:.2f} s   [User stopped speaking -> final token]")
        print(f"• Model Generation Speed   : {tps:.2f} tokens/sec ({token_count} tokens)")
        print("-------------------------------------------------------")
    else:
        print("--- Generation Metrics (Text Input) ---")
        print(f"• Model TTFT               : {ttft_ms:.1f} ms [Prompt fed to LLM -> 1st token]")
        print(f"• Model Generation Time    : {gen_time_s:.2f} s   [Prompt fed to LLM -> final token]")
        print(f"• Model Generation Speed   : {tps:.2f} tokens/sec ({token_count} tokens)")
        print("---------------------------------------")


class LLMEval:
    """
    Evaluation tracker for LLM streaming responses.
    """

    def __init__(self, asr_metrics: Optional[Dict[str, Any]] = None):
        self.start_time: float = 0.0
        self.first_token_time: Optional[float] = None
        self.end_time: float = 0.0
        self.token_count: int = 0
        self.asr_metrics: Optional[Dict[str, Any]] = asr_metrics

    def start(self, asr_metrics: Optional[Dict[str, Any]] = None) -> None:
        """Start timing a generation request."""
        self.start_time = time.perf_counter()
        self.first_token_time = None
        self.end_time = 0.0
        self.token_count = 0
        if asr_metrics is not None:
            self.asr_metrics = asr_metrics

    def record_token(self) -> None:
        """Record each newly arrived token chunk."""
        if self.first_token_time is None:
            self.first_token_time = time.perf_counter()
        self.token_count += 1

    def stop(self) -> Dict[str, Any]:
        """Stop timing and calculate final metrics."""
        self.end_time = time.perf_counter()
        speech_stopped_time = (
            self.asr_metrics.get("speech_stopped_time")
            if self.asr_metrics
            else None
        )
        transcription_time_s = (
            self.asr_metrics.get("transcription_time_s")
            if self.asr_metrics
            else None
        )
        return calculate_metrics(
            start_time=self.start_time,
            first_token_time=self.first_token_time,
            end_time=self.end_time,
            token_count=self.token_count,
            speech_stopped_time=speech_stopped_time,
            transcription_time_s=transcription_time_s,
        )

    def print_metrics(self) -> Dict[str, Any]:
        """Stop and print metrics."""
        metrics = self.stop()
        print_metrics(metrics)
        return metrics


def calculate_asr_metrics(
    speech_duration_s: float,
    transcription_time_s: float,
    speech_stopped_time: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Calculate ASR performance metrics.

    Args:
        speech_duration_s: Total duration of spoken audio in seconds.
        transcription_time_s: Total time taken to transcribe the audio in seconds.
        speech_stopped_time: Timestamp when recording finished / user stopped speaking.

    Returns:
        Dict containing transcription_time_s, transcription_time_ms, speech_duration_s,
        transcription_speed (speedup factor relative to real-time), rtf (real-time factor),
        and speech_stopped_time.
    """
    transcription_time_ms = transcription_time_s * 1000.0

    speed_factor = (
        (speech_duration_s / transcription_time_s)
        if transcription_time_s > 0
        else 0.0
    )
    rtf = (
        (transcription_time_s / speech_duration_s)
        if speech_duration_s > 0
        else 0.0
    )

    return {
        "transcription_time_s": round(transcription_time_s, 4),
        "transcription_time_ms": round(transcription_time_ms, 2),
        "speech_duration_s": round(speech_duration_s, 2),
        "transcription_speed": round(speed_factor, 2),
        "speed": round(speed_factor, 2),
        "rtf": round(rtf, 4),
        "speech_stopped_time": speech_stopped_time,
    }


def print_asr_metrics(metrics: Dict[str, Any], text: str = "") -> None:
    """
    Print formatted ASR transcription performance metrics.
    """
    time_ms = metrics.get("transcription_time_ms", 0.0)
    speech_s = metrics.get("speech_duration_s", 0.0)
    speed = metrics.get("transcription_speed", 0.0)

    print(
        f"\n--- ASR Metrics (Transcription Time: {time_ms:.0f} ms | "
        f"Speech Duration: {speech_s:.2f}s | Speed: {speed:.2f}x real-time) ---"
    )
    if text:
        print(text)
        print("-----------------------------------------------------------------")
