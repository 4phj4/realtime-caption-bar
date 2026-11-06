"""Volcengine/Doubao streaming ASR client.

This module only handles protocol and audio conversion. The Qt UI lives in
native_caption.py.
"""

from __future__ import annotations

import asyncio
import contextlib
import gzip
import json
import math
import os
import struct
import time
import uuid
from dataclasses import dataclass
from typing import Callable


ASR_URL = "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_async"
DEFAULT_RESOURCE_ID = "volc.seedasr.sauc.duration"
SOURCE_SAMPLE_RATE = 48000
TARGET_SAMPLE_RATE = 16000
CHUNK_SECONDS = 0.2
MAX_AUDIO_QUEUE_SECONDS = 0.8
MAX_ASR_SESSION_SECONDS = 8 * 60
ASR_RECONNECT_PAUSE_SECONDS = 0.5
MAX_SERVER_LAG_SECONDS = 2.0
MAX_ACTIVE_AUDIO_WITHOUT_RESPONSE_SECONDS = 8.0
ACTIVE_AUDIO_LEVEL = 3


class AsrReconnectRequired(RuntimeError):
    """Raised when continuing a stale ASR session would increase subtitle delay."""


@dataclass(frozen=True)
class VolcAsrConfig:
    app_id: str
    access_token: str
    resource_id: str = DEFAULT_RESOURCE_ID
    url: str = ASR_URL


@dataclass(frozen=True)
class AudioOutputDevice:
    name: str
    id: str
    is_default: bool = False


def load_volc_asr_config() -> tuple[VolcAsrConfig | None, list[str]]:
    app_id = os.getenv("VOLC_ASR_APP_ID", "").strip()
    access_token = os.getenv("VOLC_ASR_ACCESS_TOKEN", "").strip()
    resource_id = os.getenv("VOLC_ASR_RESOURCE_ID", DEFAULT_RESOURCE_ID).strip()

    missing = []
    if not app_id:
        missing.append("VOLC_ASR_APP_ID")
    if not access_token:
        missing.append("VOLC_ASR_ACCESS_TOKEN")
    if missing:
        return None, missing

    return VolcAsrConfig(app_id=app_id, access_token=access_token, resource_id=resource_id), []


def list_audio_output_devices() -> list[AudioOutputDevice]:
    import soundcard as sc

    default_id = str(getattr(sc.default_speaker(), "id", ""))
    devices = []
    for speaker in sc.all_speakers():
        speaker_id = str(getattr(speaker, "id", ""))
        devices.append(
            AudioOutputDevice(
                name=str(getattr(speaker, "name", "未知输出设备")),
                id=speaker_id,
                is_default=speaker_id == default_id,
            )
        )
    return devices


def get_output_speaker(sc, output_device_id: str | None):
    speakers = list(sc.all_speakers())
    if output_device_id:
        for speaker in speakers:
            if str(getattr(speaker, "id", "")) == output_device_id:
                return speaker
    return sc.default_speaker()


def get_loopback_microphone(sc, speaker):
    speaker_id = str(getattr(speaker, "id", ""))
    speaker_name = str(getattr(speaker, "name", ""))
    for candidate_id in (speaker_id, speaker_name):
        if not candidate_id:
            continue
        try:
            microphone = sc.get_microphone(id=candidate_id, include_loopback=True)
        except Exception:
            continue
        if getattr(microphone, "isloopback", False):
            return microphone

    for microphone in sc.all_microphones(include_loopback=True):
        if not getattr(microphone, "isloopback", False):
            continue
        if str(getattr(microphone, "id", "")) == speaker_id:
            return microphone
    raise RuntimeError(f"未找到输出设备的系统声音回环：{speaker_name}")


def _make_header(message_type: int, flags: int, serialization: int, compression: int) -> bytes:
    version = 0x1
    header_size = 0x1
    return bytes(
        [
            (version << 4) | header_size,
            (message_type << 4) | flags,
            (serialization << 4) | compression,
            0x00,
        ]
    )


def _frame_payload(header: bytes, payload: bytes) -> bytes:
    return header + struct.pack(">I", len(payload)) + payload


def make_full_client_request(payload: dict) -> bytes:
    body = gzip.compress(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    header = _make_header(message_type=0x1, flags=0x0, serialization=0x1, compression=0x1)
    return _frame_payload(header, body)


def make_audio_request(pcm_bytes: bytes, final: bool = False) -> bytes:
    body = gzip.compress(pcm_bytes)
    flags = 0x2 if final else 0x0
    header = _make_header(message_type=0x2, flags=flags, serialization=0x0, compression=0x1)
    return _frame_payload(header, body)


def parse_server_message(message: bytes) -> dict:
    if len(message) < 8:
        return {}

    header_size = (message[0] & 0x0F) * 4
    message_type = message[1] >> 4
    flags = message[1] & 0x0F
    compression = message[2] & 0x0F
    offset = header_size

    if message_type == 0xF:
        if len(message) < offset + 8:
            return {"error": "ASR server returned a short error frame"}
        error_code = struct.unpack(">I", message[offset : offset + 4])[0]
        offset += 4
        error_size = struct.unpack(">I", message[offset : offset + 4])[0]
        offset += 4
        error_text = message[offset : offset + error_size].decode("utf-8", errors="replace")
        return {"error": f"{error_code}: {error_text}"}

    if message_type != 0x9:
        return {}

    if flags in (0x1, 0x3):
        if len(message) < offset + 4:
            return {}
        offset += 4

    if len(message) < offset + 4:
        return {}
    payload_size = struct.unpack(">I", message[offset : offset + 4])[0]
    offset += 4
    payload = message[offset : offset + payload_size]

    if compression == 0x1 and payload:
        payload = gzip.decompress(payload)
    if not payload:
        return {}
    return json.loads(payload.decode("utf-8"))


def convert_loopback_audio_to_pcm16(audio, source_rate: int = SOURCE_SAMPLE_RATE) -> tuple[bytes, int]:
    import numpy as np

    if audio is None or audio.size == 0:
        return b"", 0

    if audio.ndim == 2:
        mono = audio.mean(axis=1)
    else:
        mono = audio

    if source_rate == TARGET_SAMPLE_RATE:
        resampled = mono
    elif source_rate % TARGET_SAMPLE_RATE == 0:
        resampled = mono[:: source_rate // TARGET_SAMPLE_RATE]
    else:
        target_len = max(1, int(len(mono) * TARGET_SAMPLE_RATE / source_rate))
        source_x = np.arange(len(mono))
        target_x = np.linspace(0, len(mono) - 1, target_len)
        resampled = np.interp(target_x, source_x, mono)

    rms = float(np.sqrt(np.mean(np.square(mono)))) if len(mono) else 0.0
    level = min(100, int(rms * 450))
    pcm = np.clip(resampled, -1.0, 1.0)
    pcm = (pcm * 32767).astype("<i2")
    return pcm.tobytes(), level


class VolcAsrStream:
    def __init__(
        self,
        config: VolcAsrConfig,
        on_partial: Callable[[str], None],
        on_final: Callable[[str], None],
        on_level: Callable[[int], None],
        on_status: Callable[[str], None],
    ) -> None:
        self.config = config
        self.on_partial = on_partial
        self.on_final = on_final
        self.on_level = on_level
        self.on_status = on_status
        self._final_keys: set[tuple[int, int, str]] = set()

    async def run_system_audio(self, should_stop: Callable[[], bool]) -> None:
        await self.run_system_audio_from_device(should_stop, None)

    async def run_system_audio_from_device(
        self, should_stop: Callable[[], bool], output_device_id: str | None
    ) -> None:
        import soundcard as sc

        speaker = get_output_speaker(sc, output_device_id)
        loopback = get_loopback_microphone(sc, speaker)

        session_index = 1
        while not should_stop():
            reconnecting = False
            try:
                await self._run_one_system_audio_session(
                    should_stop=should_stop,
                    speaker_name=str(getattr(speaker, "name", "系统声音")),
                    loopback=loopback,
                    session_index=session_index,
                )
            except AsrReconnectRequired as exc:
                if should_stop():
                    break
                reconnecting = True
                self.on_status(f"实时字幕：{exc}，正在丢弃旧音频并重连")
            except (OSError, TimeoutError) as exc:
                if should_stop():
                    break
                reconnecting = True
                self.on_status(f"实时字幕：网络连接异常（{exc}），正在重连")
            session_index += 1
            if not should_stop():
                if not reconnecting:
                    self.on_status("实时字幕：定时刷新连接，防止长时间延迟累积")
                await asyncio.sleep(ASR_RECONNECT_PAUSE_SECONDS)

    async def _run_one_system_audio_session(
        self,
        should_stop: Callable[[], bool],
        speaker_name: str,
        loopback,
        session_index: int,
    ) -> None:
        import websockets

        headers = {
            "X-Api-App-Key": self.config.app_id,
            "X-Api-Access-Key": self.config.access_token,
            "X-Api-Resource-Id": self.config.resource_id,
            "X-Api-Connect-Id": str(uuid.uuid4()),
        }
        frames_per_read = int(SOURCE_SAMPLE_RATE * CHUNK_SECONDS)
        max_queue_chunks = max(1, math.ceil(MAX_AUDIO_QUEUE_SECONDS / CHUNK_SECONDS))
        audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=max_queue_chunks)
        dropped_chunks = 0
        last_drop_report = 0
        sent_audio_seconds = 0.0
        last_response_at = time.monotonic()
        active_audio_without_response = 0.0
        lagging_response_count = 0

        self._final_keys.clear()
        self.on_status(f"连接豆包 ASR：{self.config.resource_id}")
        connection = self._connect(websockets, headers)
        async with connection as websocket:
            await websocket.send(make_full_client_request(self._initial_request()))
            session_started_at = time.monotonic()
            self.on_status(
                f"实时字幕中：{speaker_name}（第 {session_index} 段，系统声音回环）"
            )

            async def receive_loop() -> None:
                nonlocal last_response_at, active_audio_without_response
                nonlocal lagging_response_count
                async for raw_message in websocket:
                    if isinstance(raw_message, str):
                        continue
                    response = parse_server_message(raw_message)
                    last_response_at = time.monotonic()
                    active_audio_without_response = 0.0
                    if "error" in response:
                        self.on_status(f"豆包 ASR 错误：{response['error']}")
                        continue
                    self._handle_response(response)

                    audio_info = response.get("audio_info")
                    if not isinstance(audio_info, dict):
                        continue
                    try:
                        server_audio_seconds = float(audio_info.get("duration", 0)) / 1000
                    except (TypeError, ValueError):
                        continue
                    if server_audio_seconds <= 0:
                        continue
                    server_lag = sent_audio_seconds - server_audio_seconds
                    if server_lag > MAX_SERVER_LAG_SECONDS:
                        lagging_response_count += 1
                    else:
                        lagging_response_count = 0
                    if lagging_response_count >= 3:
                        raise AsrReconnectRequired(
                            f"识别已落后约 {server_lag:.1f} 秒"
                        )

            async def capture_loop() -> None:
                nonlocal dropped_chunks, last_drop_report
                nonlocal active_audio_without_response
                try:
                    with loopback.recorder(samplerate=SOURCE_SAMPLE_RATE, channels=2) as recorder:
                        while not should_stop():
                            if time.monotonic() - session_started_at >= MAX_ASR_SESSION_SECONDS:
                                break
                            audio = await asyncio.to_thread(
                                recorder.record, numframes=frames_per_read
                            )
                            pcm_bytes, level = convert_loopback_audio_to_pcm16(audio)
                            self.on_level(level)
                            if not pcm_bytes:
                                continue
                            if level >= ACTIVE_AUDIO_LEVEL:
                                active_audio_without_response += CHUNK_SECONDS
                            else:
                                active_audio_without_response = max(
                                    0.0,
                                    active_audio_without_response - CHUNK_SECONDS,
                                )
                            response_silence = time.monotonic() - last_response_at
                            if (
                                active_audio_without_response
                                >= MAX_ACTIVE_AUDIO_WITHOUT_RESPONSE_SECONDS
                                and response_silence
                                >= MAX_ACTIVE_AUDIO_WITHOUT_RESPONSE_SECONDS
                            ):
                                raise AsrReconnectRequired(
                                    f"连续 {response_silence:.0f} 秒没有新识别结果"
                                )
                            if audio_queue.full():
                                with contextlib.suppress(asyncio.QueueEmpty):
                                    audio_queue.get_nowait()
                                dropped_chunks += 1
                                if dropped_chunks == 1 or dropped_chunks - last_drop_report >= 10:
                                    dropped_seconds = dropped_chunks * CHUNK_SECONDS
                                    self.on_status(
                                        "识别端处理偏慢，已丢弃"
                                        f"{dropped_seconds:.1f}秒旧音频，优先保持字幕实时"
                                    )
                                    last_drop_report = dropped_chunks
                            await audio_queue.put(pcm_bytes)
                finally:
                    while audio_queue.full():
                        with contextlib.suppress(asyncio.QueueEmpty):
                            audio_queue.get_nowait()
                    await audio_queue.put(None)

            async def send_loop() -> None:
                nonlocal sent_audio_seconds
                next_send_at = time.monotonic()
                while True:
                    pcm_bytes = await audio_queue.get()
                    if pcm_bytes is None:
                        break
                    now = time.monotonic()
                    if next_send_at > now:
                        await asyncio.sleep(next_send_at - now)
                    await websocket.send(make_audio_request(pcm_bytes))
                    sent_audio_seconds += CHUNK_SECONDS
                    next_send_at += CHUNK_SECONDS
                    if next_send_at < time.monotonic() - CHUNK_SECONDS:
                        next_send_at = time.monotonic()
                await websocket.send(make_audio_request(b"", final=True))

            receiver = asyncio.create_task(receive_loop())
            capturer = asyncio.create_task(capture_loop())
            sender = asyncio.create_task(send_loop())
            try:
                while True:
                    if receiver.done():
                        error = receiver.exception()
                        if error is not None:
                            if isinstance(error, AsrReconnectRequired):
                                raise error
                            raise AsrReconnectRequired(f"识别连接异常：{error}") from error
                        if not should_stop() and not capturer.done():
                            raise AsrReconnectRequired("识别连接提前结束")
                    if sender.done() and not capturer.done():
                        error = sender.exception()
                        if error is not None:
                            raise AsrReconnectRequired(f"音频发送异常：{error}") from error
                        raise AsrReconnectRequired("音频发送提前结束")
                    if capturer.done():
                        error = capturer.exception()
                        if error is not None:
                            raise error
                        await sender
                        break
                    await asyncio.sleep(0.05)
                await asyncio.sleep(0.8)
            finally:
                for task in (receiver, capturer, sender):
                    task.cancel()
                await asyncio.gather(receiver, capturer, sender, return_exceptions=True)
                self.on_level(0)
                if should_stop():
                    self.on_status("实时字幕已停止")

    def _connect(self, websockets, headers: dict):
        try:
            return websockets.connect(self.config.url, additional_headers=headers, max_size=None)
        except TypeError:
            return websockets.connect(self.config.url, extra_headers=headers, max_size=None)

    def _initial_request(self) -> dict:
        return {
            "user": {"uid": "native-caption"},
            "audio": {
                "format": "pcm",
                "codec": "raw",
                "rate": TARGET_SAMPLE_RATE,
                "bits": 16,
                "channel": 1,
            },
            "request": {
                "model_name": "bigmodel",
                "enable_nonstream": True,
                "enable_itn": True,
                "enable_punc": True,
                "enable_ddc": False,
                "show_utterances": True,
                "result_type": "single",
                "end_window_size": 600,
                "force_to_speech_time": 1000,
                "enable_accelerate_text": True,
                "accelerate_score": 8,
            },
        }

    def _handle_response(self, response: dict) -> None:
        result = response.get("result")
        if isinstance(result, list):
            result = result[0] if result else {}
        if not isinstance(result, dict):
            return

        utterances = result.get("utterances") or []
        for utterance in utterances:
            if not isinstance(utterance, dict) or not utterance.get("definite"):
                continue
            text = str(utterance.get("text") or "").strip()
            if not text:
                continue
            key = (
                int(utterance.get("start_time") or 0),
                int(utterance.get("end_time") or 0),
                text,
            )
            if key in self._final_keys:
                continue
            self._final_keys.add(key)
            self.on_final(text)

        text = str(result.get("text") or "").strip()
        if text:
            self.on_partial(text)
