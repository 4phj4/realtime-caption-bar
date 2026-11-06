"""Windows native always-on-top caption bar.

Current scope:
- a frameless always-on-top caption bar
- manual test caption input and transcript export
- Windows system-audio loopback capture
- Volcengine/Doubao streaming ASR for realtime captions
"""

from __future__ import annotations

import asyncio
import ctypes
import ctypes.wintypes
import json
import platform
import re
import sys
import threading
from datetime import datetime
from pathlib import Path

try:
    from PySide6 import QtCore, QtGui, QtWidgets
except ModuleNotFoundError:
    try:
        from PyQt6 import QtCore, QtGui, QtWidgets
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "缺少桌面 GUI 依赖。请先运行：python -m pip install -r requirements.txt"
        ) from exc


Qt = QtCore.Qt
try:
    qt_signal = QtCore.Signal
except AttributeError:
    qt_signal = QtCore.pyqtSignal

APP_DIR = Path(__file__).resolve().parent
SETTINGS_PATH = APP_DIR / "native_caption_settings.json"
TRANSCRIPTS_DIR = APP_DIR / "transcripts"

HWND_TOPMOST = -1
HWND_NOTOPMOST = -2
SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOACTIVATE = 0x0010
SWP_SHOWWINDOW = 0x0040

DEFAULT_SETTINGS = {
    "caption_geometry": None,
    "control_geometry": None,
    "font_size": 36,
    "opacity": 88,
    "topmost": True,
    "audio_output_id": "",
}
MAX_CAPTION_CHARS = 20


def is_windows() -> bool:
    return platform.system().lower() == "windows"


def force_hwnd_topmost(hwnd: int, enabled: bool = True) -> None:
    """Use Win32 SetWindowPos as the final topmost guarantee on Windows."""
    if not is_windows() or not hwnd:
        return

    ctypes.windll.user32.SetWindowPos(
        ctypes.wintypes.HWND(hwnd),
        ctypes.wintypes.HWND(HWND_TOPMOST if enabled else HWND_NOTOPMOST),
        0,
        0,
        0,
        0,
        SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW,
    )


def global_mouse_pos(event: QtGui.QMouseEvent) -> QtCore.QPoint:
    if hasattr(event, "globalPosition"):
        return event.globalPosition().toPoint()
    return event.globalPos()


def load_settings() -> dict:
    if not SETTINGS_PATH.exists():
        return DEFAULT_SETTINGS.copy()
    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return DEFAULT_SETTINGS.copy()
    if not isinstance(data, dict):
        return DEFAULT_SETTINGS.copy()
    settings = DEFAULT_SETTINGS.copy()
    settings.update(data)
    return settings


def geometry_to_dict(widget: QtWidgets.QWidget) -> dict[str, int]:
    geometry = widget.geometry()
    return {
        "x": geometry.x(),
        "y": geometry.y(),
        "width": geometry.width(),
        "height": geometry.height(),
    }


def geometry_from_dict(data: object) -> QtCore.QRect | None:
    if not isinstance(data, dict):
        return None
    try:
        return QtCore.QRect(
            int(data["x"]),
            int(data["y"]),
            int(data["width"]),
            int(data["height"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def rect_is_visible(rect: QtCore.QRect) -> bool:
    for screen in QtWidgets.QApplication.screens():
        if screen.availableGeometry().intersects(rect):
            return True
    return False


def restore_geometry(widget: QtWidgets.QWidget, data: object) -> bool:
    rect = geometry_from_dict(data)
    if rect is None or rect.width() < 80 or rect.height() < 40:
        return False
    if not rect_is_visible(rect):
        return False
    widget.setGeometry(rect)
    return True


def caption_display_text(text: str) -> str:
    text = text.strip()
    if not text:
        return "等待测试字幕"
    return text[-MAX_CAPTION_CHARS:]


def sanitize_transcript_name(name: str) -> str:
    """Turn a user-entered title into a safe Windows filename component."""
    name = re.sub(r'[<>:"/\\|?*]', "_", name.strip())
    name = name.rstrip(" .")
    if name.lower().endswith(".txt"):
        name = name[:-4].rstrip(" .")
    return name[:80].rstrip(" .")


class SystemAudioMonitor(QtCore.QThread):
    level_changed = qt_signal(int)
    status_changed = qt_signal(str)

    def __init__(self, output_device_id: str | None = None) -> None:
        super().__init__()
        self._running = True
        self.output_device_id = output_device_id

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        if not is_windows():
            self.status_changed.emit("系统声音捕获目前只支持 Windows")
            self.level_changed.emit(0)
            return

        try:
            import numpy as np
            import soundcard as sc
            from volc_asr_client import get_loopback_microphone, get_output_speaker
        except ModuleNotFoundError:
            self.status_changed.emit("缺少 soundcard：请运行 python -m pip install -r requirements.txt")
            self.level_changed.emit(0)
            return

        samplerate = 48000
        frames_per_read = int(samplerate * 0.1)

        try:
            speaker = get_output_speaker(sc, self.output_device_id)
            loopback = get_loopback_microphone(sc, speaker)
            self.status_changed.emit(f"正在监听系统声音：{speaker.name}（回环）")

            with loopback.recorder(samplerate=samplerate, channels=2) as recorder:
                while self._running:
                    data = recorder.record(numframes=frames_per_read)
                    if data is None or data.size == 0:
                        self.level_changed.emit(0)
                        continue

                    rms = float(np.sqrt(np.mean(np.square(data))))
                    level = min(100, int(rms * 450))
                    self.level_changed.emit(level)
        except Exception as exc:
            self.status_changed.emit(f"系统声音捕获失败：{exc}")
            self.level_changed.emit(0)
            return

        self.status_changed.emit("系统声音监听已停止")
        self.level_changed.emit(0)


class VolcRealtimeCaptionWorker(QtCore.QThread):
    final_text = qt_signal(str)
    level_changed = qt_signal(int)
    status_changed = qt_signal(str)

    def __init__(self, config, output_device_id: str | None = None) -> None:
        super().__init__()
        self.config = config
        self.output_device_id = output_device_id
        self._running = True
        self._partial_lock = threading.Lock()
        self._latest_partial = ""
        self._partial_version = 0

    def stop(self) -> None:
        self._running = False

    def store_latest_partial(self, text: str) -> None:
        """Keep only the newest partial result instead of queueing every UI update."""
        text = text.strip()
        if not text:
            return
        with self._partial_lock:
            self._latest_partial = text
            self._partial_version += 1

    def latest_partial(self) -> tuple[int, str]:
        with self._partial_lock:
            return self._partial_version, self._latest_partial

    def run(self) -> None:
        try:
            from volc_asr_client import VolcAsrStream
        except ModuleNotFoundError as exc:
            self.status_changed.emit(f"缺少实时识别依赖：{exc.name}")
            self.level_changed.emit(0)
            return

        stream = VolcAsrStream(
            config=self.config,
            on_partial=self.store_latest_partial,
            on_final=self.final_text.emit,
            on_level=self.level_changed.emit,
            on_status=self.status_changed.emit,
        )

        try:
            asyncio.run(
                stream.run_system_audio_from_device(
                    lambda: not self._running, self.output_device_id
                )
            )
        except ModuleNotFoundError as exc:
            self.status_changed.emit(f"缺少依赖：{exc.name}，请运行 python -m pip install -r requirements.txt")
        except Exception as exc:
            self.status_changed.emit(f"实时字幕失败：{exc}")
        finally:
            self.level_changed.emit(0)


class CaptionWindow(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._drag_offset: QtCore.QPoint | None = None
        self._topmost_enabled = True
        self._font_size = DEFAULT_SETTINGS["font_size"]
        self._opacity_percent = DEFAULT_SETTINGS["opacity"]

        self.setWindowTitle("实时悬浮字幕条")
        self.setObjectName("CaptionWindow")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)

        self.caption_label = QtWidgets.QLabel("等待测试字幕")
        self.caption_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.caption_label.setWordWrap(False)
        self._apply_caption_font_style()

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(28, 10, 28, 10)
        layout.addWidget(self.caption_label)

        self.setStyleSheet(
            """
            #CaptionWindow {
                background: #050505;
                border: 1px solid rgba(255, 255, 255, 0.18);
            }
            """
        )
        self.resize(820, 76)
        self.set_caption_opacity(self._opacity_percent)

        self._topmost_timer = QtCore.QTimer(self)
        self._topmost_timer.setInterval(500)
        self._topmost_timer.timeout.connect(self.apply_topmost)
        self._topmost_timer.start()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        self.apply_topmost()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = global_mouse_pos(event) - self.frameGeometry().topLeft()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(global_mouse_pos(event) - self._drag_offset)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        self._drag_offset = None
        super().mouseReleaseEvent(event)

    def set_caption(self, text: str) -> None:
        self.caption_label.setText(caption_display_text(text))
        self.apply_topmost()

    def set_caption_font_size(self, size: int) -> None:
        self._font_size = max(24, min(size, 64))
        self._apply_caption_font_style()

    def _apply_caption_font_style(self) -> None:
        self.caption_label.setStyleSheet(
            f"color: white; font-size: {self._font_size}px; font-weight: 700;"
        )

    def set_caption_opacity(self, opacity_percent: int) -> None:
        self._opacity_percent = max(30, min(opacity_percent, 100))
        self.setWindowOpacity(self._opacity_percent / 100)

    def set_topmost_enabled(self, enabled: bool) -> None:
        self._topmost_enabled = enabled
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, enabled)
        self.show()
        self.apply_topmost()

    def apply_topmost(self) -> None:
        hwnd = int(self.winId())
        force_hwnd_topmost(hwnd, self._topmost_enabled)

    @property
    def font_size(self) -> int:
        return self._font_size

    @property
    def opacity_percent(self) -> int:
        return self._opacity_percent

    @property
    def topmost_enabled(self) -> bool:
        return self._topmost_enabled


class ControlPanel(QtWidgets.QWidget):
    def __init__(self, caption_window: CaptionWindow) -> None:
        super().__init__()
        self.caption_window = caption_window
        self.transcript_lines: list[str] = []
        self._last_committed_caption = ""
        self._transcript_dirty = False
        self._last_transcript_name = ""
        self.audio_monitor: SystemAudioMonitor | None = None
        self.asr_worker: VolcRealtimeCaptionWorker | None = None
        self._last_partial_version = 0
        self._partial_display_timer = QtCore.QTimer(self)
        self._partial_display_timer.setInterval(50)
        self._partial_display_timer.timeout.connect(self.display_latest_partial)
        self.setWindowTitle("字幕器控制面板")
        self.setMinimumWidth(420)

        self.input_box = QtWidgets.QLineEdit()
        self.input_box.setPlaceholderText("输入测试字幕，回车记录到文字稿")
        self.input_box.textChanged.connect(self.preview_caption)
        self.input_box.returnPressed.connect(self.commit_caption)

        self.commit_button = QtWidgets.QPushButton("记录这句")
        self.commit_button.clicked.connect(self.commit_caption)

        self.visible_button = QtWidgets.QPushButton("隐藏字幕条")
        self.visible_button.clicked.connect(self.toggle_caption_visible)

        self.save_button = QtWidgets.QPushButton("保存文字稿")
        self.save_button.clicked.connect(self.save_transcript)

        self.clear_button = QtWidgets.QPushButton("清空文字稿")
        self.clear_button.clicked.connect(self.clear_transcript)

        self.status_label = QtWidgets.QLabel("文字稿：0 句")
        self.status_label.setStyleSheet("color: #666;")

        self.audio_source_combo = QtWidgets.QComboBox()
        self.audio_source_combo.setMinimumWidth(280)

        self.refresh_audio_button = QtWidgets.QPushButton("刷新音频设备")
        self.refresh_audio_button.clicked.connect(self.refresh_audio_outputs)

        self.audio_button = QtWidgets.QPushButton("开始系统声音测试")
        self.audio_button.clicked.connect(self.toggle_audio_monitor)

        self.audio_level_bar = QtWidgets.QProgressBar()
        self.audio_level_bar.setRange(0, 100)
        self.audio_level_bar.setValue(0)
        self.audio_level_bar.setTextVisible(False)

        self.audio_status_label = QtWidgets.QLabel("系统声音：未开始")
        self.audio_status_label.setWordWrap(True)
        self.audio_status_label.setStyleSheet("color: #666;")

        self.asr_check_button = QtWidgets.QPushButton("检测豆包配置")
        self.asr_check_button.clicked.connect(self.check_volc_asr_config)

        self.asr_button = QtWidgets.QPushButton("开始实时字幕")
        self.asr_button.clicked.connect(self.toggle_realtime_caption)

        self.asr_status_label = QtWidgets.QLabel("豆包实时字幕：未开始")
        self.asr_status_label.setWordWrap(True)
        self.asr_status_label.setStyleSheet("color: #666;")

        self.topmost_checkbox = QtWidgets.QCheckBox("保持字幕条置顶")
        self.topmost_checkbox.setChecked(True)
        self.topmost_checkbox.toggled.connect(self.caption_window.set_topmost_enabled)

        self.font_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.font_slider.setRange(24, 64)
        self.font_slider.setValue(36)
        self.font_slider.valueChanged.connect(self.caption_window.set_caption_font_size)

        self.opacity_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(30, 100)
        self.opacity_slider.setValue(88)
        self.opacity_slider.valueChanged.connect(self.caption_window.set_caption_opacity)

        form = QtWidgets.QFormLayout()
        form.addRow("测试字幕", self.input_box)
        form.addRow("字号", self.font_slider)
        form.addRow("透明度", self.opacity_slider)
        form.addRow("音频输出", self.audio_source_combo)

        transcript_buttons = QtWidgets.QHBoxLayout()
        transcript_buttons.addWidget(self.commit_button)
        transcript_buttons.addWidget(self.save_button)
        transcript_buttons.addWidget(self.clear_button)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(transcript_buttons)
        layout.addWidget(self.topmost_checkbox)
        layout.addWidget(self.visible_button)
        layout.addWidget(self.status_label)
        layout.addWidget(self.audio_button)
        layout.addWidget(self.refresh_audio_button)
        layout.addWidget(self.audio_level_bar)
        layout.addWidget(self.audio_status_label)
        layout.addWidget(self.asr_check_button)
        layout.addWidget(self.asr_button)
        layout.addWidget(self.asr_status_label)

        hint = QtWidgets.QLabel("验收：播放电脑声音时音量条波动；开始实时字幕后，识别结果应出现在字幕条。")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #666;")
        layout.addWidget(hint)
        self.refresh_audio_outputs()

    def preview_caption(self, text: str) -> None:
        self.caption_window.set_caption(text)

    def commit_caption(self) -> None:
        text = self.input_box.text().strip()
        if not text or text == self._last_committed_caption:
            return
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.transcript_lines.append(f"[{timestamp}] {text}")
        self._last_committed_caption = text
        self._transcript_dirty = True
        self.caption_window.set_caption(text)
        self.update_transcript_status()

    def update_transcript_status(self, saved_path: Path | None = None) -> None:
        if saved_path is None:
            unsaved_hint = "（未保存）" if self._transcript_dirty else ""
            self.status_label.setText(
                f"文字稿：{len(self.transcript_lines)} 句{unsaved_hint}"
            )
        else:
            self.status_label.setText(f"已保存：{saved_path.name}")

    def prompt_transcript_name(self) -> str | None:
        default_name = self._last_transcript_name or datetime.now().strftime(
            "课程文字稿-%Y%m%d"
        )
        while True:
            name, accepted = QtWidgets.QInputDialog.getText(
                self,
                "保存文字稿",
                "请输入课程或文字稿名称：",
                text=default_name,
            )
            if not accepted:
                return None
            safe_name = sanitize_transcript_name(name)
            if safe_name:
                return safe_name
            QtWidgets.QMessageBox.warning(
                self,
                "名称无效",
                "请输入一个有效的文字稿名称。",
            )

    def save_transcript(self) -> bool:
        self.commit_caption()
        if not self.transcript_lines:
            self.status_label.setText("文字稿为空，暂无可保存内容")
            return False

        transcript_name = self.prompt_transcript_name()
        if transcript_name is None:
            return False

        TRANSCRIPTS_DIR.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{transcript_name}-{timestamp}.txt"
        path = TRANSCRIPTS_DIR / filename
        duplicate_index = 2
        while path.exists():
            path = TRANSCRIPTS_DIR / (
                f"{transcript_name}-{timestamp}-{duplicate_index}.txt"
            )
            duplicate_index += 1
        content = "\n".join(self.transcript_lines) + "\n"
        try:
            path.write_text(content, encoding="utf-8")
        except OSError as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "保存失败",
                f"文字稿保存失败：\n{exc}",
            )
            return False
        self._transcript_dirty = False
        self._last_transcript_name = transcript_name
        self.update_transcript_status(path)
        return True

    def clear_transcript(self) -> None:
        self.transcript_lines.clear()
        self._last_committed_caption = ""
        self._transcript_dirty = False
        self._last_transcript_name = ""
        self.update_transcript_status()

    def append_final_caption(self, text: str) -> None:
        text = text.strip()
        if not text or text == self._last_committed_caption:
            return
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.transcript_lines.append(f"[{timestamp}] {text}")
        self._last_committed_caption = text
        self._transcript_dirty = True
        self.update_transcript_status()

    def selected_audio_output_id(self) -> str | None:
        value = self.audio_source_combo.currentData()
        if value is None:
            return None
        value = str(value).strip()
        return value or None

    def refresh_audio_outputs(self, preferred_id: str | None = None) -> None:
        current_id = preferred_id or self.selected_audio_output_id()
        self.audio_source_combo.clear()
        try:
            from volc_asr_client import list_audio_output_devices

            devices = list_audio_output_devices()
        except Exception as exc:
            self.audio_source_combo.addItem("未能读取音频输出设备", "")
            self.audio_status_label.setText(f"音频设备读取失败：{exc}")
            return

        if not devices:
            self.audio_source_combo.addItem("未找到音频输出设备", "")
            self.audio_status_label.setText("未找到音频输出设备")
            return

        selected_index = 0
        for index, device in enumerate(devices):
            label = device.name
            if device.is_default:
                label += "（默认）"
            if "hands-free" in label.lower() or "免提" in label:
                label += "（通话模式，谨慎选择）"
            self.audio_source_combo.addItem(label, device.id)
            if current_id and device.id == current_id:
                selected_index = index
        self.audio_source_combo.setCurrentIndex(selected_index)

    def toggle_audio_monitor(self) -> None:
        if self.audio_monitor is not None and self.audio_monitor.isRunning():
            self.stop_audio_monitor()
            return
        self.start_audio_monitor()

    def start_audio_monitor(self) -> None:
        self.audio_monitor = SystemAudioMonitor(self.selected_audio_output_id())
        self.audio_monitor.level_changed.connect(self.audio_level_bar.setValue)
        self.audio_monitor.status_changed.connect(self.audio_status_label.setText)
        self.audio_monitor.finished.connect(self.on_audio_monitor_finished)
        self.audio_button.setText("停止系统声音测试")
        self.audio_status_label.setText("系统声音：启动中")
        self.audio_monitor.start()

    def stop_audio_monitor(self) -> None:
        if self.audio_monitor is None:
            return
        self.audio_monitor.stop()
        self.audio_monitor.wait(1200)
        self.audio_level_bar.setValue(0)
        self.audio_button.setText("开始系统声音测试")

    def on_audio_monitor_finished(self) -> None:
        self.audio_level_bar.setValue(0)
        self.audio_button.setText("开始系统声音测试")
        self.audio_monitor = None

    def check_volc_asr_config(self) -> bool:
        try:
            from volc_asr_client import load_volc_asr_config
        except ModuleNotFoundError:
            self.asr_status_label.setText("缺少 volc_asr_client.py")
            return False

        config, missing = load_volc_asr_config()
        if missing:
            self.asr_status_label.setText("缺少环境变量：" + "、".join(missing))
            return False
        self.asr_status_label.setText(f"豆包配置可用：{config.resource_id}")
        return True

    def toggle_realtime_caption(self) -> None:
        if self.asr_worker is not None and self.asr_worker.isRunning():
            self.stop_realtime_caption()
            return
        self.start_realtime_caption()

    def display_latest_partial(self) -> None:
        """Coalesce bursty ASR updates so the GUI never replays stale captions."""
        worker = self.asr_worker
        if worker is None:
            return
        version, text = worker.latest_partial()
        if version <= self._last_partial_version:
            return
        self._last_partial_version = version
        self.caption_window.set_caption(text)

    def start_realtime_caption(self) -> None:
        try:
            from volc_asr_client import load_volc_asr_config
        except ModuleNotFoundError:
            self.asr_status_label.setText("缺少 volc_asr_client.py")
            return

        config, missing = load_volc_asr_config()
        if missing:
            self.asr_status_label.setText("缺少环境变量：" + "、".join(missing))
            return

        if self.audio_monitor is not None and self.audio_monitor.isRunning():
            self.stop_audio_monitor()

        self.asr_worker = VolcRealtimeCaptionWorker(config, self.selected_audio_output_id())
        self._last_partial_version = 0
        self.asr_worker.final_text.connect(self.on_final_asr_text)
        self.asr_worker.level_changed.connect(self.audio_level_bar.setValue)
        self.asr_worker.status_changed.connect(self.asr_status_label.setText)
        self.asr_worker.finished.connect(self.on_realtime_caption_finished)
        self._partial_display_timer.start()
        self.asr_button.setText("停止实时字幕")
        self.asr_status_label.setText("豆包实时字幕：启动中")
        self.asr_worker.start()

    def stop_realtime_caption(self) -> None:
        if self.asr_worker is None:
            return
        self.asr_worker.stop()
        self.asr_worker.wait(1500)
        self.audio_level_bar.setValue(0)
        self.asr_button.setText("开始实时字幕")

    def on_realtime_caption_finished(self) -> None:
        self.display_latest_partial()
        self._partial_display_timer.stop()
        self.audio_level_bar.setValue(0)
        self.asr_button.setText("开始实时字幕")
        self.asr_worker = None

    def on_final_asr_text(self, text: str) -> None:
        self.caption_window.set_caption(text)
        self.append_final_caption(text)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self._transcript_dirty and self.transcript_lines:
            message_box = QtWidgets.QMessageBox(self)
            message_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
            message_box.setWindowTitle("文字稿尚未保存")
            message_box.setText("本次文字稿还有未保存的内容。")
            message_box.setInformativeText("现在关闭会丢失这些内容，是否先保存？")
            save_button = message_box.addButton(
                "保存文字稿", QtWidgets.QMessageBox.ButtonRole.AcceptRole
            )
            discard_button = message_box.addButton(
                "不保存并关闭", QtWidgets.QMessageBox.ButtonRole.DestructiveRole
            )
            cancel_button = message_box.addButton(
                "取消关闭", QtWidgets.QMessageBox.ButtonRole.RejectRole
            )
            message_box.setDefaultButton(save_button)
            message_box.exec()

            clicked_button = message_box.clickedButton()
            if clicked_button is cancel_button:
                event.ignore()
                return
            if clicked_button is save_button and not self.save_transcript():
                event.ignore()
                return
            if clicked_button is not save_button and clicked_button is not discard_button:
                event.ignore()
                return

        self.stop_realtime_caption()
        self.stop_audio_monitor()
        self.caption_window.close()
        super().closeEvent(event)
        app = QtWidgets.QApplication.instance()
        if app is not None:
            QtCore.QTimer.singleShot(0, app.quit)

    def toggle_caption_visible(self) -> None:
        if self.caption_window.isVisible():
            self.caption_window.hide()
            self.visible_button.setText("显示字幕条")
        else:
            self.caption_window.show()
            self.caption_window.apply_topmost()
            self.visible_button.setText("隐藏字幕条")

    def apply_settings(self, settings: dict) -> None:
        font_size = int(settings.get("font_size", DEFAULT_SETTINGS["font_size"]))
        opacity = int(settings.get("opacity", DEFAULT_SETTINGS["opacity"]))
        topmost = bool(settings.get("topmost", DEFAULT_SETTINGS["topmost"]))

        self.font_slider.setValue(max(24, min(font_size, 64)))
        self.opacity_slider.setValue(max(30, min(opacity, 100)))
        self.topmost_checkbox.setChecked(topmost)
        self.refresh_audio_outputs(str(settings.get("audio_output_id", "") or ""))


def save_settings(caption_window: CaptionWindow, control_panel: ControlPanel) -> None:
    settings = {
        "caption_geometry": geometry_to_dict(caption_window),
        "control_geometry": geometry_to_dict(control_panel),
        "font_size": caption_window.font_size,
        "opacity": caption_window.opacity_percent,
        "topmost": caption_window.topmost_enabled,
        "audio_output_id": control_panel.selected_audio_output_id() or "",
    }
    tmp_path = SETTINGS_PATH.with_suffix(".tmp")
    tmp_path.write_text(
        json.dumps(settings, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(SETTINGS_PATH)


def move_caption_to_bottom_center(caption_window: CaptionWindow) -> None:
    screen = QtWidgets.QApplication.primaryScreen()
    if screen is None:
        return
    area = screen.availableGeometry()
    x = area.x() + (area.width() - caption_window.width()) // 2
    y = area.y() + area.height() - caption_window.height() - 120
    caption_window.move(x, y)


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("实时悬浮字幕器")
    settings = load_settings()

    caption_window = CaptionWindow()
    if not restore_geometry(caption_window, settings.get("caption_geometry")):
        move_caption_to_bottom_center(caption_window)
    caption_window.show()

    control_panel = ControlPanel(caption_window)
    control_panel.apply_settings(settings)
    restore_geometry(control_panel, settings.get("control_geometry"))
    control_panel.show()

    def cleanup() -> None:
        control_panel.stop_realtime_caption()
        control_panel.stop_audio_monitor()
        save_settings(caption_window, control_panel)

    app.aboutToQuit.connect(cleanup)

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
