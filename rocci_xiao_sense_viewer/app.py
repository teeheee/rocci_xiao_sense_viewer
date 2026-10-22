from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import serial
import serial.tools.list_ports
from PySide6 import QtCore, QtGui, QtWidgets

from .parser import (
    BoxFrame,
    ImageFrame,
    LineFrame,
    PointFrame,
    TerminalFrame,
    parse_stream_incremental,
)


def image_frame_to_rgb(frame: ImageFrame) -> Optional[np.ndarray]:
    width = frame.width
    height = frame.height
    if width <= 0 or height <= 0:
        return None

    if frame.pixformat == 0:  # PIXFORMAT_RGB565
        expected = width * height * 2
        if frame.length < expected:
            return None
        pixels = np.frombuffer(frame.data[:expected], dtype=">u2").astype(np.uint32)
        red = ((pixels >> 11) & 0x1F) * 255 // 31
        green = ((pixels >> 5) & 0x3F) * 255 // 63
        blue = (pixels & 0x1F) * 255 // 31
        return np.stack((red, green, blue), axis=-1).astype(np.uint8).reshape(height, width, 3)

    if frame.pixformat == 1:  # PIXFORMAT_YUV422 (Y0, U, Y1, V)
        expected = width * height * 2
        if width % 2 or frame.length < expected:
            return None
        packed = np.frombuffer(frame.data[:expected], dtype=np.uint8).reshape(height, width // 2, 4).astype(np.int32)
        y0 = packed[..., 0] - 16
        u = packed[..., 1] - 128
        y1 = packed[..., 2] - 16
        v = packed[..., 3] - 128
        rgb = np.empty((height, width, 3), dtype=np.uint8)
        for column, luminance in ((0, y0), (1, y1)):
            red = (298 * luminance + 409 * v + 128) >> 8
            green = (298 * luminance - 100 * u - 208 * v + 128) >> 8
            blue = (298 * luminance + 516 * u + 128) >> 8
            rgb[:, column::2] = np.stack((red, green, blue), axis=-1).clip(0, 255).astype(np.uint8)
        return rgb

    if frame.pixformat == 2:  # PIXFORMAT_YUV420 (planar Y, U, V)
        if width % 2 or height % 2:
            return None
        luma_size = width * height
        chroma_size = luma_size // 4
        if frame.length < luma_size + chroma_size * 2:
            return None
        pixels = np.frombuffer(frame.data[:luma_size + chroma_size * 2], dtype=np.uint8)
        y = pixels[:luma_size].reshape(height, width).astype(np.int32) - 16
        u = pixels[luma_size:luma_size + chroma_size].reshape(height // 2, width // 2)
        v = pixels[luma_size + chroma_size:].reshape(height // 2, width // 2)
        u = np.repeat(np.repeat(u, 2, axis=0), 2, axis=1).astype(np.int32) - 128
        v = np.repeat(np.repeat(v, 2, axis=0), 2, axis=1).astype(np.int32) - 128
        red = (298 * y + 409 * v + 128) >> 8
        green = (298 * y - 100 * u - 208 * v + 128) >> 8
        blue = (298 * y + 516 * u + 128) >> 8
        return np.stack((red, green, blue), axis=-1).clip(0, 255).astype(np.uint8)

    if frame.pixformat == 3:  # PIXFORMAT_GRAYSCALE
        expected = width * height
        if frame.length < expected:
            return None
        gray = np.frombuffer(frame.data[:expected], dtype=np.uint8).reshape(height, width)
        return np.repeat(gray[:, :, None], 3, axis=2)

    if frame.pixformat == 4:  # PIXFORMAT_JPEG
        image = QtGui.QImage.fromData(frame.data)
        if image.isNull() or image.width() != width or image.height() != height:
            return None
        image = image.convertToFormat(QtGui.QImage.Format_RGB888)
        rows = np.frombuffer(image.bits(), dtype=np.uint8, count=image.sizeInBytes())
        return rows.reshape(height, image.bytesPerLine())[:, :width * 3].reshape(height, width, 3).copy()

    if frame.pixformat == 5:  # PIXFORMAT_RGB888
        expected = width * height * 3
        if frame.length < expected:
            return None
        return np.frombuffer(frame.data[:expected], dtype=np.uint8).reshape(height, width, 3).copy()

    return None


def rgb888_to_rgb565(image: np.ndarray) -> Optional[bytes]:
    pixels = np.asarray(image)
    if pixels.shape != (96, 96, 3) or pixels.dtype != np.uint8:
        return None
    red = (pixels[:, :, 0].astype(np.uint16) * 31 + 127) // 255
    green = (pixels[:, :, 1].astype(np.uint16) * 63 + 127) // 255
    blue = (pixels[:, :, 2].astype(np.uint16) * 31 + 127) // 255
    packed = (red << 11) | (green << 5) | blue
    return packed.astype(">u2").tobytes()


@dataclass
class CapturedFrame:
    kind: str
    payload: object
    timestamp: float


class SerialThread(QtCore.QThread):
    message_received = QtCore.Signal(object)
    terminal_received = QtCore.Signal(str)
    error = QtCore.Signal(str)
    status = QtCore.Signal(str)

    def __init__(self, port: str, baudrate: int = 921600):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.serial: Optional[serial.Serial] = None
        self._running = False
        self._buffer = bytearray()
        self._lock = threading.Lock()

    def run(self):
        self._running = True
        self.status.emit(f"Connecting to {self.port}...")
        try:
            self.serial = serial.Serial(self.port, self.baudrate, timeout=0.1)
        except serial.SerialException as exc:
            self.error.emit(f"Could not open {self.port}: {exc}")
            self._running = False
            return

        self.status.emit(f"Connected to {self.port}")
        while self._running:
            try:
                if self.serial is None:
                    break
                chunk = self.serial.read(4096)
                if not chunk:
                    continue
                self._buffer.extend(chunk)
                frames, remainder = parse_stream_incremental(self._buffer)
                self._buffer = bytearray(remainder)
                for frame in frames:
                    if isinstance(frame, ImageFrame):
                        self.message_received.emit(CapturedFrame("image", frame, time.time()))
                    elif isinstance(frame, TerminalFrame):
                        self.terminal_received.emit(frame.message)
                    elif isinstance(frame, (BoxFrame, PointFrame, LineFrame)):
                        self.message_received.emit(CapturedFrame("overlay", frame, time.time()))
                    else:
                        self.message_received.emit(CapturedFrame("unknown", frame, time.time()))
            except serial.SerialException as exc:
                self.error.emit(f"Serial error: {exc}")
                break

        self.status.emit("Disconnected")
        if self.serial is not None and self.serial.is_open:
            self.serial.close()
            self.serial = None

    def send_command(self, command: bytes | str) -> bool:
        if self.serial is None or not self.serial.is_open:
            return False
        data = command if isinstance(command, bytes) else command.encode("utf-8")
        with self._lock:
            self.serial.write(data)
            self.serial.flush()
        return True

    def send_process_frame(self, image_data: bytes) -> bool:
        if self.serial is None or not self.serial.is_open or len(image_data) != 96 * 96 * 2:
            return False
        with self._lock:
            self.serial.write(b"process\n")
            self.serial.write(image_data)
            self.serial.flush()
        return True

    def stop(self):
        self._running = False
        if self.serial is not None and self.serial.is_open:
            with self._lock:
                self.serial.close()


class ImageLabel(QtWidgets.QLabel):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(0, 0)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setContentsMargins(0, 0, 0, 0)
        self.setStyleSheet("QLabel { background-color: black; border: 1px solid #444; }")
        self._image = None
        self._overlay_items = []

    def set_image(self, image: Optional[np.ndarray]):
        self._image = image
        self.update()

    def set_overlay_items(self, items):
        self._overlay_items = items
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        qp = QtGui.QPainter(self)
        if self._image is not None:
            h, w, ch = self._image.shape
            qimage = QtGui.QImage(self._image.data, w, h, w * ch, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(qimage)
            target = self.contentsRect()
            scaled = pixmap.scaled(target.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            x = target.x() + (target.width() - scaled.width()) // 2
            y = target.y() + (target.height() - scaled.height()) // 2
            qp.drawPixmap(x, y, scaled)

        pen = QtGui.QPen(QtGui.QColor('#ff0000'))
        pen.setWidth(2)
        qp.setPen(pen)
        for item in self._overlay_items:
            if isinstance(item, BoxFrame):
                rect = QtCore.QRect(item.x, item.y, item.width, item.height)
                qp.drawRect(rect)
            elif isinstance(item, PointFrame):
                qp.drawPoint(item.x, item.y)
            elif isinstance(item, LineFrame):
                qp.drawLine(item.x0, item.y0, item.x1, item.y1)


class ViewerWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rocci Xiao Sense Viewer")
        self.resize(1200, 820)

        self.serial_thread: Optional[SerialThread] = None
        self.capture_dir = Path.cwd() / "captures"
        self.capture_dir.mkdir(exist_ok=True)
        self.live_enabled = False
        self.camera_halted = False
        self._resume_live_after_go = False
        self.loaded_frames: list[tuple[Path, np.ndarray]] = []
        self.extra_image_dirs: list[Path] = []
        self.refresh_images_on_next_frame = False
        self.loaded_image_index = 0
        self.last_frame_received_at = 0.0
        self.frame_timer = QtCore.QTimer(self)
        self.frame_timer.setInterval(1000)
        self.frame_timer.timeout.connect(self._maintain_live_view)

        widget = QtWidgets.QWidget()
        self.setCentralWidget(widget)
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        top = QtWidgets.QHBoxLayout()
        top.setSpacing(8)
        self.port_combo = QtWidgets.QComboBox()
        self.refresh_btn = QtWidgets.QPushButton("Refresh ports")
        self.connect_btn = QtWidgets.QPushButton("Connect")
        self.disconnect_btn = QtWidgets.QPushButton("Disconnect")
        self.capture_btn = QtWidgets.QPushButton("Capture frame")
        self.process_btn = QtWidgets.QPushButton("Process on chip")
        self.halt_btn = QtWidgets.QPushButton("Halt camera")
        self.go_btn = QtWidgets.QPushButton("Go")
        self.live_btn = QtWidgets.QPushButton("Live view")
        self.load_btn = QtWidgets.QPushButton("Load frames")
        self.refresh_images_btn = QtWidgets.QPushButton("Refresh images")
        self.status_label = QtWidgets.QLabel("Disconnected")

        connection_group = QtWidgets.QGroupBox("Connection")
        connection_layout = QtWidgets.QGridLayout(connection_group)
        connection_layout.setContentsMargins(10, 8, 10, 8)
        connection_layout.addWidget(self.port_combo, 0, 0, 1, 2)
        connection_layout.addWidget(self.refresh_btn, 0, 2)
        connection_layout.addWidget(self.connect_btn, 1, 0)
        connection_layout.addWidget(self.disconnect_btn, 1, 1)
        connection_layout.addWidget(self.status_label, 1, 2)
        top.addWidget(connection_group, 3)

        camera_group = QtWidgets.QGroupBox("Camera")
        camera_layout = QtWidgets.QHBoxLayout(camera_group)
        camera_layout.setContentsMargins(10, 8, 10, 8)
        camera_layout.addWidget(self.live_btn)
        camera_layout.addWidget(self.halt_btn)
        camera_layout.addWidget(self.go_btn)
        top.addWidget(camera_group, 2)
        layout.addLayout(top)

        mid = QtWidgets.QHBoxLayout()
        mid.setContentsMargins(0, 0, 0, 0)
        mid.setSpacing(8)
        self.image_label = ImageLabel()
        self.image_label.setObjectName("imageView")
        self.terminal_box = QtWidgets.QPlainTextEdit()
        self.terminal_box.setReadOnly(True)
        self.terminal_box.setPlaceholderText("Terminal output will appear here...")
        self.terminal_box.setMinimumWidth(230)
        self.auto_clear_terminal = QtWidgets.QCheckBox("Auto-clear each frame")
        self.clear_terminal_btn = QtWidgets.QPushButton("Clear")
        terminal_group = QtWidgets.QGroupBox("Terminal")
        terminal_layout = QtWidgets.QVBoxLayout(terminal_group)
        terminal_header = QtWidgets.QHBoxLayout()
        terminal_header.addWidget(self.auto_clear_terminal)
        terminal_header.addStretch(1)
        terminal_header.addWidget(self.clear_terminal_btn)
        terminal_layout.addLayout(terminal_header)
        terminal_layout.addWidget(self.terminal_box)
        mid.addWidget(self.image_label, 5)
        mid.addWidget(terminal_group, 2)
        layout.addLayout(mid, 6)

        frames_group = QtWidgets.QGroupBox("Frames")
        frames_layout = QtWidgets.QVBoxLayout(frames_group)
        frames_layout.setContentsMargins(10, 8, 10, 8)
        frames_actions = QtWidgets.QHBoxLayout()
        frames_actions.addWidget(self.capture_btn)
        frames_actions.addWidget(self.load_btn)
        frames_actions.addWidget(self.refresh_images_btn)
        frames_actions.addWidget(self.process_btn)
        frames_actions.addStretch(1)
        frames_layout.addLayout(frames_actions)
        self.file_list = QtWidgets.QListWidget()
        self.file_list.setMinimumHeight(96)
        self.file_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.file_list.setViewMode(QtWidgets.QListView.IconMode)
        self.file_list.setFlow(QtWidgets.QListView.LeftToRight)
        self.file_list.setWrapping(False)
        self.file_list.setResizeMode(QtWidgets.QListView.Adjust)
        self.file_list.setMovement(QtWidgets.QListView.Static)
        self.file_list.setIconSize(QtCore.QSize(64, 64))
        self.file_list.setGridSize(QtCore.QSize(112, 90))
        frames_layout.addWidget(self.file_list)
        layout.addWidget(frames_group, 2)

        self.refresh_btn.clicked.connect(self.refresh_ports)
        self.connect_btn.clicked.connect(self.connect_to_port)
        self.disconnect_btn.clicked.connect(self.disconnect_from_port)
        self.capture_btn.clicked.connect(self.capture_frame)
        self.process_btn.clicked.connect(self.process_loaded_frames)
        self.halt_btn.clicked.connect(self.halt_camera)
        self.go_btn.clicked.connect(self.go_camera)
        self.live_btn.clicked.connect(self.toggle_live_view)
        self.load_btn.clicked.connect(self.load_frames_from_folder)
        self.refresh_images_btn.clicked.connect(self.refresh_image_list)
        self.clear_terminal_btn.clicked.connect(self.terminal_box.clear)
        self.file_list.itemClicked.connect(self.show_selected_frame)

        self.refresh_ports()
        self.refresh_image_list()

    def refresh_ports(self):
        self.port_combo.clear()
        ports = [port.device for port in serial.tools.list_ports.comports()]
        self.port_combo.addItems(ports)
        if not ports:
            self.status_label.setText("No ports found")

    def connect_to_port(self):
        port = self.port_combo.currentText()
        if not port:
            self.status_label.setText("No port selected")
            return

        if self.serial_thread is not None and self.serial_thread.isRunning():
            self.status_label.setText("Already connected")
            return

        self.serial_thread = SerialThread(port, baudrate=921600)
        self.serial_thread.message_received.connect(self.handle_frame_message)
        self.serial_thread.terminal_received.connect(self.append_terminal_output)
        self.serial_thread.error.connect(self.on_error)
        self.serial_thread.status.connect(self.status_label.setText)
        self.serial_thread.start()
        self.last_frame_received_at = time.monotonic()
        self.frame_timer.start()

    def disconnect_from_port(self):
        self.frame_timer.stop()
        self.live_enabled = False
        if self.serial_thread is not None:
            self.serial_thread.stop()
            self.serial_thread.wait(1000)
            self.serial_thread = None
        self.status_label.setText("Disconnected")

    def append_terminal_output(self, text: str):
        if not text:
            return
        if self.auto_clear_terminal.isChecked() and text.startswith("Processing image of size"):
            self.terminal_box.clear()
        self.terminal_box.appendPlainText(text)

    def on_error(self, text: str):
        self.terminal_box.appendPlainText(f"[ERROR] {text}\n")
        self.status_label.setText("Error")

    def capture_frame(self):
        if self.serial_thread is None or not self.serial_thread.isRunning():
            self.on_error("Connect to the camera before capturing frames.")
            return
        self.serial_thread.send_command(b"capture\n")
        self.last_frame_received_at = time.monotonic()
        self.refresh_images_on_next_frame = True
        self.status_label.setText("Capture requested")

    def halt_camera(self):
        if self.serial_thread is None or not self.serial_thread.isRunning():
            self.on_error("Connect to the camera before halting it.")
            return
        if self.camera_halted:
            return
        self._resume_live_after_go = self.live_enabled
        self.live_enabled = False
        self.frame_timer.stop()
        if self.serial_thread.send_command(b"halt\n"):
            self.camera_halted = True
            self.status_label.setText("Camera halted")

    def go_camera(self):
        if self.serial_thread is None or not self.serial_thread.isRunning():
            self.on_error("Connect to the camera before resuming it.")
            return
        if not self.camera_halted:
            return
        if not self.serial_thread.send_command(b"go\n"):
            return
        self.camera_halted = False
        if self._resume_live_after_go:
            self.serial_thread.send_command(b"start\n")
            self.live_enabled = True
            self.last_frame_received_at = time.monotonic()
            self.frame_timer.start()
        self._resume_live_after_go = False
        self.status_label.setText("Camera resumed")

    def process_loaded_frames(self):
        if self.serial_thread is None or not self.serial_thread.isRunning():
            self.on_error("Connect to the camera before processing frames.")
            return
        if not self.loaded_frames:
            self.status_label.setText("Load a frame before processing")
            return
        index = self.file_list.currentRow()
        if index < 0:
            index = 0
        _, frame = self.loaded_frames[index]
        image_data = rgb888_to_rgb565(frame)
        if image_data is None:
            self.status_label.setText("Chip processing requires a 96x96 RGB image")
            return
        if self.serial_thread.send_process_frame(image_data):
            self.status_label.setText("Processing frame on chip")
        else:
            self.on_error("Could not send the complete frame to the camera.")

    def _maintain_live_view(self):
        if not self.live_enabled or self.serial_thread is None or not self.serial_thread.isRunning():
            return
        if time.monotonic() - self.last_frame_received_at >= 1.0:
            self.serial_thread.send_command(b"capture\n")
            self.last_frame_received_at = time.monotonic()
            self.status_label.setText("Live view active (1 FPS)")

    def toggle_live_view(self):
        if self.serial_thread is None or not self.serial_thread.isRunning():
            self.on_error("Connect to the camera before enabling live view.")
            return
        if self.camera_halted:
            self.status_label.setText("Resume the camera before changing live view")
            return
        self.live_enabled = not self.live_enabled
        cmd = b"start\n" if self.live_enabled else b"stop\n"
        self.serial_thread.send_command(cmd)
        if self.live_enabled:
            self.last_frame_received_at = time.monotonic()
            self.frame_timer.start()
        else:
            self.frame_timer.stop()
        self.status_label.setText("Live view enabled" if self.live_enabled else "Live view disabled")

    def load_frames_from_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select frame folder")
        if not folder:
            return
        selected_dir = Path(folder).resolve()
        if selected_dir not in self.extra_image_dirs:
            self.extra_image_dirs.append(selected_dir)
        self.refresh_image_list()

    def refresh_image_list(self):
        directories = [Path.cwd(), self.capture_dir, *self.extra_image_dirs]
        image_paths = set()
        for directory in directories:
            if directory.is_dir():
                image_paths.update(
                    path.resolve()
                    for path in directory.iterdir()
                    if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".npy"}
                )

        self.file_list.clear()
        self.loaded_frames = []
        for path in sorted(image_paths, key=lambda value: (value.name.lower(), str(value).lower())):
            image = self._read_image(path)
            if image is None:
                continue
            self.loaded_frames.append((path, image))
            qimage = QtGui.QImage(image.data, image.shape[1], image.shape[0], image.shape[1] * 3, QtGui.QImage.Format_RGB888).copy()
            pixmap = QtGui.QPixmap.fromImage(qimage).scaled(
                self.file_list.iconSize(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
            item = QtWidgets.QListWidgetItem(QtGui.QIcon(pixmap), path.name)
            item.setToolTip(str(path))
            item.setData(QtCore.Qt.UserRole, str(path))
            self.file_list.addItem(item)

        if not self.loaded_frames:
            self.file_list.setToolTip(f"No images found in {Path.cwd()}")
        else:
            self.file_list.setToolTip("")

    def _read_image(self, path: Path) -> Optional[np.ndarray]:
        try:
            if path.suffix.lower() == ".npy":
                image = np.load(path)
                if image.ndim == 2:
                    image = np.repeat(image[:, :, None], 3, axis=2)
                elif image.ndim == 3 and image.shape[2] >= 3:
                    image = image[:, :, :3]
                else:
                    return None
                if image.dtype != np.uint8:
                    image = np.clip(image, 0, 255).astype(np.uint8)
                return image

            qimage = QtGui.QImage(str(path))
            if qimage.isNull():
                return None
            qimage = qimage.convertToFormat(QtGui.QImage.Format_RGB888)
            rows = np.frombuffer(qimage.bits(), dtype=np.uint8, count=qimage.sizeInBytes())
            return rows.reshape(qimage.height(), qimage.bytesPerLine())[:, :qimage.width() * 3].reshape(
                qimage.height(), qimage.width(), 3
            ).copy()
        except (OSError, ValueError):
            return None

    def show_selected_frame(self, item):
        row = self.file_list.row(item)
        if 0 <= row < len(self.loaded_frames):
            _, frame = self.loaded_frames[row]
            self.loaded_image_index = row
            self.image_label.set_image(frame)
            self.image_label.set_overlay_items([])

    def handle_frame_message(self, payload: CapturedFrame):
        self.last_frame_received_at = time.monotonic()
        if payload.kind == "image":
            frame: ImageFrame = payload.payload
            img = image_frame_to_rgb(frame)
            if img is not None:
                self.image_label.set_image(img)
                self.save_image(img)
                if self.refresh_images_on_next_frame:
                    self.refresh_images_on_next_frame = False
                    self.refresh_image_list()
            else:
                self.status_label.setText(f"Unsupported or invalid pixel format {frame.pixformat}")
        elif payload.kind == "overlay":
            overlay = payload.payload
            items = []
            if isinstance(overlay, (BoxFrame, PointFrame, LineFrame)):
                items.append(overlay)
            self.image_label.set_overlay_items(items)

    def save_image(self, image: np.ndarray):
        self.capture_dir.mkdir(exist_ok=True)
        out = self.capture_dir / f"frame_{int(time.time() * 1000)}.npy"
        np.save(out, image)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Rocci Xiao Sense Viewer")
    parser.add_argument("--list-ports", action="store_true", help="List available COM ports and exit")
    parser.add_argument("--port", default=None, help="Serial port to open immediately")
    parser.add_argument("--headless", action="store_true", help="Start the application without showing the UI")
    args = parser.parse_args(argv)

    if args.list_ports:
        ports = [port.device for port in serial.tools.list_ports.comports()]
        print("\n".join(ports) if ports else "No ports found")
        return 0

    app = QtWidgets.QApplication([])
    window = ViewerWindow()
    if args.port:
        window.port_combo.setCurrentText(args.port)
    if not args.headless:
        window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
