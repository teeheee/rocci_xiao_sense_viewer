import struct

import numpy as np

from rocci_xiao_sense_viewer.app import image_frame_to_rgb, rgb888_to_rgb565
from rocci_xiao_sense_viewer.app import CapturedFrame, ViewerWindow
from rocci_xiao_sense_viewer.parser import (
    BoxFrame,
    CommandPacket,
    ImageFrame,
    TerminalFrame,
    parse_command,
    parse_stream,
    parse_stream_incremental,
)


MAGIC = b"\xDE\xAD\xBE\xAF"


def test_capture_frame_requests_one_frame_without_stopping_live_view():
    class FakeSerialThread:
        def __init__(self):
            self.commands = []

        def isRunning(self):
            return True

        def send_command(self, command):
            self.commands.append(command)
            return True

    class FakeTimer:
        def __init__(self):
            self.stopped = False

        def stop(self):
            self.stopped = True

    class FakeLabel:
        def setText(self, _text):
            pass

    serial_thread = FakeSerialThread()
    timer = FakeTimer()
    window = type(
        "CaptureWindow",
        (),
        {
            "serial_thread": serial_thread,
            "live_enabled": True,
            "frame_timer": timer,
            "refresh_images_on_next_frame": False,
            "status_label": FakeLabel(),
        },
    )()

    ViewerWindow.capture_frame(window)

    assert serial_thread.commands == [b"capture\n"]
    assert not timer.stopped
    assert window.live_enabled
    assert window.refresh_images_on_next_frame


def test_live_frames_save_only_one_image_after_capture_request():
    class FakeImageLabel:
        def set_image(self, _image):
            pass

    class CaptureWindow:
        refresh_images_on_next_frame = True
        image_label = FakeImageLabel()

        def __init__(self):
            self.saved_images = []
            self.refresh_count = 0

        def save_image(self, image):
            self.saved_images.append(image)

        def refresh_image_list(self):
            self.refresh_count += 1

    window = CaptureWindow()
    frame = ImageFrame(length=12, data=bytes(12), width=2, height=2, pixformat=5)
    message = CapturedFrame("image", frame, 0)

    ViewerWindow.handle_frame_message(window, message)
    ViewerWindow.handle_frame_message(window, message)

    assert len(window.saved_images) == 1
    assert window.refresh_count == 1


def test_parse_image_frame():
    payload = bytes([0x10, 0x20, 0x30, 0x40])
    packet = MAGIC + b"I" + struct.pack("<I", len(payload)) + payload
    frames = parse_stream(packet)
    assert len(frames) == 1
    assert isinstance(frames[0], ImageFrame)
    assert frames[0].length == len(payload)
    assert frames[0].data == payload


def test_parse_frame_with_dimensions_and_pixformat():
    image_data = bytes(range(12))
    header = struct.pack("<HHBI", 2, 2, 5, len(image_data))
    packet = MAGIC + b"F" + header + image_data

    frames = parse_stream(packet)

    assert len(frames) == 1
    assert frames[0] == ImageFrame(
        length=len(image_data),
        data=image_data,
        width=2,
        height=2,
        pixformat=5,
    )


def test_decode_rgb565_uses_frame_dimensions():
    pixels = bytes([0xF8, 0x00, 0x07, 0xE0])
    frame = ImageFrame(length=len(pixels), data=pixels, width=2, height=1, pixformat=0)

    rgb = image_frame_to_rgb(frame)

    assert rgb.shape == (1, 2, 3)
    np.testing.assert_array_equal(rgb, [[[255, 0, 0], [0, 255, 0]]])


def test_process_frame_conversion_emits_camera_rgb565_wire_order():
    image = np.zeros((96, 96, 3), dtype=np.uint8)
    image[0, 0] = [255, 0, 0]
    image[0, 1] = [0, 255, 0]

    payload = rgb888_to_rgb565(image)

    assert payload is not None
    assert len(payload) == 96 * 96 * 2
    assert payload[:4] == bytes([0xF8, 0x00, 0x07, 0xE0])


def test_process_frame_conversion_rejects_unsupported_size():
    image = np.zeros((48, 48, 3), dtype=np.uint8)

    assert rgb888_to_rgb565(image) is None


def test_decode_yuv422_uses_wide_color_arithmetic():
    pixels = bytes([16, 128, 235, 128])
    frame = ImageFrame(length=len(pixels), data=pixels, width=2, height=1, pixformat=1)

    rgb = image_frame_to_rgb(frame)

    assert rgb.shape == (1, 2, 3)
    np.testing.assert_array_equal(rgb, [[[0, 0, 0], [255, 255, 255]]])


def test_decode_yuv420_uses_planar_chroma():
    pixels = bytes([16, 235, 16, 235, 128, 128])
    frame = ImageFrame(length=len(pixels), data=pixels, width=2, height=2, pixformat=2)

    rgb = image_frame_to_rgb(frame)

    assert rgb.shape == (2, 2, 3)
    np.testing.assert_array_equal(
        rgb,
        [[[0, 0, 0], [255, 255, 255]], [[0, 0, 0], [255, 255, 255]]],
    )


def test_decode_grayscale_uses_camera_enum_value():
    pixels = bytes([0, 127, 255, 64])
    frame = ImageFrame(length=len(pixels), data=pixels, width=2, height=2, pixformat=3)

    rgb = image_frame_to_rgb(frame)

    np.testing.assert_array_equal(rgb[:, :, 0], [[0, 127], [255, 64]])
    np.testing.assert_array_equal(rgb[:, :, 0], rgb[:, :, 1])
    np.testing.assert_array_equal(rgb[:, :, 1], rgb[:, :, 2])


def test_parse_terminal_frame():
    payload = b"hello from camera"
    packet = MAGIC + b"T" + payload
    frames = parse_stream(packet)
    assert len(frames) == 1
    assert isinstance(frames[0], TerminalFrame)
    assert frames[0].message == payload.decode("utf-8")


def test_parse_box_frame():
    box = BoxFrame(x=10, y=20, width=30, height=40, color=0x1234)
    packet = MAGIC + b"B" + struct.pack("<hhhhH", 10, 20, 30, 40, 0x1234)
    frames = parse_stream(packet)
    assert len(frames) == 1
    assert isinstance(frames[0], BoxFrame)
    assert frames[0] == box


def test_incremental_parser_preserves_fragmented_image_after_terminal_frame():
    image_data = bytes(range(8))
    image_packet = MAGIC + b"I" + struct.pack("<I", len(image_data)) + image_data
    first_chunk = MAGIC + b"Tready" + image_packet[:12]

    frames, remainder = parse_stream_incremental(first_chunk)

    assert frames == [TerminalFrame(message="ready")]
    assert remainder == image_packet[:12]

    frames, remainder = parse_stream_incremental(remainder + image_packet[12:])

    assert frames == [ImageFrame(length=len(image_data), data=image_data)]
    assert remainder == b""


def test_incremental_parser_preserves_magic_split_after_terminal_frame():
    image_data = b"pixels"
    image_packet = MAGIC + b"I" + struct.pack("<I", len(image_data)) + image_data
    first_chunk = MAGIC + b"Tready" + MAGIC[:2]

    frames, remainder = parse_stream_incremental(first_chunk)

    assert frames == [TerminalFrame(message="ready")]
    assert remainder == MAGIC[:2]

    frames, remainder = parse_stream_incremental(remainder + image_packet[2:])

    assert frames == [ImageFrame(length=len(image_data), data=image_data)]
    assert remainder == b""


def test_parse_command_packet():
    command = parse_command(b"capture\n")
    assert command == CommandPacket(command="capture", payload="")

    command = parse_command(b"start\n")
    assert command == CommandPacket(command="start", payload="")

    command = parse_command(b"quiet\n")
    assert command == CommandPacket(command="quiet", payload="")

    command = parse_command(b"capture 123\n")
    assert command == CommandPacket(command="capture", payload="123")
