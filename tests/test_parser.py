import struct

import numpy as np

from rocci_xiao_sense_viewer.app import image_frame_to_rgb
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
    pixels = struct.pack("<HH", 0xF800, 0x07E0)
    frame = ImageFrame(length=len(pixels), data=pixels, width=2, height=1, pixformat=0)

    rgb = image_frame_to_rgb(frame)

    assert rgb.shape == (1, 2, 3)
    np.testing.assert_array_equal(rgb, [[[255, 0, 0], [0, 255, 0]]])


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
