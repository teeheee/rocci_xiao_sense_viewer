from __future__ import annotations

import dataclasses
import struct
from typing import Iterable, List

MAGIC = b"\xDE\xAD\xBE\xAF"


@dataclasses.dataclass(frozen=True)
class ImageFrame:
    length: int
    data: bytes


@dataclasses.dataclass(frozen=True)
class TerminalFrame:
    message: str


@dataclasses.dataclass(frozen=True)
class BoxFrame:
    x: int
    y: int
    width: int
    height: int
    color: int


@dataclasses.dataclass(frozen=True)
class PointFrame:
    x: int
    y: int
    color: int


@dataclasses.dataclass(frozen=True)
class LineFrame:
    x0: int
    y0: int
    x1: int
    y1: int
    color: int


@dataclasses.dataclass(frozen=True)
class CommandPacket:
    command: str
    payload: str = ""


class SerialProtocolError(ValueError):
    pass


def parse_command(raw: bytes | str) -> CommandPacket:
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    if not raw:
        raise SerialProtocolError("empty command")
    text = raw.strip()
    if not text:
        raise SerialProtocolError("empty command")
    if b" " in text:
        command, payload = text.split(b" ", 1)
    else:
        command, payload = text, b""
    return CommandPacket(command=command.decode("utf-8"), payload=payload.decode("utf-8"))


def parse_stream(data: bytes | Iterable[int]) -> List[object]:
    frames, _ = parse_stream_incremental(data)
    return frames


def parse_stream_incremental(data: bytes | Iterable[int]) -> tuple[List[object], bytes]:
    payload = bytes(data)
    frames: List[object] = []
    offset = 0

    while offset < len(payload):
        if len(payload) - offset < 4:
            break
        if payload[offset:offset + 4] != MAGIC:
            next_magic = payload.find(MAGIC, offset + 1)
            if next_magic == -1:
                suffix_length = min(len(payload) - offset, len(MAGIC) - 1)
                while suffix_length and payload[-suffix_length:] != MAGIC[:suffix_length]:
                    suffix_length -= 1
                offset = len(payload) - suffix_length
                break
            offset = next_magic
            continue

        if len(payload) - offset < 5:
            break
        frame_type = payload[offset + 4]

        if frame_type == ord("I"):
            if len(payload) - offset - 5 < 4:
                break
            length = struct.unpack_from("<I", payload, offset + 5)[0]
            total = 5 + 4 + length
            if len(payload) - offset < total:
                break
            image_data = payload[offset + 9:offset + 9 + length]
            frames.append(ImageFrame(length=length, data=image_data))
            offset += total
            continue

        if frame_type == ord("T"):
            next_magic = payload.find(MAGIC, offset + 5)
            if next_magic == -1:
                suffix_length = min(len(payload) - offset - 5, len(MAGIC) - 1)
                while suffix_length and payload[-suffix_length:] != MAGIC[:suffix_length]:
                    suffix_length -= 1
                text_end = len(payload) - suffix_length
                text = payload[offset + 5:text_end].decode("utf-8", errors="replace")
                frames.append(TerminalFrame(message=text))
                offset = text_end
                break
            text = payload[offset + 5:next_magic].decode("utf-8", errors="replace")
            frames.append(TerminalFrame(message=text))
            offset = next_magic
            continue

        if frame_type == ord("B"):
            total = 5 + 10
            if len(payload) - offset < total:
                break
            x, y, width, height, color = struct.unpack_from("<hhhhH", payload, offset + 5)
            frames.append(BoxFrame(x=x, y=y, width=width, height=height, color=color))
            offset += total
            continue

        if frame_type == ord("P"):
            total = 5 + 6
            if len(payload) - offset < total:
                break
            x, y, color = struct.unpack_from("<hhH", payload, offset + 5)
            frames.append(PointFrame(x=x, y=y, color=color))
            offset += total
            continue

        if frame_type == ord("L"):
            total = 5 + 10
            if len(payload) - offset < total:
                break
            x0, y0, x1, y1, color = struct.unpack_from("<hhhhH", payload, offset + 5)
            frames.append(LineFrame(x0=x0, y0=y0, x1=x1, y1=y1, color=color))
            offset += total
            continue

        offset += 1

    return frames, payload[offset:]


def parse_serial_bytes(data: bytes) -> List[object]:
    return parse_stream(data)
