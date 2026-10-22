from __future__ import annotations

import argparse
import json
import serial

from .parser import CommandPacket, parse_command, parse_stream


def send_command(port: str, command: str, baudrate: int = 921600) -> CommandPacket:
    with serial.Serial(port, baudrate, timeout=1) as ser:
        ser.write((command + "\n").encode("utf-8"))
        ser.flush()
        response = ser.readline().strip()
    if not response:
        return CommandPacket(command=command, payload="")
    return parse_command(response)


def dump_frames(port: str, baudrate: int = 921600) -> list:
    frames = []
    with serial.Serial(port, baudrate, timeout=0.2) as ser:
        while True:
            chunk = ser.read(4096)
            if not chunk:
                break
            frames.extend(parse_stream(chunk))
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Serial protocol parser for rocci_xiao_sense_camera")
    parser.add_argument("--port", default=None, help="Serial port, e.g. COM3 or /dev/ttyUSB0")
    parser.add_argument("--command", default=None, help="Send a command to the camera")
    parser.add_argument("--dump", action="store_true", help="Dump all serial frames from the device")
    args = parser.parse_args()

    if args.command:
        if not args.port:
            raise SystemExit("--port is required when using --command")
        packet = send_command(args.port, args.command)
        print(json.dumps(packet.__dict__, indent=2))
        return

    if args.dump:
        if not args.port:
            raise SystemExit("--port is required when using --dump")
        for frame in dump_frames(args.port):
            print(frame)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
