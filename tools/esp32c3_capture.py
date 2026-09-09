#!/usr/bin/env python3
"""Reset an ESP32-C3 over its native USB-Serial-JTAG and log what it prints.

One process owns the port for the whole capture: it drives the same DTR/RTS
sequence as `espflash reset` (espflash 4.5.0, src/connection/reset.rs,
`reset_after_flash` for the USB-Serial-JTAG pid), then reads for SECONDS.

Two rules, both learned on the bench (evidence/esp32c3-bringup/README.md):
- never share the port with espflash while reading: `espflash reset` waits
  for bytes the reader has already consumed, hangs, and leaves the chip in
  download mode;
- flush the port BEFORE the reset, never after: the ROM, the bootloader and
  the firmware banner are all out within ~50 ms of the reset edge.

The board's USB link survives this reset (no re-enumeration), so the first
byte after the reset edge is the ROM's `rst:` line.

usage: tools/esp32c3_capture.py PORT SECONDS OUTFILE
Standard library only; no pyserial.
"""
import fcntl
import os
import select
import sys
import termios
import time

port, secs, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
fd = os.open(port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
attr = termios.tcgetattr(fd)
attr[0] = 0
attr[1] = 0
attr[3] = 0  # raw
attr[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
attr[4] = attr[5] = termios.B115200
termios.tcsetattr(fd, termios.TCSANOW, attr)


def line(flag, on):
    fcntl.ioctl(
        fd,
        termios.TIOCMBIS if on else termios.TIOCMBIC,
        flag.to_bytes(4, sys.byteorder),
    )


DTR, RTS = termios.TIOCM_DTR, termios.TIOCM_RTS
termios.tcflush(fd, termios.TCIFLUSH)  # stale bytes go BEFORE the reset
time.sleep(0.1)
line(DTR, False)
time.sleep(0.1)
line(RTS, True)
line(DTR, False)
line(RTS, True)
time.sleep(0.1)
line(RTS, False)
t_reset = time.monotonic()

n = 0
with open(out, "wb") as f:
    while time.monotonic() - t_reset < secs:
        r, _, _ = select.select([fd], [], [], 0.2)
        if r:
            try:
                b = os.read(fd, 4096)
            except BlockingIOError:
                continue
            f.write(b)
            n += len(b)
print(f"captured {n} bytes in {secs:.0f} s after reset")
