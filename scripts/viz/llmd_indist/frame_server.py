"""Decode video frames and round-trip renders in a process of its own (helper of extract_llmd_features.py).

The feature extraction keeps PyAV out of the process that holds the model: with torch / torchvision loaded, an
abandoned PyAV decoder hung one pilot and crashed another. This small server imports only av, numpy and PIL (through
h264_roundtrip.py, which also guarantees the pixels are the training dataloader's), always decodes a video to its end,
and hands the frames over a pipe.

Protocol: one JSON job per line on stdin, frames on stdout. Every job carries an "id" that comes back with each of
its records, so a client that stopped reading a job early can skip what is left of it.
  {"id": n, "video": path, "frames": "all" | [i, ...]}   the frames of a video, in order
  {"id": n, "image": path, "options": {...}}              h264_roundtrip.roundtrip() of an image file; the option
                                                          "crf": "start" means the crf that gives a lone frame the
                                                          quantisation of the training start frames (CRF_START)
Each record: int32 job id, index, height, width, channels (little endian), then height * width * channels uint8
pixels. A job ends with index -1 (the number of frames sent in the height field); an error is index -2 with the
length of the message in the height field, followed by the message.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from h264_roundtrip import CRF_START, iter_video_frames, roundtrip  # noqa: E402


def main() -> None:
    out = sys.stdout.buffer

    def emit(index: int, frame: np.ndarray) -> None:
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        out.write(struct.pack("<iiiii", job["id"], index, *frame.shape))
        out.write(frame.tobytes())

    for line in sys.stdin:
        if not line.strip():
            continue
        job = json.loads(line)
        try:
            n = 0
            if "video" in job:
                want = None if job["frames"] == "all" else set(job["frames"])
                for i, frame in iter_video_frames(job["video"]):  # always to the end of the file
                    if want is None or i in want:
                        emit(i, frame)
                        n += 1
            else:
                opt = dict(job.get("options") or {})
                if opt.get("crf") == "start":
                    opt["crf"] = CRF_START
                emit(0, roundtrip(np.asarray(Image.open(job["image"]).convert("RGB")), **opt))
                n = 1
            out.write(struct.pack("<iiiii", job["id"], -1, n, 0, 0))
        except Exception as exc:  # the client raises it
            msg = repr(exc).encode()
            out.write(struct.pack("<iiiii", job["id"], -2, len(msg), 0, 0) + msg)
        out.flush()


if __name__ == "__main__":
    main()
