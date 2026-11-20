"""Decode the train pack's H.264 videos exactly like the training dataloader, and push a raw render through the
training videos' H.264 encode + decode ("round trip"). Helper of the LLMD in-distribution experiment.

Why: the detector's statistics are fitted on frames decoded from the train pack's videos, the test images are raw
lossless renders. To see how much of a score comes from the codec alone, a render is encoded here with the settings
the training videos were written with and decoded again with the decoder the training dataloader used.

Runs in Psi0's venv, CPU only (needs av, numpy, PIL; the --check also torch / torchvision / lerobot / psi). Import:
  iter_video_frames(path)       (frame index, rgb uint8 HxWx3) of every frame, in order: the dataloader's pixels
  read_video_frame(path, i)     one frame, the same pixels
  roundtrip(rgb, **overrides)   the image after a one-frame encode with the training videos' settings + that decode
  encode(rgb, **overrides)      the mp4 bytes of that encode
  x264_settings(src)            the x264 settings string a stream carries (SEI user data)
  frame_qps(src), mean_qp(src)  picture type and mean macroblock QP per frame: how hard a stream was quantised
  ENCODER                       the settings, what the training streams say, and how each was established
  VARIANTS                      the three round trips worth comparing, as roundtrip() overrides (see below)

What a one-frame round trip can and cannot reproduce (measured by --check, see ENCODER["qp"] and ENCODER["detail"]):
  roundtrip(rgb)                    the writer's exact call (crf 23). A lone frame has no later frames that refer to
                                    it, so x264 quantises it harder than a training video's frames: QP 29.9 against
                                    25.6 for the training start frames, and it keeps about 20 % less fine detail than
                                    the training frame of the same pose. The training B-frames (74 % of the frames)
                                    have as high a nominal QP (29.5), but they are predicted from the I/P frames and
                                    keep their detail (95 - 99 % of frame 0's in the frames right after it): this
                                    variant over-states the codec for them too.
  roundtrip(rgb, crf=CRF_START)     the same, with the crf at which a re-render of a training start pose gets the
                                    quantisation, and the fine detail, of the training start frames (frame 0 of every
                                    video, an I-frame). The closest single-image stand-in for a training frame.
  roundtrip(rgb, crf=0)             lossless: only the colour conversion and the chroma subsampling.
  Mid-demo frames are P/B frames predicted from their neighbours; no single image reproduces those.
  Every variant also comes back about 1 level darker (R -1.0, G -1.5, B -1.0 on these renders; -0.5 where a pixel is
  exactly grey): that is the decoder's truncating YUV -> RGB conversion, which every training frame went through too.

CLI:
  --check                   run the validations and print a report. Writes to <out> (default .runs/llmd_indist/_dev/h264):
                            report.json (every number), writer_reference.mp4 (the image encoded by SIMPLE's own
                            writer class), roundtrip_other_python.png, anchor_roundtrip_*.png and
                            domain_patch_train_raw_crf23_crfstart.png (one table patch, 4x, of the training frame,
                            a raw re-render and its two round trips)
  --in a.png --out b.png    round-trip one image (--crf / --frames as overrides); prints one json line
"""

from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import re
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterator

import av
import numpy as np
from PIL import Image

PSI0 = Path(__file__).resolve().parents[3]
SIMPLE = PSI0.parent / "SIMPLE"
TRAIN_DIR = SIMPLE / "data/psi-data/simple/G1WholebodyOpenFaucetTeleop-v0"
EXP3 = PSI0 / ".runs/vlm_camera_sweep/faucet_ep0_indist"
DEV_OUT = PSI0 / ".runs/llmd_indist/_dev/h264"

# The crf at which a ONE-frame encode of a re-render of a training start pose is quantised like the training start
# frames: mean macroblock QP 25.6 (median of frame 0 of the 99 training videos: 25.57; the 7 anchor re-renders of
# experiment 3 at crf 18.7: 25.56). At the writer's crf 23 the same one-frame encodes sit at QP 29.9.
# A cross-check without that calibration: a re-render encoded at crf 23 as frame 0 of a video whose next 40 frames
# are the real ones of its training episode gets QP 27.5 (the one-frame equivalent of crf 20.6) and lands as close to
# the training frame as the crf 18.7 one-frame encode does. The real frame 0 of that episode has QP 26.3, in between.
CRF_START = 18.7

# The round trips worth comparing against the raw image, as roundtrip() overrides: for name, kw in VARIANTS.items().
VARIANTS = {
    "crf23": {},                      # the writer's exact parameters; quantises harder than the training frames were
    "crf_start": {"crf": CRF_START},  # quantised like the training frames: the closest single-image stand-in
    "colour": {"crf": 0},             # lossless: the colour conversion and chroma subsampling alone
}

ENCODER = {
    # What encode() passes to PyAV. Without overrides this is the writer's own call.
    "call": {
        "codec": "h264",        # PyAV resolves the name to FFmpeg's libx264 encoder
        # fps of the stream. It changes the pixels: with mbtree x264 scales the quantiser with the frame duration
        # (the same one-frame encode has QP 27.5 at 25 fps and 29.9 at 50 fps), so it must stay the writer's 50.
        "rate": 50,
        "pix_fmt": "yuv420p",   # what PyAV picks for libx264 when the writer sets none
        "thread_type": "SLICE", # PyAV's default; with libx264 it means sliced threads: one slice per thread
        "thread_count": 0,      # PyAV's default (auto). x264 caps sliced threads at rows of macroblocks / 4 = 5 for 360 px
        "options": {},          # none: x264 defaults = preset medium, crf 23, profile High
        "container": "mp4",
        "frames": 1,            # copies of the image that are encoded; the decoded copy `frame` (0) is returned
    },
    # What every one of the 99 training videos says about itself (SEI user data of the first frame, mp4 tag).
    "stream": {
        "x264": "x264 - core 165 - H.264/MPEG-4 AVC codec - Copyleft 2003-2025 - http://www.videolan.org/x264.html",
        "options": "cabac=1 ref=3 deblock=1:0:0 analyse=0x3:0x113 me=hex subme=7 psy=1 psy_rd=1.00:0.00 mixed_ref=1 "
                   "me_range=16 chroma_me=1 trellis=1 8x8dct=1 cqm=0 deadzone=21,11 fast_pskip=1 chroma_qp_offset=-2 "
                   "threads=5 lookahead_threads=5 sliced_threads=1 slices=5 nr=0 decimate=1 interlaced=0 "
                   "bluray_compat=0 constrained_intra=0 bframes=3 b_pyramid=2 b_adapt=1 b_bias=0 direct=1 weightb=1 "
                   "open_gop=0 weightp=2 keyint=250 keyint_min=25 scenecut=40 intra_refresh=0 rc_lookahead=40 rc=crf "
                   "mbtree=1 crf=23.0 qcomp=0.60 qpmin=0 qpmax=69 qpstep=4 ip_ratio=1.40 aq=1:1.00",
        "muxer": "Lavf62.3.100",  # libavformat of FFmpeg 8.0, the one bundled in the PyAV 16 / 17 wheels
        "profile": "High", "level": 31, "pix_fmt": "yuv420p", "fps": 50, "time_base": "1/12800",
        "color_range": "unspecified", "colorspace": "unspecified",  # no colour tags: decoders assume BT.601 limited
        "gop": "I B B B P ... ; closed GOP, a keyframe at frame 0 and at most 250 frames later (or at a scene cut)",
    },
    # Mean macroblock QP (read from the decoder's venc_params side data), measured by --check.
    "qp": {
        "train_frame0_I": 25.6,     # frame 0 of the 99 training videos (range 24.7 .. 27.5)
        "train_all_I": 24.5,        # 201 frames, 0.56 %
        "train_all_P": 26.1,        # 9,152 frames, 25.6 %
        "train_all_B": 29.5,        # 26,455 frames, 73.9 %
        "one_frame_crf23": 29.9,    # roundtrip(rgb) of the anchor re-renders
        "one_frame_crf_start": 25.6,  # roundtrip(rgb, crf=CRF_START) of the anchor re-renders
        "spliced_crf23": 27.5,      # an anchor re-render as frame 0, followed by the next 40 real training frames
    },
    # Fine detail (texture(): mean abs difference of neighbouring pixels) as a ratio, measured by --check. The
    # nominal QP alone misleads: B-frames have a high one, but copy their detail from the I/P frames around them.
    "detail": {
        "train_frames_1_to_8_vs_frame0": (0.95, 0.99),  # B B B P B B B P after the I-frame, mean over the 99 videos
        "one_frame_crf23_vs_train_frame": 0.81,     # round-tripped anchor re-renders / the training frame of that pose
        "one_frame_crf_start_vs_train_frame": 1.02,
        "spliced_crf23_vs_train_frame": 0.93,       # the anchor as frame 0 of 41 frames, the rest real training frames
        "raw_vs_train_frame": 1.31,                 # the raw re-renders: what the codec took away
    },
    "how": [
        "writer: SIMPLE/third_party/decoupled_wbc/data/video_writer.py VideoWriter = av.open(path, 'w'); "
        "add_stream('h264', rate=fps); stream.width/height; av.VideoFrame.from_ndarray(rgb, format='rgb24'); "
        "stream.encode; no options, no pix_fmt, no colour tags. Created by Gr00tDataExporter.create(fps=50) "
        "(vcodec default 'h264') in simple/cli/replay_decoupled_wbc.py, which re-renders the teleop episodes.",
        "scripts/postprocess_psi0_sonic.py copies that mp4 into the pack byte for byte (shutil.copyfile) when "
        "--skip 0 --downsample 1; otherwise it would re-encode with the ffmpeg CLI (frame threads, another Lavf tag). "
        "The streams carry sliced threads and Lavf62.3.100, so they are the PyAV writer's output, not a re-encode.",
        "RGB -> YUV 4:2:0 happens inside PyAV's stream.encode (libswscale, bilinear, BT.601 limited range); "
        "encode() goes through the same call, so the conversion is the writer's.",
        "x264 build: the PyAV wheels of both venvs (Psi0 16.0.1, SIMPLE 17.0.0 = the version SIMPLE's uv.lock pins) "
        "bundle x264 core 165 and write the same settings string as the training streams, byte for byte.",
    ],
}

_PICT = {1: "I", 2: "P", 3: "B"}


# ----------------------------------------------------------------------------------------------------- decode


def _open(src):
    """An input container of a path, mp4 bytes or a file object, opened the way torchvision's pyav reader does.

    The decoder keeps PyAV's default threading on purpose. Frame threads (thread_type AUTO) would decode 3x faster,
    but their workers keep running between calls, and once torchvision is imported (it turns PyAV's log forwarding
    on) a worker that logs needs the GIL while the main thread holds it closing the decoder: read_video_frame hung
    that way within seconds. Slice threads only run inside a decode call, which releases the GIL.
    """
    if isinstance(src, (bytes, bytearray)):
        src = io.BytesIO(bytes(src))
    elif isinstance(src, Path):
        src = str(src)
    return av.open(src, metadata_errors="ignore")


def _name(src) -> str:
    """How an error names a source: the path, never the megabytes of an in-memory mp4."""
    return str(src) if isinstance(src, (str, Path)) else f"<{type(src).__name__}>"


def _frame_index(frame, stream) -> int:
    """The frame's position by its timestamp: pts * fps. The dataloader looks frames up by timestamp, not by count."""
    return round(frame.pts * stream.time_base * stream.average_rate)


def iter_video_frames(path) -> Iterator[tuple[int, np.ndarray]]:
    """(frame index, rgb uint8 HxWx3) of every frame of a video, decoded in order.

    The pixels are those the training dataloader fed the model: lerobot's pyav backend reads a frame with
    torchvision.io.VideoReader, which is ``frame.to_rgb().to_ndarray()`` on PyAV's decoded frame; the same call is
    made here, on a decoder opened the same way.
    """
    with _open(path) as container:
        stream = container.streams.video[0]
        for count, frame in enumerate(container.decode(stream)):
            index = _frame_index(frame, stream)
            if index != count:
                # a gap or an offset would silently pair pixels with another frame's state
                raise RuntimeError(f"{_name(path)}: decoded frame {count} has timestamp index {index}")
            yield index, frame.to_rgb().to_ndarray()


def read_video_frame(path, index: int) -> np.ndarray:
    """Frame `index` of a video (rgb uint8 HxWx3), the same pixels as iter_video_frames gives for it."""
    if index < 0:
        raise IndexError(f"frame {index}")
    with _open(path) as container:
        stream = container.streams.video[0]
        # jump to the keyframe at or before the frame, then decode forward to it
        container.seek(int(index / (stream.average_rate * stream.time_base)), stream=stream, backward=True)
        for frame in container.decode(stream):
            at = _frame_index(frame, stream)
            if at == index:
                return frame.to_rgb().to_ndarray()
            if at > index:
                break
    raise IndexError(f"{_name(path)}: no frame {index}")


# ----------------------------------------------------------------------------------------------------- encode


def _as_rgb(rgb) -> np.ndarray:
    rgb = np.ascontiguousarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected an rgb uint8 HxWx3 image, got {rgb.dtype} {rgb.shape}")
    if rgb.shape[0] % 2 or rgb.shape[1] % 2:
        raise ValueError(f"yuv420p needs an even width and height, got {rgb.shape[1]}x{rgb.shape[0]}")
    return rgb


def encode(rgb: np.ndarray, **overrides) -> bytes:
    """The mp4 bytes of the image encoded as a video with the training videos' settings.

    Overrides: any key of ENCODER["call"]; ``crf`` and ``preset`` as shorthands for options.
    """
    rgb = _as_rgb(rgb)
    cfg = {**ENCODER["call"], **overrides}
    options = {str(k): str(v) for k, v in cfg.pop("options").items()}
    for name in ("crf", "preset"):
        if name in cfg:
            options[name] = str(cfg.pop(name))
    unknown = set(cfg) - set(ENCODER["call"])
    if unknown:
        raise TypeError(f"unknown encoder settings {sorted(unknown)}")
    if cfg["frames"] < 1:
        raise ValueError(f"frames must be at least 1, got {cfg['frames']}")
    return _encode_images([rgb] * cfg["frames"], cfg, options)


def _encode_images(images, cfg: dict, options: dict) -> bytes:
    """The mp4 bytes of a sequence of rgb images, one video frame each, encoded the way SIMPLE's VideoWriter does."""
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format=cfg["container"]) as container:
        stream = container.add_stream(cfg["codec"], rate=cfg["rate"], options=options)
        stream.width, stream.height = images[0].shape[1], images[0].shape[0]
        stream.pix_fmt = cfg["pix_fmt"]
        stream.codec_context.thread_type = cfg["thread_type"]
        stream.codec_context.thread_count = cfg["thread_count"]
        for image in images:
            # like the writer: an rgb24 frame, converted to the stream's pix_fmt inside encode()
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return buffer.getvalue()


def roundtrip(rgb: np.ndarray, *, frame: int = 0, **overrides) -> np.ndarray:
    """The image after an H.264 encode with the training videos' settings and the dataloader's decode.

    Raises if the stream does not carry the training videos' x264 settings (see _require_training_settings).
    With ``frames=n`` the image is repeated n times and the decoded copy `frame` is returned.
    """
    data = encode(rgb, **overrides)
    _require_training_settings(data, overrides)
    for index, out in iter_video_frames(data):
        if index == frame:
            return out
    raise IndexError(f"the encode has no frame {frame}")


def _require_training_settings(data: bytes, overrides: dict) -> None:
    """Raise unless the encoder was set up like the training videos' one.

    x264 takes the slice count from the machine's cores: with fewer than 5 it would silently write another stream.
    Without overrides the whole settings string must be the training one; with crf / frames / rate overrides the
    encoder and its slicing must; with any other override the caller asked for another encoder and nothing is checked.
    """
    if set(overrides) - {"crf", "frames", "rate"}:
        return
    got, want = _settings_options(x264_settings(data)), _settings_options(_training_settings())
    keys = ("x264", "threads", "lookahead_threads", "sliced_threads", "slices") if overrides else tuple(want)
    wrong = {k: (got.get(k), want[k]) for k in keys if got.get(k) != want[k]}
    if wrong:
        raise RuntimeError(f"the encoder did not reproduce the training videos' x264 settings, (got, expected): {wrong}"
                           " (on a machine with fewer than 5 cores pass thread_count=5)")


# ------------------------------------------------------------------------------------- what a stream says


def _training_settings() -> str:
    return f'{ENCODER["stream"]["x264"]} - options: {ENCODER["stream"]["options"]}'


def x264_settings(src) -> str:
    """The settings string x264 embeds in the first frame of a stream ('' if there is none)."""
    data = bytes(src) if isinstance(src, (bytes, bytearray)) else Path(src).read_bytes()
    found = re.search(rb"x264 - core[^\x00]*", data)
    return found.group(0).decode("ascii", "replace") if found else ""


def _settings_options(settings: str) -> dict[str, str]:
    """{option: value} of a settings string; the encoder name and version under 'x264'."""
    head, _, opts = settings.partition(" - options: ")
    out = {"x264": head}
    for item in opts.split():
        key, _, value = item.partition("=")
        out[key] = value
    return out


def frame_qps(src, limit: int = 0) -> Iterator[tuple[int, str, np.ndarray]]:
    """(frame index, picture type I/P/B, QP of every macroblock) for the frames of a stream (the first `limit`).

    FFmpeg's H.264 decoder exports the encoder's quantisers as AVVideoEncParams side data when asked to.
    """
    with _open(src) as container:
        stream = container.streams.video[0]
        stream.codec_context.options = {"export_side_data": "venc_params"}
        for count, frame in enumerate(container.decode(stream)):
            side = [bytes(s) for s in frame.side_data if "VIDEO_ENC_PARAMS" in str(s.type)]
            if not side:
                raise RuntimeError("the decoder exported no encoder parameters")
            # struct AVVideoEncParams {uint nb_blocks; size_t blocks_offset, block_size; int type, qp; ...}
            # followed by nb_blocks x struct AVVideoBlockParams {int src_x, src_y, w, h, delta_qp}
            blocks, = struct.unpack_from("<I", side[0], 0)
            offset, size = struct.unpack_from("<QQ", side[0], 8)
            kind, base = struct.unpack_from("<ii", side[0], 24)
            if kind != 1 or size != 20:
                raise RuntimeError(f"unexpected encoder parameter layout (type {kind}, block size {size})")
            delta = np.frombuffer(side[0], dtype="<i4", count=blocks * 5, offset=offset).reshape(blocks, 5)[:, 4]
            yield count, _PICT.get(int(frame.pict_type), "?"), base + delta
            if limit and count + 1 >= limit:
                break


def mean_qp(src, frame: int = 0) -> float:
    """Mean macroblock QP of one frame of a stream: higher = quantised harder."""
    for index, _, qp in frame_qps(src, limit=frame + 1):
        if index == frame:
            return float(qp.mean())
    raise IndexError(f"no frame {frame}")


# ----------------------------------------------------------------------------------------------- the checks


def diff(a: np.ndarray, b: np.ndarray) -> dict:
    d = a.astype(np.int16) - b.astype(np.int16)
    return {"max": int(np.abs(d).max()), "mean_abs": float(np.abs(d).mean()), "mean_signed": float(d.mean())}


def mad(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).mean())


def box(x: np.ndarray, n: int = 16) -> np.ndarray:
    """The image averaged over n x n pixels around every pixel (edges repeated): its low frequencies."""
    lo = n // 2
    padded = np.pad(x.astype(np.float64), ((lo, n - 1 - lo), (lo, n - 1 - lo), (0, 0)), mode="edge")
    c = np.cumsum(np.cumsum(np.pad(padded, ((1, 0), (1, 0), (0, 0))), axis=0), axis=1)
    return (c[n:, n:] - c[:-n, n:] - c[n:, :-n] + c[:-n, :-n]) / (n * n)


def texture(x: np.ndarray) -> float:
    """Mean absolute difference of neighbouring pixels: what a codec smooths away."""
    g = x.astype(np.float64).mean(axis=2)
    return float(np.abs(np.diff(g, axis=1)).mean() + np.abs(np.diff(g, axis=0)).mean())


def load_png(path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def train_video(train_dir: Path, episode: int) -> Path:
    info = json.loads((train_dir / "meta/info.json").read_text())
    return train_dir / info["video_path"].format(episode_chunk=episode // info["chunks_size"], episode_index=episode,
                                                 video_key="observation.images.egocentric")


def ffmpeg_cli_frame(path: Path, frame: int, shape) -> np.ndarray:
    """Experiment 3's decode (extract_grid_latents.video_frame): the ffmpeg CLI, -pix_fmt rgb24."""
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-vf", f"select=eq(n\\,{frame})",
                          "-vsync", "0", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(shape)


def check_streams(train_dir: Path) -> dict:
    """What the 99 training videos say: settings string, muxer tag, stream fields, frame counts, timestamps."""
    episodes = [json.loads(line) for line in open(train_dir / "meta/episodes.jsonl")]
    settings, muxers, fields, wrong_length, irregular, keyframes = {}, {}, {}, [], [], []
    for ep in episodes:
        path = train_video(train_dir, ep["episode_index"])
        s = x264_settings(path)
        settings[s] = settings.get(s, 0) + 1
        with _open(path) as container:
            stream = container.streams.video[0]
            cc = stream.codec_context
            muxers[container.metadata.get("encoder", "")] = muxers.get(container.metadata.get("encoder", ""), 0) + 1
            # FFmpeg's enums: AVCOL_RANGE_UNSPECIFIED = 0, AVCOL_SPC_UNSPECIFIED = 2
            key = json.dumps({"codec": cc.codec.name, "profile": cc.profile, "pix_fmt": cc.pix_fmt,
                              "size": [cc.width, cc.height], "fps": str(stream.average_rate),
                              "time_base": str(stream.time_base),
                              "color_range": {0: "unspecified"}.get(int(cc.color_range), int(cc.color_range)),
                              "colorspace": {2: "unspecified"}.get(int(cc.colorspace), int(cc.colorspace)),
                              "has_b_frames": cc.has_b_frames})
            fields[key] = fields.get(key, 0) + 1
            # demux only: packet timestamps give the frame count, the spacing and the keyframes
            packets = [(p.pts, p.is_keyframe) for p in container.demux(stream) if p.pts is not None]
            step = 1 / (stream.average_rate * stream.time_base)
            pts = sorted(p for p, _ in packets)
            if pts != [i * step for i in range(len(pts))]:
                irregular.append(ep["episode_index"])
            if len(pts) != ep["length"]:
                wrong_length.append(ep["episode_index"])
            keyframes.append(sorted(int(p / step) for p, k in packets if k))
    second = [k[1] for k in keyframes if len(k) > 1]
    return {
        "videos": len(episodes), "frames": sum(ep["length"] for ep in episodes),
        "settings_strings": [{"n": n, "settings": s} for s, n in settings.items()],
        "all_equal_ENCODER_stream": list(settings) == [_training_settings()],
        "muxer_tags": muxers, "stream_fields": [{"n": n, **json.loads(k)} for k, n in fields.items()],
        "videos_with_wrong_frame_count": wrong_length, "videos_with_irregular_timestamps": irregular,
        "keyframes": {"first_is_frame0": all(k[0] == 0 for k in keyframes),
                      "per_video": sorted({len(k) for k in keyframes}),
                      "second_keyframe_min_max": [min(second), max(second)] if second else None},
    }


def check_ffprobe(train_dir: Path, episodes) -> list:
    if not shutil.which("ffprobe"):
        return [{"skipped": "no ffprobe on PATH"}]
    keys = "codec_name,profile,level,pix_fmt,color_range,color_space,color_transfer,color_primaries,has_b_frames," \
           "refs,r_frame_rate,time_base,nb_frames,bit_rate,field_order,chroma_location"
    out = []
    for ep in episodes:
        raw = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                              f"stream={keys}:format_tags=encoder", "-of", "json", str(train_video(train_dir, ep))],
                             capture_output=True, check=True, text=True).stdout
        probe = json.loads(raw)
        out.append({"episode": ep, **probe["streams"][0], "muxer": probe["format"]["tags"].get("encoder")})
    return out


def writer_reference(simple_dir: Path, image: np.ndarray, out: Path) -> dict:
    """Encode the image with SIMPLE's own VideoWriter class and compare with encode(): same file?"""
    source = simple_dir / "third_party/decoupled_wbc/data/video_writer.py"
    if not source.exists():
        return {"skipped": f"{source} not found"}
    # run the source as it is, without importing it: an import would leave a .pyc in the SIMPLE repo
    module: dict = {"__name__": "simple_video_writer"}
    exec(compile(source.read_text(), str(source), "exec"), module)
    path = out / "writer_reference.mp4"
    writer = module["VideoWriter"](str(path), image.shape[1], image.shape[0], ENCODER["call"]["rate"])
    writer.add_frame(image)
    # its stop() only waits for an empty queue, not for the frame being encoded, and flushing an encoder that is
    # still encoding hangs: wait for the worker to clear the flag it clears once the first frame is muxed
    deadline = time.monotonic() + 30
    while writer._first_frame:
        if time.monotonic() > deadline:
            return {"skipped": "SIMPLE's VideoWriter did not encode the frame within 30 s"}
        time.sleep(0.01)
    writer.stop()
    ours = encode(image)
    theirs = path.read_bytes()
    return {"writer": str(source), "file": str(path), "bytes_writer": len(theirs), "bytes_encode": len(ours),
            "file_identical": ours == theirs, "settings_identical": x264_settings(theirs) == x264_settings(ours),
            "decoded_pixels": diff(next(iter_video_frames(theirs))[1], next(iter_video_frames(ours))[1])}


def other_python(python: Path, image_path: Path, out: Path) -> dict:
    """Run this file's round trip in another venv (SIMPLE's: the PyAV version the writer ran with) and compare."""
    if not python.exists():
        return {"skipped": f"{python} not found"}
    target = out / "roundtrip_other_python.png"
    # -B: write no bytecode into that venv or repo
    done = subprocess.run([str(python), "-B", str(Path(__file__).resolve()), "--in", str(image_path), "--out",
                           str(target)], capture_output=True, text=True)
    if done.returncode:
        return {"skipped": f"{python} failed: {done.stderr.strip()[-300:]}"}
    theirs = json.loads(done.stdout.strip().splitlines()[-1])
    image = load_png(image_path)
    return {"python": str(python), "pyav_there": theirs["pyav"], "pyav_here": av.__version__,
            "mp4_identical": theirs["mp4_sha1"] == hashlib.sha1(encode(image)).hexdigest(),
            "decoded_pixels": diff(load_png(target), roundtrip(image))}


def check_settings(train_dir: Path, image: np.ndarray) -> dict:
    """The settings string of a round-tripped file against a training video's, option by option."""
    ours = _settings_options(x264_settings(encode(image)))
    theirs = _settings_options(x264_settings(train_video(train_dir, 0)))
    keys = list(dict.fromkeys([*theirs, *ours]))
    repeat = [hashlib.sha1(encode(image)).hexdigest() for _ in range(3)]
    return {"identical": ours == theirs, "options_compared": len(keys),
            "match": [k for k in keys if ours.get(k) == theirs.get(k)],
            "differ": {k: {"train": theirs.get(k), "roundtrip": ours.get(k)} for k in keys if ours.get(k) != theirs.get(k)},
            "encode_deterministic": len(set(repeat)) == 1}


def check_decode(train_dir: Path, episodes, frames) -> dict:
    """iter_video_frames against the dataloader's own decode functions and the ffmpeg CLI, on the same frames."""
    import pandas as pd
    import torch
    from psi.utils import pt_to_pil
    try:
        from lerobot.datasets import video_utils
    except ModuleNotFoundError:
        from lerobot.common.datasets import video_utils

    def loader_pixels(decoded: torch.Tensor) -> np.ndarray:
        # lerobot returns float32 CHW in [0, 1]; the repack transform turns it into the PIL image the model gets
        return np.asarray(pt_to_pil(decoded, normalized=False))

    rows, skipped = [], {}
    for ep in episodes:
        path = str(train_video(train_dir, ep))
        info = json.loads((train_dir / "meta/info.json").read_text())
        table = pd.read_parquet(train_dir / info["data_path"].format(episode_chunk=ep // info["chunks_size"],
                                                                     episode_index=ep), columns=["timestamp"])
        wanted = [f for f in frames if f < len(table)] + [len(table) - 1]
        ours = {i: img for i, img in iter_video_frames(path) if i in wanted}
        for f in wanted:
            ts = float(table["timestamp"].iloc[f])  # the timestamp the dataset queries the video with
            row = {"episode": ep, "frame": f, "timestamp": ts,
                   "vs_lerobot_pyav": diff(ours[f], loader_pixels(
                       video_utils.decode_video_frames_torchvision(path, [ts], 1e-4, "pyav")[0])),
                   "vs_read_video_frame": diff(ours[f], read_video_frame(path, f))}
            try:
                row["vs_lerobot_torchcodec"] = diff(ours[f], loader_pixels(
                    video_utils.decode_video_frames_torchcodec(path, [ts], 1e-4)[0]))
            except Exception as error:  # torchcodec needs FFmpeg shared libraries it may not find
                skipped["torchcodec"] = f"{type(error).__name__}: {str(error)[:200]}"
            if shutil.which("ffmpeg"):
                row["vs_ffmpeg_cli"] = diff(ours[f], ffmpeg_cli_frame(path, f, ours[f].shape))
            else:
                skipped["ffmpeg_cli"] = "no ffmpeg on PATH"
            rows.append(row)

    def worst(key):
        got = [r[key] for r in rows if key in r]
        return {"frames": len(got), "max": max(g["max"] for g in got),
                "mean_abs": float(np.mean([g["mean_abs"] for g in got]))} if got else None

    return {"frames_compared": len(rows), "episodes": list(episodes),
            **{k: worst(k) for k in ("vs_lerobot_pyav", "vs_lerobot_torchcodec", "vs_ffmpeg_cli",
                                     "vs_read_video_frame")},
            "skipped": skipped, "rows": rows}


def check_qp(train_dir: Path) -> dict:
    """How hard the training frames are quantised: frame 0 of every video, and every frame by picture type."""
    episodes = [json.loads(line)["episode_index"] for line in open(train_dir / "meta/episodes.jsonl")]
    start, by_type = [], {"I": [], "P": [], "B": []}
    for ep in episodes:
        for index, kind, qp in frame_qps(train_video(train_dir, ep)):
            by_type[kind].append(float(qp.mean()))
            if index == 0:
                start.append(float(qp.mean()))
    total = sum(len(v) for v in by_type.values())
    stats = lambda v: {"mean": float(np.mean(v)), "median": float(np.median(v)), "min": float(np.min(v)),
                       "max": float(np.max(v))}
    return {"frame0": {"n": len(start), **stats(start)},
            "all_frames": {k: {"n": len(v), "share": len(v) / total, **stats(v)} for k, v in by_type.items()},
            "all_frames_mean": float(np.mean(sum(by_type.values(), [])))}


def check_roundtrip(paths) -> dict:
    """Raw against round-tripped for a few of experiment 3's renders, at the writer's crf and at CRF_START."""
    rows = []
    for path in paths:
        raw = load_png(path)
        row = {"file": path.name}
        for name, kw in (("crf23", {}), ("crf_start", {"crf": CRF_START})):
            started = time.perf_counter()
            data = encode(raw, **kw)
            rt = next(iter_video_frames(data))[1]
            ms = (time.perf_counter() - started) * 1e3
            row[name] = {**diff(rt, raw), "qp": mean_qp(data), "bytes": len(data), "ms": ms,
                         # a second pass through the codec: how far is the round trip from a fixed point
                         "again_mean_abs": mad(roundtrip(rt, **kw), rt)}
        rows.append(row)
    mean = lambda name, key: float(np.mean([r[name][key] for r in rows]))
    return {"renders": len(rows),
            **{name: {key: mean(name, key) for key in ("mean_abs", "mean_signed", "max", "qp", "bytes", "ms",
                                                       "again_mean_abs")} for name in ("crf23", "crf_start")},
            "rows": rows}


def check_colour(image: np.ndarray) -> dict:
    """The colour conversions alone (no codec): RGB -> YUV 4:2:0 -> RGB with the writer's and the decoder's calls."""
    yuv = av.VideoFrame.from_ndarray(image, format="rgb24").reformat(format="yuv420p")
    back = yuv.to_rgb().to_ndarray()
    y, u, v = (np.frombuffer(p, np.uint8).reshape(p.height, p.line_size)[:, :p.width].astype(np.float64)
               for p in yuv.planes)
    rgb = image.astype(np.float64)
    ideal_y = 16 + (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]) * 219 / 255
    up = lambda c: np.repeat(np.repeat(c, 2, axis=0), 2, axis=1)
    luma, cb, cr = (y - 16) * 255 / 219, (up(u) - 128) * 255 / 224, (up(v) - 128) * 255 / 224
    ideal_rgb = np.clip(np.stack([luma + 1.402 * cr, luma - 0.344136 * cb - 0.714136 * cr, luma + 1.772 * cb], -1),
                        0, 255)
    return {"rgb_yuv_rgb": {**diff(back, image), "mean_signed_rgb": (back - rgb).mean((0, 1)).tolist()},
            "lossless_crf0": diff(roundtrip(image, crf=0), image),
            # which side shifts the brightness: the encoder's luma, or the decoder's YUV -> RGB
            "encode_side_luma_minus_exact": float((y - ideal_y).mean()),
            "decode_side_rgb_minus_exact": (back - ideal_rgb).mean((0, 1)).tolist()}


def model_transform(run_dir: Path):
    """The checkpoint's image transform up to the VLM processor (resize + centre crop), as serving builds it."""
    from torchvision.transforms import v2
    from psi.config.augmentation import CenterCrop, ResizeImage
    cfg = json.loads((run_dir / "run_config.json").read_text())["data"]["transform"]["model"]
    resize, crop = ResizeImage(size=tuple(cfg["resize"]["size"]))(), CenterCrop(size=tuple(cfg["center_crop"]["size"]))()
    transform = v2.Compose([resize, crop])
    apply = lambda img: np.asarray(transform(Image.fromarray(img)))
    probe = (np.random.default_rng(0).random((360, 640, 3)) * 255).astype(np.uint8)
    facts = {"resize": cfg["resize"]["size"], "center_crop": cfg["center_crop"]["size"],
             "interpolation": str(resize.interpolation), "img_aug_at_training": cfg["img_aug"],
             "color_jitter_at_training": cfg["color_jitter"],
             # nearest on a PIL image, 2x down: one of four pixels is kept, nothing is averaged
             "equals_every_second_pixel_from_1": bool(np.array_equal(apply(probe), probe[1::2, 1::2]))}
    return apply, facts


def check_domain(renders: Path, run_dir: Path, train_dir: Path, out: Path) -> dict:
    """The training frame against raw and round-tripped re-renders of the same pose (experiment 3's anchors)."""
    rows = [json.loads(line) for line in open(renders / "index.jsonl")]
    rec = load_png(renders / next(r["file"] for r in rows if r["kind"] == "recorded"))
    anchors = [(r["file"], load_png(renders / r["file"])) for r in rows if r["kind"] == "anchor"]
    apply, transform_facts = model_transform(run_dir)
    variants = {"crf23": {}, "crf_start": {"crf": CRF_START}}
    trips = {name: [roundtrip(img, **kw) for _, img in anchors] for name, kw in variants.items()}

    def abc(raw, rt, view) -> dict:
        return {"a_raw_vs_train": mad(view(raw), view(rec)), "b_roundtrip_vs_train": mad(view(rt), view(rec)),
                "c_raw_vs_roundtrip": mad(view(raw), view(rt))}

    views = {"full_640x360": lambda x: x, "model_180x320": apply,
             # the same split into fine detail (what the codec changes) and smooth shading (lighting, brightness)
             "detail_minus_16px_mean": lambda x: x - box(x), "smooth_16px_mean": box}
    result = {"anchors": [name for name, _ in anchors], "transform": transform_facts, "per_anchor": []}
    for name in variants:
        result[name] = {}
        for view_name, view in views.items():
            each = [abc(raw, rt, view) for (_, raw), rt in zip(anchors, trips[name])]
            result[name][view_name] = {k: float(np.mean([e[k] for e in each])) for k in each[0]}
            # the first four anchors were rendered back to back; the later ones after the grid, with a drifted scene
            result[name][view_name + "_first4"] = {k: float(np.mean([e[k] for e in each[:4]])) for k in each[0]}
        result[name]["mean_signed_roundtrip_minus_raw_rgb"] = np.mean(
            [(rt.astype(np.float64) - raw).mean((0, 1)) for (_, raw), rt in zip(anchors, trips[name])], axis=0).tolist()
        result[name]["texture"] = float(np.mean([texture(rt) for rt in trips[name]]))
        result[name]["qp"] = float(np.mean([mean_qp(encode(img, **variants[name])) for _, img in anchors[:4]]))
        # noise floor: two round-tripped re-renders of the same pose against each other
        first = trips[name][:4]
        pairs = [(i, j) for i in range(len(first)) for j in range(i + 1, len(first))]
        result[name]["repeat_roundtrip_vs_roundtrip"] = {
            "full": float(np.mean([mad(first[i], first[j]) for i, j in pairs])),
            "model_180x320": float(np.mean([mad(apply(first[i]), apply(first[j])) for i, j in pairs])),
            "detail": float(np.mean([mad(first[i] - box(first[i]), first[j] - box(first[j])) for i, j in pairs]))}
    raws = [img for _, img in anchors[:4]]
    pairs = [(i, j) for i in range(len(raws)) for j in range(i + 1, len(raws))]
    result["repeat_raw_vs_raw"] = {
        "full": float(np.mean([mad(raws[i], raws[j]) for i, j in pairs])),
        "model_180x320": float(np.mean([mad(apply(raws[i]), apply(raws[j])) for i, j in pairs])),
        "detail": float(np.mean([mad(raws[i] - box(raws[i]), raws[j] - box(raws[j])) for i, j in pairs]))}
    result["mean_signed_train_minus_raw_rgb"] = np.mean(
        [(rec.astype(np.float64) - raw).mean((0, 1)) for _, raw in anchors[:4]], axis=0).tolist()
    result["texture_train_frame"] = texture(rec)
    result["texture_raw"] = float(np.mean([texture(img) for _, img in anchors]))
    result["train_frame_qp"] = mean_qp(train_video(train_dir, 0))
    for (name, raw), rt23, rts in zip(anchors, trips["crf23"], trips["crf_start"]):
        result["per_anchor"].append({"file": name, "a": mad(raw, rec), "b_crf23": mad(rt23, rec),
                                     "b_crf_start": mad(rts, rec), "c_crf23": mad(raw, rt23),
                                     "c_crf_start": mad(raw, rts)})
    # for the eye: the same table patch of the training frame, the raw re-render and its two round trips, 4x
    patch = lambda x: np.kron(x[100:180, 260:380], np.ones((4, 4, 1), np.uint8))
    strip = np.concatenate([patch(rec), patch(anchors[0][1]), patch(trips["crf23"][0]), patch(trips["crf_start"][0])], 1)
    Image.fromarray(strip).save(out / "domain_patch_train_raw_crf23_crfstart.png")
    Image.fromarray(trips["crf23"][0]).save(out / "anchor_roundtrip_crf23.png")
    Image.fromarray(trips["crf_start"][0]).save(out / "anchor_roundtrip_crf_start.png")
    return result


def check_start_frame(renders: Path, train_dir: Path) -> dict:
    """Is a one-frame encode a fair stand-in for a training frame? No raw training frame exists, so two indirect
    measurements: how much fine detail the frames right after frame 0 keep (B/P frames of a scene that barely moves,
    with a higher nominal QP), and what the codec does to a re-render that is followed by the real next frames of
    its training video (the "splice": frame 0 then gets what the lookahead gives a real start frame)."""
    mean = lambda values: float(np.mean(list(values)))
    episodes = [json.loads(line)["episode_index"] for line in open(train_dir / "meta/episodes.jsonl")]
    last = 8
    kinds, qps, ratios = [set() for _ in range(last + 1)], [[] for _ in range(last + 1)], [[] for _ in range(last + 1)]
    for ep in episodes:
        path = train_video(train_dir, ep)
        for index, kind, qp in frame_qps(path, limit=last + 1):
            kinds[index].add(kind)
            qps[index].append(float(qp.mean()))
        energy = [texture(img) for _, img in itertools.islice(iter_video_frames(path), last + 1)]
        for index in range(last + 1):
            ratios[index].append(energy[index] / energy[0])
    result = {"after_frame0": [{"frame": i, "types": "".join(sorted(kinds[i])), "qp": mean(qps[i]),
                                "texture_ratio_to_frame0": mean(ratios[i])} for i in range(1, last + 1)]}

    rows = [json.loads(line) for line in open(renders / "index.jsonl")]
    rec = load_png(renders / next(r["file"] for r in rows if r["kind"] == "recorded"))
    video = train_video(train_dir, 0)
    result["recorded_is_episode0_frame0"] = bool(np.array_equal(rec, read_video_frame(video, 0)))
    if not result["recorded_is_episode0_frame0"]:
        return {**result, "skipped": "the recorded frame is not frame 0 of training episode 0"}
    # the anchors rendered back to back (see check_domain), and the 40 frames the lookahead sees after frame 0
    anchors = [load_png(renders / r["file"]) for r in rows if r["kind"] == "anchor"][:4]
    following = [img for _, img in itertools.islice(iter_video_frames(video), 1, 41)]
    spliced = [_encode_images([a, *following], dict(ENCODER["call"]), {}) for a in anchors]
    result["splice_settings_equal_training"] = all(x264_settings(data) == _training_settings() for data in spliced)
    detail = lambda x: x - box(x)

    def against_train(images, qp) -> dict:
        return {"qp": qp, "texture_ratio_to_train_frame": mean(texture(img) for img in images) / texture(rec),
                "vs_train_frame": mean(mad(img, rec) for img in images),
                "vs_train_frame_detail": mean(mad(detail(img), detail(rec)) for img in images)}

    result["train_frame_qp"] = mean_qp(video)
    result["raw"] = against_train(anchors, None)
    for name, kw in VARIANTS.items():
        result["one_frame_" + name] = against_train([roundtrip(a, **kw) for a in anchors],
                                                    mean(mean_qp(encode(a, **kw)) for a in anchors))
    firsts = [next(iter_video_frames(data))[1] for data in spliced]
    result["spliced_crf23"] = {**against_train(firsts, mean(mean_qp(data) for data in spliced)),
                               "frames": 1 + len(following),
                               **{"vs_one_frame_" + name: mean(mad(f, roundtrip(a, **kw)) for f, a in zip(firsts, anchors))
                                  for name, kw in VARIANTS.items()}}
    return result


def check_speed(train_dir: Path, whole_pack: bool) -> dict:
    """Time of iter_video_frames per frame over one whole video (best of 3), optionally over the whole pack."""
    path = train_video(train_dir, 0)
    times = []
    for _ in range(3):
        started = time.perf_counter()
        n = sum(1 for _ in iter_video_frames(path))
        times.append((time.perf_counter() - started) / n * 1e3)
    out = {"video": path.name, "frames": n, "ms_per_frame_best": min(times), "ms_per_frame_runs": times}
    if whole_pack:
        episodes = [json.loads(line)["episode_index"] for line in open(train_dir / "meta/episodes.jsonl")]
        started = time.perf_counter()
        n = sum(1 for ep in episodes for _ in iter_video_frames(train_video(train_dir, ep)))
        seconds = time.perf_counter() - started
        out["whole_pack"] = {"videos": len(episodes), "frames": n, "seconds": seconds, "ms_per_frame": seconds / n * 1e3}
    return out


def run_check(args: argparse.Namespace) -> None:
    import warnings
    warnings.filterwarnings("ignore")  # torchvision's video deprecation notice, once per decoded frame
    train_dir, renders, out = Path(args.train_dir), Path(args.renders), Path(args.out)
    run_dir = Path(args.run_dir or (renders.parent / "prep/run_dir.txt").read_text().strip())
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in open(renders / "index.jsonl")]
    files = [renders / r["file"] for r in rows if r["kind"] != "recorded"]
    # 10 renders spread over the whole list: near and far, straight and turned
    sample = [files[i] for i in np.linspace(0, len(files) - 1, 10).round().astype(int)]
    anchor = load_png(files[0])

    report = {"versions": {"pyav": av.__version__, "libavcodec": list(av.library_versions["libavcodec"]),
                           "libavformat": list(av.library_versions["libavformat"]),
                           "libswscale": list(av.library_versions["libswscale"]), "python": sys.version.split()[0]},
              "crf_start": CRF_START}
    steps = [("streams", lambda: check_streams(train_dir)),
             ("ffprobe", lambda: check_ffprobe(train_dir, (0, 50, 98))),
             ("settings", lambda: check_settings(train_dir, anchor)),
             ("writer_reference", lambda: writer_reference(Path(args.simple_dir), anchor, out)),
             ("other_python", lambda: other_python(Path(args.simple_dir) / ".venv/bin/python", files[0], out)),
             ("decode", lambda: check_decode(train_dir, (0, 50, 98), (0, 1, 2, 3, 57, 123, 200, 251, 300))),
             ("qp", lambda: check_qp(train_dir)),
             ("colour", lambda: check_colour(anchor)),
             ("roundtrip", lambda: check_roundtrip(sample)),
             ("domain", lambda: check_domain(renders, run_dir, train_dir, out)),
             ("start_frame", lambda: check_start_frame(renders, train_dir)),
             ("speed", lambda: check_speed(train_dir, args.speed_all))]
    for name, step in steps:
        started = time.perf_counter()
        report[name] = step()
        print(f"[{name}] {time.perf_counter() - started:.1f} s", flush=True)
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print_report(report)
    print(f"\nwrote {out / 'report.json'}")


def print_report(r: dict) -> None:
    f = lambda x: f"{x:.3f}"
    s, d, q, c, rt, dom = r["streams"], r["decode"], r["qp"], r["colour"], r["roundtrip"], r["domain"]
    print(f"\n== training streams ({s['videos']} videos, {s['frames']} frames) ==")
    print(f"x264 settings strings: {len(s['settings_strings'])} distinct; equal to ENCODER['stream']: "
          f"{s['all_equal_ENCODER_stream']}")
    print(f"muxer tags: {s['muxer_tags']}")
    print(f"stream fields: {s['stream_fields']}")
    print(f"wrong frame count: {s['videos_with_wrong_frame_count']}  irregular timestamps: "
          f"{s['videos_with_irregular_timestamps']}  keyframes: {s['keyframes']}")
    for probe in r["ffprobe"]:
        print(f"ffprobe: {probe}")
    print("\n== round trip vs training stream ==")
    st = r["settings"]
    print(f"settings string identical: {st['identical']} ({len(st['match'])}/{st['options_compared']} options match; "
          f"differ: {st['differ']}); encode deterministic: {st['encode_deterministic']}")
    print(f"SIMPLE's VideoWriter on the same image: {r['writer_reference']}")
    print(f"other venv: {r['other_python']}")
    print(f"\n== decode ({d['frames_compared']} frames of episodes {d['episodes']}) ==")
    for key in ("vs_lerobot_pyav", "vs_lerobot_torchcodec", "vs_ffmpeg_cli", "vs_read_video_frame"):
        print(f"iter_video_frames {key}: {d[key]}")
    if d["skipped"]:
        print(f"skipped: {d['skipped']}")
    print("\n== quantisation (mean macroblock QP) ==")
    print(f"training frame 0: {q['frame0']}")
    for kind, v in q["all_frames"].items():
        print(f"training {kind}-frames: n {v['n']} ({v['share']:.2%}) mean {f(v['mean'])} [{f(v['min'])}, {f(v['max'])}]")
    print(f"one-frame round trip of {rt['renders']} renders: crf 23 -> QP {f(rt['crf23']['qp'])}; "
          f"crf {r['crf_start']} -> QP {f(rt['crf_start']['qp'])}")
    print("\n== colour conversion alone (anchor re-render) ==")
    print(f"rgb -> yuv420p -> rgb: {c['rgb_yuv_rgb']}")
    print(f"lossless codec (crf 0): {c['lossless_crf0']}")
    print(f"encoder luma - exact: {f(c['encode_side_luma_minus_exact'])}; decoder rgb - exact: "
          f"{[round(x, 3) for x in c['decode_side_rgb_minus_exact']]}")
    print(f"\n== raw vs round-tripped, {rt['renders']} renders ==")
    for name in ("crf23", "crf_start"):
        v = rt[name]
        print(f"{name}: mean abs {f(v['mean_abs'])} signed {f(v['mean_signed'])} max {v['max']:.0f}; second pass moves "
              f"{f(v['again_mean_abs'])}; {v['bytes']:.0f} bytes; {v['ms']:.0f} ms per round trip")
    print(f"\n== domain check: training frame vs re-renders of its pose ({len(dom['anchors'])} anchors) ==")
    print(f"transform: {dom['transform']}")
    for name in ("crf23", "crf_start"):
        for view in ("full_640x360", "model_180x320", "detail_minus_16px_mean", "smooth_16px_mean"):
            v, v4 = dom[name][view], dom[name][view + "_first4"]
            print(f"{name:9s} {view:22s} (a) raw-train {f(v['a_raw_vs_train'])}  (b) roundtrip-train "
                  f"{f(v['b_roundtrip_vs_train'])}  (c) raw-roundtrip {f(v['c_raw_vs_roundtrip'])}   first 4: "
                  f"{f(v4['a_raw_vs_train'])} / {f(v4['b_roundtrip_vs_train'])} / {f(v4['c_raw_vs_roundtrip'])}")
        print(f"{name:9s} QP {f(dom[name]['qp'])} (training frame: {f(dom['train_frame_qp'])}); texture "
              f"{f(dom[name]['texture'])} (training frame {f(dom['texture_train_frame'])}, raw {f(dom['texture_raw'])}); "
              f"round trip - raw per channel {[round(x, 2) for x in dom[name]['mean_signed_roundtrip_minus_raw_rgb']]}; "
              f"two round-tripped repeats differ by {dom[name]['repeat_roundtrip_vs_roundtrip']}")
    print(f"two raw repeats differ by {dom['repeat_raw_vs_raw']}; training frame - raw per channel "
          f"{[round(x, 2) for x in dom['mean_signed_train_minus_raw_rgb']]}")
    for row in dom["per_anchor"]:
        print(f"  {row['file']}: a {f(row['a'])}  b {f(row['b_crf23'])} / {f(row['b_crf_start'])}  "
              f"c {f(row['c_crf23'])} / {f(row['c_crf_start'])}   (crf 23 / crf start)")
    sf = r["start_frame"]
    print("\n== is a one-frame encode a fair stand-in for a training frame? ==")
    print("training frames after frame 0 (type, QP, fine detail relative to frame 0; 99 videos): " + "  ".join(
        f"{a['frame']}:{a['types']} {a['qp']:.1f} {a['texture_ratio_to_frame0']:.3f}" for a in sf["after_frame0"]))
    if "skipped" in sf:
        print(f"skipped: {sf['skipped']}")
    else:
        print(f"re-renders of the training start pose against its training frame (QP {f(sf['train_frame_qp'])}); "
              f"splice settings equal training: {sf['splice_settings_equal_training']}")
        for name in ("raw", *("one_frame_" + v for v in VARIANTS), "spliced_crf23"):
            v = sf[name]
            print(f"  {name:20s} QP {'-' if v['qp'] is None else f(v['qp']):>6s}  detail kept "
                  f"{f(v['texture_ratio_to_train_frame'])} of the training frame's  mean abs to the training frame "
                  f"{f(v['vs_train_frame'])} (detail part {f(v['vs_train_frame_detail'])})")
        v = sf["spliced_crf23"]
        print(f"  spliced = the re-render as frame 0 of {v['frames']} frames, the rest real training frames; its frame 0 "
              "differs from the one-frame round trips by " + ", ".join(
                  f"{name} {f(v['vs_one_frame_' + name])}" for name in VARIANTS))
    sp = r["speed"]
    print(f"\n== speed ==\niter_video_frames: {f(sp['ms_per_frame_best'])} ms per frame over {sp['video']} "
          f"({sp['frames']} frames), whole pack: {sp.get('whole_pack', 'not timed (--speed-all)')}")


# ------------------------------------------------------------------------------------------------------ CLI


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true", help="run the validations and print the report")
    p.add_argument("--in", dest="src", default=None, help="image to round-trip")
    p.add_argument("--out", default=None, help="with --in: the output png; with --check: the report directory "
                                               f"(default {DEV_OUT})")
    p.add_argument("--crf", type=float, default=None, help=f"with --in: crf override (training start frames: {CRF_START})")
    p.add_argument("--frames", type=int, default=None, help="with --in: encode the image this many times")
    p.add_argument("--train-dir", default=str(TRAIN_DIR), help="the train pack")
    p.add_argument("--renders", default=str(EXP3 / "exp3_indist_signal"), help="experiment 3's render directory")
    p.add_argument("--run-dir", default=None, help="the checkpoint's run directory (default: experiment 3's)")
    p.add_argument("--simple-dir", default=str(SIMPLE), help="the SIMPLE repo: its writer and venv are compared")
    p.add_argument("--speed-all", action="store_true", help="--check: also time the decode of the whole pack")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.check:
        args.out = args.out or str(DEV_OUT)
        run_check(args)
        return
    if not (args.src and args.out):
        raise SystemExit("give --check, or --in and --out")
    overrides = {k: v for k, v in (("crf", args.crf), ("frames", args.frames)) if v is not None}
    image = load_png(args.src)
    data = encode(image, **overrides)
    result = roundtrip(image, **overrides)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result).save(args.out)
    # one json line, read back by --check when it runs this file in another venv
    print(json.dumps({"out": args.out, "pyav": av.__version__, "overrides": overrides,
                      "mp4_sha1": hashlib.sha1(data).hexdigest(), "mp4_bytes": len(data), "qp": mean_qp(data),
                      "settings_equal_training": x264_settings(data) == _training_settings(), **diff(result, image)}))


if __name__ == "__main__":
    main()
