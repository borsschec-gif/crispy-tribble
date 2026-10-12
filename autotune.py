#!/usr/bin/env python3
"""Real-time autotune for a microphone (e.g. for Discord calls).

Mic -> pitch detection (YIN) -> snap to the nearest note of a scale ->
pitch shift (two-tap crossfaded delay line) -> output device
(a virtual audio cable that Discord uses as its microphone).
"""
import argparse
import sys
import wave

import numpy as np

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_ALIASES = {"DB": "C#", "EB": "D#", "GB": "F#", "AB": "G#", "BB": "A#"}
SCALES = {
    "chromatic": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "pentatonic": [0, 2, 4, 7, 9],
    "minor-pentatonic": [0, 3, 5, 7, 10],
    "blues": [0, 3, 5, 6, 7, 10],
}


def parse_key(name):
    name = name.strip().upper()
    name = FLAT_ALIASES.get(name, name)
    if name not in NOTE_NAMES:
        raise argparse.ArgumentTypeError(f"неизвестная тональность: {name}")
    return NOTE_NAMES.index(name)


class PitchDetector:
    """YIN pitch detector working on the most recent `frame` samples."""

    def __init__(self, sr, frame=2048, fmin=70.0, fmax=1000.0, threshold=0.15):
        self.sr = sr
        self.frame = frame
        self.w = frame // 2
        self.tau_min = max(2, int(sr / fmax))
        self.tau_max = min(self.w - 1, int(sr / fmin))
        self.threshold = threshold
        self.nfft = 1 << (2 * frame - 1).bit_length()

    def detect(self, x):
        w = self.w
        x = x - x.mean()
        # r(tau) = sum_{j<w} x[j] * x[j+tau], via FFT cross-correlation
        X = np.fft.rfft(x, self.nfft)
        X0 = np.fft.rfft(x[:w], self.nfft)
        r = np.fft.irfft(X * np.conj(X0), self.nfft)[: w]
        sq = np.concatenate(([0.0], np.cumsum(x * x)))
        e0 = sq[w]
        taus = np.arange(w)
        e_tau = sq[taus + w] - sq[taus]
        d = e0 + e_tau - 2.0 * r
        d[0] = 0.0
        # cumulative mean normalized difference
        cum = np.cumsum(d[1:])
        cmndf = np.ones(w)
        cmndf[1:] = d[1:] * np.arange(1, w) / np.maximum(cum, 1e-12)

        seg = cmndf[self.tau_min : self.tau_max]
        below = np.nonzero(seg < self.threshold)[0]
        if below.size == 0:
            return 0.0
        tau = below[0] + self.tau_min
        while tau + 1 < self.tau_max and cmndf[tau + 1] < cmndf[tau]:
            tau += 1
        # parabolic interpolation for sub-sample precision
        if 1 <= tau < w - 1:
            a, b, c = cmndf[tau - 1], cmndf[tau], cmndf[tau + 1]
            denom = a - 2 * b + c
            shift = 0.5 * (a - c) / denom if abs(denom) > 1e-12 else 0.0
        else:
            shift = 0.0
        return self.sr / (tau + shift)


class PitchShifter:
    """Two read heads on a delay line, crossfaded with sin^2 windows.

    Low latency and good enough for the small (<= 1 semitone) shifts
    autotune needs.
    """

    def __init__(self, sr, grain_ms=30.0, buf_size=1 << 15):
        self.grain = sr * grain_ms / 1000.0
        self.buf = np.zeros(buf_size)
        self.mask = buf_size - 1
        self.pos = 0  # absolute write position
        self.phase = 0.0
        self.ratio = 1.0

    def process(self, x, ratio):
        n = len(x)
        idx = (self.pos + np.arange(n)) & self.mask
        self.buf[idx] = x

        ratios = np.linspace(self.ratio, ratio, n, endpoint=False)
        self.ratio = ratio
        dp = (1.0 - ratios) / self.grain
        phases = (self.phase + np.cumsum(dp)) % 1.0
        self.phase = phases[-1]

        write_pos = self.pos + np.arange(n)
        out = np.zeros(n)
        for ph in (phases, (phases + 0.5) % 1.0):
            delay = ph * self.grain + 2.0
            rp = write_pos - delay
            i0 = np.floor(rp).astype(np.int64)
            frac = rp - i0
            s = self.buf[i0 & self.mask] * (1 - frac) + self.buf[(i0 + 1) & self.mask] * frac
            out += s * np.sin(np.pi * ph) ** 2
        self.pos += n
        return out


class AutoTune:
    def __init__(self, sr, key=0, scale="major", speed=0.8, gate_db=-45.0, mix=1.0,
                 transpose=0.0):
        self.sr = sr
        self.detector = PitchDetector(sr)
        self.shifter = PitchShifter(sr)
        self.history = np.zeros(self.detector.frame)
        allowed = [(key + s) % 12 for s in SCALES[scale]]
        self.allowed = np.array(sorted(allowed))
        self.speed = float(np.clip(speed, 0.0, 1.0))
        self.gate = 10 ** (gate_db / 20)
        self.mix = mix
        self.transpose = transpose
        self.correction = 0.0  # current correction in semitones
        self.last_note = None
        self.f0 = 0.0

    def target_semitones(self, f0):
        midi = 69 + 12 * np.log2(f0 / 440.0)
        base = np.floor(midi / 12) * 12
        cands = np.concatenate([base - 12 + self.allowed, base + self.allowed,
                                base + 12 + self.allowed])
        target = cands[np.argmin(np.abs(cands - midi))]
        # small hysteresis: keep the previous note if we are near the boundary
        if self.last_note is not None and abs(midi - self.last_note) < 0.6 \
                and self.last_note in cands:
            target = self.last_note
        self.last_note = target
        return target - midi

    def process(self, x):
        x = x.astype(np.float64)
        n = len(x)
        self.history = np.concatenate([self.history[n:], x])[-self.detector.frame:]

        rms = np.sqrt(np.mean(x * x)) if n else 0.0
        f0 = self.detector.detect(self.history) if rms > self.gate else 0.0
        self.f0 = f0
        if f0 > 0:
            target = self.target_semitones(f0)
            # speed 1.0 -> instant (robotic "T-Pain"), lower -> smoother/natural
            block_s = n / self.sr
            tau = 0.002 + (1.0 - self.speed) * 0.25
            alpha = 1.0 - np.exp(-block_s / tau)
            self.correction += (target - self.correction) * alpha
        else:
            self.last_note = None
            self.correction *= 0.9  # relax back to no correction on silence/noise

        ratio = 2 ** ((self.correction + self.transpose) / 12)
        wet = self.shifter.process(x, ratio)
        return wet * self.mix + x * (1 - self.mix)


def read_wav(path):
    with wave.open(path, "rb") as f:
        sr, ch, sw = f.getframerate(), f.getnchannels(), f.getsampwidth()
        data = f.readframes(f.getnframes())
    if sw != 2:
        sys.exit("поддерживается только 16-битный WAV")
    x = np.frombuffer(data, dtype="<i2").astype(np.float64) / 32768.0
    return x.reshape(-1, ch).mean(axis=1), sr


def write_wav(path, x, sr):
    y = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(y.tobytes())


def process_file(args):
    x, sr = read_wav(args.file[0])
    tuner = make_tuner(args, sr)
    out = np.concatenate([tuner.process(x[i:i + args.block])
                          for i in range(0, len(x), args.block)])
    write_wav(args.file[1], out, sr)
    print(f"готово: {args.file[1]}")


def make_tuner(args, sr):
    return AutoTune(sr, key=args.key, scale=args.scale, speed=args.speed,
                    gate_db=args.gate, mix=args.mix, transpose=args.transpose)


VIRTUAL_CABLES = ["CABLE Input", "BlackHole", "autotune"]


def resolve_device(devices, query, kind, hostapi=None):
    """Find a device index by number or name substring.

    Windows lists the same device once per host API (MME, WASAPI, ...), so
    several matches are normal: prefer the one on `hostapi`, else the first.
    """
    if query is None or isinstance(query, int):
        return query
    key = "max_input_channels" if kind == "input" else "max_output_channels"
    matches = [i for i, d in enumerate(devices)
               if query.lower() in d["name"].lower() and d[key] > 0]
    if not matches:
        raise LookupError(f"устройство '{query}' не найдено")
    same_api = [i for i in matches if devices[i]["hostapi"] == hostapi]
    return (same_api or matches)[0]


def auto_virtual_cable(devices, hostapi):
    for name in VIRTUAL_CABLES:
        try:
            return resolve_device(devices, name, "output", hostapi)
        except LookupError:
            pass
    return None


def run_live(args):
    import sounddevice as sd

    sr = args.samplerate
    tuner = make_tuner(args, sr)
    monitor = []

    def callback(indata, outdata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        y = tuner.process(indata[:, 0]) * args.gain
        y = np.clip(y, -1, 1).astype(np.float32)
        outdata[:] = y[:, None]
        if monitor:
            monitor[0].write(y[:, None].copy())

    devices = list(sd.query_devices())
    try:
        inp = resolve_device(devices, args.input, "input")
        if inp is None:
            inp = sd.default.device[0]
            if inp is None or inp < 0:
                raise LookupError("не найден микрофон по умолчанию, укажи его через -i")
        api = devices[inp]["hostapi"]
        out = resolve_device(devices, args.output, "output", api)
        if out is None:
            out = auto_virtual_cable(devices, api)
        if out is None:
            raise LookupError("не найден виртуальный кабель (VB-Audio CABLE / BlackHole). "
                              "Установи его и перезагрузи компьютер, или укажи выход через -o")
        mon_dev = resolve_device(devices, args.monitor, "output", api)
    except LookupError as e:
        print(sd.query_devices())
        sys.exit(f"\nОшибка: {e}")
    print(f"Микрофон: {devices[inp]['name']}\nВыход:    {devices[out]['name']}")

    try:
        stream = sd.Stream(device=(inp, out), samplerate=sr,
                           blocksize=args.block, channels=(1, 1), dtype="float32",
                           latency="low", callback=callback)
    except Exception as e:
        sys.exit(f"Не удалось открыть устройства: {e}\n"
                 "Подсказка: укажи номера устройств из одного Host API "
                 "(см. --list), например -i 1 -o 5.")

    if mon_dev is not None:
        mon = sd.OutputStream(device=mon_dev, samplerate=sr, channels=1,
                              dtype="float32", latency="low", blocksize=args.block)
        mon.start()
        monitor.append(mon)

    key_name = NOTE_NAMES[args.key]
    print(f"Автотюн включён: {key_name} {args.scale}, скорость {args.speed}. "
          "Ctrl+C — выход.")
    with stream:
        try:
            while True:
                sd.sleep(200)
                f0 = tuner.f0
                if f0 > 0:
                    midi = 69 + 12 * np.log2(f0 / 440)
                    note = NOTE_NAMES[int(round(midi)) % 12]
                    print(f"\r{f0:7.1f} Hz  ~{note:<2}  коррекция {tuner.correction:+.2f} пт ",
                          end="", flush=True)
                else:
                    print("\r" + " " * 50, end="\r", flush=True)
        except KeyboardInterrupt:
            print("\nВыход.")


def main():
    p = argparse.ArgumentParser(description="Автотюн для микрофона в реальном времени")
    p.add_argument("--list", action="store_true", help="показать аудиоустройства")
    p.add_argument("-i", "--input", type=lambda s: int(s) if s.isdigit() else s,
                   help="микрофон (номер или часть названия)")
    p.add_argument("-o", "--output", type=lambda s: int(s) if s.isdigit() else s,
                   help="выход: виртуальный кабель (например 'CABLE Input')")
    p.add_argument("-m", "--monitor", type=lambda s: int(s) if s.isdigit() else s,
                   help="(необязательно) наушники, чтобы слышать себя")
    p.add_argument("-k", "--key", type=parse_key, default=parse_key("C"),
                   help="тональность: C, C#, D, Eb, ... (по умолчанию C)")
    p.add_argument("-s", "--scale", choices=SCALES, default="chromatic",
                   help="лад (по умолчанию chromatic — подходит к любой песне)")
    p.add_argument("--speed", type=float, default=0.9,
                   help="скорость подтяжки 0..1 (1 = роботизированный эффект)")
    p.add_argument("--transpose", type=float, default=0.0,
                   help="сдвиг в полутонах (например 12 или -5)")
    p.add_argument("--mix", type=float, default=1.0, help="0..1, доля обработанного звука")
    p.add_argument("--gain", type=float, default=1.0, help="громкость на выходе")
    p.add_argument("--gate", type=float, default=-45.0,
                   help="порог тишины в дБ (ниже — без обработки)")
    p.add_argument("--samplerate", type=int, default=48000)
    p.add_argument("--block", type=int, default=256, help="размер блока (задержка)")
    p.add_argument("--file", nargs=2, metavar=("IN.wav", "OUT.wav"),
                   help="обработать файл вместо живого звука (для проверки)")
    args = p.parse_args()

    if args.list:
        import sounddevice as sd
        print(sd.query_devices())
        return
    if args.file:
        process_file(args)
        return
    run_live(args)


if __name__ == "__main__":
    try:
        main()
    finally:
        # double-clicked .exe: keep the console open so errors are readable
        if getattr(sys, "frozen", False) and sys.stdin and sys.stdin.isatty():
            input("\nНажми Enter, чтобы закрыть окно...")
