#!/usr/bin/env python3
"""Real-time autotune for a microphone (e.g. for Discord calls).

Mic -> pitch detection (YIN) -> snap to the nearest note of a scale ->
pitch shift (two-tap crossfaded delay line) -> output device
(a virtual audio cable that Discord uses as its microphone).
"""
import argparse
import json
import os
import sys
import wave

import numpy as np

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
HOTKEY_DEFAULT = "Ctrl+Alt+A"  # toggles autotune; changeable in the window
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
        # a strong upper harmonic (e.g. vowel "i") can make an early, shallower
        # dip; prefer a clearly deeper minimum shortly after it (not an octave)
        hi = min(self.tau_max, int(tau * 1.25) + 1)
        if hi > tau + 1:
            alt = tau + 1 + int(np.argmin(cmndf[tau + 1:hi]))
            if cmndf[alt] < cmndf[tau] - 0.05:
                tau = alt
        # octave-up error: the real period (2x or 3x longer) repeats much better
        for mult in (2, 3):
            lo, hi = tau * mult - 3, tau * mult + 4
            if hi >= self.tau_max:
                break
            alt = lo + int(np.argmin(cmndf[lo:hi]))
            if cmndf[alt] < cmndf[tau] - 0.08:
                tau = alt
                break
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


class TalkBox:
    """Talk box / vocoder: a synth playing the exact tuned note, shaped by the
    vowels (spectral envelope) of the voice. Streaming STFT, ~20 ms latency.

    The signal is always delayed by the same amount, so turning the effect
    on and off does not click.
    """

    def __init__(self, sr, n_fft=1024, hop=256, lifter=40):
        self.sr = sr
        self.n = n_fft
        self.hop = hop
        self.lifter = lifter
        self.win = np.sqrt(np.hanning(n_fft + 1)[:-1])  # sqrt-hann, analysis+synthesis
        self.norm = 2.0  # sum of hann windows at 75% overlap
        self.voice = np.zeros(n_fft)
        self.carrier = np.zeros(n_fft)
        self.ola = np.zeros(n_fft)
        self.in_v = np.zeros(0)
        self.in_c = np.zeros(0)
        self.out = np.zeros(hop)  # pre-roll so every call can return len(x) samples
        self.phase = 0.0
        self.amount = 0.0  # last applied amount, ramped to avoid clicks
        self.rng = np.random.default_rng(0)

    def _carrier(self, n, freq):
        if freq <= 0:
            return self.rng.standard_normal(n) * 0.3  # unvoiced: noise keeps consonants
        inc = 2 * np.pi * freq / self.sr
        ph = self.phase + inc * np.arange(1, n + 1)
        self.phase = ph[-1] % (2 * np.pi)
        k = np.arange(1, min(60, int(self.sr * 0.45 / freq)) + 1)[:, None]
        return (np.sin(k * ph) / k).sum(axis=0) * 0.6  # band-limited sawtooth

    def _envelope(self, mag):
        cep = np.fft.irfft(np.log(mag + 1e-9))
        cep[self.lifter:-self.lifter + 1] = 0.0
        return np.exp(np.fft.rfft(cep).real)

    def process(self, x, freq, amount):
        self.in_v = np.concatenate([self.in_v, x])
        self.in_c = np.concatenate([self.in_c, self._carrier(len(x), freq)])
        h, out = self.hop, [self.out]
        while len(self.in_v) >= h:
            self.voice = np.concatenate([self.voice[h:], self.in_v[:h]])
            self.carrier = np.concatenate([self.carrier[h:], self.in_c[:h]])
            self.in_v, self.in_c = self.in_v[h:], self.in_c[h:]
            if amount > 0 or self.amount > 0:
                V = np.fft.rfft(self.voice * self.win)
                C = np.fft.rfft(self.carrier * self.win)
                env_v = self._envelope(np.abs(V))
                env_c = self._envelope(np.abs(C))
                Y = C * (env_v / env_c)
                # match the loudness of the voice in this frame
                Y *= np.sqrt(np.sum(np.abs(V) ** 2) / (np.sum(np.abs(Y) ** 2) + 1e-12))
                self.ola += np.fft.irfft(Y, self.n) * self.win
            talk = self.ola[:h] / self.norm
            self.ola = np.concatenate([self.ola[h:], np.zeros(h)])
            dry = self.voice[:h]
            a = np.linspace(self.amount, amount, h, endpoint=False)
            self.amount = amount
            # equal-power crossfade: the two signals are mostly uncorrelated
            out.append(dry * np.sqrt(1 - a) + talk * np.sqrt(a))
        self.out = np.concatenate(out)
        y, self.out = self.out[:len(x)], self.out[len(x):]
        return y


class AutoTune:
    def __init__(self, sr, key=0, scale="major", speed=0.8, gate_db=-45.0, mix=1.0,
                 transpose=0.0, talkbox=0.0):
        self.sr = sr
        self.detector = PitchDetector(sr)
        self.shifter = PitchShifter(sr)
        self.history = np.zeros(self.detector.frame)
        self.set_scale(key, scale)
        self.speed = float(np.clip(speed, 0.0, 1.0))
        self.gate = 10 ** (gate_db / 20)
        self.mix = mix
        self.transpose = transpose
        self.talkbox = talkbox
        self.tb = TalkBox(sr)
        self.correction = 0.0  # current correction in semitones
        self.f0 = 0.0

    def set_scale(self, key, scale):
        self.allowed = np.array(sorted((key + s) % 12 for s in SCALES[scale]))
        self.last_note = None

    @property
    def gate_db(self):
        return 20 * np.log10(self.gate)

    @gate_db.setter
    def gate_db(self, value):
        self.gate = 10 ** (value / 20)

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
        voice = wet * self.mix + x * (1 - self.mix)
        return self.tb.process(voice, f0 * ratio, float(np.clip(self.talkbox, 0.0, 1.0)))


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
    eff = effective_settings(args, load_settings())
    tuner = AutoTune(sr, key=parse_key(eff["key"]), scale=eff["scale"], speed=eff["speed"],
                     gate_db=eff["gate"], mix=eff["mix"], transpose=eff["transpose"],
                     talkbox=eff["talkbox"])
    out = np.concatenate([tuner.process(x[i:i + args.block])
                          for i in range(0, len(x), args.block)]) * eff["gain"]
    write_wav(args.file[1], out, sr)
    print(f"готово: {args.file[1]}")


# our own macOS device first, then VB-Cable (Windows), BlackHole, a Linux null sink
VIRTUAL_CABLES = ["Autotune Mic", "CABLE Input", "BlackHole", "autotune"]

# macOS: BlackHole built under our name (see .github/workflows), bundled in the .app
MAC_DRIVER = "AutotuneMic.driver"
MAC_HAL_DIR = "/Library/Audio/Plug-Ins/HAL"


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


# virtual cables and system aliases are not real microphones
NOT_MICS = ["CABLE Output", "BlackHole", "Autotune Mic", "Monitor of", "Mapper", "Переназначение",
            "Первичный драйвер", "Primary Sound Capture", "Stereo Mix", "Стерео микшер"]


def app_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def settings_dir():
    if sys.platform == "darwin" and getattr(sys, "frozen", False):
        # never write inside the signed .app bundle
        d = os.path.expanduser("~/Library/Application Support/Autotune")
        os.makedirs(d, exist_ok=True)
        return d
    return app_dir()


SETTINGS_PATH = os.path.join(settings_dir(), "autotune_settings.json")


def load_settings():
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_settings(settings):
    try:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"(не удалось сохранить настройки: {e})")


def list_mics(devices, hostapi):
    return [i for i, d in enumerate(devices)
            if d["max_input_channels"] > 0 and d["hostapi"] == hostapi
            and not any(x.lower() in d["name"].lower() for x in NOT_MICS)]


def interactive():
    return sys.stdin is not None and sys.stdin.isatty()


def choose_mic(devices, hostapi, default):
    mics = list_mics(devices, hostapi) or [default]
    print("\nВыбери микрофон:")
    for n, i in enumerate(mics, 1):
        mark = "  (по умолчанию)" if i == default else ""
        print(f"  {n}. {devices[i]['name']}{mark}")
    while True:
        ans = input("Номер (Enter — по умолчанию): ").strip()
        if not ans:
            return default
        if ans.isdigit() and 1 <= int(ans) <= len(mics):
            return mics[int(ans) - 1]
        print("Нет такого номера, попробуй ещё раз.")


def install_vb_cable(say=print):
    """Download VB-Audio Virtual Cable from the official site and run its setup."""
    import re
    import tempfile
    import urllib.parse
    import urllib.request
    import webbrowser
    import zipfile

    page = "https://vb-audio.com/Cable/"
    try:
        say("Скачиваю VB-Audio Virtual Cable с vb-audio.com ...")
        req = urllib.request.Request(page, headers={"User-Agent": "Mozilla/5.0"})
        html = urllib.request.urlopen(req, timeout=20).read().decode("utf-8", "replace")
        m = re.search(r'href="([^"]*VBCABLE_Driver_Pack\d+\.zip)"', html)
        if not m:
            raise RuntimeError("ссылка на установщик не найдена")
        url = urllib.parse.urljoin(page, m.group(1))
        folder = tempfile.mkdtemp(prefix="vbcable_")
        zip_path = os.path.join(folder, "vbcable.zip")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r, open(zip_path, "wb") as f:
            f.write(r.read())
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(folder)
        is64 = sys.maxsize > 2**32
        setup = os.path.join(folder, "VBCABLE_Setup_x64.exe" if is64 else "VBCABLE_Setup.exe")
        if not os.path.exists(setup):
            raise RuntimeError("в архиве нет установщика")
        import ctypes
        say("Сейчас Windows спросит разрешение — нажми «Да», затем «Install Driver».")
        rc = ctypes.windll.shell32.ShellExecuteW(None, "runas", setup, None, folder, 1)
        if rc <= 32:
            raise RuntimeError("установщик не запустился")
        say("\nКогда установка закончится, ПЕРЕЗАГРУЗИ компьютер и снова запусти autotune.exe.")
    except Exception as e:
        say(f"Автоустановка не удалась ({e}). Открываю сайт — скачай и установи вручную.")
        webbrowser.open(page)


def bundled_mac_driver():
    base = getattr(sys, "_MEIPASS", app_dir())
    # the build puts it in Autotune.app/Contents/Resources; _MEIPASS is Contents/Frameworks
    for path in (os.path.join(base, "..", "Resources", MAC_DRIVER), os.path.join(base, MAC_DRIVER)):
        path = os.path.realpath(path)
        if os.path.isdir(path):
            return path
    return None


def mac_driver_install_script(src):
    import shlex
    dst = os.path.join(MAC_HAL_DIR, MAC_DRIVER)
    return (f"mkdir -p {shlex.quote(MAC_HAL_DIR)} && rm -rf {shlex.quote(dst)} && "
            f"cp -R {shlex.quote(src)} {shlex.quote(dst)} && "
            f"xattr -dr com.apple.quarantine {shlex.quote(dst)} ; "
            "killall -9 coreaudiod")


def install_mac_driver(sd, say=print):
    """Install the bundled 'Autotune Mic' device (asks for the admin password)."""
    import subprocess
    import time

    src = bundled_mac_driver()
    if src is None:
        say("Виртуальный микрофон не встроен в эту сборку. Установи BlackHole: "
            "https://existential.audio/blackhole/")
        return False
    script = mac_driver_install_script(src).replace("\\", "\\\\").replace('"', '\\"')
    r = subprocess.run(["osascript", "-e",
                        f'do shell script "{script}" with administrator privileges'],
                       capture_output=True, text=True)
    if r.returncode != 0:
        say(f"Установка отменена или не удалась: {r.stderr.strip()}")
        return False
    for _ in range(20):  # coreaudiod restarts in a few seconds
        time.sleep(0.5)
        sd._terminate()
        sd._initialize()
        if auto_virtual_cable(list(sd.query_devices()), None) is not None:
            return True
    say("Микрофон установлен, но пока не появился. Перезапусти программу.")
    return False


DEFAULTS = {"key": "C", "scale": "chromatic", "speed": 0.9, "transpose": 0.0,
            "mix": 1.0, "talkbox": 0.0, "gain": 1.0, "gate": -45.0,
            "enabled": True, "hotkey": HOTKEY_DEFAULT,
            "listen": False, "listen_volume": 0.8}


def effective_settings(args, settings):
    """Saved settings, overridden by options given on the command line."""
    eff = {k: settings.get(k, v) for k, v in DEFAULTS.items()}
    for k in DEFAULTS:
        v = getattr(args, k, None)
        if v is not None:
            eff[k] = NOTE_NAMES[v] if k == "key" else v
    if eff["scale"] not in SCALES:
        eff["scale"] = DEFAULTS["scale"]
    try:
        parse_key(eff["key"])
    except argparse.ArgumentTypeError:
        eff["key"] = DEFAULTS["key"]
    return eff


class CableMissing(LookupError):
    pass


class Listen:
    """Plays the processed voice into headphones on its own stream.

    The main callback only drops blocks into a short queue, so a slow or
    unplugged headphone device can never stall what Discord hears.
    """

    MAX_BLOCKS = 4  # caps the extra headphone latency at a few blocks
    PRIME_BLOCKS = 2  # jitter buffer: mic and headphones run on different clocks

    def __init__(self, sd, sr, block, device, channels, start=True):
        from collections import deque
        self.queue = deque()
        self.pending = np.zeros(0, dtype=np.float32)
        self.primed = False
        self.volume = 1.0
        self.stream = sd.OutputStream(device=device, samplerate=sr, channels=channels,
                                      dtype="float32", latency="low", blocksize=block,
                                      callback=self._callback)
        if start:
            self.stream.start()

    def start(self):
        self.stream.start()

    def push(self, y):
        self.queue.append(y)
        while len(self.queue) > self.MAX_BLOCKS:
            self.queue.popleft()

    def _callback(self, outdata, frames, time, status):
        out = np.zeros(frames, dtype=np.float32)
        if not self.primed and len(self.queue) >= self.PRIME_BLOCKS:
            self.primed = True
        filled = 0
        while self.primed and filled < frames:
            if not len(self.pending):
                if not self.queue:
                    self.primed = False  # underrun: refill a little before playing on
                    break
                self.pending = self.queue.popleft()
            n = min(frames - filled, len(self.pending))
            out[filled:filled + n] = self.pending[:n]
            self.pending = self.pending[n:]
            filled += n
        outdata[:] = (out * self.volume)[:, None]

    def close(self):
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass


class Engine:
    """Audio stream around an AutoTune whose parameters may change at any time."""

    def __init__(self, sd, sr, block, eff):
        self.sd = sd
        self.sr = sr
        self.block = block
        self.tuner = AutoTune(sr, key=parse_key(eff["key"]), scale=eff["scale"],
                              speed=eff["speed"], gate_db=eff["gate"], mix=eff["mix"],
                              transpose=eff["transpose"], talkbox=eff["talkbox"])
        self.gain = eff["gain"]
        self.enabled = eff["enabled"]
        self.listen_volume = eff["listen_volume"]
        self.level = 0.0
        self.stream = None
        self.listen = None  # headphones stream (normal mode)
        self.cable = None  # virtual cable stream (shared mode)
        self.mode = None  # "normal" | "shared"
        self.io = None  # (mic, virtual cable, cable channels)
        self.listen_cfg = None  # (headphones, channels)

    def _process(self, indata):
        x = indata[:, 0]
        if self.enabled:
            y = self.tuner.process(x)
        else:
            y = x.astype(np.float64)
            self.tuner.f0 = 0.0
        y = np.clip(y * self.gain, -1, 1).astype(np.float32)
        self.level = float(np.abs(y).max()) if len(y) else 0.0
        return y

    def callback(self, indata, outdata, frames, time, status):
        """Normal mode: mic -> virtual cable; headphones (if any) via a queue."""
        y = self._process(indata)
        outdata[:] = y[:, None]  # broadcast to every output channel
        listen = self.listen
        if listen is not None:
            listen.volume = self.listen_volume
            listen.push(y.copy())

    def callback_shared(self, indata, outdata, frames, time, status):
        """Shared mode (headphones = the mic's device): mic -> headphones directly,
        virtual cable via a queue."""
        y = self._process(indata)
        outdata[:] = (y * self.listen_volume)[:, None]
        cable = self.cable
        if cable is not None:
            cable.push(y.copy())

    def start(self, inp, out, mon=None, out_channels=1):
        """(Re)open the mic -> virtual cable stream (e.g. after picking another mic)."""
        self.io = (inp, out, out_channels)
        if mon is not None:
            self.listen_cfg = (mon, 1)
        self._reopen(main=True)

    def set_listen(self, device, channels=1):
        """Start/stop hearing yourself in headphones (device index or None)."""
        self.listen_cfg = (device, channels) if device is not None else None
        self._reopen(main=False)

    def _shares_device(self):
        return (self.listen_cfg is not None and self.io is not None
                and self.listen_cfg[0] in self.io[:2])

    def _reopen(self, main):
        """Apply a change of mic or headphones.

        Two PortAudio streams on one CoreAudio device are unreliable (the second
        one often stays silent), so when the headphones are the mic's own device
        (AirPods) a single duplex stream serves both, and Discord's virtual cable
        is fed through a short queue instead.
        """
        shared = self._shares_device()
        if not main and not shared and self.mode == "normal" and self.stream is not None:
            # just the headphones changed, on their own device: leave Discord's stream alone
            self._close_listen()
            if self.listen_cfg is not None:
                device, channels = self.listen_cfg
                try:
                    self.listen = Listen(self.sd, self.sr, self.block, device, channels)
                except Exception:
                    self.listen_cfg = None
                    raise
            return
        if self.io is None:
            return
        if main and not shared and self.mode == "normal":
            # another mic, headphones on their own device: keep them playing
            self._close_main()
            mic, cable, cable_channels = self.io
            self.stream = self.sd.Stream(device=(mic, cable), samplerate=self.sr,
                                         blocksize=self.block, channels=(1, cable_channels),
                                         dtype="float32", latency="low",
                                         callback=self.callback)
            self.stream.start()
            if self.listen is None and self.listen_cfg is not None:
                device, channels = self.listen_cfg
                try:
                    self.listen = Listen(self.sd, self.sr, self.block, device, channels)
                except Exception:
                    self.listen_cfg = None
                    raise
            return
        was_shared = self.mode == "shared"
        self.stop()
        if shared or was_shared:
            import time
            time.sleep(0.15)  # CoreAudio releases a stopped device asynchronously
        try:
            self._build()
        except Exception:
            if self.listen_cfg is None:
                raise
            # Discord must keep hearing us even if the headphones fail
            self.listen_cfg = None
            self.stop()
            self._build()
            self._start_all()
            raise
        self._start_all()

    def _build(self):
        mic, cable, cable_channels = self.io
        if self._shares_device():
            device, channels = self.listen_cfg
            self.cable = Listen(self.sd, self.sr, self.block, cable, cable_channels,
                                start=False)
            try:
                self.stream = self.sd.Stream(device=(mic, device), samplerate=self.sr,
                                             blocksize=self.block, channels=(1, channels),
                                             dtype="float32", latency="low",
                                             callback=self.callback_shared)
            except Exception:
                self._close_cable()
                raise
            self.mode = "shared"
        else:
            # write the same signal to both channels: some apps only listen to one
            self.stream = self.sd.Stream(device=(mic, cable), samplerate=self.sr,
                                         blocksize=self.block, channels=(1, cable_channels),
                                         dtype="float32", latency="low",
                                         callback=self.callback)
            self.mode = "normal"
            if self.listen_cfg is not None:
                device, channels = self.listen_cfg
                self.listen = Listen(self.sd, self.sr, self.block, device, channels,
                                     start=False)

    def _start_all(self):
        for out in (self.listen, self.cable):
            if out is not None:
                out.start()
        self.stream.start()

    def _close_main(self):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
        self.stream = None

    def _close_listen(self):
        old, self.listen = self.listen, None
        if old is not None:
            old.close()

    def _close_cable(self):
        old, self.cable = self.cable, None
        if old is not None:
            old.close()

    def stop(self):
        self._close_listen()
        self._close_cable()
        self._close_main()
        self.mode = None


# virtual cables and system aliases are not headphones
NOT_HEADPHONES = ["CABLE Input", "Autotune Mic", "BlackHole", "autotune", "Mapper",
                  "Переназначение", "Primary Sound", "Первичный"]


def list_headphones(devices, hostapi):
    return [i for i, d in enumerate(devices)
            if d["max_output_channels"] > 0 and d["hostapi"] == hostapi
            and not any(x.lower() in d["name"].lower() for x in NOT_HEADPHONES)]


def default_input(sd, devices):
    default_in = sd.default.device[0]
    if default_in is None or default_in < 0:
        mics = [i for i, d in enumerate(devices) if d["max_input_channels"] > 0]
        if not mics:
            raise LookupError("в системе нет ни одного микрофона")
        default_in = mics[0]
    return default_in


def pick_devices(sd, args, settings):
    """Return (devices, hostapi, default_in, saved/explicit mic or None, out, monitor)."""
    devices = list(sd.query_devices())
    default_in = default_input(sd, devices)
    api = devices[default_in]["hostapi"]
    out = resolve_device(devices, args.output, "output", api)
    if out is None:
        out = auto_virtual_cable(devices, api)
    if out is None:
        raise CableMissing("не найден виртуальный кабель (VB-Audio CABLE / BlackHole). "
                           "Установи его и перезагрузи компьютер, или укажи выход через -o")
    inp = None
    if args.input is not None:
        inp = resolve_device(devices, args.input, "input", api)
    elif not args.choose and settings.get("mic"):
        try:
            inp = resolve_device(devices, settings["mic"], "input", api)
        except LookupError:
            print(f"Сохранённый микрофон «{settings['mic']}» не найден.")
    mon = resolve_device(devices, args.monitor, "output", api)
    return devices, api, default_in, inp, out, mon


def discord_device_name(out_name):
    if "CABLE" in out_name:
        return "CABLE Output (VB-Audio Virtual Cable)"
    if "autotune mic" in out_name.lower():
        return "Autotune Mic"
    if "autotune" in out_name.lower():
        return "Monitor of autotune"
    return out_name


def run_live(args):
    """Console mode (--nogui)."""
    import sounddevice as sd

    settings = load_settings()
    try:
        devices, api, default_in, inp, out, mon = pick_devices(sd, args, settings)
        if inp is None:
            inp = choose_mic(devices, api, default_in) if interactive() else default_in
            settings["mic"] = devices[inp]["name"]
            save_settings(settings)
    except CableMissing as e:
        print("Не найден виртуальный кабель — без него Discord не увидит автотюн.")
        if sys.platform == "darwin" and interactive() and \
                input("Установить виртуальный микрофон «Autotune Mic»? [Y/n] ").strip().lower() \
                in ("", "y", "д", "yes", "да"):
            if install_mac_driver(sd):
                return run_live(args)
            sys.exit(1)
        if sys.platform == "win32" and interactive() and \
                input("Установить VB-Audio Virtual Cable сейчас? [Y/n] ").strip().lower() \
                in ("", "y", "д", "yes", "да"):
            install_vb_cable()
            return
        sys.exit(f"\nОшибка: {e}")
    except LookupError as e:
        print(sd.query_devices())
        sys.exit(f"\nОшибка: {e}")

    eff = effective_settings(args, settings)
    engine = Engine(sd, args.samplerate, args.block, eff)
    print(f"\nМикрофон: {devices[inp]['name']}   (сменить: запусти с --choose)")
    print(f"В Discord выбери устройство ввода: {discord_device_name(devices[out]['name'])}\n")
    try:
        engine.start(inp, out, mon, min(2, devices[out]["max_output_channels"]))
    except Exception as e:
        sys.exit(f"Не удалось открыть устройства: {e}\n"
                 "Подсказка: укажи номера устройств из одного Host API "
                 "(см. --list), например -i 1 -o 5.")

    print(f"Автотюн включён: {eff['key']} {eff['scale']}, скорость {eff['speed']}. "
          "Ctrl+C — выход.")
    tuner = engine.tuner
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
    finally:
        engine.stop()


SCALE_LABELS = {
    "chromatic": "Все ноты (подходит к любой песне)",
    "major": "Мажор",
    "minor": "Минор",
    "pentatonic": "Пентатоника",
    "minor-pentatonic": "Минорная пентатоника",
    "blues": "Блюз",
}

PRESETS = {
    "Естественно": {"speed": 0.5, "transpose": 0.0, "mix": 1.0, "talkbox": 0.0},
    "T-Pain": {"speed": 1.0, "transpose": 0.0, "mix": 1.0, "talkbox": 0.0},
    # Ye "All the Love" (Bully): hard-tuned vocal blended with a talk box
    "Бабияй": {"speed": 1.0, "transpose": 0.0, "mix": 1.0, "talkbox": 0.65},
    "Чистый ток-бокс": {"speed": 1.0, "transpose": 0.0, "mix": 1.0, "talkbox": 1.0},
    "Бурундук": {"speed": 0.9, "transpose": 12.0, "mix": 1.0, "talkbox": 0.0},
    "Демон": {"speed": 0.9, "transpose": -8.0, "mix": 1.0, "talkbox": 0.0},
    "Сброс": {"speed": 0.9, "transpose": 0.0, "mix": 1.0, "talkbox": 0.0, "gain": 1.0,
              "gate": -45.0},
}

# Discord dark theme
C_LINE = "#1f2023"
C_MAIN = "#313338"
C_CARD = "#2b2d31"
C_INPUT = "#1e1f22"
C_HOVER = "#35373c"
C_SELECTED = "#404249"
C_TEXT = "#f2f3f5"
C_MUTED = "#b5bac1"
C_FAINT = "#949ba4"
C_BLURPLE = "#5865f2"
C_BLURPLE_HOVER = "#4752c4"
C_GREEN = "#23a55a"
C_RED = "#f23f43"


# ---------------------------------------------------------------- hotkeys
# A system-wide shortcut that toggles autotune while Discord (or a game) is
# focused. Windows: RegisterHotKey; macOS: Carbon RegisterEventHotKey. Neither
# needs extra permissions. Elsewhere the shortcut works only in our window.

HOTKEY_MODS = ["Ctrl", "Alt", "Shift", "Cmd"]
FKEYS = [f"F{i}" for i in range(1, 13)]

# macOS virtual key codes (HIToolbox/Events.h)
MAC_KEYCODES = {
    "A": 0x00, "S": 0x01, "D": 0x02, "F": 0x03, "H": 0x04, "G": 0x05, "Z": 0x06, "X": 0x07,
    "C": 0x08, "V": 0x09, "B": 0x0B, "Q": 0x0C, "W": 0x0D, "E": 0x0E, "R": 0x0F, "Y": 0x10,
    "T": 0x11, "1": 0x12, "2": 0x13, "3": 0x14, "4": 0x15, "6": 0x16, "5": 0x17, "9": 0x19,
    "7": 0x1A, "8": 0x1C, "0": 0x1D, "O": 0x1F, "U": 0x20, "I": 0x22, "P": 0x23, "L": 0x25,
    "J": 0x26, "K": 0x28, "N": 0x2D, "M": 0x2E,
    "F1": 0x7A, "F2": 0x78, "F3": 0x63, "F4": 0x76, "F5": 0x60, "F6": 0x61, "F7": 0x62,
    "F8": 0x64, "F9": 0x65, "F10": 0x6D, "F11": 0x67, "F12": 0x6F,
}


def parse_hotkey(text):
    """'Ctrl+Alt+A' -> ({'Ctrl', 'Alt'}, 'A'); None if invalid."""
    parts = [p.strip() for p in (text or "").split("+") if p.strip()]
    if not parts:
        return None
    aliases = {"control": "Ctrl", "ctrl": "Ctrl", "alt": "Alt", "option": "Alt",
               "shift": "Shift", "cmd": "Cmd", "command": "Cmd", "win": "Cmd"}
    mods = set()
    for p in parts[:-1]:
        m = aliases.get(p.lower())
        if m is None:
            return None
        mods.add(m)
    key = parts[-1].upper()
    if not (key in FKEYS or (len(key) == 1 and key.isalnum() and key.isascii())):
        return None
    if not mods and key not in FKEYS:
        return None  # a bare letter would fire while typing in chat
    return mods, key


def hotkey_text(mods, key, platform=None):
    """Display form, with the platform's names for the keys."""
    platform = platform or sys.platform
    names = {"Alt": "Option" if platform == "darwin" else "Alt",
             "Cmd": "Cmd" if platform == "darwin" else "Win"}
    return "+".join([names.get(m, m) for m in HOTKEY_MODS if m in mods] + [key])


def canonical_hotkey(mods, key):
    return "+".join([m for m in HOTKEY_MODS if m in mods] + [key])


class GlobalHotkey:
    """Counts presses of a system-wide shortcut; the GUI polls `presses`."""

    def __init__(self, mods, key):
        self.mods, self.key = mods, key
        self.presses = 0
        self.error = None
        self._stop = None

    def start(self):
        try:
            if sys.platform == "win32":
                self._start_windows()
            elif sys.platform == "darwin":
                self._start_mac()
            else:
                self.error = "только когда окно программы активно"
        except Exception as e:  # never let a shortcut problem break the app
            self.error = f"не удалось включить ({e})"
        return self.error is None

    def stop(self):
        if self._stop:
            self._stop()
            self._stop = None

    def _start_windows(self):
        import ctypes
        import threading
        from ctypes import wintypes

        user32, kernel32 = ctypes.windll.user32, ctypes.windll.kernel32
        user32.RegisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.UINT,
                                          wintypes.UINT]
        user32.GetMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND,
                                       wintypes.UINT, wintypes.UINT]
        user32.PostThreadMessageW.argtypes = [wintypes.DWORD, wintypes.UINT, wintypes.WPARAM,
                                              wintypes.LPARAM]
        flags = {"Alt": 0x1, "Ctrl": 0x2, "Shift": 0x4, "Cmd": 0x8}
        mod = sum(flags[m] for m in self.mods) | 0x4000  # MOD_NOREPEAT
        vk = 0x70 + FKEYS.index(self.key) if self.key in FKEYS else ord(self.key)
        ready = threading.Event()
        state = {}

        def loop():
            state["tid"] = kernel32.GetCurrentThreadId()
            if not user32.RegisterHotKey(None, 1, mod, vk):
                self.error = "это сочетание уже занято другой программой"
                ready.set()
                return
            ready.set()
            msg = wintypes.MSG()
            while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
                if msg.message == 0x0312:  # WM_HOTKEY
                    self.presses += 1
            user32.UnregisterHotKey(None, 1)

        t = threading.Thread(target=loop, daemon=True)
        t.start()
        ready.wait(2)

        def stop():
            if "tid" in state:
                user32.PostThreadMessageW(state["tid"], 0x0012, 0, 0)  # WM_QUIT
            t.join(1)

        self._stop = stop

    def _start_mac(self):
        import ctypes
        from ctypes import c_uint32, c_void_p, POINTER, byref

        carbon = ctypes.CDLL("/System/Library/Frameworks/Carbon.framework/Carbon")

        class EventTypeSpec(ctypes.Structure):
            _fields_ = [("eventClass", c_uint32), ("eventKind", c_uint32)]

        class EventHotKeyID(ctypes.Structure):
            _fields_ = [("signature", c_uint32), ("id", c_uint32)]

        handler_t = ctypes.CFUNCTYPE(ctypes.c_int32, c_void_p, c_void_p, c_void_p)
        carbon.GetApplicationEventTarget.restype = c_void_p
        carbon.InstallEventHandler.argtypes = [c_void_p, handler_t, c_uint32,
                                               POINTER(EventTypeSpec), c_void_p,
                                               POINTER(c_void_p)]
        carbon.RegisterEventHotKey.argtypes = [c_uint32, c_uint32, EventHotKeyID, c_void_p,
                                               c_uint32, POINTER(c_void_p)]
        carbon.UnregisterEventHotKey.argtypes = [c_void_p]
        carbon.RemoveEventHandler.argtypes = [c_void_p]

        def on_hotkey(_call, _event, _data):
            self.presses += 1
            return 0

        self._handler = handler_t(on_hotkey)  # keep a reference: called from C
        target = carbon.GetApplicationEventTarget()
        spec = EventTypeSpec(int.from_bytes(b"keyb", "big"), 5)  # kEventHotKeyPressed
        handler_ref = c_void_p()
        status = carbon.InstallEventHandler(target, self._handler, 1, byref(spec), None,
                                            byref(handler_ref))
        if status != 0:
            raise OSError(f"InstallEventHandler {status}")
        flags = {"Cmd": 0x100, "Shift": 0x200, "Alt": 0x800, "Ctrl": 0x1000}
        mod = sum(flags[m] for m in self.mods)
        hk_ref = c_void_p()
        status = carbon.RegisterEventHotKey(MAC_KEYCODES[self.key], mod,
                                            EventHotKeyID(int.from_bytes(b"AuTu", "big"), 1),
                                            target, 0, byref(hk_ref))
        if status != 0:
            carbon.RemoveEventHandler(handler_ref)
            self.error = "это сочетание уже занято системой или другой программой"
            return

        def stop():
            carbon.UnregisterEventHotKey(hk_ref)
            carbon.RemoveEventHandler(handler_ref)

        self._stop = stop


def hotkey_from_tk_event(event, platform=None):
    """Read a shortcut from a Tk <KeyPress>; None for a lone modifier or unknown key."""
    platform = platform or sys.platform
    state = event.state
    mods = set()
    if state & 0x4:
        mods.add("Ctrl")
    if state & 0x1:
        mods.add("Shift")
    if platform == "win32":
        if state & 0x20000:
            mods.add("Alt")
    elif platform == "darwin":
        if state & 0x10:
            mods.add("Alt")
        if state & 0x8:
            mods.add("Cmd")
    else:
        if state & 0x8:
            mods.add("Alt")
        if state & 0x40:
            mods.add("Cmd")
    sym = event.keysym
    key = None
    if sym.upper() in FKEYS:
        key = sym.upper()
    elif len(sym) == 1 and sym.isalnum() and sym.isascii():
        key = sym.upper()
    elif platform == "darwin":
        # Option+letter types a special character; recover the physical key
        # (Tk on macOS keeps the virtual key code in the top byte)
        vk = (event.keycode >> 24) & 0xFF
        key = next((k for k, v in MAC_KEYCODES.items() if v == vk), None)
    if key is None:
        return None
    return mods, key


def play_toggle_sound(on):
    """Short cue in the user's own speakers (Discord does not hear it)."""
    import threading

    def run():
        try:
            if sys.platform == "win32":
                import winsound
                winsound.Beep(880 if on else 440, 90)
            elif sys.platform == "darwin":
                import subprocess
                name = "Pop" if on else "Bottle"
                subprocess.run(["afplay", f"/System/Library/Sounds/{name}.aiff"],
                               capture_output=True)
        except Exception:
            pass

    threading.Thread(target=run, daemon=True).start()


def asset_path(name):
    base = getattr(sys, "_MEIPASS", app_dir())
    return os.path.join(base, "assets", name)


def set_window_icon(root):
    try:
        if sys.platform == "win32":
            root.iconbitmap(asset_path("icon.ico"))
        else:
            import tkinter as tk
            root._icon = tk.PhotoImage(file=asset_path("icon.png"))
            root.iconphoto(True, root._icon)
    except Exception:
        pass  # running without the assets folder: keep the default icon


def run_gui(args):
    """Discord-styled settings window; every change applies immediately."""
    from tkinter import messagebox

    import customtkinter as ctk
    import sounddevice as sd

    ctk.set_appearance_mode("dark")
    root = ctk.CTk(fg_color=C_MAIN)
    root.title("Автотюн для Discord")
    set_window_icon(root)
    root.geometry("780x780")
    root.minsize(700, 520)

    family = {"win32": "Segoe UI", "darwin": "Helvetica Neue"}.get(sys.platform, "DejaVu Sans")

    def font(size, weight="normal"):
        return ctk.CTkFont(family=family, size=size, weight=weight)

    settings = load_settings()
    try:
        try:
            devices, api, default_in, inp, out, mon = pick_devices(sd, args, settings)
        except CableMissing:
            if sys.platform != "darwin" or bundled_mac_driver() is None:
                raise
            root.withdraw()
            if not messagebox.askyesno(
                    "Автотюн", "Чтобы Discord услышал автотюн, нужно один раз установить "
                    "виртуальный микрофон «Autotune Mic».\n\nmacOS попросит пароль "
                    "администратора. Установить?"):
                root.destroy()
                return
            msgs = []
            if not install_mac_driver(sd, say=msgs.append):
                messagebox.showerror("Автотюн", "\n".join(msgs) or "Не удалось установить.")
                root.destroy()
                return
            root.deiconify()
            devices, api, default_in, inp, out, mon = pick_devices(sd, args, settings)
    except CableMissing:
        root.withdraw()
        if sys.platform == "darwin":
            messagebox.showerror("Автотюн", "Не найден виртуальный микрофон. Установи BlackHole "
                                 "(existential.audio/blackhole) и запусти программу снова.")
        elif sys.platform == "win32" and messagebox.askyesno(
                "Автотюн", "Не найден виртуальный аудиокабель — без него Discord не услышит "
                "автотюн.\n\nСкачать и установить бесплатный VB-Audio Virtual Cable сейчас?"):
            msgs = []
            install_vb_cable(say=msgs.append)
            messagebox.showinfo("Автотюн", "\n".join(msgs))
        else:
            messagebox.showerror("Автотюн", "Установи VB-Audio Virtual Cable (vb-audio.com/Cable), "
                                 "перезагрузи компьютер и запусти программу снова.")
        root.destroy()
        return
    except LookupError as e:
        root.withdraw()
        messagebox.showerror("Автотюн", f"Ошибка: {e}")
        root.destroy()
        return

    eff = effective_settings(args, settings)
    engine = Engine(sd, args.samplerate, args.block, eff)
    tuner = engine.tuner

    mics = list_mics(devices, api)
    if inp is None:
        inp = default_in
    if inp not in mics:
        mics.insert(0, inp)
    mic_names = [devices[i]["name"] for i in mics]

    save_job = [None]

    def schedule_save():
        if save_job[0] is not None:
            root.after_cancel(save_job[0])
        save_job[0] = root.after(500, lambda: save_settings(settings))

    root.grid_columnconfigure(0, weight=1)
    root.grid_rowconfigure(0, weight=1)

    # ---------- main area ----------
    main = ctk.CTkFrame(root, corner_radius=0, fg_color=C_MAIN)
    main.grid(row=0, column=0, sticky="nsew")
    main.grid_columnconfigure(0, weight=1)
    main.grid_rowconfigure(2, weight=1)
    top = ctk.CTkFrame(main, height=48, corner_radius=0, fg_color=C_MAIN)
    top.grid(row=0, column=0, sticky="we")
    top.pack_propagate(False)
    ctk.CTkLabel(top, text="Автотюн", font=font(15, "bold"),
                 text_color=C_TEXT).pack(side="left", padx=(16, 0))
    ctk.CTkFrame(top, width=1, height=24, fg_color=C_SELECTED).pack(side="left", padx=12)
    ctk.CTkLabel(top, text="Все изменения применяются сразу, прямо во время звонка",
                 font=font(12), text_color=C_FAINT).pack(side="left")
    ctk.CTkFrame(main, height=1, corner_radius=0, fg_color=C_LINE).grid(row=1, column=0,
                                                                         sticky="we")

    body = ctk.CTkScrollableFrame(main, fg_color=C_MAIN, corner_radius=0,
                                  scrollbar_button_color=C_LINE)
    body.grid(row=2, column=0, sticky="nsew")
    body.grid_columnconfigure(0, weight=1)

    def card(title, row):
        frame = ctk.CTkFrame(body, fg_color=C_CARD, corner_radius=8)
        frame.grid(row=row, column=0, sticky="we", padx=16, pady=(16 if row == 0 else 0, 12))
        frame.grid_columnconfigure(1, weight=1)
        if title:
            ctk.CTkLabel(frame, text=title, font=font(11, "bold"), text_color=C_FAINT,
                         anchor="w").grid(row=0, column=0, columnspan=3, sticky="w",
                                          padx=16, pady=(12, 4))
        return frame

    # --- live card: note, level, master switch
    live = card(None, 0)
    note_lbl = ctk.CTkLabel(live, text="—", font=font(44, "bold"), text_color=C_TEXT,
                            width=120)
    note_lbl.grid(row=0, column=0, rowspan=2, padx=(16, 8), pady=14)
    info_lbl = ctk.CTkLabel(live, text="Спой что-нибудь", font=font(13), text_color=C_MUTED,
                            anchor="w")
    info_lbl.grid(row=0, column=1, sticky="sw", pady=(18, 2))
    meter = ctk.CTkProgressBar(live, height=8, corner_radius=4, fg_color=C_INPUT,
                               progress_color=C_GREEN)
    meter.grid(row=1, column=1, sticky="we", pady=(2, 18))
    meter.set(0)
    enabled_var = ctk.BooleanVar(value=engine.enabled)
    switch = ctk.CTkSwitch(live, text="Автотюн", variable=enabled_var, font=font(13, "bold"),
                           text_color=C_TEXT, progress_color=C_GREEN,
                           button_color="white", button_hover_color="#e3e5e8",
                           fg_color="#80848e", switch_width=44, switch_height=22)
    switch.grid(row=0, column=2, rowspan=2, padx=20)

    def set_enabled(value):
        enabled_var.set(value)
        engine.enabled = value
        settings["enabled"] = value
        schedule_save()

    switch.configure(command=lambda: set_enabled(enabled_var.get()))

    # --- keyboard shortcut (works while Discord is focused)
    hk_row = ctk.CTkFrame(live, fg_color="transparent")
    hk_row.grid(row=2, column=0, columnspan=3, sticky="w", padx=16, pady=(0, 14))
    ctk.CTkLabel(hk_row, text="Вкл/выкл с клавиатуры", font=font(12),
                 text_color=C_FAINT).pack(side="left")
    hk_btn = ctk.CTkButton(hk_row, text="", height=28, width=130, corner_radius=4,
                           fg_color=C_INPUT, hover_color=C_HOVER, text_color=C_TEXT,
                           font=font(12, "bold"))
    hk_btn.pack(side="left", padx=10)
    hk_status = ctk.CTkLabel(hk_row, text="", font=font(12), text_color=C_FAINT)
    hk_status.pack(side="left")
    hotkey = {"obj": None, "parsed": None, "seen": 0, "capturing": False}

    def apply_hotkey(text):
        if hotkey["obj"] is not None:
            hotkey["obj"].stop()
        parsed = parse_hotkey(text) or parse_hotkey(HOTKEY_DEFAULT)
        hk = GlobalHotkey(*parsed)
        ok = hk.start()
        hotkey.update(obj=hk, parsed=parsed, seen=0)
        hk_btn.configure(text=hotkey_text(*parsed))
        if ok:
            hk_status.configure(text="работает везде, даже когда открыт Discord",
                                text_color=C_FAINT)
        else:
            hk_status.configure(text=hk.error,
                                text_color=C_RED if "занято" in hk.error else C_FAINT)
        settings["hotkey"] = canonical_hotkey(*parsed)
        schedule_save()

    def toggle_by_hotkey():
        set_enabled(not engine.enabled)
        play_toggle_sound(engine.enabled)

    def start_capture():
        if hotkey["obj"] is not None:
            hotkey["obj"].stop()  # so pressing the old shortcut now does not toggle
        hotkey["capturing"] = True
        hk_btn.configure(text="Нажми сочетание…")
        hk_status.configure(text="например Ctrl+Alt+A или F9  ·  Esc — отмена",
                            text_color=C_FAINT)

    def on_key(event):
        pressed = hotkey_from_tk_event(event)
        if hotkey["capturing"]:
            if event.keysym == "Escape":
                hotkey["capturing"] = False
                apply_hotkey(settings.get("hotkey", HOTKEY_DEFAULT))
            elif pressed is not None:
                mods, key = pressed
                if not mods and key not in FKEYS:
                    hk_status.configure(text="добавь Ctrl, Alt или Shift", text_color=C_RED)
                else:
                    hotkey["capturing"] = False
                    apply_hotkey(canonical_hotkey(mods, key))
            return "break"
        hk = hotkey["obj"]
        if hk is not None and hk.error and pressed == hotkey["parsed"]:
            toggle_by_hotkey()  # no system-wide shortcut: at least work in our window
            return "break"
        return None

    hk_btn.configure(command=start_capture)
    root.bind_all("<KeyPress>", on_key)

    # --- presets
    pre = card("ПРЕСЕТЫ", 1)
    pre_grid = ctk.CTkFrame(pre, fg_color="transparent")
    pre_grid.grid(row=1, column=0, columnspan=3, sticky="we", padx=12, pady=(2, 12))
    for c in range(4):
        pre_grid.grid_columnconfigure(c, weight=1, uniform="presets")
    preset_buttons = {}

    def highlight_preset(name):
        for n, b in preset_buttons.items():
            sel = n == name
            b.configure(fg_color=C_BLURPLE if sel else C_INPUT,
                        hover_color=C_BLURPLE_HOVER if sel else C_HOVER,
                        text_color="white" if sel else C_MUTED)
        settings["preset"] = name
        schedule_save()

    # --- devices
    dev = card("УСТРОЙСТВА", 2)

    def option_menu(parent, values, command, width):
        return ctk.CTkOptionMenu(parent, values=values, command=command, width=width,
                                 height=36, corner_radius=4, font=font(13),
                                 fg_color=C_INPUT, button_color=C_INPUT,
                                 button_hover_color=C_HOVER, text_color=C_TEXT,
                                 dropdown_fg_color="#111214", dropdown_hover_color=C_BLURPLE,
                                 dropdown_text_color=C_TEXT, dropdown_font=font(13),
                                 dynamic_resizing=False)

    def open_mic(index):
        try:
            engine.start(index, out, mon, min(2, devices[out]["max_output_channels"]))
            settings["mic"] = devices[index]["name"]
            schedule_save()
        except Exception as e:
            messagebox.showerror("Автотюн", f"Не удалось открыть микрофон:\n{e}")

    ctk.CTkLabel(dev, text="Микрофон", font=font(13), text_color=C_MUTED,
                 anchor="w").grid(row=1, column=0, sticky="w", padx=16, pady=6)
    mic_menu = option_menu(dev, mic_names, lambda name: open_mic(mics[mic_names.index(name)]),
                           420)
    mic_menu.set(devices[inp]["name"])
    mic_menu.grid(row=1, column=1, sticky="w", padx=(0, 16), pady=6)
    hint = ctk.CTkFrame(dev, fg_color="transparent")
    hint.grid(row=2, column=0, columnspan=3, sticky="w", padx=16, pady=(4, 14))
    ctk.CTkLabel(hint, text="●", text_color=C_GREEN, font=font(12)).pack(side="left")
    ctk.CTkLabel(hint, text="  В Discord выбери микрофон:  ", font=font(13),
                 text_color=C_MUTED).pack(side="left")
    ctk.CTkLabel(hint, text=discord_device_name(devices[out]["name"]), font=font(13, "bold"),
                 text_color=C_TEXT).pack(side="left")

    # --- hear yourself in headphones
    lis = card("ПРОСЛУШИВАНИЕ", 3)
    phones = list_headphones(devices, api)
    phone_names = [devices[i]["name"] for i in phones]
    default_out = sd.default.device[1]
    phone_idx = None
    if settings.get("listen_device") in phone_names:
        phone_idx = phones[phone_names.index(settings["listen_device"])]
    elif default_out in phones:
        phone_idx = default_out
    elif phones:
        phone_idx = phones[0]

    listen_var = ctk.BooleanVar(value=bool(eff["listen"]) and phone_idx is not None)
    listen_switch = ctk.CTkSwitch(lis, text="Слушать себя", variable=listen_var,
                                  font=font(13, "bold"), text_color=C_TEXT,
                                  progress_color=C_GREEN, button_color="white",
                                  button_hover_color="#e3e5e8", fg_color="#80848e",
                                  switch_width=44, switch_height=22)
    listen_switch.grid(row=1, column=0, columnspan=2, sticky="w", padx=16, pady=6)
    listen_hint = ctk.CTkLabel(lis, text="звук с эффектом — как тебя слышат в Discord. "
                               "Только в наушниках, иначе будет эхо", font=font(11),
                               text_color=C_FAINT, anchor="w")
    listen_hint.grid(row=2, column=0, columnspan=3, sticky="w", padx=16)

    def apply_listen():
        on = listen_var.get() and phone_idx is not None
        settings["listen"] = bool(listen_var.get())
        schedule_save()
        try:
            if on:
                engine.set_listen(phone_idx, min(2, devices[phone_idx]["max_output_channels"]))
            else:
                engine.set_listen(None)
        except Exception as e:
            listen_var.set(False)
            settings["listen"] = False
            engine.set_listen(None)
            messagebox.showerror("Автотюн", f"Не удалось открыть наушники:\n{e}")

    def pick_phones(name):
        nonlocal phone_idx
        phone_idx = phones[phone_names.index(name)]
        settings["listen_device"] = name
        schedule_save()
        if listen_var.get():
            apply_listen()

    ctk.CTkLabel(lis, text="Наушники", font=font(13), text_color=C_MUTED,
                 anchor="w").grid(row=3, column=0, sticky="w", padx=16, pady=6)
    phone_menu = option_menu(lis, phone_names or ["нет устройств вывода"], pick_phones, 420)
    if phone_idx is not None:
        phone_menu.set(devices[phone_idx]["name"])
    else:
        phone_menu.configure(state="disabled")
        listen_switch.configure(state="disabled")
    phone_menu.grid(row=3, column=1, columnspan=2, sticky="w", padx=(0, 16), pady=6)

    ctk.CTkLabel(lis, text="Громкость", font=font(13), text_color=C_MUTED,
                 anchor="w").grid(row=4, column=0, sticky="w", padx=16, pady=(6, 14))
    listen_val = ctk.CTkLabel(lis, text=f"{eff['listen_volume'] * 100:.0f}%",
                              font=font(13, "bold"), text_color=C_TEXT, width=64, anchor="e")
    listen_val.grid(row=4, column=2, padx=(8, 16), pady=(6, 14))

    def on_listen_volume(value):
        engine.listen_volume = value / 100
        settings["listen_volume"] = value / 100
        listen_val.configure(text=f"{value:.0f}%")
        schedule_save()

    listen_slider = ctk.CTkSlider(lis, from_=0, to=100, number_of_steps=100, height=18,
                                  command=on_listen_volume, fg_color=C_INPUT,
                                  progress_color=C_BLURPLE, button_color="white",
                                  button_hover_color="#e3e5e8")
    listen_slider.set(eff["listen_volume"] * 100)
    listen_slider.grid(row=4, column=1, sticky="we", pady=(6, 14))
    lis.grid_columnconfigure(1, weight=1)
    listen_switch.configure(command=apply_listen)

    # --- key and scale
    ks = card("ТОНАЛЬНОСТЬ", 4)

    def on_scale(_=None):
        settings["key"] = key_seg.get()
        settings["scale"] = scale_keys[scale_names.index(scale_menu.get())]
        tuner.set_scale(parse_key(settings["key"]), settings["scale"])
        schedule_save()

    key_seg = ctk.CTkSegmentedButton(ks, values=NOTE_NAMES, command=on_scale, height=32,
                                     font=font(13, "bold"), corner_radius=4,
                                     fg_color=C_INPUT, unselected_color=C_INPUT,
                                     unselected_hover_color=C_HOVER,
                                     selected_color=C_BLURPLE,
                                     selected_hover_color=C_BLURPLE_HOVER,
                                     text_color=C_TEXT)
    key_seg.set(eff["key"])
    key_seg.grid(row=1, column=0, columnspan=3, sticky="we", padx=16, pady=6)
    ctk.CTkLabel(ks, text="Лад", font=font(13), text_color=C_MUTED,
                 anchor="w").grid(row=2, column=0, sticky="w", padx=16, pady=(6, 14))
    scale_keys = list(SCALE_LABELS)
    scale_names = [SCALE_LABELS[k] for k in scale_keys]
    scale_menu = option_menu(ks, scale_names, on_scale, 320)
    scale_menu.set(SCALE_LABELS[eff["scale"]])
    scale_menu.grid(row=2, column=1, sticky="w", padx=(0, 16), pady=(6, 14))

    # --- sound sliders
    snd = card("ЗВУК", 5)

    def set_speed(v): tuner.speed = v
    def set_transpose(v): tuner.transpose = v
    def set_mix(v): tuner.mix = v
    def set_talkbox(v): tuner.talkbox = v
    def set_gain(v): engine.gain = v
    def set_gate(v): tuner.gate_db = v

    def pct(v):
        return f"{v * 100:.0f}%"

    # (setting, title, hint, min, max, steps, to_ui, from_ui, fmt, apply)
    slider_defs = [
        ("speed", "Скорость подтяжки", "100% — роботизированный эффект", 0, 100, 100,
         lambda v: v * 100, lambda u: u / 100, pct, set_speed),
        ("talkbox", "Ток-бокс", "синтезатор, который «поёт» твоими гласными", 0, 100, 100,
         lambda v: v * 100, lambda u: u / 100, pct, set_talkbox),
        ("transpose", "Сдвиг высоты", "в полутонах, 12 = октава", -12, 12, 24,
         lambda v: v, lambda u: float(round(u)), lambda v: f"{v:+.0f}", set_transpose),
        ("mix", "Сила эффекта", "смесь с чистым голосом", 0, 100, 100,
         lambda v: v * 100, lambda u: u / 100, pct, set_mix),
        ("gain", "Громкость", "", 0, 200, 200,
         lambda v: v * 100, lambda u: u / 100, pct, set_gain),
        ("gate", "Порог шума", "тише этого — без обработки", -70, -20, 50,
         lambda v: v, lambda u: float(round(u)), lambda v: f"{v:.0f} дБ", set_gate),
    ]
    sliders = {}
    for r, (key, title, hint_text, lo, hi, steps, to_ui, from_ui, fmt, apply) in \
            enumerate(slider_defs, start=1):
        last = r == len(slider_defs)
        labels = ctk.CTkFrame(snd, fg_color="transparent")
        labels.grid(row=r, column=0, sticky="w", padx=16, pady=(6, 14 if last else 6))
        ctk.CTkLabel(labels, text=title, font=font(13), text_color=C_TEXT, anchor="w",
                     height=18).pack(anchor="w")
        if hint_text:
            ctk.CTkLabel(labels, text=hint_text, font=font(11), text_color=C_FAINT,
                         anchor="w", height=14).pack(anchor="w")
        val_lbl = ctk.CTkLabel(snd, text=fmt(eff[key]), font=font(13, "bold"),
                               text_color=C_TEXT, width=64, anchor="e")
        val_lbl.grid(row=r, column=2, padx=(8, 16), pady=(6, 14 if last else 6))

        def on_move(value, key=key, from_ui=from_ui, fmt=fmt, apply=apply, val_lbl=val_lbl,
                    manual=True):
            v = from_ui(value)
            apply(v)
            val_lbl.configure(text=fmt(v))
            settings[key] = v
            if manual:
                highlight_preset(None)
            schedule_save()

        slider = ctk.CTkSlider(snd, from_=lo, to=hi, number_of_steps=steps, height=18,
                               command=on_move, fg_color=C_INPUT, progress_color=C_BLURPLE,
                               button_color="white", button_hover_color="#e3e5e8")
        slider.set(to_ui(eff[key]))
        slider.grid(row=r, column=1, sticky="we", pady=(6, 14 if last else 6))
        sliders[key] = (slider, to_ui, on_move)

    def apply_preset(name):
        for k, v in PRESETS[name].items():
            slider, to_ui, on_move = sliders[k]
            slider.set(to_ui(v))
            on_move(slider.get(), manual=False)
        highlight_preset(None if name == "Сброс" else name)

    for i, name in enumerate(PRESETS):
        b = ctk.CTkButton(pre_grid, text=name, height=36, corner_radius=4, font=font(13),
                          command=lambda n=name: apply_preset(n))
        b.grid(row=i // 4, column=i % 4, sticky="we", padx=4, pady=4)
        preset_buttons[name] = b
    highlight_preset(settings.get("preset") if settings.get("preset") in PRESETS else None)

    # ---------- live updates ----------
    shown = {}

    def show(widget, **kw):
        if shown.get(widget) != kw:
            shown[widget] = kw
            widget.configure(**kw)

    def refresh():
        f0 = tuner.f0
        if not engine.enabled:
            show(note_lbl, text="—", text_color=C_FAINT)
            show(info_lbl, text="Автотюн выключен — голос идёт без обработки")
        elif f0 > 0:
            midi = 69 + 12 * np.log2(f0 / 440)
            n = int(round(midi + tuner.correction))
            name = f"{NOTE_NAMES[n % 12]}{n // 12 - 1}"
            show(note_lbl, text=name, text_color=C_TEXT)
            show(info_lbl, text=f"{f0:.0f} Гц   ·   подтяжка {tuner.correction:+.2f} пт")
        else:
            show(note_lbl, text="—", text_color=C_FAINT)
            show(info_lbl, text="Спой что-нибудь")
        meter.set(min(1.0, engine.level))
        hk = hotkey["obj"]
        if hk is not None and hk.presses != hotkey["seen"]:
            n = hk.presses - hotkey["seen"]
            hotkey["seen"] = hk.presses
            if n % 2:
                toggle_by_hotkey()
        root.after(80, refresh)

    def on_close():
        save_settings(settings)
        if hotkey["obj"] is not None:
            hotkey["obj"].stop()
        engine.stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    apply_hotkey(eff["hotkey"])
    set_enabled(engine.enabled)
    open_mic(inp)
    if listen_var.get():
        apply_listen()
    refresh()
    root.mainloop()


def main():
    # Russian messages must not crash on legacy Windows code pages (cp1252, cp866)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description="Автотюн для микрофона в реальном времени")
    p.add_argument("--list", action="store_true", help="показать аудиоустройства")
    p.add_argument("-i", "--input", type=lambda s: int(s) if s.isdigit() else s,
                   help="микрофон (номер или часть названия)")
    p.add_argument("--choose", action="store_true",
                   help="заново выбрать микрофон из списка")
    p.add_argument("-o", "--output", type=lambda s: int(s) if s.isdigit() else s,
                   help="выход: виртуальный кабель (например 'CABLE Input')")
    p.add_argument("-m", "--monitor", type=lambda s: int(s) if s.isdigit() else s,
                   help="(необязательно) наушники, чтобы слышать себя")
    p.add_argument("-k", "--key", type=parse_key,
                   help="тональность: C, C#, D, Eb, ... (по умолчанию C)")
    p.add_argument("-s", "--scale", choices=SCALES,
                   help="лад (по умолчанию chromatic — подходит к любой песне)")
    p.add_argument("--speed", type=float,
                   help="скорость подтяжки 0..1 (1 = роботизированный эффект)")
    p.add_argument("--transpose", type=float,
                   help="сдвиг в полутонах (например 12 или -5)")
    p.add_argument("--mix", type=float, help="0..1, доля обработанного звука")
    p.add_argument("--talkbox", type=float,
                   help="0..1, эффект ток-бокса (синтезатор, который «говорит» твоим голосом)")
    p.add_argument("--gain", type=float, help="громкость на выходе")
    p.add_argument("--gate", type=float,
                   help="порог тишины в дБ (ниже — без обработки)")
    p.add_argument("--nogui", action="store_true",
                   help="работать в консоли без окна настроек")
    p.add_argument("--samplerate", type=int, default=48000)
    p.add_argument("--block", type=int, default=256, help="размер блока (задержка)")
    p.add_argument("--print-driver-install", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--file", nargs=2, metavar=("IN.wav", "OUT.wav"),
                   help="обработать файл вместо живого звука (для проверки)")
    args = p.parse_args()

    if args.list:
        import sounddevice as sd
        print(sd.query_devices())
        return
    if args.print_driver_install:
        src = bundled_mac_driver()
        if src is None:
            sys.exit("driver not bundled")
        print(mac_driver_install_script(src))
        return
    if args.file:
        process_file(args)
        return
    if not args.nogui:
        try:
            import customtkinter  # noqa: F401
        except ImportError:
            print("Окно недоступно (нет customtkinter), работаю в консоли.")
        else:
            run_gui(args)
            return
    run_live(args)


if __name__ == "__main__":
    try:
        main()
    finally:
        # double-clicked .exe: keep the console open so errors are readable
        if getattr(sys, "frozen", False) and sys.stdin and sys.stdin.isatty():
            input("\nНажми Enter, чтобы закрыть окно...")
