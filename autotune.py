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


# virtual cables and system aliases are not real microphones
NOT_MICS = ["CABLE Output", "BlackHole", "Monitor of", "Mapper", "Переназначение",
            "Первичный драйвер", "Primary Sound Capture", "Stereo Mix", "Стерео микшер"]


def app_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


SETTINGS_PATH = os.path.join(app_dir(), "autotune_settings.json")


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


DEFAULTS = {"key": "C", "scale": "chromatic", "speed": 0.9, "transpose": 0.0,
            "mix": 1.0, "talkbox": 0.0, "gain": 1.0, "gate": -45.0,
            "enabled": True}


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
        self.level = 0.0
        self.stream = None
        self.monitor = None

    def callback(self, indata, outdata, frames, time, status):
        x = indata[:, 0]
        if self.enabled:
            y = self.tuner.process(x)
        else:
            y = x.astype(np.float64)
            self.tuner.f0 = 0.0
        y = np.clip(y * self.gain, -1, 1).astype(np.float32)
        self.level = float(np.abs(y).max()) if len(y) else 0.0
        outdata[:] = y[:, None]
        mon = self.monitor
        if mon is not None:
            try:
                mon.write(y[:, None].copy())
            except Exception:
                pass

    def start(self, inp, out, mon=None):
        self.stop()
        self.stream = self.sd.Stream(device=(inp, out), samplerate=self.sr,
                                     blocksize=self.block, channels=(1, 1),
                                     dtype="float32", latency="low", callback=self.callback)
        if mon is not None:
            m = self.sd.OutputStream(device=mon, samplerate=self.sr, channels=1,
                                     dtype="float32", latency="low", blocksize=self.block)
            m.start()
            self.monitor = m
        self.stream.start()

    def stop(self):
        for s in (self.stream, self.monitor):
            if s is not None:
                try:
                    s.stop()
                    s.close()
                except Exception:
                    pass
        self.stream = self.monitor = None


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
        engine.start(inp, out, mon)
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

# shown in the sidebar like Discord text channels
PRESETS = {
    "естественно": {"speed": 0.5, "transpose": 0.0, "mix": 1.0, "talkbox": 0.0},
    "t-pain": {"speed": 1.0, "transpose": 0.0, "mix": 1.0, "talkbox": 0.0},
    # Ye "All the Love" (Bully): hard-tuned vocal blended with a talk box
    "kanye-all-the-love": {"speed": 1.0, "transpose": 0.0, "mix": 1.0, "talkbox": 0.65},
    "чистый-ток-бокс": {"speed": 1.0, "transpose": 0.0, "mix": 1.0, "talkbox": 1.0},
    "бурундук": {"speed": 0.9, "transpose": 12.0, "mix": 1.0, "talkbox": 0.0},
    "демон": {"speed": 0.9, "transpose": -8.0, "mix": 1.0, "talkbox": 0.0},
    "сброс": {"speed": 0.9, "transpose": 0.0, "mix": 1.0, "talkbox": 0.0, "gain": 1.0,
              "gate": -45.0},
}

# Discord dark theme
C_RAIL = "#1e1f22"
C_SIDE = "#2b2d31"
C_MAIN = "#313338"
C_CARD = "#2b2d31"
C_PANEL = "#232428"
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


def run_gui(args):
    """Discord-styled settings window; every change applies immediately."""
    from tkinter import messagebox

    import customtkinter as ctk
    import sounddevice as sd

    ctk.set_appearance_mode("dark")
    root = ctk.CTk(fg_color=C_MAIN)
    root.title("Автотюн для Discord")
    root.geometry("1000x720")
    root.minsize(860, 560)

    family = "Segoe UI" if sys.platform == "win32" else "DejaVu Sans"

    def font(size, weight="normal"):
        return ctk.CTkFont(family=family, size=size, weight=weight)

    settings = load_settings()
    try:
        devices, api, default_in, inp, out, mon = pick_devices(sd, args, settings)
    except CableMissing:
        root.withdraw()
        if sys.platform == "win32" and messagebox.askyesno(
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

    root.grid_columnconfigure(2, weight=1)
    root.grid_rowconfigure(0, weight=1)

    # ---------- server rail ----------
    rail = ctk.CTkFrame(root, width=72, corner_radius=0, fg_color=C_RAIL)
    rail.grid(row=0, column=0, sticky="ns")
    rail.grid_propagate(False)
    ctk.CTkLabel(rail, text="AT", width=48, height=48, corner_radius=16, fg_color=C_BLURPLE,
                 text_color="white", font=font(16, "bold")).pack(pady=(12, 8))
    ctk.CTkFrame(rail, width=32, height=2, fg_color=C_SELECTED).pack()

    # ---------- channel sidebar ----------
    side = ctk.CTkFrame(root, width=240, corner_radius=0, fg_color=C_SIDE)
    side.grid(row=0, column=1, sticky="ns")
    side.grid_propagate(False)
    side.grid_rowconfigure(2, weight=1)
    side.grid_columnconfigure(0, weight=1)

    head = ctk.CTkFrame(side, height=48, corner_radius=0, fg_color=C_SIDE)
    head.grid(row=0, column=0, sticky="we")
    head.pack_propagate(False)
    ctk.CTkLabel(head, text="Автотюн", font=font(15, "bold"), text_color=C_TEXT,
                 anchor="w").pack(side="left", padx=16)
    ctk.CTkFrame(side, height=1, corner_radius=0, fg_color=C_RAIL).grid(row=1, column=0,
                                                                         sticky="nwe")

    channels = ctk.CTkFrame(side, fg_color="transparent")
    channels.grid(row=2, column=0, sticky="nwe", padx=8, pady=(16, 0))
    ctk.CTkLabel(channels, text="ПРЕСЕТЫ", font=font(11, "bold"), text_color=C_FAINT,
                 anchor="w").pack(fill="x", padx=8, pady=(0, 4))

    preset_buttons = {}

    def highlight_preset(name):
        for n, b in preset_buttons.items():
            sel = n == name
            b.configure(fg_color=C_SELECTED if sel else "transparent",
                        text_color=C_TEXT if sel else C_FAINT)
        settings["preset"] = name
        schedule_save()

    # ---------- user panel (bottom of sidebar) ----------
    panel = ctk.CTkFrame(side, height=56, corner_radius=0, fg_color=C_PANEL)
    panel.grid(row=3, column=0, sticky="we")
    panel.grid_propagate(False)
    panel.grid_columnconfigure(1, weight=1)
    avatar = ctk.CTkLabel(panel, text="♪", width=32, height=32, corner_radius=16,
                          fg_color=C_BLURPLE, text_color="white", font=font(16, "bold"))
    avatar.grid(row=0, column=0, rowspan=2, padx=(10, 8), pady=11)
    panel_title = ctk.CTkLabel(panel, text="—", font=font(13, "bold"), text_color=C_TEXT,
                               anchor="w", height=16)
    panel_title.grid(row=0, column=1, sticky="sw", pady=(10, 0))
    panel_status = ctk.CTkLabel(panel, text="", font=font(11), text_color=C_FAINT,
                                anchor="w", height=14)
    panel_status.grid(row=1, column=1, sticky="nw", pady=(0, 10))
    mute_btn = ctk.CTkButton(panel, text="🎙", width=32, height=32, corner_radius=4,
                             fg_color="transparent", hover_color=C_HOVER, font=font(16))
    mute_btn.grid(row=0, column=2, rowspan=2, padx=8)

    # ---------- main area ----------
    main = ctk.CTkFrame(root, corner_radius=0, fg_color=C_MAIN)
    main.grid(row=0, column=2, sticky="nsew")
    main.grid_columnconfigure(0, weight=1)
    main.grid_rowconfigure(2, weight=1)
    top = ctk.CTkFrame(main, height=48, corner_radius=0, fg_color=C_MAIN)
    top.grid(row=0, column=0, sticky="we")
    top.pack_propagate(False)
    ctk.CTkLabel(top, text="#", font=font(22), text_color=C_FAINT).pack(side="left",
                                                                        padx=(16, 6))
    ctk.CTkLabel(top, text="настройки-голоса", font=font(15, "bold"),
                 text_color=C_TEXT).pack(side="left")
    ctk.CTkFrame(top, width=1, height=24, fg_color=C_SELECTED).pack(side="left", padx=12)
    ctk.CTkLabel(top, text="Все изменения применяются сразу, прямо во время звонка",
                 font=font(12), text_color=C_FAINT).pack(side="left")
    ctk.CTkFrame(main, height=1, corner_radius=0, fg_color=C_RAIL).grid(row=1, column=0,
                                                                         sticky="we")

    body = ctk.CTkScrollableFrame(main, fg_color=C_MAIN, corner_radius=0,
                                  scrollbar_button_color=C_RAIL)
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
        mute_btn.configure(text_color=C_TEXT if value else C_RED)
        schedule_save()

    switch.configure(command=lambda: set_enabled(enabled_var.get()))
    mute_btn.configure(command=lambda: set_enabled(not engine.enabled))

    # --- devices
    dev = card("УСТРОЙСТВА", 1)

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
            engine.start(index, out, mon)
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

    # --- key and scale
    ks = card("ТОНАЛЬНОСТЬ", 2)

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
    snd = card("ЗВУК", 3)

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
        highlight_preset(None if name == "сброс" else name)

    for name in PRESETS:
        b = ctk.CTkButton(channels, text=f"#  {name}", anchor="w", height=34, corner_radius=4,
                          fg_color="transparent", hover_color=C_HOVER, text_color=C_FAINT,
                          font=font(14), command=lambda n=name: apply_preset(n))
        b.pack(fill="x", pady=1)
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
            show(panel_title, text="Без эффекта")
            show(panel_status, text="выключено", text_color=C_RED)
        elif f0 > 0:
            midi = 69 + 12 * np.log2(f0 / 440)
            n = int(round(midi + tuner.correction))
            name = f"{NOTE_NAMES[n % 12]}{n // 12 - 1}"
            show(note_lbl, text=name, text_color=C_TEXT)
            show(info_lbl, text=f"{f0:.0f} Гц   ·   подтяжка {tuner.correction:+.2f} пт")
            show(panel_title, text=name)
            show(panel_status, text="в эфире", text_color=C_GREEN)
        else:
            show(note_lbl, text="—", text_color=C_FAINT)
            show(info_lbl, text="Спой что-нибудь")
            show(panel_title, text="Автотюн")
            show(panel_status, text="в эфире", text_color=C_GREEN)
        meter.set(min(1.0, engine.level))
        root.after(80, refresh)

    def on_close():
        save_settings(settings)
        engine.stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    set_enabled(engine.enabled)
    open_mic(inp)
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
