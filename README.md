# Автотюн для микрофона в Discord

Утилита забирает звук с микрофона, подтягивает голос к ближайшей ноте
и отправляет его в виртуальный аудиокабель. Discord слушает этот кабель
как обычный микрофон.

```
Микрофон -> autotune.py -> виртуальный кабель -> Discord
```

## 1. Установка

1. Установи [Python 3.9+](https://www.python.org/downloads/). На Windows при установке включи галочку «Add to PATH».
2. В папке проекта выполни: `pip install -r requirements.txt`
3. Поставь виртуальный аудиокабель:
   - **Windows**: [VB-Audio Virtual Cable](https://vb-audio.com/Cable/) (бесплатный). После установки перезагрузи компьютер.
   - **macOS**: [BlackHole 2ch](https://github.com/ExistentialAudio/BlackHole).
   - **Linux (PipeWire/PulseAudio)**: `pactl load-module module-null-sink sink_name=autotune`
     (в Discord выбирай «Monitor of autotune»).

## 2. Запуск

```bash
python autotune.py --list                      # список устройств
python autotune.py -o "CABLE Input"            # Windows
python autotune.py -o "BlackHole"              # macOS
python autotune.py -i "USB Mic" -o "CABLE Input" -m "Headphones"   # указать микрофон и слушать себя
```

На Windows можно просто дважды кликнуть `start_windows.bat`.

## 3. Настройка Discord

Настройки → Голос и видео:
- **Устройство ввода**: `CABLE Output (VB-Audio Virtual Cable)` (macOS: `BlackHole 2ch`).
- **Выключи** шумоподавление (Krisp), эхоподавление и автоматическую регулировку
  усиления. Они «съедают» эффект и могут резать голос.
- Режим ввода — «Голосовая активация» или «Push-to-talk», как тебе удобнее.

## Параметры

| Параметр | Что делает |
|---|---|
| `-k A` | тональность (C, C#, D, Eb, …) |
| `-s major` | лад: `chromatic` (по умолчанию, подходит к любой песне), `major`, `minor`, `pentatonic`, `minor-pentatonic`, `blues` |
| `--speed 1.0` | скорость подтяжки: `1.0` — роботизированный эффект в стиле T-Pain, `0.5` — мягкая естественная коррекция |
| `--transpose 12` | сдвиг на N полутонов (12 = октава вверх, -5 = ниже) |
| `--mix 0.7` | смесь обработанного и чистого голоса |
| `--gain 1.5` | громкость на выходе |
| `--gate -40` | порог тишины в дБ (выше — меньше шума в паузах) |
| `--block 128` | меньше задержка, но выше нагрузка на процессор |
| `--file in.wav out.wav` | обработать файл (16-бит WAV), удобно для проверки и для видео |

Пример «как в треках»: `python autotune.py -o "CABLE Input" -k A -s minor --speed 1.0`

## Если что-то не работает

- **Ошибка при открытии устройств**: вход и выход должны быть из одного Host API
  (в `--list` указано справа, например `MME` или `Windows WASAPI`). Укажи номера: `-i 1 -o 5`.
- **Треск или прерывания**: увеличь `--block 512`.
- **Робот срабатывает на шум**: подними `--gate -38`.
