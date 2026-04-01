"""
Утилиты для работы с аудиофайлами
"""

import numpy as np
import librosa
import soundfile as sf
import os
from config import SAMPLE_RATE, DURATION, AUDIO_DIR, RESULTS_DIR


def load_audio(file_path, sr=SAMPLE_RATE):

    try:
        signal, sample_rate = librosa.load(file_path, sr=sr)
        return signal, sample_rate
    except Exception as e:
        print(f"Ошибка загрузки файла {file_path}: {e}")
        return None, None


def save_audio(file_path, signal, sr=SAMPLE_RATE):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    sf.write(file_path, signal, sr)
    print(f"Сохранено: {file_path}")


def generate_synthetic_signal(genre, sr=SAMPLE_RATE, duration=DURATION):

    t = np.linspace(0, duration, int(sr * duration))

    if genre == 'Jazz':
        # Джаз: мягкие гармоники, свинг
        signal = (0.5 * np.sin(2 * np.pi * 440 * t) +
                  0.3 * np.sin(2 * np.pi * 880 * t) +
                  0.2 * np.sin(2 * np.pi * 1320 * t))
        vibrato = 1 + 0.05 * np.sin(2 * np.pi * 5 * t)
        signal = signal * vibrato

    elif genre == 'Rock':
        # Рок: искаженный, насыщенный
        signal = (0.4 * np.sin(2 * np.pi * 220 * t) +
                  0.3 * np.tanh(np.sin(2 * np.pi * 440 * t)) +
                  0.2 * np.sin(2 * np.pi * 880 * t))

    elif genre == 'Classical':
        # Классика: чистые тона
        signal = (0.6 * np.sin(2 * np.pi * 330 * t) +
                  0.3 * np.sin(2 * np.pi * 660 * t) +
                  0.2 * np.sin(2 * np.pi * 990 * t))

    elif genre == 'Electronic':
        # Электроника: синтезированные звуки
        signal = (0.4 * np.sin(2 * np.pi * 440 * t) +
                  0.2 * np.sin(2 * np.pi * 880 * t * 2) +
                  0.1 * np.sin(2 * np.pi * 1760 * t))
        sawtooth = 2 * (t * 440 % 1) - 1
        signal = signal + 0.2 * sawtooth

    elif genre == 'Voice':
        # Голос: форманты
        signal = (0.5 * np.sin(2 * np.pi * 200 * t) * np.exp(-t / 2) +
                  0.3 * np.sin(2 * np.pi * 800 * t) * np.exp(-t / 2) +
                  0.2 * np.sin(2 * np.pi * 1200 * t) * np.exp(-t / 2))
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 5 * t)
        signal = signal * envelope

    else:
        signal = np.sin(2 * np.pi * 440 * t)

    # Нормализация
    signal = signal / (np.max(np.abs(signal)) + 1e-10)

    return signal


def ensure_audio_files():
    os.makedirs(AUDIO_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    audio_files = {}

    # Музыкальные файлы
    for genre in ['Jazz', 'Rock', 'Classical', 'Electronic']:
        file_path = os.path.join(AUDIO_DIR, f'{genre.lower()}.wav')
        if os.path.exists(file_path):
            signal, sr = load_audio(file_path)
            print(f"✓ Загружен: {file_path}")
        else:
            print(f"Файл {file_path} не найден. Создаю синтетический сигнал для {genre}")
            signal = generate_synthetic_signal(genre)
            sr = SAMPLE_RATE
            save_audio(file_path, signal, sr)

        audio_files[genre] = {'signal': signal, 'sr': sr, 'path': file_path}

    # Голосовой файл
    voice_path = os.path.join(AUDIO_DIR, 'voice.wav')
    if os.path.exists(voice_path):
        signal, sr = load_audio(voice_path)
        print(f"✓ Загружен: {voice_path}")
    else:
        print(f"Файл {voice_path} не найден. Создаю синтетический голос")
        signal = generate_synthetic_signal('Voice')
        sr = SAMPLE_RATE
        save_audio(voice_path, signal, sr)

    audio_files['Voice'] = {'signal': signal, 'sr': sr, 'path': voice_path}

    # Шумовой файл
    noise_path = os.path.join(AUDIO_DIR, 'noise.wav')
    if os.path.exists(noise_path):
        signal, sr = load_audio(noise_path)
        print(f"✓ Загружен: {noise_path}")
    else:
        print(f"Файл {noise_path} не найден. Создаю белый шум")
        # Белый шум
        duration = 5
        signal = np.random.randn(int(SAMPLE_RATE * duration)) * 0.1
        sr = SAMPLE_RATE
        save_audio(noise_path, signal, sr)

    audio_files['Noise'] = {'signal': signal, 'sr': sr, 'path': noise_path}

    return audio_files