"""
Обработка шума: добавление шума и шумоподавление
"""

import numpy as np
import librosa
from config import SAMPLE_RATE, SNR_VALUES


def add_noise_with_snr(clean_signal, noise, snr_db):
    """
    Добавление шума к сигналу с заданным SNR

    Parameters:
    - clean_signal: чистый сигнал
    - noise: шумовой сигнал
    - snr_db: желаемое SNR в дБ

    Returns:
    - noisy_signal: зашумленный сигнал
    - actual_snr: фактическое SNR
    """
    # Выравнивание длины
    if len(noise) < len(clean_signal):
        repeats = int(np.ceil(len(clean_signal) / len(noise)))
        noise = np.tile(noise, repeats)[:len(clean_signal)]
    else:
        noise = noise[:len(clean_signal)]

    # Вычисление мощностей
    power_signal = np.mean(clean_signal ** 2)
    power_noise = np.mean(noise ** 2)

    if power_noise == 0:
        power_noise = 1e-10

    # Вычисление коэффициента масштабирования
    # SNR = 10 * log10(P_signal / (alpha^2 * P_noise))
    # alpha = sqrt(P_signal / (P_noise * 10^(SNR/10)))
    target_power_noise = power_signal / (10 ** (snr_db / 10))
    scale_factor = np.sqrt(target_power_noise / (power_noise + 1e-10))

    scaled_noise = noise * scale_factor
    noisy_signal = clean_signal + scaled_noise

    # Вычисление фактического SNR
    actual_snr = 10 * np.log10(power_signal / (np.mean(scaled_noise ** 2) + 1e-10))

    return noisy_signal, actual_snr


def denoise_deepfilternet(noisy_signal, sr=SAMPLE_RATE):
    """
    Шумоподавление с помощью DeepFilterNet2

    DeepFilterNet2 - нейросетевая модель для подавления шума

    Parameters:
    - noisy_signal: зашумленный сигнал
    - sr: частота дискретизации

    Returns:
    - enhanced: очищенный сигнал
    """
    try:
        from df import enhance

        # DeepFilterNet2 работает с частотой 48 кГц
        if sr != 48000:
            noisy_resampled = librosa.resample(noisy_signal, orig_sr=sr, target_sr=48000)
            enhanced_resampled = enhance(noisy_resampled, sr=48000)
            enhanced = librosa.resample(enhanced_resampled, orig_sr=48000, target_sr=sr)
        else:
            enhanced = enhance(noisy_signal, sr=sr)

        return enhanced

    except ImportError:
        print("⚠️ DeepFilterNet2 не установлен. Установите: pip install deepfilternet")
        return noisy_signal
    except Exception as e:
        print(f"⚠️ Ошибка DeepFilterNet2: {e}")
        return noisy_signal


def generate_noise_signal(sr=SAMPLE_RATE, duration=5, noise_type='white'):
    """
    Генерация шумового сигнала

    Parameters:
    - sr: частота дискретизации
    - duration: длительность в секундах
    - noise_type: тип шума ('white', 'pink', 'brown')

    Returns:
    - noise: шумовой сигнал
    """
    n_samples = int(sr * duration)

    if noise_type == 'white':
        # Белый шум (равномерная спектральная плотность)
        noise = np.random.randn(n_samples) * 0.1

    elif noise_type == 'pink':
        # Розовый шум (1/f спектр)
        white = np.random.randn(n_samples)
        # Фильтр для создания 1/f спектра
        pink = np.fft.ifft(np.fft.fft(white) / np.sqrt(np.arange(1, n_samples + 1))).real
        noise = pink / (np.max(np.abs(pink)) + 1e-10) * 0.1

    elif noise_type == 'brown':
        # Коричневый шум (1/f^2 спектр)
        white = np.random.randn(n_samples)
        # Интегрирование
        brown = np.cumsum(white)
        brown = brown / (np.max(np.abs(brown)) + 1e-10) * 0.1
        noise = brown

    else:
        noise = np.random.randn(n_samples) * 0.1

    return noise


def batch_noise_experiment(clean_signal, noise, snr_values, sr=SAMPLE_RATE):
    """
    Проведение серии экспериментов с разными SNR

    Parameters:
    - clean_signal: чистый сигнал
    - noise: шумовой сигнал
    - snr_values: список значений SNR
    - sr: частота дискретизации

    Returns:
    - results: список результатов
    """
    results = []

    for snr_target in snr_values:
        print(f"\n--- SNR целевой: {snr_target} дБ ---")

        # Добавление шума
        noisy_signal, actual_snr = add_noise_with_snr(clean_signal, noise, snr_target)
        print(f"  Фактическое SNR: {actual_snr:.2f} дБ")

        # Шумоподавление
        enhanced_signal = denoise_deepfilternet(noisy_signal, sr)

        results.append({
            'snr_target': snr_target,
            'actual_snr': actual_snr,
            'noisy': noisy_signal,
            'enhanced': enhanced_signal
        })

    return results