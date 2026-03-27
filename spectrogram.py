"""
Реализация спектрограммы и мел-спектрограммы
"""

import numpy as np
from scipy.fft import fft
import librosa
from config import SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS


def stft_manual(signal, n_fft=N_FFT, hop_length=HOP_LENGTH, window='hann'):
    """
    Самостоятельная реализация STFT (Short-Time Fourier Transform)

    Parameters:
    - signal: входной сигнал
    - n_fft: размер окна FFT
    - hop_length: шаг между окнами
    - window: тип окна ('hann' или 'rect')

    Returns:
    - stft_matrix: матрица STFT (частота x время)
    """
    # Создание окна
    if window == 'hann':
        win = np.hanning(n_fft)
    else:
        win = np.ones(n_fft)

    # Количество кадров
    n_frames = 1 + (len(signal) - n_fft) // hop_length

    # Инициализация матрицы STFT (только положительные частоты)
    stft_matrix = np.zeros((n_fft // 2 + 1, n_frames), dtype=complex)

    for i in range(n_frames):
        start = i * hop_length
        frame = signal[start:start + n_fft] * win
        fft_result = fft(frame)
        stft_matrix[:, i] = fft_result[:n_fft // 2 + 1]

    return stft_matrix


def create_mel_filterbank_manual(n_fft=N_FFT, sr=SAMPLE_RATE, n_mels=N_MELS, fmin=0, fmax=None):
    """
    Самостоятельное создание банка мел-фильтров

    Parameters:
    - n_fft: размер FFT
    - sr: частота дискретизации
    - n_mels: количество мел-фильтров
    - fmin: минимальная частота (Гц)
    - fmax: максимальная частота (Гц)

    Returns:
    - mel_filters: матрица мел-фильтров
    """
    if fmax is None:
        fmax = sr / 2

    # Преобразование герц в мелы
    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    def mel_to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    # Границы в мелах
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)

    # Равномерно распределенные точки в мелах
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    # Частотные бины FFT
    fft_freqs = np.linspace(0, sr, n_fft + 1)[:n_fft // 2 + 1]

    # Создание треугольных фильтров
    filters = np.zeros((n_mels, len(fft_freqs)))

    for i in range(n_mels):
        left = hz_points[i]
        center = hz_points[i + 1]
        right = hz_points[i + 2]

        for j, freq in enumerate(fft_freqs):
            if freq < left or freq > right:
                filters[i, j] = 0
            elif freq <= center:
                filters[i, j] = (freq - left) / (center - left)
            else:
                filters[i, j] = (right - freq) / (right - center)

    # Нормализация
    energies = np.sum(filters, axis=1)
    filters = filters / energies[:, np.newaxis]

    return filters


def mel_spectrogram_manual(signal, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS):
    """
    Самостоятельная реализация мел-спектрограммы

    Parameters:
    - signal: входной сигнал
    - sr: частота дискретизации
    - n_fft: размер FFT
    - hop_length: шаг между окнами
    - n_mels: количество мел-фильтров

    Returns:
    - mel_spec: мел-спектрограмма
    """
    # STFT
    stft_matrix = stft_manual(signal, n_fft, hop_length)

    # Спектр мощности
    power_spectrum = np.abs(stft_matrix) ** 2

    # Банк мел-фильтров
    mel_filters = create_mel_filterbank_manual(n_fft, sr, n_mels)

    # Применение фильтров
    mel_spec = np.dot(mel_filters, power_spectrum)

    # Логарифмическое масштабирование (дБ)
    mel_spec_db = 10 * np.log10(mel_spec + 1e-10)

    return mel_spec_db, mel_filters


def mel_spectrogram_librosa(signal, sr=SAMPLE_RATE, n_mels=N_MELS):
    """
    Библиотечная реализация мел-спектрограммы (librosa)
    """
    mel_spec = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_mels=n_mels,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db


def spectrogram_librosa(signal, sr=SAMPLE_RATE):
    """
    Библиотечная реализация спектрограммы
    """
    stft = librosa.stft(signal, n_fft=N_FFT, hop_length=HOP_LENGTH)
    spec_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    return spec_db