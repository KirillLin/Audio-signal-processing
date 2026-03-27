"""
Спектральные признаки аудиосигнала
"""

import numpy as np
import librosa
from scipy.fft import fft
from config import SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS, N_MFCC
from spectrogram import stft_manual, mel_spectrogram_manual


# ============================================================================
# СПЕКТРАЛЬНЫЙ ЦЕНТРОИД
# ============================================================================

def spectral_centroid_manual(stft_matrix, sr=SAMPLE_RATE):
    """
    Самостоятельная реализация спектрального центроида

    Спектральный центроид - средневзвешенная частота спектра,
    указывает на "яркость" звука. Чем выше значение, тем ярче звук.

    Parameters:
    - stft_matrix: матрица STFT
    - sr: частота дискретизации

    Returns:
    - centroids: массив центроидов для каждого кадра
    """
    freqs = np.linspace(0, sr / 2, stft_matrix.shape[0])
    n_frames = stft_matrix.shape[1]
    centroids = np.zeros(n_frames)

    for i in range(n_frames):
        spectrum = np.abs(stft_matrix[:, i])
        total_energy = np.sum(spectrum)

        if total_energy > 0:
            centroids[i] = np.sum(freqs * spectrum) / total_energy
        else:
            centroids[i] = 0

    return centroids


def spectral_centroid_librosa(signal, sr=SAMPLE_RATE):
    """Библиотечная реализация спектрального центроида"""
    centroid = librosa.feature.spectral_centroid(
        y=signal,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    return centroid[0]


# ============================================================================
# СПЕКТРАЛЬНЫЙ СПАД
# ============================================================================

def spectral_rolloff_manual(stft_matrix, sr=SAMPLE_RATE, roll_percent=0.85):
    """
    Самостоятельная реализация спектрального спада

    Спектральный спад - частота, ниже которой сосредоточено roll_percent энергии

    Parameters:
    - stft_matrix: матрица STFT
    - sr: частота дискретизации
    - roll_percent: процент энергии (0.85 = 85%)

    Returns:
    - rolloffs: массив частот спада для каждого кадра
    """
    freqs = np.linspace(0, sr / 2, stft_matrix.shape[0])
    n_frames = stft_matrix.shape[1]
    rolloffs = np.zeros(n_frames)

    for i in range(n_frames):
        spectrum = np.abs(stft_matrix[:, i])
        total_energy = np.sum(spectrum)

        if total_energy > 0:
            threshold = roll_percent * total_energy
            cumulative = 0
            for j, energy in enumerate(spectrum):
                cumulative += energy
                if cumulative >= threshold:
                    rolloffs[i] = freqs[j]
                    break
        else:
            rolloffs[i] = 0

    return rolloffs


def spectral_rolloff_librosa(signal, sr=SAMPLE_RATE, roll_percent=0.85):
    """Библиотечная реализация спектрального спада"""
    rolloff = librosa.feature.spectral_rolloff(
        y=signal,
        sr=sr,
        roll_percent=roll_percent,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    return rolloff[0]


# ============================================================================
# СПЕКТРАЛЬНАЯ ШИРИНА
# ============================================================================

def spectral_bandwidth_manual(stft_matrix, sr=SAMPLE_RATE):
    """
    Самостоятельная реализация спектральной ширины

    Спектральная ширина - взвешенное стандартное отклонение частот
    от спектрального центроида. Характеризует "ширину" спектра.

    Parameters:
    - stft_matrix: матрица STFT
    - sr: частота дискретизации

    Returns:
    - bandwidth: массив спектральной ширины для каждого кадра
    """
    freqs = np.linspace(0, sr / 2, stft_matrix.shape[0])
    n_frames = stft_matrix.shape[1]
    bandwidth = np.zeros(n_frames)

    for i in range(n_frames):
        spectrum = np.abs(stft_matrix[:, i])
        total_energy = np.sum(spectrum)

        if total_energy > 0:
            centroid = np.sum(freqs * spectrum) / total_energy
            variance = np.sum(((freqs - centroid) ** 2) * spectrum) / total_energy
            bandwidth[i] = np.sqrt(variance)
        else:
            bandwidth[i] = 0

    return bandwidth


def spectral_bandwidth_librosa(signal, sr=SAMPLE_RATE):
    """Библиотечная реализация спектральной ширины"""
    bandwidth = librosa.feature.spectral_bandwidth(
        y=signal,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    return bandwidth[0]


# ============================================================================
# ЧАСТОТА ПЕРЕСЕЧЕНИЯ НУЛЯ
# ============================================================================

def zcr_manual(signal, frame_length=512, hop_length=256):
    """
    Самостоятельная реализация частоты пересечения нуля (ZCR)

    ZCR - количество пересечений нуля в сигнале за единицу времени.
    Высокое значение характерно для шумных сигналов, низкое - для тональных.

    Parameters:
    - signal: входной сигнал
    - frame_length: длина кадра
    - hop_length: шаг между кадрами

    Returns:
    - zcr: массив значений ZCR для каждого кадра
    """
    n_frames = 1 + (len(signal) - frame_length) // hop_length
    zcr = np.zeros(n_frames)

    for i in range(n_frames):
        start = i * hop_length
        frame = signal[start:start + frame_length]
        # Подсчет пересечений нуля
        zero_crossings = np.sum(np.abs(np.diff(np.sign(frame))) > 0)
        zcr[i] = zero_crossings / frame_length

    return zcr


def zcr_librosa(signal, frame_length=512, hop_length=256):
    """Библиотечная реализация частоты пересечения нуля"""
    zcr = librosa.feature.zero_crossing_rate(
        y=signal,
        frame_length=frame_length,
        hop_length=hop_length
    )
    return zcr[0]


# ============================================================================
# MFCC (Мел-частотные кепстральные коэффициенты)
# ============================================================================

def mfcc_manual(signal, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_mels=N_MELS):
    """
    Самостоятельная реализация MFCC

    MFCC - коэффициенты, описывающие тембр звука.
    Широко используются в распознавании речи и классификации музыки.

    Parameters:
    - signal: входной сигнал
    - sr: частота дискретизации
    - n_mfcc: количество MFCC коэффициентов
    - n_mels: количество мел-фильтров

    Returns:
    - mfccs: матрица MFCC (n_mfcc x время)
    """
    # Получаем мел-спектрограмму
    mel_spec, _ = mel_spectrogram_manual(signal, sr, n_mels=n_mels)

    n_frames = mel_spec.shape[1]
    mfccs = np.zeros((n_mfcc, n_frames))

    # DCT-II преобразование
    for i in range(n_frames):
        for k in range(n_mfcc):
            mfccs[k, i] = np.sum(
                mel_spec[:, i] * np.cos(np.pi * k * (np.arange(n_mels) + 0.5) / n_mels)
            )

    return mfccs


def mfcc_librosa(signal, sr=SAMPLE_RATE, n_mfcc=N_MFCC):
    """Библиотечная реализация MFCC"""
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    return mfcc


# ============================================================================
# ЦВЕТНОСТЬ (CHROMA)
# ============================================================================

def chroma_manual(signal, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """
    Самостоятельная реализация цветности (Chroma)

    Цветность - распределение энергии по 12 музыкальным нотам.
    Используется для гармонического анализа музыки.

    Parameters:
    - signal: входной сигнал
    - sr: частота дискретизации
    - n_fft: размер FFT
    - hop_length: шаг между окнами

    Returns:
    - chroma: матрица цветности (12 x время)
    - note_names: названия нот
    """
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    # Частоты нот (A4 = 440 Hz)
    a4_freq = 440
    note_freqs = []
    for i in range(-48, 48):  # 4 октавы вверх и вниз
        note_freqs.append(a4_freq * (2 ** (i / 12)))

    # STFT
    stft_matrix = stft_manual(signal, n_fft, hop_length)
    freqs = np.linspace(0, sr / 2, stft_matrix.shape[0])
    n_frames = stft_matrix.shape[1]

    # Инициализация хромы
    chroma = np.zeros((12, n_frames))

    for i in range(n_frames):
        spectrum = np.abs(stft_matrix[:, i])

        # Для каждой ноты суммируем энергию во всех октавах
        for note_idx in range(12):
            note_energy = 0
            for freq in note_freqs:
                if freq > sr / 2:
                    continue
                bin_idx = np.argmin(np.abs(freqs - freq))
                if bin_idx < len(spectrum):
                    note_in_octave = int(round(12 * np.log2(freq / a4_freq))) % 12
                    if note_in_octave == note_idx:
                        note_energy += spectrum[bin_idx] ** 2

            chroma[note_idx, i] = np.sqrt(note_energy)

    # Нормализация
    for i in range(n_frames):
        total = np.sum(chroma[:, i])
        if total > 0:
            chroma[:, i] = chroma[:, i] / total

    return chroma, note_names


def chroma_librosa(signal, sr=SAMPLE_RATE):
    """Библиотечная реализация цветности"""
    chroma = librosa.feature.chroma_stft(
        y=signal,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    return chroma