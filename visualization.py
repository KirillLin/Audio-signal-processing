"""
Визуализация результатов
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import os
from config import SAMPLE_RATE, N_FFT, HOP_LENGTH, COLORS


def setup_plotting_style():
    """Настройка стиля графиков"""
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 150
    plt.rcParams['font.size'] = 10


def plot_waveform(signal, sr, title, save_path=None):
    """Визуализация сигнала во времени"""
    plt.figure(figsize=(12, 4))
    time = np.arange(len(signal)) / sr
    plt.plot(time, signal, color='blue', linewidth=0.8)
    plt.xlabel('Время (с)')
    plt.ylabel('Амплитуда')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_spectrogram(signal, sr, title, save_path=None, mel=False):
    """Визуализация спектрограммы"""
    plt.figure(figsize=(12, 5))

    if mel:
        mel_spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128)
        spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        librosa.display.specshow(spec_db, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB', label='Уровень (дБ)')
    else:
        stft = librosa.stft(signal, n_fft=N_FFT, hop_length=HOP_LENGTH)
        spec_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        librosa.display.specshow(spec_db, sr=sr, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB', label='Уровень (дБ)')

    plt.xlabel('Время (с)')
    plt.ylabel('Частота (Гц)')
    plt.title(title)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_mel_spectrogram_comparison(signal, sr, title, save_path=None):
    """Сравнение самостоятельной и библиотечной мел-спектрограммы"""
    from spectrogram import mel_spectrogram_manual

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Самостоятельная реализация
    mel_manual, _ = mel_spectrogram_manual(signal, sr)
    im1 = axes[0].imshow(
        mel_manual,
        aspect='auto',
        origin='lower',
        extent=[0, len(signal) / sr, 0, mel_manual.shape[0]]
    )
    axes[0].set_xlabel('Время (с)')
    axes[0].set_ylabel('Мел-полоса')
    axes[0].set_title('Мел-спектрограмма (самостоятельная)')
    plt.colorbar(im1, ax=axes[0], label='дБ')

    # Библиотечная реализация
    mel_lib = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128)
    mel_lib_db = librosa.power_to_db(mel_lib, ref=np.max)
    im2 = librosa.display.specshow(mel_lib_db, sr=sr, x_axis='time', y_axis='mel', ax=axes[1])
    axes[1].set_title('Мел-спектрограмма (librosa)')
    plt.colorbar(im2, ax=axes[1], format='%+2.0f дБ')

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_spectral_features(signal, sr, title, save_path=None):
    """Визуализация спектральных признаков"""
    from spectral_features import (
        spectral_centroid_librosa,
        spectral_rolloff_librosa,
        spectral_bandwidth_librosa,
        zcr_librosa
    )

    # Вычисление признаков
    centroid = spectral_centroid_librosa(signal, sr)
    rolloff = spectral_rolloff_librosa(signal, sr)
    bandwidth = spectral_bandwidth_librosa(signal, sr)
    zcr = zcr_librosa(signal)

    fig, axes = plt.subplots(4, 1, figsize=(12, 10))

    # Спектрограмма
    ax1 = axes[0]
    stft = librosa.stft(signal, n_fft=N_FFT, hop_length=HOP_LENGTH)
    spec_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    librosa.display.specshow(spec_db, sr=sr, x_axis='time', y_axis='hz', ax=ax1)
    ax1.set_title('Спектрограмма')
    ax1.set_ylabel('Частота (Гц)')

    # Спектральный центроид
    ax2 = axes[1]
    time = np.arange(len(centroid)) * (HOP_LENGTH / sr)
    ax2.plot(time, centroid, color=COLORS['centroid'], linewidth=1.5)
    ax2.set_ylabel('Частота (Гц)')
    ax2.set_title('Спектральный центроид')
    ax2.grid(True, alpha=0.3)

    # Спектральный спад
    ax3 = axes[2]
    ax3.plot(time, rolloff, color=COLORS['rolloff'], linewidth=1.5)
    ax3.set_ylabel('Частота (Гц)')
    ax3.set_title('Спектральный спад (85%)')
    ax3.grid(True, alpha=0.3)

    # Спектральная ширина и ZCR
    ax4 = axes[3]
    ax4.plot(time, bandwidth, color=COLORS['bandwidth'], linewidth=1.5, label='Спектральная ширина')
    ax4.set_xlabel('Время (с)')
    ax4.set_ylabel('Частота (Гц)')
    ax4.set_title('Спектральная ширина')
    ax4.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_mfcc(signal, sr, title, save_path=None):
    """Визуализация MFCC"""
    from spectral_features import mfcc_librosa

    mfcc = mfcc_librosa(signal, sr)

    plt.figure(figsize=(12, 5))
    librosa.display.specshow(mfcc, sr=sr, x_axis='time', hop_length=HOP_LENGTH)
    plt.colorbar(label='Значение коэффициента')
    plt.xlabel('Время (с)')
    plt.ylabel('MFCC коэффициент')
    plt.title(f'MFCC - {title}')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_chroma(signal, sr, title, save_path=None):
    """Визуализация цветности"""
    from spectral_features import chroma_librosa

    chroma = chroma_librosa(signal, sr)
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    plt.figure(figsize=(12, 5))
    librosa.display.specshow(chroma, sr=sr, x_axis='time', y_axis='chroma', hop_length=HOP_LENGTH)
    plt.colorbar(label='Нормированная энергия')
    plt.xlabel('Время (с)')
    plt.ylabel('Нота')
    plt.yticks(np.arange(12), note_names)
    plt.title(f'Цветность - {title}')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_quality_comparison(results, save_path=None):
    """Визуализация сравнения метрик качества"""
    from quality_metrics import snr_manual, sdr_manual, si_sdr_manual

    snr_targets = []
    snr_noisy = []
    snr_enhanced = []
    sdr_noisy = []
    sdr_enhanced = []
    si_sdr_noisy = []
    si_sdr_enhanced = []

    for r in results:
        snr_targets.append(r['snr_target'])

        # Вычисление метрик
        snr_n = snr_manual(r['original'], r['noisy'])
        snr_e = snr_manual(r['original'], r['enhanced'])
        sdr_n = sdr_manual(r['original'], r['noisy'])
        sdr_e = sdr_manual(r['original'], r['enhanced'])
        si_sdr_n = si_sdr_manual(r['original'], r['noisy'])
        si_sdr_e = si_sdr_manual(r['original'], r['enhanced'])

        snr_noisy.append(snr_n)
        snr_enhanced.append(snr_e)
        sdr_noisy.append(sdr_n)
        sdr_enhanced.append(sdr_e)
        si_sdr_noisy.append(si_sdr_n)
        si_sdr_enhanced.append(si_sdr_e)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # SNR
    axes[0].plot(snr_targets, snr_noisy, 'o-', color=COLORS['noisy'],
                 label='Зашумленный', linewidth=2, markersize=8)
    axes[0].plot(snr_targets, snr_enhanced, 's-', color=COLORS['enhanced'],
                 label='Очищенный', linewidth=2, markersize=8)
    axes[0].plot(snr_targets, snr_targets, '--', color=COLORS['ideal'],
                 label='Идеальный', alpha=0.5)
    axes[0].set_xlabel('Целевой SNR (дБ)')
    axes[0].set_ylabel('Измеренный SNR (дБ)')
    axes[0].set_title('SNR')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # SDR
    axes[1].plot(snr_targets, sdr_noisy, 'o-', color=COLORS['noisy'],
                 label='Зашумленный', linewidth=2, markersize=8)
    axes[1].plot(snr_targets, sdr_enhanced, 's-', color=COLORS['enhanced'],
                 label='Очищенный', linewidth=2, markersize=8)
    axes[1].set_xlabel('Целевой SNR (дБ)')
    axes[1].set_ylabel('SDR (дБ)')
    axes[1].set_title('SDR')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # SI-SDR
    axes[2].plot(snr_targets, si_sdr_noisy, 'o-', color=COLORS['noisy'],
                 label='Зашумленный', linewidth=2, markersize=8)
    axes[2].plot(snr_targets, si_sdr_enhanced, 's-', color=COLORS['enhanced'],
                 label='Очищенный', linewidth=2, markersize=8)
    axes[2].set_xlabel('Целевой SNR (дБ)')
    axes[2].set_ylabel('SI-SDR (дБ)')
    axes[2].set_title('SI-SDR')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()