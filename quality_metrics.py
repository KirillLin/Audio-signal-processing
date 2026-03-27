"""
Метрики качества звука
"""

import numpy as np
import librosa
from config import SAMPLE_RATE, PESQ_TARGET_SR, STOI_TARGET_SR


# ============================================================================
# SNR (Signal-to-Noise Ratio)
# ============================================================================

def snr_manual(original, processed):
    """
    Самостоятельная реализация SNR (Signal-to-Noise Ratio)

    SNR = 10 * log10(P_signal / P_noise)
    где P_noise = mean((original - processed)^2)

    Parameters:
    - original: исходный (чистый) сигнал
    - processed: обработанный сигнал

    Returns:
    - snr: значение SNR в дБ
    """
    # Выравнивание длины
    min_len = min(len(original), len(processed))
    original = original[:min_len]
    processed = processed[:min_len]

    # Мощность сигнала
    power_signal = np.mean(original ** 2)

    # Мощность шума (искажения)
    power_noise = np.mean((original - processed) ** 2)

    if power_noise == 0:
        return np.inf

    snr = 10 * np.log10(power_signal / power_noise)
    return snr


# ============================================================================
# SDR (Signal-to-Distortion Ratio)
# ============================================================================

def sdr_manual(original, processed):
    """
    Самостоятельная реализация SDR (Signal-to-Distortion Ratio)

    SDR = 10 * log10(||original||^2 / ||original - processed||^2)

    Parameters:
    - original: исходный сигнал
    - processed: обработанный сигнал

    Returns:
    - sdr: значение SDR в дБ
    """
    min_len = min(len(original), len(processed))
    original = original[:min_len]
    processed = processed[:min_len]

    # Энергия сигнала
    energy_signal = np.sum(original ** 2)

    # Энергия искажения
    energy_distortion = np.sum((original - processed) ** 2)

    if energy_distortion == 0:
        return np.inf

    sdr = 10 * np.log10(energy_signal / energy_distortion)
    return sdr


# ============================================================================
# SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)
# ============================================================================

def si_sdr_manual(original, processed):
    """
    Самостоятельная реализация SI-SDR (Scale-Invariant SDR)

    SI-SDR = 10 * log10(||alpha * target||^2 / ||alpha * target - processed||^2)
    где alpha = (target^T * processed) / ||target||^2

    Эта метрика инвариантна к масштабированию сигнала.

    Parameters:
    - original: исходный сигнал
    - processed: обработанный сигнал

    Returns:
    - si_sdr: значение SI-SDR в дБ
    """
    min_len = min(len(original), len(processed))
    original = original[:min_len]
    processed = processed[:min_len]

    # Вычисление коэффициента масштабирования
    alpha = np.dot(original, processed) / (np.dot(original, original) + 1e-10)

    # Масштабированная версия исходного сигнала
    scaled_target = alpha * original

    # Энергия масштабированного сигнала
    energy_target = np.sum(scaled_target ** 2)

    # Энергия искажения
    energy_distortion = np.sum((scaled_target - processed) ** 2)

    if energy_distortion == 0:
        return np.inf

    si_sdr = 10 * np.log10(energy_target / energy_distortion)
    return si_sdr


# ============================================================================
# PESQ (Perceptual Evaluation of Speech Quality)
# ============================================================================

def pesq_metric(original, processed, sr=SAMPLE_RATE):
    """
    Вычисление PESQ (требуется библиотека pesq)

    PESQ - стандарт ITU-T P.862 для оценки качества речи
    Диапазон: -0.5 до 4.5 (чем выше, тем лучше)

    Parameters:
    - original: исходный сигнал
    - processed: обработанный сигнал
    - sr: частота дискретизации

    Returns:
    - pesq_score: значение PESQ или None если библиотека не установлена
    """
    try:
        from pesq import pesq

        # PESQ работает с частотами 8 кГц или 16 кГц
        if sr == 8000 or sr == 16000:
            target_sr = sr
            orig_resampled = original
            proc_resampled = processed
        else:
            target_sr = PESQ_TARGET_SR
            orig_resampled = librosa.resample(original, orig_sr=sr, target_sr=target_sr)
            proc_resampled = librosa.resample(processed, orig_sr=sr, target_sr=target_sr)

        # Выравнивание длины
        min_len = min(len(orig_resampled), len(proc_resampled))
        orig_resampled = orig_resampled[:min_len]
        proc_resampled = proc_resampled[:min_len]

        # Вычисление PESQ
        mode = 'wb' if target_sr == 16000 else 'nb'
        pesq_score = pesq(target_sr, orig_resampled, proc_resampled, mode)

        return pesq_score

    except ImportError:
        print("⚠️ PESQ не установлен. Установите: pip install pesq")
        return None
    except Exception as e:
        print(f"⚠️ Ошибка вычисления PESQ: {e}")
        return None


# ============================================================================
# STOI (Short-Time Objective Intelligibility)
# ============================================================================

def stoi_metric(original, processed, sr=SAMPLE_RATE):
    """
    Вычисление STOI (требуется библиотека pystoi)

    STOI оценивает разборчивость речи
    Диапазон: 0 до 1 (чем выше, тем лучше)

    Parameters:
    - original: исходный сигнал
    - processed: обработанный сигнал
    - sr: частота дискретизации

    Returns:
    - stoi_score: значение STOI или None если библиотека не установлена
    """
    try:
        import pystoi

        # STOI требует частоту >= 10 кГц
        if sr < 10000:
            target_sr = STOI_TARGET_SR
            orig_resampled = librosa.resample(original, orig_sr=sr, target_sr=target_sr)
            proc_resampled = librosa.resample(processed, orig_sr=sr, target_sr=target_sr)
            sr_use = target_sr
        else:
            orig_resampled = original
            proc_resampled = processed
            sr_use = sr

        # Выравнивание длины
        min_len = min(len(orig_resampled), len(proc_resampled))
        orig_resampled = orig_resampled[:min_len]
        proc_resampled = proc_resampled[:min_len]

        stoi_score = pystoi.stoi(orig_resampled, proc_resampled, sr_use)
        return stoi_score

    except ImportError:
        print("⚠️ STOI не установлен. Установите: pip install pystoi")
        return None
    except Exception as e:
        print(f"⚠️ Ошибка вычисления STOI: {e}")
        return None


# ============================================================================
# NISQA (Non-Intrusive Speech Quality Assessment) - Упрощенная версия
# ============================================================================

def nisqa_metric(processed, sr=SAMPLE_RATE):
    """
    Упрощенная реализация NISQA (Non-Intrusive Speech Quality Assessment)

    Оценивает качество речи без эталонного сигнала
    Возвращает MOS (Mean Opinion Score) от 1 до 5

    Parameters:
    - processed: обработанный сигнал
    - sr: частота дискретизации

    Returns:
    - mos: оценка качества от 1 до 5
    """
    # Вычисление характеристик сигнала
    rms = np.sqrt(np.mean(processed ** 2))

    # Спектральные характеристики
    stft = librosa.stft(processed, n_fft=2048, hop_length=512)
    spectrum = np.abs(stft)

    # Спектральная энтропия (чем выше, тем больше шума)
    spectrum_norm = spectrum / (np.sum(spectrum, axis=0, keepdims=True) + 1e-10)
    spectral_entropy = -np.sum(spectrum_norm * np.log2(spectrum_norm + 1e-10), axis=0)
    entropy_mean = np.mean(spectral_entropy)

    # Частота пересечения нуля
    zcr = librosa.feature.zero_crossing_rate(processed)
    zcr_mean = np.mean(zcr)

    # Эвристическая оценка MOS
    # Низкая энтропия и низкий ZCR -> хорошее качество
    quality_score = 5.0

    # Штраф за высокую энтропию (шумность)
    if entropy_mean > 6:
        quality_score -= (entropy_mean - 6) * 0.5
    elif entropy_mean < 4:
        quality_score += (4 - entropy_mean) * 0.2

    # Штраф за высокий ZCR
    if zcr_mean > 0.1:
        quality_score -= (zcr_mean - 0.1) * 10

    # Штраф за низкую громкость
    if rms < 0.05:
        quality_score -= 1

    # Ограничение диапазона
    mos = np.clip(quality_score, 1.0, 5.0)

    return mos


# ============================================================================
# DNSMOS (Deep Noise Suppression MOS) - Упрощенная версия
# ============================================================================

def dnsmos_metric(processed, sr=SAMPLE_RATE):
    """
    Упрощенная реализация DNSMOS (Deep Noise Suppression Mean Opinion Score)

    Оценивает качество шумоподавления
    Возвращает три оценки: SIG (качество сигнала), BAK (качество фона), OVRL (общее)

    Parameters:
    - processed: обработанный сигнал
    - sr: частота дискретизации

    Returns:
    - scores: dict с оценками SIG, BAK, OVRL
    """
    # Вычисление характеристик сигнала
    rms = np.sqrt(np.mean(processed ** 2))

    # Спектральные характеристики
    stft = librosa.stft(processed, n_fft=2048, hop_length=512)
    spectrum = np.abs(stft)
    power_spec = spectrum ** 2

    # Спектральный центроид
    freqs = np.linspace(0, sr / 2, spectrum.shape[0])
    centroid = np.sum(freqs[:, np.newaxis] * spectrum, axis=0) / (np.sum(spectrum, axis=0) + 1e-10)
    centroid_mean = np.mean(centroid)

    # ZCR
    zcr = librosa.feature.zero_crossing_rate(processed)
    zcr_mean = np.mean(zcr)

    # Спектральная энтропия
    spec_norm = spectrum / (np.sum(spectrum, axis=0, keepdims=True) + 1e-10)
    entropy = -np.sum(spec_norm * np.log2(spec_norm + 1e-10), axis=0)
    entropy_mean = np.mean(entropy)

    # SIG - качество сигнала (основано на центроиде и RMS)
    # Идеальный центроид для речи ~ 200-800 Гц
    if centroid_mean < 500:
        sig_score = 4.5
    elif centroid_mean < 1000:
        sig_score = 4.0
    elif centroid_mean < 2000:
        sig_score = 3.0
    else:
        sig_score = 2.0

    # Штраф за низкую громкость
    if rms < 0.05:
        sig_score -= 1
    elif rms > 0.5:
        sig_score -= 0.5

    # BAK - качество фона (основано на ZCR и энтропии)
    # Низкий ZCR и низкая энтропия -> чистый фон
    bak_score = 5.0
    if zcr_mean > 0.05:
        bak_score -= (zcr_mean - 0.05) * 10
    if entropy_mean > 5:
        bak_score -= (entropy_mean - 5) * 0.3

    # OVRL - общее качество
    ovrl_score = (sig_score + bak_score) / 2

    # Ограничение диапазона
    sig_score = np.clip(sig_score, 1.0, 5.0)
    bak_score = np.clip(bak_score, 1.0, 5.0)
    ovrl_score = np.clip(ovrl_score, 1.0, 5.0)

    return {
        'SIG': sig_score,
        'BAK': bak_score,
        'OVRL': ovrl_score
    }