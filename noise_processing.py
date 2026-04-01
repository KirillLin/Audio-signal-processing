"""
Обработка шума: добавление шума и шумоподавление
"""

import numpy as np
import librosa
from scipy.signal import wiener, medfilt
import noisereduce as nr
from config import SAMPLE_RATE, SNR_VALUES


def add_noise_with_snr(clean_signal, noise, snr_db):
    """Добавление шума к сигналу с заданным SNR"""
    if len(noise) < len(clean_signal):
        repeats = int(np.ceil(len(clean_signal) / len(noise)))
        noise = np.tile(noise, repeats)[:len(clean_signal)]
    else:
        noise = noise[:len(clean_signal)]

    power_signal = np.mean(clean_signal ** 2)
    power_noise = np.mean(noise ** 2)

    if power_noise == 0:
        power_noise = 1e-10

    target_power_noise = power_signal / (10 ** (snr_db / 10))
    scale_factor = np.sqrt(target_power_noise / (power_noise + 1e-10))

    scaled_noise = noise * scale_factor
    noisy_signal = clean_signal + scaled_noise
    actual_snr = 10 * np.log10(power_signal / (np.mean(scaled_noise ** 2) + 1e-10))

    return noisy_signal, actual_snr


def find_best_noise_sample(noisy_signal, sr, min_duration=0.3):
    """
    Поиск лучшего участка для оценки шума

    Parameters:
    - noisy_signal: зашумленный сигнал
    - sr: частота дискретизации
    - min_duration: минимальная длительность участка (сек)

    Returns:
    - noise_sample: участок с шумом (всегда возвращает массив)
    """
    # Проверка входных данных
    if noisy_signal is None or len(noisy_signal) == 0:
        print("   Предупреждение: пустой сигнал, создаю шумовой участок")
        return np.random.randn(int(sr * 0.5)) * 0.1

    # Если сигнал слишком короткий
    if len(noisy_signal) < sr * min_duration:
        print(f"   Предупреждение: сигнал слишком короткий, использую первые 0.3 сек")
        return noisy_signal[:int(sr * 0.3)]

    try:
        # Анализ сигнала
        window_size = int(sr * 0.05)  # 50 мс
        hop_size = int(sr * 0.025)    # 25 мс

        # Вычисляем энергию в каждом окне
        energies = []
        for i in range(0, len(noisy_signal) - window_size, hop_size):
            window = noisy_signal[i:i + window_size]
            energies.append(np.sum(window ** 2))

        if len(energies) == 0:
            print("   Предупреждение: не удалось вычислить энергию, использую первые 0.3 сек")
            return noisy_signal[:int(sr * 0.3)]

        # Находим 5 самых тихих участков
        quiet_indices = np.argsort(energies)[:5]

        # Выбираем участок с наименьшей вариацией
        best_variance = float('inf')
        best_start = 0
        noise_duration = int(sr * 0.5)

        for idx in quiet_indices:
            start = idx * hop_size
            end = min(start + noise_duration, len(noisy_signal))

            if end - start < sr * 0.2:  # Слишком короткий участок
                continue

            segment = noisy_signal[start:end]
            variance = np.var(segment)

            if variance < best_variance:
                best_variance = variance
                best_start = start

        # Формируем результат
        end = min(best_start + noise_duration, len(noisy_signal))
        noise_sample = noisy_signal[best_start:end]

        # Проверка результата
        if len(noise_sample) < sr * 0.2:
            print("   Предупреждение: найденный участок слишком короткий, использую первые 0.3 сек")
            return noisy_signal[:int(sr * 0.3)]

        return noise_sample

    except Exception as e:
        print(f"   Предупреждение: ошибка поиска шума ({e}), использую первые 0.3 сек")
        return noisy_signal[:int(sr * 0.3)]


def denoise_noisereduce(noisy_signal, sr, prop_decrease=0.55, snr_db=None):
    """
    Шумоподавление с помощью noisereduce с улучшенными настройками
    """
    try:
        # Поиск лучшего участка для оценки шума
        noise_sample = find_best_noise_sample(noisy_signal, sr)

        # Адаптивный коэффициент подавления
        if snr_db is not None:
            if snr_db < 6:
                prop_decrease = 0.70
            elif snr_db < 12:
                prop_decrease = 0.55
            else:
                prop_decrease = 0.40

        # Применяем шумоподавление
        enhanced = nr.reduce_noise(
            y=noisy_signal,
            sr=sr,
            y_noise=noise_sample,
            prop_decrease=prop_decrease,
            stationary=False,
            time_constant_s=0.4,
            freq_mask_smooth_hz=500
        )

        return enhanced

    except Exception as e:
        print(f"   Ошибка в noisereduce: {e}")
        return noisy_signal


def denoise_wiener(noisy_signal, sr, mysize=7):
    """Винеровская фильтрация"""
    try:
        from scipy.signal import wiener
        enhanced = wiener(noisy_signal, mysize=mysize)
        return enhanced
    except Exception as e:
        print(f"   Ошибка в Wiener: {e}")
        return noisy_signal


def denoise_median(noisy_signal, sr, kernel_size=5):
    """Медианная фильтрация"""
    try:
        from scipy.signal import medfilt
        enhanced = medfilt(noisy_signal, kernel_size=kernel_size)
        return enhanced
    except Exception as e:
        print(f"   Ошибка в Median: {e}")
        return noisy_signal


def denoise_spectral_gate(noisy_signal, sr, threshold_db=-20):
    """Спектральное шумоподавление"""
    try:
        n_fft = 2048
        hop_length = 512
        stft = librosa.stft(noisy_signal, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)

        noise_floor = np.percentile(magnitude, 10, axis=1, keepdims=True)
        threshold = noise_floor * (10 ** (threshold_db / 20))
        magnitude_clean = np.where(magnitude > threshold, magnitude, threshold * 0.1)

        stft_clean = magnitude_clean * np.exp(1j * phase)
        enhanced = librosa.istft(stft_clean, hop_length=hop_length)
        enhanced = enhanced[:len(noisy_signal)]

        return enhanced
    except Exception as e:
        print(f"   Ошибка в Spectral Gate: {e}")
        return noisy_signal


def denoise_fft_threshold(noisy_signal, sr, threshold_ratio=0.15):
    """FFT шумоподавление"""
    try:
        fft_spectrum = np.fft.rfft(noisy_signal)
        magnitude = np.abs(fft_spectrum)
        threshold = np.percentile(magnitude, threshold_ratio * 100)
        magnitude_clean = np.where(magnitude > threshold, magnitude, 0)
        fft_clean = magnitude_clean * np.exp(1j * np.angle(fft_spectrum))
        enhanced = np.fft.irfft(fft_clean)
        return enhanced
    except Exception as e:
        print(f"   Ошибка в FFT Threshold: {e}")
        return noisy_signal


def denoise_main(noisy_signal, sr, method='wiener', **kwargs):
    """
    Основная функция шумоподавления
    """
    methods = {
        'wiener': denoise_wiener,
        'median': denoise_median,
        'spectral': denoise_spectral_gate,
        'fft': denoise_fft_threshold,
        'noisereduce': denoise_noisereduce
    }

    if method not in methods:
        method = 'wiener'

    return methods[method](noisy_signal, sr, **kwargs)


def generate_noise_signal(sr=SAMPLE_RATE, duration=5, noise_type='white'):
    """Генерация шумового сигнала"""
    n_samples = int(sr * duration)

    if noise_type == 'white':
        noise = np.random.randn(n_samples) * 0.1
    elif noise_type == 'pink':
        white = np.random.randn(n_samples)
        pink = np.fft.ifft(np.fft.fft(white) / np.sqrt(np.arange(1, n_samples + 1))).real
        noise = pink / (np.max(np.abs(pink)) + 1e-10) * 0.1
    elif noise_type == 'brown':
        white = np.random.randn(n_samples)
        brown = np.cumsum(white)
        noise = brown / (np.max(np.abs(brown)) + 1e-10) * 0.1
    else:
        noise = np.random.randn(n_samples) * 0.1

    return noise


def compare_denoise_methods(clean_signal, noisy_signal, sr):
    """
    Сравнение различных методов шумоподавления
    """
    from quality_metrics import snr_manual

    methods = {
        'Винеровский фильтр (рекомендуемый)': lambda x, s: denoise_wiener(x, s, mysize=7),
        'Медианный фильтр': lambda x, s: denoise_median(x, s, kernel_size=5),
        'Спектральный порог': denoise_spectral_gate,
        'FFT порог': denoise_fft_threshold,
        'Noisereduce': lambda x, s: denoise_noisereduce(x, s, prop_decrease=0.55)
    }

    results = {}

    print("\n" + "="*70)
    print("СРАВНЕНИЕ МЕТОДОВ ШУМОПОДАВЛЕНИЯ")
    print("="*70)
    print(f"{'Метод':<35} | {'SNR до':<10} | {'SNR после':<10} | {'Улучшение':<10}")
    print("-" * 70)

    snr_before = snr_manual(clean_signal, noisy_signal)

    for name, method in methods.items():
        try:
            enhanced = method(noisy_signal, sr)
            snr_after = snr_manual(clean_signal, enhanced)
            improvement = snr_after - snr_before

            results[name] = {
                'signal': enhanced,
                'snr_before': snr_before,
                'snr_after': snr_after,
                'improvement': improvement
            }

            status = "✓" if improvement > 0 else "✗"
            print(f"{name:<35} | {snr_before:>8.2f} | {snr_after:>8.2f} | {improvement:>8.2f} {status}")

        except Exception as e:
            print(f"{name:<35} | Ошибка: {e}")

    print("="*70)

    # Находим лучший метод
    if results:
        best_method = max(results.items(), key=lambda x: x[1]['improvement'])
        print(f"\n🏆 Лучший метод: {best_method[0]}")
        print(f"   Улучшение SNR: {best_method[1]['improvement']:.2f} дБ")
        return results, best_method
    else:
        print("\n⚠️ Не удалось выполнить сравнение методов")
        return {}, None


def batch_noise_experiment(clean_signal, noise, snr_values, sr=SAMPLE_RATE, method='wiener'):
    """
    Проведение серии экспериментов с разными SNR
    """
    from quality_metrics import snr_manual, sdr_manual, si_sdr_manual

    results = []

    for snr_target in snr_values:
        print(f"\n--- SNR целевой: {snr_target} дБ ---")

        # Добавление шума
        noisy_signal, actual_snr = add_noise_with_snr(clean_signal, noise, snr_target)
        print(f"  Фактическое SNR: {actual_snr:.2f} дБ")

        # Шумоподавление
        print(f"  Метод: {method}")
        enhanced_signal = denoise_main(noisy_signal, sr, method=method)

        # Вычисление метрик
        snr_n = snr_manual(clean_signal, noisy_signal)
        snr_e = snr_manual(clean_signal, enhanced_signal)
        sdr_n = sdr_manual(clean_signal, noisy_signal)
        sdr_e = sdr_manual(clean_signal, enhanced_signal)
        si_sdr_n = si_sdr_manual(clean_signal, noisy_signal)
        si_sdr_e = si_sdr_manual(clean_signal, enhanced_signal)

        improvement = snr_e - snr_n
        status = "✓ Улучшение" if improvement > 0 else "✗ Ухудшение"

        print(f"  Зашумленный: SNR={snr_n:.2f}, SDR={sdr_n:.2f}, SI-SDR={si_sdr_n:.2f}")
        print(f"  Очищенный:  SNR={snr_e:.2f}, SDR={sdr_e:.2f}, SI-SDR={si_sdr_e:.2f}")
        print(f"  {status}: {abs(improvement):.2f} дБ")

        results.append({
            'snr_target': snr_target,
            'actual_snr': actual_snr,
            'original': clean_signal,
            'noisy': noisy_signal,
            'enhanced': enhanced_signal,
            'snr_noisy': snr_n,
            'snr_enhanced': snr_e,
            'sdr_noisy': sdr_n,
            'sdr_enhanced': sdr_e,
            'si_sdr_noisy': si_sdr_n,
            'si_sdr_enhanced': si_sdr_e,
            'improvement': improvement
        })

    return results