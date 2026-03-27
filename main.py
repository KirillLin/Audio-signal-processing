"""
Главный файл для запуска лабораторной работы
"""

import os
import numpy as np
from config import SAMPLE_RATE, SNR_VALUES, MUSIC_GENRES, RESULTS_DIR
from audio_utils import ensure_audio_files, save_audio
from noise_processing import batch_noise_experiment, generate_noise_signal
from visualization import (
    setup_plotting_style,
    plot_waveform,
    plot_spectrogram,
    plot_mel_spectrogram_comparison,
    plot_spectral_features,
    plot_mfcc,
    plot_chroma,
    plot_quality_comparison
)
from spectral_features import (
    spectral_centroid_librosa,
    spectral_rolloff_librosa,
    spectral_bandwidth_librosa,
    zcr_librosa
)
from report_generator import generate_report, generate_subjective_form
from quality_metrics import snr_manual, sdr_manual, si_sdr_manual


def analyze_music_file(signal, sr, genre):
    """
    Полный анализ музыкального файла
    """
    print(f"\n{'=' * 60}")
    print(f"АНАЛИЗ: {genre}")
    print('=' * 60)

    genre_folder = os.path.join(RESULTS_DIR, genre.replace(' ', '_'))
    os.makedirs(genre_folder, exist_ok=True)

    # 1. Визуализация сигнала
    plot_waveform(signal, sr, f'Осциллограмма - {genre}',
                  os.path.join(genre_folder, 'waveform.png'))

    # 2. Спектрограмма
    plot_spectrogram(signal, sr, f'Спектрограмма - {genre}',
                     os.path.join(genre_folder, 'spectrogram.png'))

    # 3. Мел-спектрограмма (сравнение)
    plot_mel_spectrogram_comparison(signal, sr, f'Мел-спектрограмма - {genre}',
                                    os.path.join(genre_folder, 'mel_spectrogram.png'))

    # 4. Спектральные признаки
    plot_spectral_features(signal, sr, f'Спектральные признаки - {genre}',
                           os.path.join(genre_folder, 'spectral_features.png'))

    # 5. MFCC
    plot_mfcc(signal, sr, genre, os.path.join(genre_folder, 'mfcc.png'))

    # 6. Цветность
    plot_chroma(signal, sr, genre, os.path.join(genre_folder, 'chroma.png'))

    # Вычисление средних значений признаков
    centroid = spectral_centroid_librosa(signal, sr)
    rolloff = spectral_rolloff_librosa(signal, sr)
    bandwidth = spectral_bandwidth_librosa(signal, sr)
    zcr = zcr_librosa(signal)

    results = {
        'genre': genre,
        'centroid_mean': np.mean(centroid),
        'rolloff_mean': np.mean(rolloff),
        'bandwidth_mean': np.mean(bandwidth),
        'zcr_mean': np.mean(zcr)
    }

    print(f"\nСредние значения признаков для {genre}:")
    print(f"  Спектральный центроид: {results['centroid_mean']:.2f} Гц")
    print(f"  Спектральный спад: {results['rolloff_mean']:.2f} Гц")
    print(f"  Спектральная ширина: {results['bandwidth_mean']:.2f} Гц")
    print(f"  ZCR: {results['zcr_mean']:.4f}")

    return results


def main():
    """
    Основная функция
    """
    print("=" * 70)
    print("ЛАБОРАТОРНАЯ РАБОТА ПО ЦИФРОВОЙ ОБРАБОТКЕ АУДИОСИГНАЛОВ")
    print("Вариант 12: Мел-спектрограмма, MFCC, спектральная ширина, SNR, SDR, DeepFilterNet2")
    print("=" * 70)

    # Настройка стиля графиков
    setup_plotting_style()

    # Создание папок
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Загрузка аудиофайлов
    print("\nЗагрузка аудиофайлов...")
    audio_files = ensure_audio_files()

    # ========================================================================
    # ЧАСТЬ 2-3: Анализ музыкальных файлов
    # ========================================================================
    print("\n" + "=" * 70)
    print("ЧАСТЬ 2-3: АНАЛИЗ МУЗЫКАЛЬНЫХ ФАЙЛОВ И СПЕКТРАЛЬНЫХ ПРИЗНАКОВ")
    print("=" * 70)

    music_results = []
    for genre in MUSIC_GENRES:
        if genre in audio_files:
            signal = audio_files[genre]['signal']
            sr = audio_files[genre]['sr']
            result = analyze_music_file(signal, sr, genre)
            music_results.append(result)

    # ========================================================================
    # ЧАСТЬ 4-5: Эксперименты с шумом
    # ========================================================================
    print("\n" + "=" * 70)
    print("ЧАСТЬ 4-5: ЭКСПЕРИМЕНТЫ С ШУМОМ И ОЦЕНКА КАЧЕСТВА")
    print("=" * 70)

    # Голосовой сигнал
    voice_signal = audio_files['Voice']['signal']
    voice_sr = audio_files['Voice']['sr']

    # Шумовой сигнал
    noise_signal = audio_files['Noise']['signal']

    print(f"\nГолосовой сигнал: длина {len(voice_signal) / voice_sr:.2f} сек")
    print(f"Шумовой сигнал: длина {len(noise_signal) / voice_sr:.2f} сек")
    print(f"Диапазон SNR: {SNR_VALUES[0]}-{SNR_VALUES[-1]} дБ, шаг {SNR_VALUES[1] - SNR_VALUES[0]} дБ")

    # Проведение эксперимента
    noise_experiment_results = batch_noise_experiment(voice_signal, noise_signal, SNR_VALUES, voice_sr)

    # Сохранение результатов и вычисление метрик
    processed_results = []
    for r in noise_experiment_results:
        # Сохранение аудиофайлов
        save_audio(os.path.join(RESULTS_DIR, f'noisy_snr_{r["snr_target"]:.0f}dB.wav'), r['noisy'], voice_sr)
        save_audio(os.path.join(RESULTS_DIR, f'enhanced_snr_{r["snr_target"]:.0f}dB.wav'), r['enhanced'], voice_sr)

        # Визуализация спектрограмм
        plot_spectrogram(r['noisy'], voice_sr, f'Зашумленный сигнал (SNR={r["snr_target"]} дБ)',
                         os.path.join(RESULTS_DIR, f'spectrogram_noisy_{r["snr_target"]:.0f}dB.png'))
        plot_spectrogram(r['enhanced'], voice_sr, f'Очищенный сигнал (SNR={r["snr_target"]} дБ)',
                         os.path.join(RESULTS_DIR, f'spectrogram_enhanced_{r["snr_target"]:.0f}dB.png'))

        # Вычисление метрик для сравнения
        snr_n = snr_manual(voice_signal, r['noisy'])
        snr_e = snr_manual(voice_signal, r['enhanced'])
        sdr_n = sdr_manual(voice_signal, r['noisy'])
        sdr_e = sdr_manual(voice_signal, r['enhanced'])
        si_sdr_n = si_sdr_manual(voice_signal, r['noisy'])
        si_sdr_e = si_sdr_manual(voice_signal, r['enhanced'])

        processed_results.append({
            'snr_target': r['snr_target'],
            'actual_snr': r['actual_snr'],
            'original': voice_signal,
            'noisy': r['noisy'],
            'enhanced': r['enhanced'],
            'snr_noisy': snr_n,
            'snr_enhanced': snr_e,
            'sdr_noisy': sdr_n,
            'sdr_enhanced': sdr_e,
            'si_sdr_noisy': si_sdr_n,
            'si_sdr_enhanced': si_sdr_e
        })

        print(f"\nSNR {r['snr_target']:.0f} дБ:")
        print(f"  Зашумленный: SNR={snr_n:.2f}, SDR={sdr_n:.2f}, SI-SDR={si_sdr_n:.2f}")
        print(f"  Очищенный:  SNR={snr_e:.2f}, SDR={sdr_e:.2f}, SI-SDR={si_sdr_e:.2f}")
        print(f"  Улучшение:  SNR={snr_e - snr_n:.2f} дБ")

    # Визуализация сравнения метрик
    plot_quality_comparison(processed_results, os.path.join(RESULTS_DIR, 'quality_comparison.png'))

    # ========================================================================
    # ЧАСТЬ 5: Субъективная оценка
    # ========================================================================
    generate_subjective_form(processed_results)

    # ========================================================================
    # ЧАСТЬ 7: Генерация отчета
    # ========================================================================
    generate_report(music_results, processed_results, voice_sr)

    print("\n" + "=" * 70)
    print("РАБОТА ЗАВЕРШЕНА УСПЕШНО!")
    print(f"Все результаты сохранены в папке '{RESULTS_DIR}'")
    print("=" * 70)


if __name__ == "__main__":
    main()
