
import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf

from config import SAMPLE_RATE, SNR_VALUES, MUSIC_GENRES, RESULTS_DIR
from audio_utils import ensure_audio_files, save_audio
from noise_processing import (
    batch_noise_experiment,
    generate_noise_signal,
    compare_denoise_methods,
    denoise_main,
    denoise_wiener
)
from visualization import (
    setup_plotting_style,
    plot_waveform,
    plot_spectrogram,
    plot_mel_spectrogram_comparison,
    plot_spectral_features,
    plot_mfcc,
    plot_chroma,
    plot_quality_comparison,
    plot_methods_comparison
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
    """Полный анализ музыкального файла"""
    print(f"\n{'='*60}")
    print(f"АНАЛИЗ: {genre}")
    print('='*60)

    genre_folder = os.path.join(RESULTS_DIR, genre.replace(' ', '_'))
    os.makedirs(genre_folder, exist_ok=True)

    # Визуализации
    plot_waveform(signal, sr, f'Осциллограмма - {genre}',
                  os.path.join(genre_folder, 'waveform.png'))
    plot_spectrogram(signal, sr, f'Спектрограмма - {genre}',
                     os.path.join(genre_folder, 'spectrogram.png'))
    plot_mel_spectrogram_comparison(signal, sr, f'Мел-спектрограмма - {genre}',
                                    os.path.join(genre_folder, 'mel_spectrogram.png'))
    plot_spectral_features(signal, sr, f'Спектральные признаки - {genre}',
                          os.path.join(genre_folder, 'spectral_features.png'))
    plot_mfcc(signal, sr, genre, os.path.join(genre_folder, 'mfcc.png'))
    plot_chroma(signal, sr, genre, os.path.join(genre_folder, 'chroma.png'))

    # Вычисление средних значений
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
    """Основная функция"""
    print("="*70)
    print("ЛАБОРАТОРНАЯ РАБОТА ПО ЦИФРОВОЙ ОБРАБОТКЕ АУДИОСИГНАЛОВ")
    print("Вариант 12: Мел-спектрограмма, MFCC, спектральная ширина, SNR, SDR")
    print("Шумоподавление: Винеровский фильтр (оптимальный линейный фильтр)")
    print("="*70)

    # Настройка стиля
    setup_plotting_style()

    # Создание папок
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Загрузка аудиофайлов
    print("\nЗагрузка аудиофайлов...")
    audio_files = ensure_audio_files()

    print("\n" + "="*70)
    print("ЧАСТЬ 2-3: АНАЛИЗ МУЗЫКАЛЬНЫХ ФАЙЛОВ И СПЕКТРАЛЬНЫХ ПРИЗНАКОВ")
    print("="*70)

    music_results = []
    for genre in MUSIC_GENRES:
        if genre in audio_files:
            signal = audio_files[genre]['signal']
            sr = audio_files[genre]['sr']
            result = analyze_music_file(signal, sr, genre)
            music_results.append(result)

    print("\n" + "="*70)
    print("ЧАСТЬ 4-5: ЭКСПЕРИМЕНТЫ С ШУМОМ И ОЦЕНКА КАЧЕСТВА")
    print("="*70)

    # Голосовой сигнал
    voice_signal = audio_files['Voice']['signal']
    voice_sr = audio_files['Voice']['sr']

    # Шумовой сигнал
    noise_signal = audio_files['Noise']['signal']

    print(f"\nГолосовой сигнал: длина {len(voice_signal)/voice_sr:.2f} сек")
    print(f"Шумовой сигнал: длина {len(noise_signal)/voice_sr:.2f} сек")
    print(f"Диапазон SNR: {SNR_VALUES[0]}-{SNR_VALUES[-1]} дБ, шаг {SNR_VALUES[1]-SNR_VALUES[0]} дБ")
    print(f"Метод шумоподавления: Винеровский фильтр")

    noise_results = batch_noise_experiment(
        voice_signal,
        noise_signal,
        SNR_VALUES,
        voice_sr,
        method='wiener' #поменять на другой для эксперимента
    )

    for r in noise_results:
        save_audio(os.path.join(RESULTS_DIR, f'noisy_snr_{r["snr_target"]:.0f}dB.wav'),
                   r['noisy'], voice_sr)
        save_audio(os.path.join(RESULTS_DIR, f'enhanced_snr_{r["snr_target"]:.0f}dB.wav'),
                   r['enhanced'], voice_sr)

        plot_spectrogram(r['noisy'], voice_sr,
                        f'Зашумленный сигнал (SNR={r["snr_target"]} дБ)',
                        os.path.join(RESULTS_DIR, f'spectrogram_noisy_{r["snr_target"]:.0f}dB.png'))
        plot_spectrogram(r['enhanced'], voice_sr,
                        f'Очищенный сигнал (SNR={r["snr_target"]} дБ)',
                        os.path.join(RESULTS_DIR, f'spectrogram_enhanced_{r["snr_target"]:.0f}dB.png'))

    print("\n" + "="*70)
    print("СРАВНЕНИЕ МЕТОДОВ ШУМОПОДАВЛЕНИЯ")
    print("="*70)

    test_snr = 6
    test_noisy = None
    for r in noise_results:
        if r['snr_target'] == test_snr:
            test_noisy = r['noisy']
            break

    best_method_info = None
    if test_noisy is not None:
        methods_results, best_method = compare_denoise_methods(voice_signal, test_noisy, voice_sr)

        plot_methods_comparison(voice_signal, test_noisy, voice_sr, methods_results,
                               os.path.join(RESULTS_DIR, 'methods_comparison.png'))

        best_method_info = best_method

        best_enhanced = best_method[1]['signal']
        save_audio(os.path.join(RESULTS_DIR, f'best_method_enhanced.wav'),
                   best_enhanced, voice_sr)

    plot_quality_comparison(noise_results, os.path.join(RESULTS_DIR, 'quality_comparison.png'))

    generate_subjective_form(noise_results)

    generate_report(music_results, noise_results, voice_sr, best_method_info)
    print("\n" + "="*70)
    print("СТАТИСТИКА ВЫПОЛНЕНИЯ")
    print("="*70)

    if noise_results:
        improvements = [r['improvement'] for r in noise_results]
        print(f"\n Результаты шумоподавления (Винеровский фильтр):")
        for r in noise_results:
            status = "" if r['improvement'] > 0 else ""
            print(f"   SNR {r['snr_target']:2d} дБ: {r['snr_noisy']:.2f} → {r['snr_enhanced']:.2f} дБ ({status} {abs(r['improvement']):.2f} дБ)")

        print(f"\n Среднее улучшение: {np.mean(improvements):.2f} дБ")
        print(f" Максимальное улучшение: {max(improvements):.2f} дБ")

    png_files = []
    for root, dirs, files in os.walk('results'):
        for file in files:
            if file.endswith('.png'):
                png_files.append(os.path.join(root, file))

    print(f"\n Создано PNG графиков: {len(png_files)}")

    print("\n" + "="*70)
    print("РАБОТА ЗАВЕРШЕНА УСПЕШНО!")
    print(f"Все результаты сохранены в папке '{RESULTS_DIR}'")
    print("="*70)


if __name__ == "__main__":
    main()