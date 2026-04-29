
import os
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

SAMPLE_RATE = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
RESULTS_DIR = "voiceResults"

def nisqa_metric(processed_signal, sr=SAMPLE_RATE):
    rms = np.sqrt(np.mean(processed_signal ** 2))

    stft = librosa.stft(processed_signal, n_fft=2048, hop_length=512)
    spectrum = np.abs(stft)
    spectrum_norm = spectrum / (np.sum(spectrum, axis=0, keepdims=True) + 1e-10)
    spectral_entropy = -np.sum(spectrum_norm * np.log2(spectrum_norm + 1e-10), axis=0)
    entropy_mean = np.mean(spectral_entropy)

    zcr = librosa.feature.zero_crossing_rate(processed_signal)
    zcr_mean = np.mean(zcr)

    centroid = librosa.feature.spectral_centroid(y=processed_signal, sr=sr)
    centroid_mean = np.mean(centroid)

    quality_score = 5.0

    if entropy_mean > 6:
        quality_score -= (entropy_mean - 6) * 0.5
    elif entropy_mean < 4:
        quality_score += (4 - entropy_mean) * 0.2

    if zcr_mean > 0.1:
        quality_score -= (zcr_mean - 0.1) * 10
    if rms < 0.05:
        quality_score -= 1
    elif rms > 0.5:
        quality_score -= 0.5

    if centroid_mean > 4000:
        quality_score -= 1
    elif centroid_mean < 200:
        quality_score -= 0.5

    return np.clip(quality_score, 1.0, 5.0)


def dnsmos_metric(processed_signal, sr=SAMPLE_RATE):

    rms = np.sqrt(np.mean(processed_signal ** 2))


    stft = librosa.stft(processed_signal, n_fft=2048, hop_length=512)
    spectrum = np.abs(stft)
    freqs = np.linspace(0, sr / 2, spectrum.shape[0])
    centroid = np.sum(freqs[:, np.newaxis] * spectrum, axis=0) / (np.sum(spectrum, axis=0) + 1e-10)
    centroid_mean = np.mean(centroid)


    zcr = librosa.feature.zero_crossing_rate(processed_signal)
    zcr_mean = np.mean(zcr)


    spectrum_norm = spectrum / (np.sum(spectrum, axis=0, keepdims=True) + 1e-10)
    entropy = -np.sum(spectrum_norm * np.log2(spectrum_norm + 1e-10), axis=0)
    entropy_mean = np.mean(entropy)


    if centroid_mean < 500:
        sig_score = 4.5
    elif centroid_mean < 1000:
        sig_score = 4.0
    elif centroid_mean < 2000:
        sig_score = 3.0
    else:
        sig_score = 2.0

    if rms < 0.05:
        sig_score -= 1

    bak_score = 5.0
    if zcr_mean > 0.1:
        bak_score -= (zcr_mean - 0.1) * 10
    if entropy_mean > 6:
        bak_score -= (entropy_mean - 6) * 0.5


    ovrl_score = (sig_score + bak_score) / 2

    return {
        'SIG': np.clip(sig_score, 1.0, 5.0),
        'BAK': np.clip(bak_score, 1.0, 5.0),
        'OVRL': np.clip(ovrl_score, 1.0, 5.0)
    }


def spectral_centroid_metric(signal, sr=SAMPLE_RATE):

    centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)
    return np.mean(centroid)


def spectral_rolloff_metric(signal, sr=SAMPLE_RATE, roll_percent=0.85):
    rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr, roll_percent=roll_percent)
    return np.mean(rolloff)


def spectral_bandwidth_metric(signal, sr=SAMPLE_RATE):

    bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr)
    return np.mean(bandwidth)


def zcr_metric(signal, frame_length=512, hop_length=256):

    zcr = librosa.feature.zero_crossing_rate(y=signal, frame_length=frame_length, hop_length=hop_length)
    return np.mean(zcr)

def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)


def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)


def create_mel_filterbank_manual(n_fft, sr, n_mels=128, fmin=0, fmax=None):
    if fmax is None:
        fmax = sr / 2

    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    fft_freqs = np.linspace(0, sr, n_fft + 1)[:n_fft // 2 + 1]

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

    energies = np.sum(filters, axis=1)
    filters = filters / energies[:, np.newaxis]

    return filters


def mel_spectrogram_manual(signal, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS):
    win = np.hanning(n_fft)
    n_frames = 1 + (len(signal) - n_fft) // hop_length
    stft_matrix = np.zeros((n_fft // 2 + 1, n_frames), dtype=complex)

    for i in range(n_frames):
        start = i * hop_length
        frame = signal[start:start + n_fft] * win
        fft_result = np.fft.fft(frame)
        stft_matrix[:, i] = fft_result[:n_fft // 2 + 1]

    power_spectrum = np.abs(stft_matrix) ** 2

    mel_filters = create_mel_filterbank_manual(n_fft, sr, n_mels)

    mel_spec = np.dot(mel_filters, power_spectrum)
    mel_spec_db = 10 * np.log10(mel_spec + 1e-10)

    return mel_spec_db


def plot_mel_spectrogram(signal, sr, filename, save_path):
    try:
        mel_spec = mel_spectrogram_manual(signal, sr)

        plt.figure(figsize=(12, 6))

        plt.imshow(mel_spec, aspect='auto', origin='lower',
                   extent=[0, len(signal)/sr, 0, mel_spec.shape[0]])

        plt.colorbar(label='Уровень (дБ)')
        plt.xlabel('Время (с)')
        plt.ylabel('Мел-полоса')
        plt.title(f'Мел-спектрограмма - {filename}')
        plt.tight_layout()

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return True
    except Exception as e:
        print(f"    Ошибка при построении мел-спектрограммы: {e}")
        return False


def analyze_audio_file(file_path, output_dir):

    filename = os.path.basename(file_path)
    name_without_ext = os.path.splitext(filename)[0]

    print(f"\n{'='*60}")
    print(f"Анализ: {filename}")
    print('='*60)

    # Загрузка файла
    try:
        signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        duration = len(signal) / sr
        print(f"  Длительность: {duration:.2f} сек")
        print(f"  Частота дискретизации: {sr} Гц")
    except Exception as e:
        print(f"  ✗ Ошибка загрузки: {e}")
        return None


    dnsmos = dnsmos_metric(signal, sr)
    print(f"    DNSMOS (SIG - качество сигнала):   {dnsmos['SIG']:.2f}")
    print(f"    DNSMOS (BAK - качество фона):      {dnsmos['BAK']:.2f}")
    print(f"    DNSMOS (OVRL - общее качество):    {dnsmos['OVRL']:.2f}")

    nisqa = nisqa_metric(signal, sr)
    print(f"    NISQA (общее качество):            {nisqa:.2f}")
    centroid = spectral_centroid_metric(signal, sr)
    rolloff = spectral_rolloff_metric(signal, sr)
    bandwidth = spectral_bandwidth_metric(signal, sr)
    zcr = zcr_metric(signal)

    print(f"\n  Спектральные признаки:")
    print(f"    Спектральный центроид:   {centroid:.2f} Гц")
    print(f"    Спектральный спад (85%): {rolloff:.2f} Гц")
    print(f"    Спектральная ширина:     {bandwidth:.2f} Гц")
    print(f"    Частота пересечения нуля: {zcr:.4f}")

    print("\n  Построение мел-спектрограммы:")
    mel_path = os.path.join(output_dir, f"{name_without_ext}_mel_spectrogram.png")
    if plot_mel_spectrogram(signal, sr, name_without_ext, mel_path):
        print(f"    ✓ Сохранена: {mel_path}")

    return {
        'filename': filename,
        'duration': duration,
        'dnsmos_sig': dnsmos['SIG'],
        'dnsmos_bak': dnsmos['BAK'],
        'dnsmos_ovrl': dnsmos['OVRL'],
        'nisqa': nisqa,
        'centroid': centroid,
        'rolloff': rolloff,
        'bandwidth': bandwidth,
        'zcr': zcr
    }

def main():
    print("="*70)
    print("АНАЛИЗ АУДИОФАЙЛОВ (ОЗВУЧКА И КОНВЕРТАЦИЯ ГОЛОСА)")
    print("="*70)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    current_dir = os.getcwd()
    print(f"\nРабочая директория: {current_dir}")

    # Поиск всех WAV файлов в рабочей директории
    print("\nПоиск WAV файлов...")
    wav_files = [f for f in os.listdir(current_dir) if f.lower().endswith('.wav')]

    if not wav_files:
        print("  ✗ Не найдено WAV файлов в рабочей директории!")
        print("\nПожалуйста, поместите WAV файлы в текущую папку и запустите программу снова.")
        return

    print(f"  Найдено WAV файлов: {len(wav_files)}")
    for f in wav_files:
        file_size = os.path.getsize(os.path.join(current_dir, f)) / 1024
        print(f"    - {f} ({file_size:.1f} KB)")

    all_results = []

    for wav_file in wav_files:
        file_path = os.path.join(current_dir, wav_file)
        result = analyze_audio_file(file_path, RESULTS_DIR)
        if result:
            all_results.append(result)

    if all_results:
        print("\n" + "="*70)
        print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
        print("="*70)

        print("\n| Файл | DNSMOS (OVRL) | NISQA | Центроид (Гц) | Ширина (Гц) | ZCR |")
        print("|------|---------------|-------|---------------|-------------|-----|")

        for r in all_results:
            # Сокращаем имя файла для отображения
            short_name = r['filename'][:30] + '...' if len(r['filename']) > 30 else r['filename']
            print(f"| {short_name} | {r['dnsmos_ovrl']:.2f} | {r['nisqa']:.2f} | "
                  f"{r['centroid']:.0f} | {r['bandwidth']:.0f} | {r['zcr']:.4f} |")

        # Сохранение таблицы в CSV
        import csv
        csv_path = os.path.join(RESULTS_DIR, "metrics_summary.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Файл', 'Длительность (сек)', 'DNSMOS_SIG', 'DNSMOS_BAK',
                           'DNSMOS_OVRL', 'NISQA', 'Центроид (Гц)',
                           'Спад (Гц)', 'Ширина (Гц)', 'ZCR'])
            for r in all_results:
                writer.writerow([
                    r['filename'],
                    f"{r['duration']:.2f}",
                    f"{r['dnsmos_sig']:.2f}",
                    f"{r['dnsmos_bak']:.2f}",
                    f"{r['dnsmos_ovrl']:.2f}",
                    f"{r['nisqa']:.2f}",
                    f"{r['centroid']:.2f}",
                    f"{r['rolloff']:.2f}",
                    f"{r['bandwidth']:.2f}",
                    f"{r['zcr']:.6f}"
                ])

        print(f"\n✓ Таблица сохранена: {csv_path}")

    print("\n" + "="*70)
    print(f"РЕЗУЛЬТАТЫ СОХРАНЕНЫ В ПАПКЕ '{RESULTS_DIR}'")
    print("="*70)


    print("\nСозданные файлы:")
    for root, dirs, files in os.walk(RESULTS_DIR):
        for f in files:
            print(f"  - {os.path.join(RESULTS_DIR, f)}")


def process_specific_files(file_list):

    print("="*70)
    print("АНАЛИЗ КОНКРЕТНЫХ АУДИОФАЙЛОВ")
    print("="*70)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    current_dir = os.getcwd()

    all_results = []

    for filename in file_list:
        file_path = os.path.join(current_dir, filename)
        if os.path.exists(file_path) and filename.lower().endswith('.wav'):
            result = analyze_audio_file(file_path, RESULTS_DIR)
            if result:
                all_results.append(result)
        else:
            print(f"\n⚠️ Файл не найден или не является WAV: {filename}")

    return all_results


if __name__ == "__main__":
    main()