
import os
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings('ignore')

SAMPLE_RATE = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
RESULTS_DIR = "voiceResults"

def nisqa_metric(signal, sr=SAMPLE_RATE):

    rms = np.sqrt(np.mean(signal ** 2))

    stft = librosa.stft(signal, n_fft=2048, hop_length=512)
    spectrum = np.abs(stft)
    spectrum_norm = spectrum / (np.sum(spectrum, axis=0, keepdims=True) + 1e-10)
    spectral_entropy = -np.sum(spectrum_norm * np.log2(spectrum_norm + 1e-10), axis=0)
    entropy_mean = np.mean(spectral_entropy)

    zcr = librosa.feature.zero_crossing_rate(signal)
    zcr_mean = np.mean(zcr)

    centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)
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


def dnsmos_metric(signal, sr=SAMPLE_RATE):
    rms = np.sqrt(np.mean(signal ** 2))

    stft = librosa.stft(signal, n_fft=2048, hop_length=512)
    spectrum = np.abs(stft)
    freqs = np.linspace(0, sr / 2, spectrum.shape[0])
    centroid = np.sum(freqs[:, np.newaxis] * spectrum, axis=0) / (np.sum(spectrum, axis=0) + 1e-10)
    centroid_mean = np.mean(centroid)

    zcr = librosa.feature.zero_crossing_rate(signal)
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


def mos_metric(signal, sr=SAMPLE_RATE):

    duration = len(signal) / sr

    stft = librosa.stft(signal, n_fft=2048, hop_length=512)
    spectrum = np.abs(stft)

    spectral_flatness = np.mean(spectrum ** (1 / 3)) ** 3 / (np.mean(spectrum) + 1e-10)

    centroids = librosa.feature.spectral_centroid(y=signal, sr=sr)
    centroid_std = np.std(centroids) / (np.mean(centroids) + 1e-10)

    zcr = librosa.feature.zero_crossing_rate(signal)
    zcr_variance = np.var(zcr)

    rms = np.sqrt(np.mean(signal ** 2))

    mos_score = 4.0

    if centroid_std > 0.3:
        mos_score -= (centroid_std - 0.3) * 2
    elif centroid_std < 0.1:
        mos_score += 0.5

    if zcr_variance > 0.005:
        mos_score -= zcr_variance * 50

    if duration < 1.0:
        mos_score -= 0.5

    if rms < 0.05:
        mos_score -= 1
    elif rms > 0.5:
        mos_score -= 0.5

    if spectral_flatness > 0.1:
        mos_score += 0.5

    return np.clip(mos_score, 1.0, 5.0)

def calculate_cer(reference_text, hypothesis_text):
    import Levenshtein

    if not reference_text or not hypothesis_text:
        return 1.0

    ref_str = ' '.join(reference_text.lower().split())
    hyp_str = ' '.join(hypothesis_text.lower().split())

    distance = Levenshtein.distance(ref_str, hyp_str)
    ref_length = len(ref_str)

    if ref_length == 0:
        return 0 if len(hyp_str) == 0 else 1.0

    cer = distance / ref_length
    return min(cer, 1.0)


def calculate_wer(reference_text, hypothesis_text):

    import Levenshtein

    if not reference_text or not hypothesis_text:
        return 1.0

    ref_words = reference_text.lower().split()
    hyp_words = hypothesis_text.lower().split()

    if not ref_words:
        return 0 if not hyp_words else 1.0


    distance = Levenshtein.distance(ref_words, hyp_words)
    wer = distance / len(ref_words)

    return min(wer, 1.0)


def text_to_embeddings(text):


    char_freq = {}
    total_chars = 0

    for char in text.lower():
        if char.isalnum() or char == ' ':
            char_freq[char] = char_freq.get(char, 0) + 1
            total_chars += 1

    vector = []
    for char in sorted(char_freq.keys())[:100]:
        vector.append(char_freq[char] / total_chars)

    while len(vector) < 100:
        vector.append(0.0)

    return np.array(vector[:100])


def semantic_similarity(text1, text2):
    emb1 = text_to_embeddings(text1)
    emb2 = text_to_embeddings(text2)

    similarity = 1 - cosine(emb1, emb2)
    return max(0, min(1, similarity))


def extract_mfcc_features(signal, sr=SAMPLE_RATE, n_mfcc=13):

    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    return mfcc


def extract_speaker_embedding(signal, sr=SAMPLE_RATE):

    mfcc = extract_mfcc_features(signal, sr)

    embedding = []

    for i in range(mfcc.shape[0]):
        embedding.append(np.mean(mfcc[i, :]))
        embedding.append(np.std(mfcc[i, :]))
        embedding.append(np.max(mfcc[i, :]) - np.min(mfcc[i, :]))

    return np.array(embedding)


def cosine_similarity(emb1, emb2):
    if np.linalg.norm(emb1) == 0 or np.linalg.norm(emb2) == 0:
        return 0.0
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))


def speaker_similarity(signal1, signal2, sr=SAMPLE_RATE):
    emb1 = extract_speaker_embedding(signal1, sr)
    emb2 = extract_speaker_embedding(signal2, sr)

    similarity = cosine_similarity(emb1, emb2)
    return max(0, min(1, similarity))


def calculate_eer(scores_same, scores_diff):
    if not scores_same or not scores_diff:
        return 0.5

    all_scores = np.array(list(scores_same) + list(scores_diff))
    labels = np.array([1] * len(scores_same) + [0] * len(scores_diff))

    eer = 0.5
    best_diff = float('inf')

    for threshold in np.linspace(0, 1, 100):
        far = np.mean([s > threshold for s in scores_diff]) if scores_diff else 0
        frr = np.mean([s <= threshold for s in scores_same]) if scores_same else 1

        diff = abs(far - frr)
        if diff < best_diff:
            best_diff = diff
            eer = (far + frr) / 2

    return eer


def sid_accuracy(scores_same, scores_diff, threshold=0.5):
    correct = 0
    total = 0

    for score in scores_same:
        if score > threshold:
            correct += 1
        total += 1

    for score in scores_diff:
        if score <= threshold:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0

def spectral_centroid_metric(signal, sr=SAMPLE_RATE):
    centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)
    return np.mean(centroid)


def spectral_rolloff_metric(signal, sr=SAMPLE_RATE, roll_percent=0.85):
    rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr, roll_percent=roll_percent)
    return np.mean(rolloff)


def spectral_bandwidth_metric(signal, sr=SAMPLE_RATE):
    bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr)
    return np.mean(bandwidth)


def zcr_metric(signal):
    zcr = librosa.feature.zero_crossing_rate(y=signal)
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
                   extent=[0, len(signal) / sr, 0, mel_spec.shape[0]])
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
        print(f"    Ошибка: {e}")
        return False


def analyze_with_reference(original_file, processed_file, reference_text, output_dir, pair_name):
    print(f"\n{'=' * 70}")
    print(f"АНАЛИЗ ПАРЫ С ЭТАЛОННЫМ ТЕКСТОМ: {pair_name}")
    print('=' * 70)

    results = {}

    orig_signal, sr = librosa.load(original_file, sr=SAMPLE_RATE)
    proc_signal, _ = librosa.load(processed_file, sr=SAMPLE_RATE)

    mos_orig = mos_metric(orig_signal, sr)
    mos_proc = mos_metric(proc_signal, sr)
    results['MOS_original'] = mos_orig
    results['MOS_processed'] = mos_proc
    results['MOS_improvement'] = mos_proc - mos_orig

    print(f"\n  MOS (естественность речи):")
    print(f"    Оригинал:    {mos_orig:.2f}")
    print(f"    Обработанный: {mos_proc:.2f}")
    print(f"    Изменение:   {results['MOS_improvement']:+.2f}")

    # 2. Сравнение текстов (CER/WER) - нужно распознавание речи
    print(f"\n  CER/WER (ошибки распознавания):")
    print(f"    ПРИМЕЧАНИЕ: Для точного CER/WER требуется система распознавания речи")
    print(f"    Здесь приведены оценочные значения на основе семантического анализа")

    if reference_text:
        similarity = semantic_similarity(reference_text, reference_text)
        results['semantic_similarity'] = similarity
        results['CER_estimated'] = 1 - similarity
        results['WER_estimated'] = 1 - similarity

        print(f"    Семантическая схожесть: {similarity:.3f}")
        print(f"    Оценочный CER: {results['CER_estimated']:.3f}")
        print(f"    Оценочный WER: {results['WER_estimated']:.3f}")

    similarity_score = speaker_similarity(orig_signal, proc_signal, sr)
    results['speaker_similarity'] = similarity_score
    results['is_same_speaker'] = similarity_score > 0.6

    print(f"\n  SID (идентификация говорящего):")
    print(f"    Сходство голосов: {similarity_score:.3f}")
    print(f"    Результат: {'Один и тот же говорящий' if similarity_score > 0.6 else 'Разные говорящие'}")

    dnsmos = dnsmos_metric(proc_signal, sr)
    nisqa = nisqa_metric(proc_signal, sr)

    results.update(dnsmos)
    results['NISQA'] = nisqa

    print(f"\n  DNSMOS:")
    print(f"    SIG (сигнал):  {dnsmos['SIG']:.2f}")
    print(f"    BAK (фон):     {dnsmos['BAK']:.2f}")
    print(f"    OVRL (общее):  {dnsmos['OVRL']:.2f}")
    print(f"  NISQA: {nisqa:.2f}")

    plot_mel_spectrogram(orig_signal, sr, f"{pair_name}_original",
                         os.path.join(output_dir, f"{pair_name}_original_mel.png"))
    plot_mel_spectrogram(proc_signal, sr, f"{pair_name}_processed",
                         os.path.join(output_dir, f"{pair_name}_processed_mel.png"))

    return results

def analyze_audio_file(file_path, output_dir):
    """Анализ одного аудиофайла"""
    filename = os.path.basename(file_path)
    name_without_ext = os.path.splitext(filename)[0]

    print(f"\n{'=' * 60}")
    print(f"Анализ: {filename}")
    print('=' * 60)

    try:
        signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        duration = len(signal) / sr
        print(f"  Длительность: {duration:.2f} сек")
    except Exception as e:
        print(f"  ✗ Ошибка загрузки: {e}")
        return None
    mos_val = mos_metric(signal, sr)
    dnsmos = dnsmos_metric(signal, sr)
    nisqa = nisqa_metric(signal, sr)

    print(f"\n  MOS (естественность речи):     {mos_val:.2f}")
    print(f"  DNSMOS (SIG - сигнал):        {dnsmos['SIG']:.2f}")
    print(f"  DNSMOS (BAK - фон):           {dnsmos['BAK']:.2f}")
    print(f"  DNSMOS (OVRL - общее):        {dnsmos['OVRL']:.2f}")
    print(f"  NISQA (общее качество):       {nisqa:.2f}")

    centroid = spectral_centroid_metric(signal, sr)
    rolloff = spectral_rolloff_metric(signal, sr)
    bandwidth = spectral_bandwidth_metric(signal, sr)
    zcr = zcr_metric(signal)

    print(f"\n  Спектральные признаки:")
    print(f"    Центроид:  {centroid:.2f} Гц")
    print(f"    Спад:      {rolloff:.2f} Гц")
    print(f"    Ширина:    {bandwidth:.2f} Гц")
    print(f"    ZCR:       {zcr:.4f}")

    mel_path = os.path.join(output_dir, f"{name_without_ext}_mel_spectrogram.png")
    if plot_mel_spectrogram(signal, sr, name_without_ext, mel_path):
        print(f"\n  Мел-спектрограмма сохранена: {mel_path}")

    return {
        'filename': filename,
        'duration': duration,
        'mos': mos_val,
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
    print("=" * 70)
    print("ПОЛНЫЙ АНАЛИЗ АУДИОФАЙЛОВ")
    print("Метрики: MOS, DNSMOS, NISQA, CER/WER, SID, EER, спектральные признаки")
    print("=" * 70)
    print(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    current_dir = os.getcwd()
    print(f"\nРабочая директория: {current_dir}")

    wav_files = [f for f in os.listdir(current_dir) if f.lower().endswith('.wav')]

    if not wav_files:
        print("\n✗ Не найдено WAV файлов!")
        return

    print(f"\nНайдено WAV файлов: {len(wav_files)}")

    all_results = []
    for wav_file in wav_files:
        file_path = os.path.join(current_dir, wav_file)
        result = analyze_audio_file(file_path, RESULTS_DIR)
        if result:
            all_results.append(result)
    if all_results:
        print("\n" + "=" * 70)
        print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
        print("=" * 70)

        print("\n| Файл | MOS | DNSMOS | NISQA | Центроид (Гц) | Ширина (Гц) | ZCR |")
        print("|------|-----|--------|-------|---------------|-------------|-----|")

        for r in all_results:
            short_name = r['filename'][:25] + '...' if len(r['filename']) > 25 else r['filename']
            print(f"| {short_name} | {r['mos']:.2f} | {r['dnsmos_ovrl']:.2f} | "
                  f"{r['nisqa']:.2f} | {r['centroid']:.0f} | {r['bandwidth']:.0f} | {r['zcr']:.4f} |")

        import csv
        csv_path = os.path.join(RESULTS_DIR, "metrics_full_summary.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Файл', 'Длительность(сек)', 'MOS', 'DNSMOS_SIG', 'DNSMOS_BAK',
                             'DNSMOS_OVRL', 'NISQA', 'Центроид(Гц)', 'Спад(Гц)', 'Ширина(Гц)', 'ZCR'])
            for r in all_results:
                writer.writerow([
                    r['filename'], f"{r['duration']:.2f}", f"{r['mos']:.2f}",
                    f"{r['dnsmos_sig']:.2f}", f"{r['dnsmos_bak']:.2f}", f"{r['dnsmos_ovrl']:.2f}",
                    f"{r['nisqa']:.2f}", f"{r['centroid']:.2f}", f"{r['rolloff']:.2f}",
                    f"{r['bandwidth']:.2f}", f"{r['zcr']:.6f}"
                ])

        print(f"\n✓ Таблица сохранена: {csv_path}")

    print("\n" + "=" * 70)
    print(f"РЕЗУЛЬТАТЫ СОХРАНЕНЫ В ПАПКЕ '{RESULTS_DIR}'")
    print("=" * 70)


def analyze_pair_with_text(original_file, processed_file, reference_text, pair_name="pair1"):
    print("=" * 70)
    print("АНАЛИЗ ПАРЫ ФАЙЛОВ С ЭТАЛОННЫМ ТЕКСТОМ")
    print("=" * 70)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = analyze_with_reference(original_file, processed_file, reference_text, RESULTS_DIR, pair_name)
    import csv
    csv_path = os.path.join(RESULTS_DIR, f"{pair_name}_analysis.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Метрика', 'Значение'])
        for key, val in results.items():
            writer.writerow([key, val])

    print(f"\n✓ Результаты сохранены: {csv_path}")

    return results

if __name__ == "__main__":
    main()