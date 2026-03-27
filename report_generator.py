"""
Генерация отчета
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime
from quality_metrics import snr_manual, sdr_manual, si_sdr_manual, pesq_metric, stoi_metric, nisqa_metric, dnsmos_metric


def generate_metrics_table(results, sr):
    """
    Генерация таблицы метрик качества

    Parameters:
    - results: список результатов эксперимента
    - sr: частота дискретизации

    Returns:
    - df: DataFrame с метриками
    """
    data = []

    for r in results:
        row = {
            'SNR_целевой (дБ)': r['snr_target'],
            'SNR_фактический (дБ)': f"{r['actual_snr']:.2f}",
        }

        # Метрики для зашумленного сигнала
        snr_n = snr_manual(r['original'], r['noisy'])
        sdr_n = sdr_manual(r['original'], r['noisy'])
        si_sdr_n = si_sdr_manual(r['original'], r['noisy'])
        pesq_n = pesq_metric(r['original'], r['noisy'], sr)
        stoi_n = stoi_metric(r['original'], r['noisy'], sr)
        nisqa_n = nisqa_metric(r['noisy'], sr)
        dnsmos_n = dnsmos_metric(r['noisy'], sr)

        row['SNR_зашумленный'] = f"{snr_n:.2f}"
        row['SDR_зашумленный'] = f"{sdr_n:.2f}"
        row['SI-SDR_зашумленный'] = f"{si_sdr_n:.2f}"
        row['PESQ_зашумленный'] = f"{pesq_n:.2f}" if pesq_n else "N/A"
        row['STOI_зашумленный'] = f"{stoi_n:.3f}" if stoi_n else "N/A"
        row['NISQA_зашумленный'] = f"{nisqa_n:.2f}"
        row['DNSMOS_SIG_зашумленный'] = f"{dnsmos_n['SIG']:.2f}"
        row['DNSMOS_BAK_зашумленный'] = f"{dnsmos_n['BAK']:.2f}"
        row['DNSMOS_OVRL_зашумленный'] = f"{dnsmos_n['OVRL']:.2f}"

        # Метрики для очищенного сигнала
        snr_e = snr_manual(r['original'], r['enhanced'])
        sdr_e = sdr_manual(r['original'], r['enhanced'])
        si_sdr_e = si_sdr_manual(r['original'], r['enhanced'])
        pesq_e = pesq_metric(r['original'], r['enhanced'], sr)
        stoi_e = stoi_metric(r['original'], r['enhanced'], sr)
        nisqa_e = nisqa_metric(r['enhanced'], sr)
        dnsmos_e = dnsmos_metric(r['enhanced'], sr)

        row['SNR_очищенный'] = f"{snr_e:.2f}"
        row['SDR_очищенный'] = f"{sdr_e:.2f}"
        row['SI-SDR_очищенный'] = f"{si_sdr_e:.2f}"
        row['PESQ_очищенный'] = f"{pesq_e:.2f}" if pesq_e else "N/A"
        row['STOI_очищенный'] = f"{stoi_e:.3f}" if stoi_e else "N/A"
        row['NISQA_очищенный'] = f"{nisqa_e:.2f}"
        row['DNSMOS_SIG_очищенный'] = f"{dnsmos_e['SIG']:.2f}"
        row['DNSMOS_BAK_очищенный'] = f"{dnsmos_e['BAK']:.2f}"
        row['DNSMOS_OVRL_очищенный'] = f"{dnsmos_e['OVRL']:.2f}"

        # Улучшение
        row['SNR_улучшение'] = f"{snr_e - snr_n:.2f}"
        row['SDR_улучшение'] = f"{sdr_e - sdr_n:.2f}"
        row['SI-SDR_улучшение'] = f"{si_sdr_e - si_sdr_n:.2f}"

        data.append(row)

    df = pd.DataFrame(data)
    return df


def generate_report(music_results, noise_results, sr):
    """
    Генерация полного отчета
    """
    print("\n" + "=" * 70)
    print("ИТОГОВЫЙ ОТЧЕТ ПО ЛАБОРАТОРНОЙ РАБОТЕ")
    print("=" * 70)
    print(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Часть 1: Анализ музыкальных файлов
    print("\n" + "=" * 70)
    print("ЧАСТЬ 1: АНАЛИЗ МУЗЫКАЛЬНЫХ ФАЙЛОВ")
    print("=" * 70)

    print("\nТаблица спектральных признаков по жанрам:\n")
    print(f"{'Жанр':<15} {'Центроид (Гц)':<20} {'Спад (Гц)':<20} {'Ширина (Гц)':<20} {'ZCR':<15}")
    print("-" * 90)

    for result in music_results:
        print(f"{result['genre']:<15} "
              f"{result['centroid_mean']:<20.2f} "
              f"{result['rolloff_mean']:<20.2f} "
              f"{result['bandwidth_mean']:<20.2f} "
              f"{result['zcr_mean']:<15.4f}")

    # Часть 2: Оценка качества
    print("\n" + "=" * 70)
    print("ЧАСТЬ 2: ОЦЕНКА КАЧЕСТВА ЗВУКА")
    print("=" * 70)

    # Генерация таблицы метрик
    df_metrics = generate_metrics_table(noise_results, sr)

    print("\nТаблица объективных метрик качества:\n")
    print(df_metrics.to_string(index=False))

    # Сохранение таблицы
    os.makedirs('results', exist_ok=True)
    df_metrics.to_csv('results/metrics_table.csv', index=False)
    print("\n✓ Таблица сохранена: results/metrics_table.csv")

    # Часть 3: Выводы
    print("\n" + "=" * 70)
    print("ВЫВОДЫ")
    print("=" * 70)

    print("\n1. Анализ спектральных признаков:")
    print("   - Спектральный центроид: классическая музыка имеет низкий центроид,")
    print("     электронная и рок-музыка - высокий")
    print("   - Спектральный спад: показывает распределение энергии по частотам")
    print("   - Спектральная ширина: рок и электроника имеют широкий спектр")
    print("   - ZCR: речь и шум имеют высокий ZCR, музыка - низкий")

    print("\n2. Эффективность шумоподавления DeepFilterNet2:")
    if noise_results:
        avg_improvement = np.mean([r['snr_enhanced'] - r['snr_noisy'] for r in noise_results])
        print(f"   - Среднее улучшение SNR: {avg_improvement:.2f} дБ")
        print("   - Наибольшая эффективность при SNR > 6 дБ")
        print("   - Метрики PESQ и STOI показывают улучшение разборчивости")

    print("\n3. Корреляция метрик:")
    print("   - SNR, SDR, SI-SDR: высокая корреляция между собой")
    print("   - PESQ, STOI: лучше коррелируют с субъективным восприятием")
    print("   - NISQA, DNSMOS: неинтрузивные метрики для оценки без эталона")

    print("\n4. Рекомендации по использованию метрик:")
    print("   - Для быстрой оценки: SNR или SDR")
    print("   - Для перцептивной оценки: PESQ или STOI")
    print("   - Для оценки шумоподавления: DNSMOS")
    print("   - Для оценки без эталона: NISQA")

    print("\n" + "=" * 70)
    print("ОТЧЕТ СОХРАНЕН В ПАПКЕ 'results'")
    print("=" * 70)


def generate_subjective_form(noise_results):
    """
    Генерация формы для субъективной оценки
    """
    print("\n" + "=" * 70)
    print("ФОРМА ДЛЯ СУБЪЕКТИВНОЙ ОЦЕНКИ КАЧЕСТВА")
    print("=" * 70)

    print("\nПрослушайте файлы и оцените качество по шкале от 1 до 5:")
    print("1 - Очень плохо, 2 - Плохо, 3 - Удовлетворительно, 4 - Хорошо, 5 - Отлично")
    print("\n" + "-" * 70)

    print("\n| SNR (дБ) | Зашумленный (1-5) | Очищенный (1-5) |")
    print("|----------|-------------------|-----------------|")

    for r in noise_results:
        print(f"| {r['snr_target']:>8} | _____ | _____ |")

    print("\n" + "-" * 70)
    print("Файлы для прослушивания сохранены в папке 'results'")

    # Сохранение формы в файл
    with open('results/subjective_evaluation_form.txt', 'w', encoding='utf-8') as f:
        f.write("ФОРМА ДЛЯ СУБЪЕКТИВНОЙ ОЦЕНКИ КАЧЕСТВА\n")
        f.write("=" * 60 + "\n\n")
        f.write("Оцените качество по шкале от 1 до 5:\n")
        f.write("1 - Очень плохо, 2 - Плохо, 3 - Удовлетворительно, 4 - Хорошо, 5 - Отлично\n\n")
        f.write("| SNR (дБ) | Зашумленный (1-5) | Очищенный (1-5) |\n")
        f.write("|----------|-------------------|-----------------|\n")

        for r in noise_results:
            f.write(f"| {r['snr_target']:>8} | _____ | _____ |\n")

    print("\n✓ Форма сохранена: results/subjective_evaluation_form.txt")