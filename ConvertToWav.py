import os
import librosa
import soundfile as sf


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header(text):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def get_user_input(prompt, default=None, input_type=str, min_val=None, max_val=None):
    while True:
        if default is not None:
            user_input = input(f"{prompt} [{default}]: ").strip()
            if user_input == "":
                return default
        else:
            user_input = input(f"{prompt}: ").strip()
            if user_input == "" and default is not None:
                return default

        try:
            if input_type == int:
                value = int(user_input)
            elif input_type == float:
                value = float(user_input)
            else:
                return user_input

            if min_val is not None and value < min_val:
                print(f"  ⚠️ Значение должно быть >= {min_val}")
                continue
            if max_val is not None and value > max_val:
                print(f"  ⚠️ Значение должно быть <= {max_val}")
                continue

            return value

        except ValueError:
            print(f"  ⚠️ Введите число (тип: {input_type.__name__})")


def get_interval_input(file_name, file_duration, default_start=0, default_end=5):

    print(f"\n  Файл: {file_name}")
    print(f"  Длительность: {file_duration:.1f} сек")

    while True:
        print("\n  Варианты обрезки:")
        print("    1 - Обрезать с начала (0 - 5 сек)")
        print("    2 - Обрезать с указанного интервала")
        print("    3 - Взять весь файл")
        print("    4 - Пропустить файл")

        choice = input("\n  Выберите вариант [1-4]: ").strip()

        if choice == "1":
            start = 0
            end = min(5, file_duration)
            print(f"  ✓ Обрезано: {start:.1f} - {end:.1f} сек")
            return start, end, True

        elif choice == "2":
            print(f"\n  Введите интервал (от 0 до {file_duration:.1f} сек):")
            start = get_user_input("    Начало (сек)", default_start, float, 0, file_duration)
            end = get_user_input("    Конец (сек)", default_end, float, start, file_duration)
            print(f"  ✓ Обрезано: {start:.1f} - {end:.1f} сек")
            return start, end, True

        elif choice == "3":
            print(f"  ✓ Беру весь файл ({file_duration:.1f} сек)")
            return 0, file_duration, True

        elif choice == "4":
            print(f"  ✗ Файл пропущен")
            return 0, 0, False

        else:
            print("  ⚠️ Неверный выбор. Введите 1, 2, 3 или 4")


def interactive_batch_convert():

    clear_screen()

    print_header("КОНВЕРТАЦИЯ MP3 → WAV")

    print("\n📁 ШАГ 1: Укажите папки")
    print("-" * 40)

    input_folder = get_user_input(
        "  Путь к папке с MP3 файлами",
        default="D:/COS-3/Mp3",
        input_type=str
    )

    # Проверяем существование папки
    while not os.path.exists(input_folder):
        print(f"  ⚠️ Папка '{input_folder}' не существует!")
        input_folder = get_user_input(
            "  Введите правильный путь",
            default="D:/COS-3/Mp3",
            input_type=str
        )

    output_folder = get_user_input(
        "  Путь для сохранения WAV файлов",
        default="D:/COS-3/Wav",
        input_type=str
    )

    os.makedirs(output_folder, exist_ok=True)

    print("\n⚙️ ШАГ 2: Настройки обработки")
    print("-" * 40)

    # Частота дискретизации
    sr_options = {
        "1": 8000,
        "2": 16000,
        "3": 22050,
        "4": 44100,
        "5": 48000
    }

    print("\n  Выберите частоту дискретизации:")
    print("    1 - 8000 Гц (телефонное качество)")
    print("    2 - 16000 Гц (стандарт для речи)")
    print("    3 - 22050 Гц (рекомендуется для лабораторной)")
    print("    4 - 44100 Гц (CD качество)")
    print("    5 - 48000 Гц (DVD качество)")

    sr_choice = get_user_input("  Ваш выбор", default="3", input_type=str)
    sr = sr_options.get(sr_choice, 22050)
    print(f"  ✓ Частота: {sr} Гц")

    # Режим обрезки
    print("\n  Выберите режим обрезки:")
    print("    1 - Одинаковый интервал для всех файлов")
    print("    2 - Индивидуальный интервал для каждого файла")
    print("    3 - Взять весь файл без обрезки")

    trim_mode = get_user_input("  Ваш выбор", default="1", input_type=int, min_val=1, max_val=3)

    # Переменные для режима 1
    default_start = 0
    default_end = 5

    if trim_mode == 1:
        print("\n  Настройки интервала (для всех файлов):")
        default_start = get_user_input("    Начало (сек)", default=0, input_type=float, min_val=0)
        default_end = get_user_input("    Конец (сек)", default=5, input_type=float, min_val=default_start)
        print(f"  ✓ Интервал: {default_start:.1f} - {default_end:.1f} сек")

    print("\n🔍 ШАГ 3: Поиск MP3 файлов")
    print("-" * 40)

    mp3_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.mp3')]

    if not mp3_files:
        print(f"  ⚠️ В папке '{input_folder}' нет MP3 файлов!")
        return

    print(f"  Найдено MP3 файлов: {len(mp3_files)}")
    for i, f in enumerate(mp3_files, 1):
        print(f"    {i}. {f}")

    print("\n▶️ ШАГ 4: Подтверждение")
    print("-" * 40)
    print(f"  Папка исходников: {input_folder}")
    print(f"  Папка результатов: {output_folder}")
    print(f"  Частота: {sr} Гц")
    if trim_mode == 1:
        print(f"  Обрезка: {default_start:.1f} - {default_end:.1f} сек")
    elif trim_mode == 2:
        print(f"  Обрезка: индивидуальная для каждого файла")
    else:
        print(f"  Обрезка: весь файл")

    confirm = get_user_input("\n  Начать конвертацию? (да/нет)", default="да", input_type=str)
    if confirm.lower() not in ['да', 'yes', 'y', 'д']:
        print("  ✗ Конвертация отменена")
        return

    print("\n📀 ШАГ 5: Конвертация")
    print("-" * 40)

    success_count = 0
    error_count = 0
    skip_count = 0

    for i, filename in enumerate(mp3_files, 1):
        input_path = os.path.join(input_folder, filename)
        output_name = filename.replace('.mp3', '.wav').replace('.MP3', '.wav')
        output_path = os.path.join(output_folder, output_name)

        print(f"\n  [{i}/{len(mp3_files)}] {filename}")

        try:
            # Загрузка аудио
            signal, loaded_sr = librosa.load(input_path, sr=sr, mono=True)
            file_duration = len(signal) / sr

            # Определение интервала обрезки
            if trim_mode == 1:
                start = default_start
                end = default_end
                process_file = True

            elif trim_mode == 2:
                start, end, process_file = get_interval_input(filename, file_duration, default_start, default_end)

            else:
                # Весь файл
                start = 0
                end = file_duration
                process_file = True

            if not process_file:
                skip_count += 1
                continue

            # Обрезка
            start_sample = int(start * sr)
            end_sample = int(end * sr)

            if start_sample >= len(signal):
                print(f"    ⚠️ Начало {start:.1f} сек превышает длину, беру весь файл")
                trimmed = signal
            elif end_sample > len(signal):
                print(f"    ⚠️ Конец {end:.1f} сек превышает длину, обрезаю до конца")
                trimmed = signal[start_sample:]
            else:
                trimmed = signal[start_sample:end_sample]

            result_duration = len(trimmed) / sr
            print(f"    Результат: {result_duration:.1f} сек")

            # Сохранение
            sf.write(output_path, trimmed, sr)
            print(f"    ✓ Сохранён: {output_name}")
            success_count += 1

        except Exception as e:
            print(f"    ✗ Ошибка: {e}")
            error_count += 1

    print_header("РЕЗУЛЬТАТЫ КОНВЕРТАЦИИ")
    print(f"  Всего файлов: {len(mp3_files)}")
    print(f"  Успешно: {success_count}")
    print(f"  Пропущено: {skip_count}")
    print(f"  Ошибок: {error_count}")
    print(f"\n  Файлы сохранены в: {output_folder}")

    check = get_user_input("\n  Показать информацию о созданных файлах? (да/нет)", default="да", input_type=str)
    if check.lower() in ['да', 'yes', 'y', 'д']:
        print("\n📊 ИНФОРМАЦИЯ О ФАЙЛАХ")
        print("-" * 40)

        wav_files = [f for f in os.listdir(output_folder) if f.endswith('.wav')]
        if wav_files:
            print(f"\n{'Файл':<30} {'Длительность':<15}")
            print("-" * 45)
            for wav in sorted(wav_files):
                filepath = os.path.join(output_folder, wav)
                try:
                    sig, _ = librosa.load(filepath, sr=None)
                    dur = len(sig) / sr
                    print(f"{wav:<30} {dur:<15.1f} сек")
                except:
                    print(f"{wav:<30} ошибка чтения")
        else:
            print("  Нет WAV файлов в папке")

    print("\n✅ Готово!")


def quick_convert():
    clear_screen()

    print_header("БЫСТРАЯ КОНВЕРТАЦИЯ MP3 → WAV")

    # Только самые важные параметры
    input_folder = get_user_input("Папка с MP3 файлами", default="D:/COS-3/Mp3")
    output_folder = get_user_input("Папка для WAV файлов", default="D:/COS-3/Wav")

    # Проверяем существование
    while not os.path.exists(input_folder):
        print(f"  Папка '{input_folder}' не существует!")
        input_folder = get_user_input("Введите правильный путь", default="D:/COS-3/Mp3")

    os.makedirs(output_folder, exist_ok=True)

    # Стандартные параметры
    sr = 22050
    start_time = 0
    end_time = 5

    print(f"\nПараметры:")
    print(f"  Частота: {sr} Гц")
    print(f"  Обрезка: {start_time} - {end_time} сек")

    confirm = get_user_input("\nНачать конвертацию?", default="да")
    if confirm.lower() not in ['да', 'yes', 'y', 'д']:
        print("Отменено")
        return

    # Поиск и конвертация
    mp3_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.mp3')]

    if not mp3_files:
        print("Нет MP3 файлов!")
        return

    success = 0
    for filename in mp3_files:
        input_path = os.path.join(input_folder, filename)
        output_name = filename.replace('.mp3', '.wav').replace('.MP3', '.wav')
        output_path = os.path.join(output_folder, output_name)

        try:
            signal, _ = librosa.load(input_path, sr=sr, mono=True)

            if len(signal) > sr * end_time:
                signal = signal[:int(sr * end_time)]

            sf.write(output_path, signal, sr)
            print(f"  ✓ {filename} → {output_name}")
            success += 1

        except Exception as e:
            print(f"  ✗ {filename}: {e}")

    print(f"\n✅ Готово! Обработано {success} из {len(mp3_files)} файлов")
    print(f"Файлы сохранены в: {output_folder}")


if __name__ == "__main__":
    print_header("КОНВЕРТЕР MP3 → WAV")
    print("  Выберите режим работы:")
    print("    1 - Полный интерактивный режим (все вопросы)")
    print("    2 - Быстрый режим (только папки, остальное по умолчанию)")
    print("    3 - Выход")

    choice = get_user_input("\n  Ваш выбор", default="1", input_type=int, min_val=1, max_val=3)

    if choice == 1:
        interactive_batch_convert()
    elif choice == 2:
        quick_convert()
    else:
        print("\n  До свидания!")