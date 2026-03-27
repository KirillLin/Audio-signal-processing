"""
Конфигурационный файл
"""

# Параметры обработки
SAMPLE_RATE = 22050
DURATION = 3  # секунд для синтетических сигналов

# Параметры спектрального анализа
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
N_MFCC = 13

# Диапазон SNR для экспериментов (3-18 дБ, шаг 3)
SNR_VALUES = [3, 6, 9, 12, 15, 18]

# Музыкальные жанры для анализа
MUSIC_GENRES = ['Jazz', 'Rock', 'Classical', 'Electronic']

# Пути к файлам
AUDIO_DIR = 'audio'
RESULTS_DIR = 'results'
MODELS_DIR = 'models'

# Целевые частоты для метрик
PESQ_TARGET_SR = 16000
STOI_TARGET_SR = 16000

# Цвета для графиков
COLORS = {
    'centroid': 'red',
    'rolloff': 'green',
    'bandwidth': 'blue',
    'zcr': 'purple',
    'noisy': 'orange',
    'enhanced': 'blue',
    'ideal': 'gray'
}