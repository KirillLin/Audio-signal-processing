"""
Конфигурационный файл
"""

# Параметры обработки
SAMPLE_RATE = 22050
DURATION = 3

# Параметры спектрального анализа
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
N_MFCC = 13

SNR_VALUES = [3, 6, 9, 12, 15, 18]

MUSIC_GENRES = ['Jazz', 'Rock', 'Classical', 'Electronic']

AUDIO_DIR = 'audio'
RESULTS_DIR = 'results'
MODELS_DIR = 'models'

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