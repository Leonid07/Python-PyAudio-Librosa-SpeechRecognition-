#Speech Recognition
#PyAudio
#librosa
import matplotlib.pyplot as plt
import librosa.display
import librosa
import seaborn as sns
import speech_recognition as speech_r
import speech_recognition as sr 
import pyaudio
import wave
from matplotlib import pyplot as plt

CHUNK = 1024 # определяет форму ауди сигнала
FRT = pyaudio.paInt16 # шестнадцатибитный формат задает значение амплитуды
CHAN = 1 # канал записи звука
RT = 44100 # частота 
REC_SEC = 5 #длина записи
OUTPUT = "output.wav"

while True:
    comand = input("Введите команду ")
##### Запись голоса и транскрипция
    if comand == "rec":
        p = pyaudio.PyAudio()
        print("start")
        stream = p.open(format=FRT,channels=CHAN,rate=RT,input=True,frames_per_buffer=CHUNK) # открываем поток для записи
        frames = [] # формируем выборку данных фреймов
        for i in range(0, int(RT / CHUNK * REC_SEC)):
            data = stream.read(CHUNK)
            frames.append(data)
        print("done")

        stream.stop_stream() # останавливаем и закрываем поток 
        stream.close()
        p.terminate()

        w = wave.open(OUTPUT, 'wb')
        w.setnchannels(CHAN)
        w.setsampwidth(p.get_sample_size(FRT))
        w.setframerate(RT)
        w.writeframes(b''.join(frames))
        w.close()

        sample = speech_r.WavFile('C:\\Users\\User\\Desktop\\diplom\\output.wav')

        r = speech_r.Recognizer()

        with sample as audio:
            content = r.record(audio)

        with sample as audio:
            content = r.record(audio)
            r.adjust_for_ambient_noise(audio)

        user_audio_file = sr.AudioFile("C:\\Users\\User\\Desktop\\diplom\\output.wav")
        with user_audio_file as source:
            user_audio = r.record(source)
        text = r.recognize_google(user_audio, language='ru-RU')
        print(text)
#####
##### Тепловая карта
    if comand == "HM":
        y, sr = librosa.load(OUTPUT)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        print(tempo)
        print(beat_frames)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=8192, n_mfcc=12)
        sns.heatmap(mfcc)
        plt.show()
##### Хромограмма
    if comand == "CH":
        y, sr = librosa.load(OUTPUT)
        chromagram = librosa.feature.chroma_cqt(y=y, sr=sr)
        sns.heatmap(chromagram)
        plt.show()
#####
##### Звуковая дорожка
    if comand == "ST":
        y, sr = librosa.load(OUTPUT)
        plt.figure(figsize=(14, 5))
        librosa.display.waveshow(y, sr)
        plt.show()
#####
##### Спектрограмма
    if comand == "SP":
        y,sr=librosa.load(OUTPUT,sr=None,duration=None)
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        plt.figure(figsize=(14,5))
        plt.subplot(311)
        plt.plot(cent[0])
        plt.xlabel('sample')
        plt.ylabel('frequency')
        plt.show()
    if comand == "stop":
        break
#####