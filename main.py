#Shay Bechor & Gil Ashkenazi

import numpy as np
import pyaudio
import time

sample_rate = 22050
chunk = 2048
avg_num_fft = 10
samples_per_fft = chunk * avg_num_fft
freq_step = float(sample_rate) / samples_per_fft

notes = str.split('C C# D D# E F F# G G# A A# B')
target_frequencies = [82.41, 110.00, 146.83, 196.00, 246.94, 329.63]


def freq_to_midi(f):
    m = 69 + 12 * np.log2(f / 440.0)
    return m


def note_name(m):
    name = notes[m % 12] + str(m // 12 - 1)
    return name


buffer = np.zeros(samples_per_fft, dtype=np.float32)
num_frames = 0
window = 0.5 * (1 - np.cos(np.linspace(0, 2 * np.pi, samples_per_fft)))

# Initialize audio
stream = pyaudio.PyAudio().open(format=pyaudio.paInt16, channels=1, rate=sample_rate, input=True,
                                frames_per_buffer=chunk)
stream.start_stream()

while stream.is_active():

    data_in = np.frombuffer(stream.read(chunk), np.int16)

    buffer[:-chunk] = buffer[chunk:]
    buffer[-chunk:] = data_in

    num_frames += 1

    fft = np.fft.rfft(buffer * window)

    frequency = (np.abs(fft[65:337]).argmax() + 65) * freq_step

    midi = freq_to_midi(frequency)
    midi_round = int(round(midi))
    name = note_name(midi_round)
    num_frames += 1

    if num_frames >= avg_num_fft:

        closest_frequency = min(target_frequencies, key=lambda x: abs(x - frequency))
        deviation = frequency - closest_frequency

        print(f"sampled frequency: {frequency:.2f} Hz")
        print(f"closest frequency: {closest_frequency:.2f} Hz")
        print(f"closest note name: {name}")

        if deviation > -1 and deviation < 1:
            print("In tune")
        elif deviation > 1:
            print("Tune down")
        elif deviation < -1:
            print("Tune up")

            # reset for next sample
        num_frames = 0
        print("\n")
        time.sleep(0.5)