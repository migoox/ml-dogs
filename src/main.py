import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

def specgram():
  sample_rate, samples = wavfile.read('audio.wav')
  frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

  plt.pcolormesh(times, frequencies, spectrogram)
  plt.imshow(spectrogram)
  plt.ylabel('Frequency [Hz]')
  plt.xlabel('Time [sec]')
  plt.show()

def main():
  specgram()

if __name__ == "__main__":
  main()