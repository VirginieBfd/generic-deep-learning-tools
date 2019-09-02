import numpy as np
from scipy.signal import welch
from scipy.fftpack import fft


class TimeFrequencyTransformator(object):
    def __init__(self, frequency, n_samples, t_n):
        """
        :type frequency: flot
        :type n_samples: int
        :type t_n: float
        """
        self._frequency = frequency
        self._n_samples = n_samples
        self._t_n = t_n
        self._T = t_n / n_samples

    @property
    def T(self):
        """
        :rtype: float
        """
        return self._T

    @property
    def frequency(self):
        """
        :rtype: float
        """
        return self._frequency

    @property
    def n_samples(self):
        """
        :rtype: int
        """
        return self._n_samples

    @property
    def t_n(self):
        """
        :rtype: float
        """
        return self._t_n

    def get_fft_values(self, y_values):
        """The Fourier Transform transforms a signal to the frequency-domain space and therefore shows at which
        frequencies the component signals oscillate. The Fast Fourier Transform (FFT) is an efficient algorithm for
        calculating the Discrete Fourier Transform (DFT) and is the de facto standard to calculate a Fourier Transform.
        :type y_values: numpy.array
        :rtype: numpy.array, numpy.array
        """
        f_values = np.linspace(0.0, 1.0 / (2.0 * self.T), self.n_samples // 2)
        fft_values_ = fft(y_values)
        fft_values = 2.0 / self.n_samples * np.abs(fft_values_[0 : self.n_samples // 2])
        return f_values, fft_values

    def get_values(self, y_values):
        """
        :type y_values: numpy.array
        :rtype: numpy.array, numpy.array
        """
        y_values = y_values
        x_values = [self.T * kk for kk in range(0, len(y_values))]
        return x_values, y_values

    def get_psd_values(self, y_values):
        """Closely related to the Fourier Transform is the concept of Power Spectral Density. Similar to the FFT, it
        describes the frequency spectrum of a signal. But in addition to the FFT it also takes the power distribution at
        each frequency (bin) into account. Generally speaking the locations of the peaks in the frequency spectrum will
        be the same as in the FFT-case, but the height and width of the peaks will differ. The surface below the peaks
        corresponds with the power distribution at that frequency.
        :type y_values: numpy.array
        :rtype: numpy.array, numpy.array
        """
        return welch(y_values, fs=self.frequency)

    def get_autocorr_values(self, y_values):
        """The auto-correlation function calculates the correlation of a signal with a time-delayed version of itself.
        The idea behind it is that if a signal contain a pattern which repeats itself after a time-period of tau
        seconds, there will be a high correlation between the signal and a tau sec delayed version of the signal.
        :type y_values: numpy.array
        :rtype: numpy.array, numpy.array
        """
        x_values = np.array([self.T * jj for jj in range(0, self.n_samples)])
        y_values = np.correlate(y_values, y_values, mode="full")
        y_values = y_values[len(y_values) // 2 :]
        return x_values, y_values
