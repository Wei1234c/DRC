import json
import time
import wave

import matplotlib.pyplot as plt
import numpy as np
import pyaudio
from matplotlib import colors as mcolors
from scipy import signal

import thinkdsp


try:
    import IPython.display
except ImportError as e:
    print(e)

SOUND_SPEED = 34320  # 34320 cm/sec
B = bel = np.log(10) / 2
dB = decibel = bel / 10

PI2 = np.pi * 2
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name) for name, color in colors.items())
COLOR_NAMES = [name for hsv, name in by_hsv]

# DEFAULT_AMPLITUDE = 1
DEFAULT_FRAMERATE = 44100
CHUNK_SIZE = 1024
CHUNKS_PER_SEC = int(DEFAULT_FRAMERATE / CHUNK_SIZE)
DATA_TYPE = np.complex
# DATA_TYPE = np.int16
SAMPLE_WIDTH = 2
SAMPLE_WIDTH_DTYPES = {1: np.uint8, 2: np.int16, 4: np.float32, 8: np.float64}
# SAMPLE_WIDTH_DTYPES = {1: np.uint8, 2: np.int16, 4: np.int32, 8: np.int16}
NCHANNELS = 2

LOG_TIME_RESOLUTION_SEC = 0.1
LOG_WINDOW_SIZE = int(CHUNKS_PER_SEC * LOG_TIME_RESOLUTION_SEC * 2)

REC_DURATION = 5
FILE_NAME = 'output.wav'

SHOW_FRAMES = 300
ALPHA = 0.6
CHANNEL_COLORS = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']
CHANNEL_COLORS.extend(COLOR_NAMES)
LEGEND_LOC = 'upper right'



class Sound(pyaudio.PyAudio):

    def __init__(self):
        super().__init__()


    def __enter__(self):
        return self


    def __exit__(self, *args):
        self.__del__()


    def __del__(self):
        # self.terminate()
        pass


    @classmethod
    def scan_devices(cls, hostApi = 0):
        input_devices = {}
        output_devices = {}

        with cls() as s:
            info = s.get_host_api_info_by_index(hostApi)
            numdevices = info.get('deviceCount')

            for i in range(numdevices):
                device_info = s.get_device_info_by_host_api_device_index(hostApi, i)

                if device_info.get('maxInputChannels') > 0:
                    input_devices[i] = dict(name = device_info['name'],
                                            maxInputChannels = device_info['maxInputChannels'],
                                            maxOutputChannels = device_info['maxOutputChannels'],
                                            defaultSampleRate = device_info['defaultSampleRate'])

                if device_info.get('maxOutputChannels') > 0:
                    output_devices[i] = dict(name = device_info['name'],
                                             maxInputChannels = device_info['maxInputChannels'],
                                             maxOutputChannels = device_info['maxOutputChannels'],
                                             defaultSampleRate = device_info['defaultSampleRate'])

        return dict(info = info, input_devices = input_devices, output_devices = output_devices)


    @classmethod
    def _get_spectrum_freq_idx(cls, freq, chunk_size, framerate):
        max_freq = framerate / 2
        freq_segments = chunk_size // 2
        freq_resolution = max_freq / freq_segments
        freq_idx = np.round(freq / freq_resolution).astype(int)
        return freq_idx


    @classmethod
    def _set_spectrum_properties(cls, obj, chunk_size, framerate):
        obj.chunk_size = chunk_size
        obj.framerate = framerate
        obj.max_freq = framerate / 2
        obj.freq_segments = obj.chunk_size // 2
        obj.freq_resolution = obj.max_freq / obj.freq_segments
        obj.fs = np.linspace(0, obj.max_freq, obj.freq_segments)
        obj.freq_to_idx = lambda freq: np.round(freq / obj.freq_resolution).astype(int)
        obj.idx_to_freq = lambda idx: idx * obj.freq_resolution


    @classmethod
    def _get_datatype_max_value(cls, data_type):
        data_type = np.dtype(data_type)
        width = data_type.itemsize
        max_value = np.iinfo(data_type).max if data_type.kind in 'iu' else 1.0
        return max_value, data_type, width


    @property
    def datatype_max_value(self):
        max_value, _, _ = self._get_datatype_max_value(self.data_type)
        return max_value


    @classmethod
    def _normalize(cls, x, force = False, amp = 1.0):
        max_val = np.max(abs(x))
        if force or max_val > amp:
            x = x / max_val * amp
        return x


    @classmethod
    def _is_float(cls, data):
        data_type = data.dtype.type
        return issubclass(data_type, float) or \
               issubclass(data_type, np.float) or \
               issubclass(data_type, np.float16) or \
               issubclass(data_type, np.float32)


    @classmethod
    def _is_complex(cls, data):
        return issubclass(data.dtype.type, np.complex)


    @classmethod
    def _convert_datatype(cls, from_data, astype = DATA_TYPE):
        max_value, data_type, width = cls._get_datatype_max_value(astype)
        converted_data = from_data

        if from_data.dtype != astype and (cls._is_float(from_data) or cls._is_complex(from_data)):
            val_max = np.max(abs(from_data))
            if val_max > 1:
                from_data = from_data / val_max
            converted_data = (from_data * max_value).astype(data_type)

        return converted_data, width


    def _play_data(self, data, format, framerate = DEFAULT_FRAMERATE):
        nchannels = data.shape[0]
        stream = self.open(format = format, channels = nchannels, rate = framerate, output = True)

        frames = data.T.ravel().tostring()
        num_frames = data.shape[1]
        stream.write(frames, num_frames)
        stream.stop_stream()
        stream.close()


    @classmethod
    def pi_to_pi(cls, theta, boundry = np.pi, unwrap = False, discont = np.pi, rad_to_degree = False):
        theta = np.asarray(theta)
        theta = theta - np.round(theta / (2 * boundry)) * (2 * boundry)
        if unwrap:
            theta = np.unwrap(theta, discont = discont)
        if rad_to_degree:
            theta = np.rad2deg(theta)
        return theta


    @classmethod
    def powers(cls, signals):
        n_samples = signals.shape[-1]
        hs = np.fft.rfft(signals * 2 / n_samples)
        return abs(hs) ** 2


    @classmethod
    def power_sums(cls, signals):
        return np.sum(cls.powers(signals), axis = -1)


    @classmethod
    def play_wave_file(cls, file_name):
        with WaveFile(file_name) as wf:
            wf.play()


    @classmethod
    def play_thinkdsp_wave(cls, thinkdsp_wave):
        with Channel.load_thinkdsp_wave(thinkdsp_wave) as c:
            c.play()


    def make_audio(self):
        return IPython.display.Audio(data = self.data, rate = self.framerate)


    @classmethod
    def _save_wave_file(cls, frames, file_name = FILE_NAME, sample_width = SAMPLE_WIDTH, channels = NCHANNELS,
                        framerate = DEFAULT_FRAMERATE):

        with wave.open(file_name, 'wb') as wf:
            wf.setsampwidth(sample_width)
            wf.setnchannels(channels)
            wf.setframerate(framerate)
            wf.writeframes(b''.join(frames))

        return WaveFile(file_name)


    @classmethod
    def _save_channels(cls, channels, file_name = FILE_NAME, sample_width = SAMPLE_WIDTH,
                       framerate = DEFAULT_FRAMERATE):

        data_type = SAMPLE_WIDTH_DTYPES[sample_width]
        nchannels = len(channels)
        data, _ = cls._convert_datatype(np.vstack(channels), astype = data_type)
        frames = [data.T.ravel().tostring()]

        return cls._save_wave_file(frames, file_name = file_name, sample_width = sample_width, channels = nchannels,
                                   framerate = framerate)


    @classmethod
    def record(cls, duration = REC_DURATION, wire = True, save_wave_file = True, file_name = FILE_NAME,
               input_device_index = None, nchannels_per_input = 1, sample_width = SAMPLE_WIDTH,
               framerate = DEFAULT_FRAMERATE, chunk = CHUNK_SIZE):

        with Streamer(input_device_indices = input_device_index, nchannels_per_input = nchannels_per_input, wire = wire,
                      save_file = save_wave_file, file_name = file_name, sample_width = sample_width,
                      framerate = framerate, chunk_size = chunk) as streamer:

            start_time = time.time()
            while time.time() - start_time < duration:
                streamer.read_chunk()



class Channel(thinkdsp.Wave, Sound):

    def __init__(self, ys = np.zeros(3, dtype = DATA_TYPE), ts = None, framerate = DEFAULT_FRAMERATE):

        ys = np.asarray(ys).ravel()
        ys, _ = self._convert_datatype(ys)
        thinkdsp.Wave.__init__(self, ys, ts, framerate)
        Sound.__init__(self)


    @property
    def real(self):
        return Channel(self.ys.real, self.ts, self.framerate)


    @property
    def imag(self):
        return Channel(self.ys.imag, self.ts, self.framerate)


    @property
    def data_type(self):
        return self.ys.dtype


    @property
    def data(self):
        return self.ys.reshape((1, -1))


    def get_data_chunks(self, chunk_size = CHUNK_SIZE):

        data = self.data.ravel()
        length = len(data)
        start = 0

        while start < length:
            end = start + chunk_size
            chunk = data[start:end]

            if end > length:
                chunk = np.pad(chunk, pad_width = (0, end - length), mode = 'constant', constant_values = 0)

            yield chunk

            start += chunk_size


    @property
    def format(self):
        width = 4 if issubclass(self.data_type.type, np.complex) else self.data_type.itemsize
        return self.get_format_from_width(width)


    def __add__(self, other):
        wave = thinkdsp.Wave.__add__(self, other)
        return Channel.load_thinkdsp_wave(wave)


    def __or__(self, other):
        wave = thinkdsp.Wave.__or__(self, other)
        return Channel.load_thinkdsp_wave(wave)


    def __mul__(self, other):
        wave = thinkdsp.Wave.__mul__(self, other)
        return Channel.load_thinkdsp_wave(wave)


    def apodize(self, denom = 20, duration = 0.1):
        self.ys = thinkdsp.apodize(self.ys, self.framerate, denom = denom, duration = duration).astype(self.data_type)
        return self


    def convolve(self, other):
        wave = thinkdsp.Wave.convolve(self, other)
        return Channel.load_thinkdsp_wave(wave)


    def diff(self):
        wave = thinkdsp.Wave.diff(self)
        return Channel.load_thinkdsp_wave(wave)


    def cumsum(self):
        wave = thinkdsp.Wave.cumsum(self)
        return Channel.load_thinkdsp_wave(wave)


    def _clear(self):
        self.ys = np.zeros_like(self.ys)


    def hamming(self):
        self.ys = (self.ys * np.hamming(len(self.ys))).astype(self.data_type)


    def window(self, window):
        self.ys = (self.ys * window).astype(self.data_type)


    def scale(self, factor):
        self.ys = (self.ys * factor).astype(self.data_type)


    def slice(self, i, j):
        ys = self.ys[i:j].copy()
        ts = self.ts[i:j].copy()
        return Channel(ys, ts, self.framerate)


    def trim(self, diff_threshold = 350, margin = 3):

        def trim_by_diff_threshold(ys, diff_threshold, margin):

            def get_boundries(ys, diff_threshold, margin):
                ys_diff = np.diff(ys)
                idx = np.argwhere(ys_diff > diff_threshold).ravel()
                idx_start, idx_end = 0, len(ys)

                if len(idx) > 1:
                    idx_start = idx[0] + margin
                    idx_end = idx[1] - margin

                return idx_start, idx_end


            idx_start, idx_end = get_boundries(ys, diff_threshold, margin)

            return ys[idx_start:idx_end].copy()


        ys = trim_by_diff_threshold(self.ys, diff_threshold = diff_threshold, margin = margin)

        return Channel(ys, framerate = self.framerate)


    def play(self):
        self._play_data(self.data.astype(np.float32) if issubclass(self.data_type.type, np.complex) else
                        self.data,
                        self.format, self.framerate)


    def freqs_amps(self):
        spectrum = self.make_spectrum()
        amps_freqs = np.vstack(spectrum.peaks())

        freqs_amps = np.zeros((amps_freqs.shape[0], 4))
        freqs_amps[:, 0] = amps_freqs[:, 1]  # freq
        freqs_amps[:, 1] = 2 * amps_freqs[:, 0] / len(self)  # amps
        freqs_amps[:, 2] = 20 * np.log10(freqs_amps[:, 1])  # dB
        freqs_amps[:, 3] = freqs_amps[:, 1] ** 2  # power
        return freqs_amps


    @classmethod
    def load_thinkdsp_wave(cls, thinkdsp_wave, data_type = DATA_TYPE):
        # if np.max(abs(thinkdsp_wave.ys)) > 1:
        #     thinkdsp_wave.normalize()

        # ys, _ = cls._convert_datatype(thinkdsp_wave.ys, astype = data_type)
        ys = thinkdsp_wave.ys
        channel = cls(ys, thinkdsp_wave.ts, thinkdsp_wave.framerate)
        return channel


    def save(self, file_name = FILE_NAME, data_type = DATA_TYPE):
        data, width = self._convert_datatype(self.ys, data_type)
        if self._is_complex(data):
            data, width = data.astype(np.float32), 4
        frames = [data.ravel().tostring()]
        return self._save_wave_file(frames, file_name = file_name, sample_width = width, channels = 1,
                                    framerate = self.framerate)


    def merge(self, channel, file_name = FILE_NAME):
        ys, width = self._convert_datatype(channel.ys, self.data_type)
        data = np.vstack((self.ys, ys))
        frames = [data.T.ravel().tostring()]
        return self._save_wave_file(frames, file_name = file_name, sample_width = width, channels = NCHANNELS,
                                    framerate = self.framerate)


    def show(self, show_frames = SHOW_FRAMES, color = None, label = None, alpha = ALPHA, **kwargs):
        show_frames = show_frames if isinstance(show_frames, slice) else slice(-show_frames, None)
        ax = plt.gca()
        ax.plot(self.ts[show_frames], self.ys[show_frames], color = color, label = label, alpha = alpha, **kwargs)


    def _normalize_and_average(self, ys):
        length = len(ys)
        ys = ys / self.datatype_max_value
        ys = 2 * ys / length
        return ys


    def make_normailized_spectrum(self, full = False):
        self.ys = self._normalize_and_average(self.ys)
        return self.make_spectrum(full)


    def make_spectrum(self, full = True):
        sp = thinkdsp.Wave.make_spectrum(self, full)
        # sp.hs[0] = 0
        return Timber(sp)


    def make_envelope(self):
        return Channel(np.abs(signal.hilbert(self.ys)), framerate = self.framerate)


    def make_envelope_spectrum(self, full = False):
        sp = self.make_envelope().make_spectrum(full)
        sp.hs[0] = 0
        return sp


    def make_harmonic_phase_shift_spectrum(self, ratio = 2, freqs = None, unwrap = False, rad_to_degree = True,
                                           full = False):
        sp = self.make_spectrum(full)
        idx_base = sp.freqs_indices(freqs) if freqs is not None else np.arange(len(sp.fs))
        idx_harmonic = (idx_base / ratio).astype(int)

        freqs_base = sp.fs[idx_base]
        freqs_harmonic = sp.fs[idx_harmonic]
        phase_shifts = sp.delta_phases(freqs_base, freqs_harmonic, unwrap = unwrap, rad_to_degree = rad_to_degree)
        phase_shifts = self.pi_to_pi(phase_shifts, unwrap = unwrap, rad_to_degree = rad_to_degree)

        ph_series = np.zeros_like(sp.fs)
        ph_series[idx_base] = phase_shifts
        return sp.fs, ph_series


    def make_envelope_phase_shift_spectrum(self, freqs = None, unwrap = False, rad_to_degree = True, full = False):
        sp = self.make_spectrum(full)
        spe = self.make_envelope_spectrum(full)
        idx = sp.freqs_indices(freqs) if freqs is not None else np.arange(len(sp.fs))

        phase_shifts = (np.angle(sp.hs) - np.angle(spe.hs))[idx]
        phase_shifts = self.pi_to_pi(phase_shifts, unwrap = unwrap, rad_to_degree = rad_to_degree)

        ph_series = np.zeros_like(sp.fs)
        ph_series[idx] = phase_shifts
        return sp.fs, ph_series


    def rank_by_amps(self):
        return self.make_spectrum().rank_by_amps()


    def filter(self, low_pass_cutoff = None, high_pass_cutoff = None, band_pass = None, band_stop = None,
               amp_min = None, full = True):
        sp = self.make_spectrum(full)
        filtered = sp.filter(low_pass_cutoff = low_pass_cutoff, high_pass_cutoff = high_pass_cutoff,
                             band_pass = band_pass, band_stop = band_stop, amp_min = amp_min)
        return filtered.make_wave()


    def filter_freqs_pick(self, freqs):
        sp = self.make_spectrum()
        filtered = sp.filter_freqs_pick(freqs)
        return filtered.make_wave()


    def freqs_attributes(self, freqs):
        freqs, indices, fs, phases, amps = self.make_spectrum().freqs_attributes(freqs)
        return freqs, indices, fs, phases, amps


    def delta_phases(self, freq_base, freqs, unwrap = False, rad_to_degree = True):
        return self.make_spectrum().delta_phases(freq_base, freqs, unwrap = unwrap, rad_to_degree = rad_to_degree)


    def delay(self, delta_t):
        sp = self.make_spectrum()
        delayed = sp.delay(delta_t)
        return delayed.make_wave()



class WaveFile(Sound, wave.Wave_read):

    def __init__(self, f):
        self.file_name = f
        wave.Wave_read.__init__(self, f)
        Sound.__init__(self)
        self._initialize()
        wave.Wave_read.close(self)  # close Wave_read


    @property
    def data_type(self):
        return SAMPLE_WIDTH_DTYPES[self.sample_width]


    @property
    def format(self):
        return self.get_format_from_width(4 if issubclass(self.data.dtype.type, np.complex) else
                                          self.sample_width)


    @property
    def data(self):
        return np.vstack([channel.ys for channel in self.channels])


    def data_normalized(self, amp = 1.0, use_datatype_max_value = True):
        max_value = self.datatype_max_value if use_datatype_max_value else np.max(abs(self.data))
        return (amp * self.data / max_value).astype(np.float32)


    def normalize(self, amp = 1.0, use_datatype_max_value = True):
        normailized_data = self.data_normalized(amp = amp, use_datatype_max_value = use_datatype_max_value)

        for i, channel in enumerate(self.channels):
            channel.ys = normailized_data[i]

        self.sample_width = 4


    def get_data_chunks(self, chunk_size = CHUNK_SIZE):
        return self.get_chunks(self.data, chunk_size)


    @classmethod
    def get_chunks(cls, data, chunk_size = CHUNK_SIZE):

        length = data.shape[1]
        start = 0

        while start < length:
            end = start + chunk_size
            chunk = data[:, start:end]

            if end > length:
                chunk = np.pad(chunk, pad_width = [(0, 0), (0, end - length)], mode = 'constant', constant_values = 0)

            yield chunk

            start += chunk_size


    @property
    def nchannels(self):
        return len(self.channels)


    @property
    def nframes(self):
        return self.data.shape[1]


    def _initialize(self):
        params = self.getparams()._asdict()
        self.framerate = params['framerate']
        self.sample_width = params['sampwidth']
        nchannels = params['nchannels']
        nframes = params['nframes']

        self.ts = np.arange(nframes) / self.framerate
        frames = self.readframes(nframes)
        data = np.fromstring(frames, self.data_type)

        self.channels = [Channel(data[i::nchannels], self.ts, self.framerate) for i in range(nchannels)]


    def apodize(self, denom = 20, duration = 0.1):
        # for channel in self.channels:
        #     channel.ys = thinkdsp.apodize(channel.ys, channel.framerate, denom = denom, duration = duration).astype(
        #         channel.data_type)
        for channel in self.channels:
            channel.apodize(denom = denom, duration = duration)


    def make_spectrums(self, full = True):
        sps = Timbers()
        sps.timbers = [ch.make_spectrum(full) for ch in self.channels]
        return sps


    def filter(self, low_pass_cutoff = None, high_pass_cutoff = None, band_pass = None, band_stop = None,
               amp_min = None):
        return Blender.dump_thinkdsp_waves([channel.filter(low_pass_cutoff = low_pass_cutoff,
                                                           high_pass_cutoff = high_pass_cutoff, band_pass = band_pass,
                                                           band_stop = band_stop, amp_min = amp_min) for channel in
                                            self.channels], sample_width = self.sample_width)


    def filter_freqs_pick(self, freqs):
        return Blender.dump_thinkdsp_waves([channel.filter_freqs_pick(freqs) for channel in self.channels],
                                           sample_width = self.sample_width)


    def delay(self, delta_t):
        return Blender.dump_thinkdsp_waves([channel.delay(delta_t) for channel in self.channels],
                                           sample_width = self.sample_width)


    def make_envelopes(self):
        channels = [ch.make_envelope() for ch in self.channels]
        return Blender.dump_thinkdsp_waves(channels)


    def make_envelope_spectrums(self, full = False):
        sps = Timbers()
        sps.timbers = [ch.make_envelope_spectrum(full) for ch in self.channels]
        return sps


    def play(self):
        try:
            self._play_data(self.data.astype(np.float32) if issubclass(self.data.dtype.type, np.complex) else
                            self.data,
                            self.format, self.framerate)
        except OSError as e:
            print(e, '\nWave file has {} channels.'.format(self.nchannels))


    def show(self, show_frames = SHOW_FRAMES, colors = CHANNEL_COLORS, alpha = ALPHA, **kwargs):
        for i in range(self.nchannels):
            self.channels[i].show(show_frames = show_frames,
                                  color = colors[i] if len(colors) >= self.nchannels else None,
                                  label = 'channel {}'.format(i), alpha = alpha, **kwargs)
        plt.legend(loc = LEGEND_LOC)
        plt.tight_layout()  # plt.show()


    def swap_channels(self, swap_idx = (0, 1)):
        if len(self.channels) >= 2:
            temp = self.channels[swap_idx[0]]
            self.channels[swap_idx[0]] = self.channels[swap_idx[1]]
            self.channels[swap_idx[1]] = temp
            self.save()
            self.__init__(self.file_name)


    def save(self, file_name = None):
        file_name = file_name if file_name else self.file_name
        frames = [self.data.astype(self.data_type).T.ravel().tostring()]
        return self._save_wave_file(frames,
                                    file_name = file_name,
                                    sample_width = self.sample_width,
                                    channels = self.nchannels,
                                    framerate = self.framerate)


    def segment(self, start = None, duration = None):
        channels_data = [channel.segment(start = start, duration = duration).ys for channel in self.channels]
        return self._save_channels(channels_data, sample_width = self.sample_width, framerate = self.framerate)



class Timbers:

    def __init__(self, spectrums = None):
        if spectrums:
            self.timbers = [Timber(sp) for sp in spectrums]


    @classmethod
    def extract(cls, wav_file, start = 0, duration = None):
        wf = WaveFile(wav_file)
        spectrums = [ch.segment(start = start, duration = duration).make_spectrum() for ch in wf.channels]
        return Timbers(spectrums)


    def filter(self, low_pass_cutoff = None, high_pass_cutoff = None, band_pass = None, band_stop = None,
               amp_min = None):
        filtered_timbers = [
            timber.filter(low_pass_cutoff = low_pass_cutoff, high_pass_cutoff = high_pass_cutoff, band_pass = band_pass,
                          band_stop = band_stop, amp_min = amp_min) for timber in self.timbers]
        return Timbers(filtered_timbers)


    def filter_freqs_pick(self, freqs):
        filtered_timbers = [timber.filter_freqs_pick(freqs) for timber in self.timbers]
        return Timbers(filtered_timbers)


    def delay(self, delta_t):
        delayed_timbers = [timber.delay(delta_t) for timber in self.timbers]
        return Timbers(delayed_timbers)


    def plot(self, freq_lims = (1, DEFAULT_FRAMERATE // 2), xticks = None, **options):
        i = 0
        for timber in self.timbers:
            timber.plot(freq_lims = freq_lims, xticks = xticks, color = 'C' + str(i), **options)
            i += 1


    def make_waves(self):
        wfs = [timber.make_wave() for timber in self.timbers]
        wf = Blender.dump_thinkdsp_waves(wfs)
        return wf


    def play(self):
        wf = self.make_waves()
        wf.play()



class Timber(thinkdsp.Spectrum):

    def __init__(self, spectrum):
        super().__init__(spectrum.hs, spectrum.fs, spectrum.framerate, spectrum.full)
        self.phases = self.angles


    @classmethod
    def extract(cls, wav_file, channel_idx = 0, start = 0, duration = None):
        seg = WaveFile(wav_file).channels[channel_idx].segment(start = start, duration = duration)
        sp = seg.make_spectrum()

        return Timber(sp)


    def dump(self, file_name = 'profile.json'):
        profile = {'hs_real'  : self.hs.real.tolist(), 'hs_imag': self.hs.imag.tolist(), 'fs': self.fs.tolist(),
                   'framerate': self.framerate, 'full': self.full}

        with open(file_name, 'w') as f:
            json.dump(profile, f)


    @classmethod
    def load(cls, file_name = 'profile.json'):
        with open(file_name, 'r') as f:
            profile = json.load(f)

        profile['hs'] = np.array(profile['hs_real']) + 1j * np.array(profile['hs_imag'])
        sp = thinkdsp.Spectrum(profile['hs'], profile['fs'], profile['framerate'], profile['full'])

        return Timber(sp)


    def band_pass(self, low_cutoff, high_cutoff, factor = 0):
        fs = abs(self.fs)
        indices = (fs < low_cutoff) | (fs > high_cutoff)
        self.hs[indices] *= factor


    def filter(self, low_pass_cutoff = None, high_pass_cutoff = None, band_pass = None, band_stop = None,
               amp_min = None):
        filtered = self.copy()

        if high_pass_cutoff:
            filtered.high_pass(high_pass_cutoff)

        if low_pass_cutoff:
            filtered.low_pass(low_pass_cutoff)

        if band_pass:
            filtered.band_pass(*band_pass)

        if band_stop:
            filtered.band_stop(*band_stop)

        if amp_min is not None:
            idx = filtered.amps < amp_min
            filtered.hs[idx] = 0

        return filtered


    def filter_freqs_pick(self, freqs):
        filtered = self.copy()

        indices = np.indices(filtered.fs.shape)
        indices_picked = filtered.freqs_indices(freqs)
        indices_not_picked = np.setxor1d(indices, indices_picked)
        filtered.hs[indices_not_picked] = 0

        return filtered


    def freqs_attributes(self, freqs):
        indices = self.freqs_indices(freqs)
        fs = self.fs[indices]
        phases = self.angles[indices]
        amps = self.amps[indices]
        return freqs, indices, fs, phases, amps


    def freqs_indices(self, freqs):
        freqs = np.array(freqs).reshape((-1, 1))
        indices = np.argmin((self.fs - freqs) ** 2, axis = -1)
        return indices


    def rank_by_amps(self):
        idx = np.argsort(-self.amps)
        freqs = self.fs[idx]
        amps = self.amps[idx]

        return freqs, amps


    def above_amps(self, amp_min):
        idx = self.amps > amp_min
        freqs = self.fs[idx]
        phases = self.phases[idx]
        amps = self.amps[idx]

        return freqs, phases, amps


    def delta_phases(self, freq_base, freqs, unwrap = False, rad_to_degree = True):
        freqs = np.array(freqs)
        _, _, _, phs, _ = self.freqs_attributes([freq_base])
        _, _, _, phases, _ = self.freqs_attributes(freqs)

        time_elapsed = ((PI2 - phs) % PI2) / PI2 / freq_base  # 由初始的 phase 走到 phase = PI2 (0) 所需要的時間
        final_phases = Sound.pi_to_pi(phases + freqs * time_elapsed * PI2)
        final_phase_base = Sound.pi_to_pi(phs[0] + freq_base * time_elapsed * PI2)
        return Sound.pi_to_pi(final_phases - final_phase_base, unwrap = unwrap, rad_to_degree = rad_to_degree)


    def delay(self, delta_t):
        self.hs = self.hs * np.exp(-1j * PI2 * self.fs * delta_t)
        return self


    def plot(self, freq_lims = (1, DEFAULT_FRAMERATE // 2), xticks = None, **options):
        super().plot(**options)
        plt.xscale('log')

        if freq_lims:
            plt.xlim(freq_lims)

        if xticks:
            plt.xticks(xticks, xticks, fontsize = 12, rotation = 70)


    def make_wave(self, data_type = DATA_TYPE):
        w = super().make_wave()
        ch = Channel(w.ys.astype(data_type), w.ts, w.framerate)
        return ch


    def play(self):
        self.make_wave().play()



class SoundDevice(Sound):

    def __init__(self, device_index = None, framerate = DEFAULT_FRAMERATE, chunk_size = CHUNK_SIZE,
                 sample_width = SAMPLE_WIDTH, hostApi = 0):

        super().__init__()

        self._set_attributes(framerate = framerate, chunk_size = chunk_size, sample_width = sample_width)


    def _setup_stream(self, device_index, device_type, type_channels, input = False, output = False, nchannels = None,
                      hostApi = 0):

        self.device_index = device_index if device_index else self.get_host_api_info_by_index(hostApi).get(device_type)

        max_channels = self.get_device_info_by_host_api_device_index(hostApi, self.device_index)[type_channels]

        # PyAudio on Ubuntu always gets "max_channels" wrongly as 32.
        max_channels = min(2, max_channels)

        self.nchannels = min(nchannels, max_channels) if nchannels else max_channels

        kwargs = {'defaultInputDevice' : {'input_device_index': self.device_index},
                  'defaultOutputDevice': {'output_device_index': self.device_index}}[device_type]

        self.stream = self.open(format = self.format, channels = self.nchannels, rate = self.framerate, input = input,
                                output = output, frames_per_buffer = self.chunk_size, **kwargs)


    def _set_attributes(self, framerate = DEFAULT_FRAMERATE, chunk_size = CHUNK_SIZE, sample_width = SAMPLE_WIDTH):
        self.framerate = framerate
        self.chunk_size = chunk_size
        self.sample_width = sample_width
        self.format = self.get_format_from_width(sample_width)
        self.data_type = SAMPLE_WIDTH_DTYPES[sample_width]


    def __exit__(self, *args):
        self._clean_up_streams()
        super().__exit__(*args)


    def _clean_up_streams(self):
        try:
            self.stream.stop_stream()
            self.stream.close()
        except OSError as e:
            print(e)



class InputDevice(SoundDevice):

    def __init__(self, device_index = None, framerate = DEFAULT_FRAMERATE, nchannels = None, chunk_size = CHUNK_SIZE,
                 sample_width = SAMPLE_WIDTH, hostApi = 0):
        SoundDevice.__init__(self, device_index = device_index, framerate = framerate, chunk_size = chunk_size,
                             sample_width = sample_width, hostApi = hostApi)

        self._setup_stream(device_index = device_index, device_type = 'defaultInputDevice',
                           type_channels = 'maxInputChannels', input = True, nchannels = nchannels, hostApi = hostApi)


    def read_chunk(self):
        data = np.fromstring(self.stream.read(self.chunk_size), self.data_type)
        data = np.vstack([data[i::self.nchannels] for i in range(self.nchannels)])
        return data



class OutputDevice(SoundDevice):

    def __init__(self, device_index = None, framerate = DEFAULT_FRAMERATE, nchannels = None, chunk_size = CHUNK_SIZE,
                 sample_width = SAMPLE_WIDTH, hostApi = 0):
        SoundDevice.__init__(self, device_index = device_index, framerate = framerate, chunk_size = chunk_size,
                             sample_width = sample_width, hostApi = hostApi)

        self._setup_stream(device_index = device_index, device_type = 'defaultOutputDevice',
                           type_channels = 'maxOutputChannels', output = True, nchannels = nchannels, hostApi = hostApi)


    def _write_chunk(self, frames):
        self.stream.write(frames)
        return frames


    def write_chunk(self, data):
        data = data.astype(self.data_type)
        frames = data.T.ravel().tostring()
        return self._write_chunk(frames)



class Streamer(Sound):

    def __init__(self, input_device_indices = None, nchannels_per_input = 1, output_device_indices = None, wire = True,
                 sample_width = SAMPLE_WIDTH, framerate = DEFAULT_FRAMERATE, chunk_size = CHUNK_SIZE,
                 save_file = False, file_name = FILE_NAME):

        super().__init__()

        self.wire = wire
        self.save_file = save_file
        self.file_name = file_name

        if input_device_indices is None:
            input_device_indices = [None]

        self.input_devices = [
            InputDevice(idx, framerate = framerate, nchannels = nchannels_per_input, chunk_size = chunk_size,
                        sample_width = sample_width) for idx in input_device_indices]

        self.n_input_channels = sum([device.nchannels for device in self.input_devices])

        self.output_device = OutputDevice(device_index = output_device_indices, framerate = framerate,
                                          nchannels = self.n_input_channels, chunk_size = chunk_size,
                                          sample_width = sample_width)

        self.frames = []


    def __exit__(self, *args):
        self._stop()

        for input_device in self.input_devices:
            input_device.__exit__()
        self.output_device.__exit__()

        super().__exit__(*args)


    def _read_chunk(self):
        channel_chunks = [input_device.read_chunk() for input_device in self.input_devices]
        data = np.vstack(channel_chunks)
        return data


    def _read_chunk_normalized(self, amp = 1.0, use_datatype_max_value = True):
        data = self._read_chunk()
        max_value = self.input_devices[0].datatype_max_value if use_datatype_max_value else np.max(abs(data))
        return (amp * data / max_value).astype(np.float32)


    def read_chunk(self, filter = None):
        data = self._read_chunk()
        data_filtered = filter.filter(data) if filter else data
        self.write_chunk(data_filtered)
        return data_filtered


    def write_normalized_chunk(self, data):
        data, _ = self._convert_datatype(data, self.output_device.data_type)
        self.output_device.write_chunk(data)


    def write_chunk(self, data):
        data = data.astype(self.output_device.data_type)
        data_string = data.T.ravel().tostring()

        if self.wire:
            self.output_device._write_chunk(data_string)

        if self.save_file:
            self.frames.append(data_string)


    def _stop(self):
        self._save_to_file()


    def _save_to_file(self):
        if self.save_file:
            self._save_wave_file(self.frames, file_name = self.file_name,
                                 sample_width = self.output_device.sample_width, channels = self.n_input_channels,
                                 framerate = self.output_device.framerate)



class Blender(Sound):

    @classmethod
    def _merge_channels(cls, channels, time_sets = None, framerates = None, default_framerate = DEFAULT_FRAMERATE):

        nchannel = len(channels)

        # set time_sets
        if not time_sets:
            if not framerates:
                framerates = [default_framerate] * nchannel
            time_sets = [np.arange(len(channels[i])) / framerates[i] for i in range(nchannel)]

        tss = np.union1d(time_sets, time_sets)
        ts_channels = np.zeros((nchannel + 1, len(tss)), dtype = channels[0].dtype)
        ts_channels[0] = tss

        for i in range(nchannel):
            mask = np.in1d(tss, time_sets[i])
            ts_channels[i + 1, mask] = channels[i]

        return ts_channels


    @classmethod
    def _merge_thinkdsp_waves(cls, thinkdsp_waves):
        nchannel = len(thinkdsp_waves)
        channels = [thinkdsp_waves[i].ys for i in range(nchannel)]
        time_sets = [thinkdsp_waves[i].ts for i in range(nchannel)]
        return cls._merge_channels(channels, time_sets)


    @classmethod
    def dump_thinkdsp_waves(cls, thinkdsp_waves, file_name = FILE_NAME, sample_width = SAMPLE_WIDTH):
        return cls._save_channels(channels = cls._merge_thinkdsp_waves(thinkdsp_waves)[1:], file_name = file_name,
                                  sample_width = sample_width, framerate = thinkdsp_waves[0].framerate)


    @classmethod
    def merge_wave_files(cls, files, file_name = FILE_NAME):
        wave_files = [WaveFile(f) for f in files]

        widths = [wf.sample_width for wf in wave_files]
        framerates = [wf.framerate for wf in wave_files]
        assert len(np.unique(widths)) == 1
        assert len(np.unique(framerates)) == 1

        time_sets = [channel.ts for wf in wave_files for channel in wf.channels]
        amp_sets = [channel.ys for wf in wave_files for channel in wf.channels]
        ts_channels = cls._merge_channels(amp_sets, time_sets)
        channels = ts_channels[1:]

        return cls._save_channels(channels, file_name = file_name, sample_width = widths[0], framerate = framerates[0])


    @classmethod
    def merge_channels_data(cls, data, file_name = FILE_NAME):
        data = np.vstack(data)
        data = cls._normalize(data)
        return cls.dump_thinkdsp_waves([Channel(ys) for ys in data], file_name = file_name)
