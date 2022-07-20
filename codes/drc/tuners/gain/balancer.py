import time

from ...measurements import Sampler
from ...sound import np, Sound, Streamer



class GainBalancer:
    GAIN_DEFAULT = 1.0
    GAIN_MUTE = 0.0
    GAIN_RANGE = (0.3, 1.0)
    BINS = 51
    CHUNK_SIZE = 1024 * 10
    WAIT_BETWEEN_ms = 100
    MEASURE_SPL_WAIT_BETWEEN_MS = 1000


    @classmethod
    def _profile(cls, gain_cell, gains, input_device_idx = None,
                 wait_between_ms = WAIT_BETWEEN_ms, chunk_size = CHUNK_SIZE):

        gain_cell.set_gain(1)
        powers = []

        with Streamer(input_device_indices = [input_device_idx], chunk_size = chunk_size,
                      wire = False) as mic:

            for gain in gains:
                gain_cell.set_gain(gain)
                time.sleep(wait_between_ms / 1000)

                powers.append(Sound.power_sums(mic.read_chunk()))

        return powers


    @classmethod
    def optimize(cls, gain_cell, input_device_idx = None,
                 gain_range = GAIN_RANGE, bins = BINS,
                 wait_between_ms = WAIT_BETWEEN_ms, chunk_size = CHUNK_SIZE):

        gains = np.linspace(min(gain_range), max(gain_range), bins)
        powers = cls._profile(gain_cell, gains, input_device_idx = input_device_idx,
                              wait_between_ms = wait_between_ms, chunk_size = chunk_size)

        optimized_gain = gains[np.argmin(powers)]
        gain_cell.set_gain(optimized_gain)

        return gains, powers, optimized_gain


    @classmethod
    def balance(cls, gain_cell, gain_cell_base,
                input_device_idx = None, freq_lims = Sampler.MEAN_SPL_FREQ_LIMITS,
                wait_between_ms = MEASURE_SPL_WAIT_BETWEEN_MS):

        gain_cell.set_gain(cls.GAIN_MUTE)
        gain_cell_base.set_gain(cls.GAIN_DEFAULT)
        time.sleep(wait_between_ms / 1000)
        spl_base = Sampler.get_spl(input_device_idx, freq_lims = freq_lims)

        gain_cell_base.set_gain(cls.GAIN_MUTE)
        gain_cell.set_gain(cls.GAIN_DEFAULT)
        time.sleep(wait_between_ms / 1000)
        spl = Sampler.get_spl(input_device_idx, freq_lims = freq_lims)

        optimized_gain_dB = spl_base - spl
        gain_cell.set_dB(optimized_gain_dB)
        time.sleep(wait_between_ms / 1000)
        spl_balanced = Sampler.get_spl(input_device_idx, freq_lims = freq_lims)

        gain_cell_base.set_gain(cls.GAIN_DEFAULT)

        return spl_base, spl, optimized_gain_dB, spl_balanced
