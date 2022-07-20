import time

from ...sound import np, Sound, Streamer, SOUND_SPEED



class DelayAligner:
    PROBE_RANGE = (0.5, 1.5)
    BINS = 51
    CHUNK_SIZE = 1024 * 10
    WAIT_BETWEEN_ms = 100


    @classmethod
    def _gen_ms_delays(cls, estimated_distance_difference_cm,
                       probe_range, bins,
                       sound_speed_cm_sec = SOUND_SPEED):

        estimated_delay_ms = estimated_distance_difference_cm / sound_speed_cm_sec * 1000
        ms_delays = np.linspace(estimated_delay_ms * min(probe_range),
                                estimated_delay_ms * max(probe_range),
                                bins)

        return estimated_delay_ms, ms_delays


    @classmethod
    def _profile(cls, fractional_delay, ms_delays, input_device_idx = None,
                 wait_between_ms = WAIT_BETWEEN_ms, chunk_size = CHUNK_SIZE):

        fractional_delay.set_delayed_percentage(0)
        max_delayed_ms = fractional_delay.max_delayed_ms
        powers = []

        with Streamer(input_device_indices = [input_device_idx], chunk_size = chunk_size,
                      wire = False) as mic:

            for delay_ms in ms_delays:
                assert delay_ms <= max_delayed_ms, f'max delay_ms of {fractional_delay} is {max_delayed_ms}'

                fractional_delay.set_delayed_percentage(delay_ms / max_delayed_ms)
                time.sleep(wait_between_ms / 1000)

                powers.append(Sound.power_sums(mic.read_chunk()))

        return powers


    @classmethod
    def optimize(cls, fractional_delay, estimated_distance_difference_cm, input_device_idx = None,
                 probe_range = PROBE_RANGE, bins = BINS,
                 wait_between_ms = WAIT_BETWEEN_ms, chunk_size = CHUNK_SIZE, sound_speed_cm_sec = SOUND_SPEED):

        _, ms_delays = cls._gen_ms_delays(estimated_distance_difference_cm = estimated_distance_difference_cm,
                                          probe_range = probe_range, bins = bins,
                                          sound_speed_cm_sec = sound_speed_cm_sec)

        powers = cls._profile(fractional_delay, ms_delays, input_device_idx = input_device_idx,
                              wait_between_ms = wait_between_ms, chunk_size = chunk_size)

        optimized_delay_ms = ms_delays[np.argmin(powers)]
        fractional_delay.set_delayed_ms(optimized_delay_ms)
        distance_difference_cm = optimized_delay_ms * sound_speed_cm_sec / 1000

        return ms_delays, powers, optimized_delay_ms, distance_difference_cm
