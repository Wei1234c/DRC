from .delay.aligner import DelayAligner
from .gain.balancer import GainBalancer



class Tuner:

    @staticmethod
    def delay_alignment_and_gain_balancing(gain_cell, gain_cell_base,
                                           fractional_delay, estimated_distance_difference_cm,
                                           input_device_idx = None):
        gain_cell.set_gain(1)
        fractional_delay.set_delayed_percentage(0)

        ms_delays, delay_powers, optimized_delay_ms, distance_difference_cm = \
            DelayAligner.optimize(fractional_delay = fractional_delay,
                                  estimated_distance_difference_cm = estimated_distance_difference_cm,
                                  input_device_idx = input_device_idx)
        print(f'optimized_delay_ms: {optimized_delay_ms}\ndistance_difference_cm:{distance_difference_cm}')

        spl_base, spl, optimized_gain_dB, spl_balanced = \
            GainBalancer.balance(gain_cell, gain_cell_base, input_device_idx = 1)
        print(f'optimized_gain_dB: {optimized_gain_dB}')

        return (ms_delays, delay_powers, optimized_delay_ms, distance_difference_cm), \
               (spl_base, spl, optimized_gain_dB, spl_balanced)
