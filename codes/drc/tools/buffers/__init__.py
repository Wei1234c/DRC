from collections import Sequence

import numpy as np


SMOOTHING_WINDOW_SIZE = 20



class KalmanBuffer:

    def __init__(self, window_size = SMOOTHING_WINDOW_SIZE, process_variance = 1e-5,
                 estimate_of_measurement_variance = 1e-3):
        self.window_size = window_size
        self.process_variance = process_variance
        self.estimate_of_measurement_variance = estimate_of_measurement_variance

        self.posteri_estimate_of_x = 0.
        self.posteri_error_estimate = 1.
        self.priori_estimate_of_x = 0.
        self.priori_error_estimate = 0.
        self.blending_factor = 0.


    def __call__(self, value):
        # predict phase
        self.priori_error_estimate = self.posteri_error_estimate + self.process_variance

        # update phase
        self.blending_factor = self.priori_error_estimate / (
                self.priori_error_estimate + self.estimate_of_measurement_variance)

        self.posteri_estimate_of_x = self.posteri_estimate_of_x * (
                1 - self.blending_factor) + self.blending_factor * value

        self.posteri_error_estimate = (1 - self.blending_factor) * self.priori_error_estimate

        return self.posteri_estimate_of_x



class SmoothingBuffer:

    def __init__(self, window_size = SMOOTHING_WINDOW_SIZE, method = 'hanning'):
        self.window_size = window_size
        self.buffer = np.zeros(window_size)
        windows = {'average': np.ones(window_size) / window_size,
                   'hanning': np.hanning(window_size) / np.hanning(window_size).sum(),
                   'hamming': np.hamming(window_size) / np.hamming(window_size).sum(), }
        self.window = windows[method]


    def __call__(self, value):
        value = np.asarray([value]).ravel()
        length = min(len(value), self.window_size)
        self.buffer = np.roll(self.buffer, length)
        self.buffer[:length] = value[:length]
        result = self.buffer * self.window
        return result.sum()



class RingBuffer(Sequence):

    # https: // github.com / eric - wieser / numpy_ringbuffer

    def __init__(self, capacity, dtype = float, allow_overwrite = True):
        """
        Create a new ring buffer with the given capacity and element type

        Parameters
        ----------
        capacity: int
            The maximum capacity of the ring buffer
        dtype: data-type, optional
            Desired type of buffer elements. Use a type like (float, 2) to
            produce a buffer with shape (N, 2)
        allow_overwrite: bool
            If false, throw an IndexError when trying to append to an alread
            full buffer
        """
        self._arr = np.empty(capacity, dtype)
        self._left_index = 0
        self._right_index = 0
        self._capacity = capacity
        self._allow_overwrite = allow_overwrite


    def _unwrap(self):
        """ Copy the data from this buffer into unwrapped form """
        return np.concatenate((self._arr[self._left_index:min(self._right_index, self._capacity)],
                               self._arr[:max(self._right_index - self._capacity, 0)]))


    def _fix_indices(self):
        """
        Enforce our invariant that 0 <= self._left_index < self._capacity
        """
        if self._left_index >= self._capacity:
            self._left_index -= self._capacity
            self._right_index -= self._capacity
        elif self._left_index < 0:
            self._left_index += self._capacity
            self._right_index += self._capacity


    @property
    def is_full(self):
        """ True if there is no more space in the buffer """
        return len(self) == self._capacity


    # numpy compatibility
    def __array__(self):
        return self._unwrap()


    @property
    def dtype(self):
        return self._arr.dtype


    @property
    def shape(self):
        return (len(self),) + self._arr.shape[1:]


    # these mirror methods from deque
    @property
    def maxlen(self):
        return self._capacity


    def append(self, value):
        if self.is_full:
            if not self._allow_overwrite:
                raise IndexError('append to a full RingBuffer with overwrite disabled')
            elif not len(self):
                return
            else:
                self._left_index += 1

        self._arr[self._right_index % self._capacity] = value
        self._right_index += 1
        self._fix_indices()


    def appendleft(self, value):
        if self.is_full:
            if not self._allow_overwrite:
                raise IndexError('append to a full RingBuffer with overwrite disabled')
            elif not len(self):
                return
            else:
                self._right_index -= 1

        self._left_index -= 1
        self._fix_indices()
        self._arr[self._left_index] = value


    def pop(self):
        if len(self) == 0:
            raise IndexError("pop from an empty RingBuffer")
        self._right_index -= 1
        self._fix_indices()
        res = self._arr[self._right_index % self._capacity]
        return res


    def popleft(self):
        if len(self) == 0:
            raise IndexError("pop from an empty RingBuffer")
        res = self._arr[self._left_index]
        self._left_index += 1
        self._fix_indices()
        return res


    def extend(self, values):
        lv = len(values)
        if len(self) + lv > self._capacity:
            if not self._allow_overwrite:
                raise IndexError('extend a RingBuffer such that it would overflow, with overwrite disabled')
            elif not len(self):
                return
        if lv >= self._capacity:
            # wipe the entire array! - this may not be threadsafe
            self._arr[...] = values[-self._capacity:]
            self._right_index = self._capacity
            self._left_index = 0
            return

        ri = self._right_index % self._capacity
        sl1 = np.s_[ri:min(ri + lv, self._capacity)]
        sl2 = np.s_[:max(ri + lv - self._capacity, 0)]
        self._arr[sl1] = values[:sl1.stop - sl1.start]
        self._arr[sl2] = values[sl1.stop - sl1.start:]
        self._right_index += lv

        self._left_index = max(self._left_index, self._right_index - self._capacity)
        self._fix_indices()


    def extendleft(self, values):
        lv = len(values)
        if len(self) + lv > self._capacity:
            if not self._allow_overwrite:
                raise IndexError('extend a RingBuffer such that it would overflow, with overwrite disabled')
            elif not len(self):
                return
        if lv >= self._capacity:
            # wipe the entire array! - this may not be threadsafe
            self._arr[...] = values[:self._capacity]
            self._right_index = self._capacity
            self._left_index = 0
            return

        self._left_index -= lv
        self._fix_indices()
        li = self._left_index
        sl1 = np.s_[li:min(li + lv, self._capacity)]
        sl2 = np.s_[:max(li + lv - self._capacity, 0)]
        self._arr[sl1] = values[:sl1.stop - sl1.start]
        self._arr[sl2] = values[sl1.stop - sl1.start:]

        self._right_index = min(self._right_index, self._left_index + self._capacity)


    # implement Sequence methods
    def __len__(self):
        return self._right_index - self._left_index


    def __getitem__(self, item):
        # handle simple (b[1]) and basic (b[np.array([1, 2, 3])]) fancy indexing specially
        if not isinstance(item, tuple):
            item_arr = np.asarray(item)
            if issubclass(item_arr.dtype.type, np.integer):
                item_arr = (item_arr + self._left_index) % self._capacity
                return self._arr[item_arr]

        # for everything else, get it right at the expense of efficiency
        return self._unwrap()[item]


    def __iter__(self):
        # alarmingly, this is comparable in speed to using itertools.chain
        return iter(self._unwrap())


    # Everything else
    def __repr__(self):
        return '<RingBuffer of {!r}>'.format(np.asarray(self))



class CircularBuffer:

    def __init__(self, size = 0):
        self.size = size
        self.buffer = np.zeros(size)
        self.idx_new = 0
        self.idx_0 = 0


    def __getitem__(self, index):
        index = (self.idx_0 + index) % self.size
        return self.buffer[index]


    def new_value(self, value):
        self.buffer[self.idx_new] = value
        self.idx_0 = self.idx_new
        self.idx_new = (self.idx_new - 1 + self.size) % self.size
