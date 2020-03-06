import pickle
import numpy as np
import time
import sys


class Recorder(object):
    def __init__(self, metrics_to_record):
        """
        Simple object consisting of a recording tape (dictionary) that can be extended and saved
        Keys are strings and values are lists of recorded quantities
        (could be reward, loss, action, parameters, gradients, evaluation metric, etc.)
        """
        self.tape = {}

        for metric_name in metrics_to_record:
            self.tape[metric_name] = []

    def write_to_tape(self, new_values_dict):
        """
        Appends to tape all values for corresponding keys defined in dict
        If some keys present on tape do not have a new value in 'new_values_dict',
        we add None instead (so that all lists have the same length.
        """

        # new_values_dict is not allowed to contain un-initialised keys
        assert all([key in self.tape.keys() for key in new_values_dict.keys()])

        for key in self.tape.keys():
            if key in new_values_dict.keys():
                self.tape[key].append(new_values_dict[key])
            else:
                self.tape[key].append(None)

    def save(self, filename):
        """
        Saves the tape (dictionary) in .pkl file
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.tape, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def init_from_pickle_file(cls, filename):
        """
        Initialises Recorder() from a .pkl file containing a tape (dictionary)
        """
        with open(filename, 'rb') as f:
            loaded_tape = pickle.load(f)
        instance = cls(metrics_to_record=loaded_tape.keys())
        instance.tape = loaded_tape
        return instance


class TrainingIterator(object):
    def __init__(self, max_itr, heartbeat_ite=int(sys.maxsize), heartbeat_time=float('inf')):
        self.max_itr = max_itr
        self.heartbeat_time = heartbeat_time
        self.heartbeat_ite = heartbeat_ite
        self._vals = {}
        self._heartbeat = False
        self._itr = 0

    def random_idx(self, N, size):
        return np.random.randint(0, N, size=size)

    @property
    def itr(self):
        return self._itr

    @property
    def heartbeat(self):
        if self._heartbeat:
            self._heartbeat = False
            return True
        else:
            return False

    @property
    def elapsed(self):
        return self.__elapsed

    def itr_message(self):
        return f'==> Itr {self.itr + 1}/{self.max_itr} (elapsed:{self.elapsed:.2f})'

    def record(self, key, value):
        if key in self._vals:
            self._vals[key].append(value)
        else:
            self._vals[key] = [value]

    def update(self, dict):
        for (key, value) in dict.items():
            self.record(key, value)

    def pop(self, key):
        vals = self._vals.get(key, [])
        del self._vals[key]
        return vals

    def pop_mean(self, key):
        return np.mean(self.pop(key))

    def pop_all_means(self):
        r = {}
        for key in dict(self._vals):
            r.update({key: self.pop_mean(key)})
        return r

    def __iter__(self):
        prev_time = time.time()
        self._heartbeat = False
        for i in range(self.max_itr):
            self._itr = i
            cur_time = time.time()
            if (cur_time - prev_time) > self.heartbeat_time or (i == self.max_itr - 1) or (
                    self.itr % self.heartbeat_ite == 0):
                self._heartbeat = True
                self.__elapsed = cur_time - prev_time
                prev_time = cur_time
            yield self
            self._heartbeat = False

    def touch(self):
        self._itr += 1

        if (self.itr == self.max_itr) or (self.itr % self.heartbeat_ite == 0):
            self._heartbeat = True
