import copy
import pickle
import numpy as np
import time


def remove_nones(input_list):
    return [x for x in input_list if x is not None]


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
        assert all([key in self.tape.keys() for key in new_values_dict.keys()]), \
            f"self.tape.keys()={self.tape.keys()}\nnew_values_dict.keys()={new_values_dict.keys()}"

        for key in self.tape.keys():
            if key in new_values_dict.keys():
                self.tape[key].append(copy.deepcopy(new_values_dict[key]))
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
