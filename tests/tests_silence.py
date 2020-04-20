import unittest

import numpy as np

from .context import pyspeech
from pyspeech.conf import framing 
import pyspeech.dsp.silence as psil
import pyspeech.dsp.processing as sp


class TestsSilRemove(unittest.TestCase):

    def test_remove_max_amp_20_from_40(self):
        framing['size'] = 200  # ms
        framing['stride'] = 100  # ms
        signal = sp.Signal(np.arange(-20, 20), 20)

