import os
import unittest

import numpy as np
from scipy.io import wavfile

from tests import signal01
from .context import pyspeech
from pyspeech.conf import framing 
import pyspeech.dsp.silence as silence
import pyspeech.dsp.processing as sp


class TestsSilRemove(unittest.TestCase):

    def test_leq_signal_size(self):
        framing['size'] = 25  # ms
        framing['stride'] = 10  # ms
        voiced = silence.remove(signal01)
        self.assertTrue(0 < voiced.size <= signal01.size,
                       'Voiced signal has size 0 or greater than original')

