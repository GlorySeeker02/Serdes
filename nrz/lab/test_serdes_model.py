import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serdes_model import SerdesParams, Transmitter, Receiver, apply_channel
from serdes_model.analysis import compute_ber

p = SerdesParams()
tx = Transmitter(p)
bits = tx.generate_bits()
wave = apply_channel(tx.transmit(bits), p.dt, p)
dec, ph, dfe, z = Receiver(p).run(wave, p.nbits)
print(compute_ber(dec, bits))          # (0.0, 0)