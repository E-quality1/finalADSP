"""
OFDM System Parameters and Utilities
"""
import numpy as np

class OFDMParams:
    def __init__(self):
        # Basic OFDM parameters (IEEE 802.11-like)
        self.N = 64               # Number of subcarriers
        self.CP = 16              # Cyclic prefix length
        self.BW = 20e6           # Bandwidth (20 MHz)
        
        # Derived parameters
        self.subcarrier_spacing = self.BW / self.N  # Subcarrier spacing
        self.symbol_duration = 1 / self.subcarrier_spacing  # OFDM symbol duration
        self.sampling_rate = self.BW                # Sampling rate
        
        # 64-QAM configuration
        self.mod_order = 64       # 64-QAM
        self.bits_per_symbol = int(np.log2(self.mod_order))
        
        # Channel parameters
        self.delay_spread = 200e-9  # 200 ns RMS delay spread
        self.doppler_freq = 0      # Will be set for mobility scenarios
        self.num_taps = 5          # Number of channel taps
        
        # Simulation parameters
        self.snr_db = np.arange(5, 16, 2)  # SNR range: 5-15 dB
        self.num_frames = 1000             # Number of OFDM frames per SNR
        self.pilot_pattern = self._create_pilot_pattern()
        self.pilot_value = 1+0j      # BPSK pilot value
        
    def _create_pilot_pattern(self):
        """Create a pilot pattern for IEEE 802.11-style pilots"""
        pilots = np.zeros(self.N, dtype=bool)
        # IEEE 802.11 pilot positions: [-21, -7, 7, 21] in centered spectrum
        # For N=64, center is at index 32, so:
        # -21 -> index 32-21 = 11
        # -7  -> index 32-7  = 25  
        # 7   -> index 32+7  = 39
        # 21  -> index 32+21 = 53
        # But we need to account for the fact that after fftshift in receiver,
        # natural order index i maps to centered position (i - N//2)
        # So for centered position p, we need natural index (p + N//2) % N
        pilot_carriers = np.array([-21, -7, 7, 21])
        pilot_indices = (pilot_carriers + self.N // 2) % self.N
        pilots[pilot_indices] = True
        return pilots

# Create a global instance for easy access
params = OFDMParams()

# Utility functions
def db2linear(db):
    """Convert dB to linear scale"""
    return 10 ** (db / 10)

def linear2db(linear):
    """Convert linear scale to dB"""
    return 10 * np.log10(linear)
