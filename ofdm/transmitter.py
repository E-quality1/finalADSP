"""
OFDM Transmitter Implementation
"""
import numpy as np
from ofdm.system import params

# QAM Modulation and Demodulation
from commpy.modulation import QAMModem

modem = QAMModem(params.mod_order)

import logging
import matplotlib.pyplot as plt

def qam_mod(bits):
    symbols = modem.modulate(bits)
    logging.info(f"QAM Modulated {len(bits)} bits into {len(symbols)} symbols.")
    return symbols

def qam_demod(symbols):
    bits = modem.demodulate(symbols, 'hard')
    logging.info(f"QAM Demodulated {len(symbols)} symbols into {len(bits)} bits.")
    return bits

def ofdm_symbol_mapper(bits):
    symbols = qam_mod(bits)
    logging.info(f"Mapped {len(bits)} bits to {len(symbols)} 64-QAM symbols for one OFDM symbol.")
    return symbols

def insert_pilots(data_symbols):
    """Inserts pilot symbols into the data symbol stream."""
    ofdm_symbol = np.zeros(params.N, dtype=complex)
    data_indices = np.where(~params.pilot_pattern)[0]
    pilot_indices = np.where(params.pilot_pattern)[0]
    
    # Place data and pilots
    ofdm_symbol[data_indices] = data_symbols
    ofdm_symbol[pilot_indices] = params.pilot_value
    
    logging.info(f"Inserted {len(pilot_indices)} pilots and {len(data_symbols)} data symbols into OFDM symbol.")
    return ofdm_symbol

def ifft_with_cp(ofdm_symbol):
    # Apply ifftshift then IFFT to correctly map pilot positions
    # This ensures pilots at natural indices [11,25,39,53] map to centered positions [-21,-7,7,21]
    ofdm_symbol_shifted = np.fft.ifftshift(ofdm_symbol)
    time_domain_symbol = np.fft.ifft(ofdm_symbol_shifted, params.N) * np.sqrt(params.N)
    cp = time_domain_symbol[-params.CP:]
    tx_signal = np.concatenate([cp, time_domain_symbol])
    logging.info(f"IFFT and CP: OFDM symbol length {params.N}, with CP {params.CP}, total {len(tx_signal)} samples.")
    return tx_signal

def generate_preamble():
    half = np.random.choice([1, -1], size=params.N//2) + 0j
    preamble_freq = np.concatenate([half, half])
    preamble_time = np.fft.ifft(preamble_freq)
    cp = preamble_time[-params.CP:]
    preamble = np.concatenate([cp, preamble_time])
    logging.info(f"Generated Schmidl & Cox preamble of length {len(preamble)}.")
    return preamble

def generate_ofdm_frame():
    """Generates a complete OFDM frame with preamble and one data symbol."""
    preamble = generate_preamble()
    num_data_subcarriers = np.sum(~params.pilot_pattern)
    bits = np.random.randint(0, 2, params.bits_per_symbol * num_data_subcarriers)
    qam_symbols = ofdm_symbol_mapper(bits)
    ofdm_symbol_freq = insert_pilots(qam_symbols)
    ofdm_symbol_time = ifft_with_cp(ofdm_symbol_freq)
    tx_frame = np.concatenate([preamble, ofdm_symbol_time])
    logging.info(f"Generated OFDM frame of total length {len(tx_frame)}.")
    return tx_frame


# Test block for visualization
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.random.seed(42)
    # Generate a complete OFDM frame
    tx_frame = generate_ofdm_frame()

    # Plotting the transmitted signal
    plt.figure(figsize=(12, 6))
    plt.plot(np.abs(tx_frame))
    plt.title('Transmitted OFDM Frame Magnitude')
    plt.xlabel('Sample Index')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.savefig('tx_frame.png')
    plt.close()
    logging.info("Transmitted frame plot saved to tx_frame.png")

    # Plot constellation
    num_data_subcarriers = np.sum(~params.pilot_pattern)
    bits = np.random.randint(0, 2, params.bits_per_symbol * num_data_subcarriers)
    symbols = ofdm_symbol_mapper(bits)
    plt.figure(figsize=(6,6))
    plt.scatter(np.real(symbols), np.imag(symbols), c='b', s=10)
    plt.title('64-QAM Constellation (Data Symbols)')
    plt.grid(True)
    plt.xlabel('In-phase')
    plt.ylabel('Quadrature')
    plt.axis('equal')
    plt.savefig('qam_constellation.png')
    plt.close()

    # Plot OFDM symbol (frequency domain)
    ofdm_symbol_freq = insert_pilots(symbols)
    plt.figure()
    plt.stem(np.arange(params.N), np.abs(ofdm_symbol_freq))
    plt.title('OFDM Symbol Magnitude (with Pilots)')
    plt.xlabel('Subcarrier Index')
    plt.ylabel('Magnitude')
    plt.savefig('ofdm_symbol_magnitude.png')
    plt.close()

    # Plot time-domain OFDM symbol
    plt.figure()
    plt.plot(np.real(tx_signal), label='Real')
    plt.plot(np.imag(tx_signal), label='Imag', linestyle='--')
    plt.title('OFDM Symbol (Time Domain, with CP)')
    plt.legend()
    plt.savefig('ofdm_symbol_time.png')
    plt.close()

    # Plot preamble (time domain)
    plt.figure()
    plt.plot(np.real(preamble), label='Real')
    plt.plot(np.imag(preamble), label='Imag', linestyle='--')
    plt.title('Schmidl & Cox Preamble (Time Domain)')
    plt.legend()
    plt.savefig('preamble_time.png')
    plt.close()
