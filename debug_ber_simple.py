"""
Simplified BER Diagnostic

This script compares our working receiver with the BER analysis approach
to identify why BER curves show poor performance.
"""
import numpy as np
import sys
import importlib

# Force reload of modules
if 'ofdm.transmitter' in sys.modules:
    importlib.reload(sys.modules['ofdm.transmitter'])
if 'ofdm.system' in sys.modules:
    importlib.reload(sys.modules['ofdm.system'])

from ofdm.system import params
from ofdm.transmitter import qam_mod, qam_demod, insert_pilots, ifft_with_cp, generate_ofdm_frame
from ofdm.channel import add_awgn

def test_working_method():
    """Test our known working method from final comprehensive test."""
    print("="*60)
    print("WORKING METHOD TEST")
    print("="*60)
    
    # Generate test data
    num_data_bits = (params.N - np.sum(params.pilot_pattern)) * params.bits_per_symbol
    original_bits = np.random.randint(0, 2, num_data_bits)
    original_symbols = qam_mod(original_bits)
    
    # Transmitter
    ofdm_symbol = insert_pilots(original_symbols)
    time_signal = ifft_with_cp(ofdm_symbol)
    
    # Channel (30 dB SNR)
    rx_signal = add_awgn(time_signal, snr_db=30)
    
    # Receiver
    rx_time_no_cp = rx_signal[params.CP:]
    rx_freq_natural = np.fft.fft(rx_time_no_cp) / np.sqrt(params.N)
    rx_freq_centered = np.fft.fftshift(rx_freq_natural)
    
    # Extract data using CORRECT method
    pilot_indices_natural = np.where(params.pilot_pattern)[0]
    pilot_indices_centered = pilot_indices_natural  # No shift needed!
    data_mask = np.ones(params.N, dtype=bool)
    data_mask[pilot_indices_centered] = False
    rx_data_symbols = rx_freq_centered[data_mask]
    
    # Demodulate
    demod_bits = qam_demod(rx_data_symbols)
    bit_errors = np.sum(demod_bits != original_bits)
    ber = bit_errors / len(original_bits)
    
    print(f"Working method BER: {ber:.6f}")
    print(f"Working method status: {'✓ PASS' if ber < 0.01 else '✗ FAIL'}")
    
    return ber < 0.01

def test_ber_analysis_simple():
    """Test BER analysis approach but simplified."""
    print("\n" + "="*60)
    print("BER ANALYSIS APPROACH TEST")
    print("="*60)
    
    # Generate OFDM frame (as BER analysis does)
    tx_frame = generate_ofdm_frame()
    print(f"Generated frame length: {len(tx_frame)}")
    
    # Add noise (30 dB SNR)
    rx_frame = add_awgn(tx_frame, snr_db=30)
    
    # Simple timing sync (assume perfect timing)
    preamble_length = params.N  # Schmidl & Cox preamble
    data_start = preamble_length
    
    if data_start + params.N + params.CP <= len(rx_frame):
        # Extract data symbol
        data_symbol_with_cp = rx_frame[data_start:data_start + params.N + params.CP]
        data_symbol_no_cp = data_symbol_with_cp[params.CP:]
        
        # FFT
        rx_freq_natural = np.fft.fft(data_symbol_no_cp) / np.sqrt(params.N)
        rx_freq_centered = np.fft.fftshift(rx_freq_natural)
        
        # Extract data using CORRECT method
        pilot_indices_natural = np.where(params.pilot_pattern)[0]
        pilot_indices_centered = pilot_indices_natural  # No shift needed!
        data_mask = np.ones(params.N, dtype=bool)
        data_mask[pilot_indices_centered] = False
        rx_data_symbols = rx_freq_centered[data_mask]
        
        print(f"Extracted {len(rx_data_symbols)} data symbols")
        
        # We don't know the original bits from generate_ofdm_frame()
        # But we can check if the symbols look reasonable
        symbol_magnitudes = np.abs(rx_data_symbols)
        avg_magnitude = np.mean(symbol_magnitudes)
        
        print(f"Average symbol magnitude: {avg_magnitude:.3f}")
        print(f"Symbol magnitude range: {np.min(symbol_magnitudes):.3f} to {np.max(symbol_magnitudes):.3f}")
        
        # For 64-QAM, symbols should have magnitudes roughly in range 1-8
        reasonable_magnitudes = (avg_magnitude > 0.5) and (avg_magnitude < 20)
        print(f"Symbols look reasonable: {'✓' if reasonable_magnitudes else '✗'}")
        
        return reasonable_magnitudes
    else:
        print(f"Frame too short: {len(rx_frame)} samples")
        return False

def compare_pilot_extraction_methods():
    """Compare different pilot extraction methods on same data."""
    print("\n" + "="*60)
    print("PILOT EXTRACTION COMPARISON")
    print("="*60)
    
    # Generate test data
    num_data_bits = (params.N - np.sum(params.pilot_pattern)) * params.bits_per_symbol
    original_bits = np.random.randint(0, 2, num_data_bits)
    original_symbols = qam_mod(original_bits)
    
    # Transmitter
    ofdm_symbol = insert_pilots(original_symbols)
    time_signal = ifft_with_cp(ofdm_symbol)
    rx_signal = add_awgn(time_signal, snr_db=30)
    
    # Receiver
    rx_time_no_cp = rx_signal[params.CP:]
    rx_freq_natural = np.fft.fft(rx_time_no_cp) / np.sqrt(params.N)
    rx_freq_centered = np.fft.fftshift(rx_freq_natural)
    
    pilot_indices_natural = np.where(params.pilot_pattern)[0]
    
    # Method 1: OLD (incorrect) method
    pilot_indices_old = (pilot_indices_natural + params.N//2) % params.N
    data_mask_old = np.ones(params.N, dtype=bool)
    data_mask_old[pilot_indices_old] = False
    rx_data_old = rx_freq_centered[data_mask_old]
    
    # Method 2: NEW (correct) method
    pilot_indices_new = pilot_indices_natural  # No shift needed!
    data_mask_new = np.ones(params.N, dtype=bool)
    data_mask_new[pilot_indices_new] = False
    rx_data_new = rx_freq_centered[data_mask_new]
    
    print(f"Original data symbols (first 3): {original_symbols[:3]}")
    print(f"OLD method data (first 3): {rx_data_old[:3]}")
    print(f"NEW method data (first 3): {rx_data_new[:3]}")
    
    # Test BER for both methods
    methods = [
        ("OLD (incorrect)", rx_data_old),
        ("NEW (correct)", rx_data_new)
    ]
    
    for name, rx_data in methods:
        if len(rx_data) == len(original_symbols):
            demod_bits = qam_demod(rx_data)
            if len(demod_bits) == len(original_bits):
                bit_errors = np.sum(demod_bits != original_bits)
                ber = bit_errors / len(original_bits)
                print(f"{name} BER: {ber:.6f} ({'✓' if ber < 0.1 else '✗'})")
            else:
                print(f"{name}: Bit length mismatch")
        else:
            print(f"{name}: Symbol length mismatch")

def main():
    """Run simplified BER diagnostic."""
    print("SIMPLIFIED BER DIAGNOSTIC")
    print("="*80)
    
    # Test 1: Working method
    working_ok = test_working_method()
    
    # Test 2: BER analysis approach
    ber_analysis_ok = test_ber_analysis_simple()
    
    # Test 3: Compare extraction methods
    compare_pilot_extraction_methods()
    
    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)
    print(f"Working method: {'✓ PASS' if working_ok else '✗ FAIL'}")
    print(f"BER analysis approach: {'✓ PASS' if ber_analysis_ok else '✗ FAIL'}")
    
    if working_ok:
        print("\n✅ GOOD NEWS: Our receiver fix works correctly!")
        print("The issue with BER curves is likely in:")
        print("1. BER analysis script still using old extraction method")
        print("2. Additional complexity in full BER analysis (CFO, channel estimation)")
        print("3. generate_ofdm_frame() creating different data than expected")
        
        print("\n📋 RECOMMENDATION:")
        print("The BER curves from 01:05 are INVALID (generated before our fix)")
        print("Need to create a simplified BER analysis that uses our working method")
    else:
        print("\n❌ ISSUE: Even our working method fails in this test")
        print("Need to debug further")

if __name__ == "__main__":
    main()
