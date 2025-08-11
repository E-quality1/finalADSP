"""
Pilot Value Diagnostic

This script investigates why pilot values are incorrect even though
pilot positions are now correct at [-21, -7, 7, 21].
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
from ofdm.transmitter import qam_mod, insert_pilots, ifft_with_cp
from ofdm.channel import add_awgn

def trace_pilot_values():
    """Trace pilot values through the transmission chain."""
    print("="*60)
    print("PILOT VALUE TRACING")
    print("="*60)
    
    # Step 1: Create OFDM symbol with known data
    print("Step 1: Create OFDM symbol")
    
    # Use simple data for clarity
    num_data_bits = (params.N - np.sum(params.pilot_pattern)) * params.bits_per_symbol
    # Create alternating pattern for easy identification
    test_bits = np.array([i % 2 for i in range(num_data_bits)])
    data_symbols = qam_mod(test_bits)
    
    print(f"Test data (first 6 bits): {test_bits[:6]}")
    print(f"Data symbols (first 2): {data_symbols[:2]}")
    
    # Insert pilots
    ofdm_symbol = insert_pilots(data_symbols)
    pilot_indices_natural = np.where(params.pilot_pattern)[0]
    
    print(f"Pilot indices: {pilot_indices_natural}")
    print(f"Pilot values in OFDM symbol: {ofdm_symbol[pilot_indices_natural]}")
    print(f"Expected pilot value: {params.pilot_value}")
    
    # Step 2: Transmitter processing
    print(f"\nStep 2: Transmitter processing")
    time_signal = ifft_with_cp(ofdm_symbol)
    print(f"Time signal length: {len(time_signal)}")
    
    # Step 3: Channel (no noise for clarity)
    print(f"\nStep 3: Channel (no noise)")
    rx_signal = time_signal.copy()  # Perfect channel
    
    # Step 4: Receiver processing
    print(f"\nStep 4: Receiver processing")
    rx_time_no_cp = rx_signal[params.CP:]
    rx_freq_natural = np.fft.fft(rx_time_no_cp) / np.sqrt(params.N)
    rx_freq_centered = np.fft.fftshift(rx_freq_natural)
    
    print(f"After receiver FFT (natural order):")
    print(f"Pilots at natural indices: {rx_freq_natural[pilot_indices_natural]}")
    
    print(f"\nAfter fftshift (centered spectrum):")
    
    # Method 1: Current receiver approach
    pilot_indices_after_fftshift = (pilot_indices_natural + params.N//2) % params.N
    extracted_pilots_current = rx_freq_centered[pilot_indices_after_fftshift]
    print(f"Current method pilots: {extracted_pilots_current}")
    
    # Method 2: Direct search at expected positions
    expected_pilot_carriers = np.array([-21, -7, 7, 21])
    expected_indices = expected_pilot_carriers + params.N//2
    extracted_pilots_expected = rx_freq_centered[expected_indices]
    print(f"Expected position pilots: {extracted_pilots_expected}")
    
    # Step 5: Data extraction
    print(f"\nStep 5: Data extraction")
    data_mask = np.ones(params.N, dtype=bool)
    data_mask[pilot_indices_after_fftshift] = False
    extracted_data = rx_freq_centered[data_mask]
    
    print(f"Original data symbols (first 3): {data_symbols[:3]}")
    print(f"Extracted data (first 3): {extracted_data[:3]}")
    
    # Check if data matches
    if len(extracted_data) == len(data_symbols):
        data_errors = np.sum(~np.isclose(extracted_data, data_symbols, rtol=0.1))
        print(f"Data symbol errors: {data_errors}/{len(data_symbols)}")
    else:
        print(f"Data length mismatch: {len(extracted_data)} vs {len(data_symbols)}")
    
    return extracted_pilots_current, extracted_pilots_expected, extracted_data, data_symbols

def test_pilot_only_transmission():
    """Test transmission with pilots only (no data) to isolate pilot issues."""
    print("\n" + "="*60)
    print("PILOT-ONLY TRANSMISSION TEST")
    print("="*60)
    
    # Create OFDM symbol with pilots only
    ofdm_symbol = np.zeros(params.N, dtype=complex)
    pilot_indices_natural = np.where(params.pilot_pattern)[0]
    ofdm_symbol[pilot_indices_natural] = params.pilot_value
    
    print(f"Transmitting pilots only at indices: {pilot_indices_natural}")
    print(f"Pilot values: {params.pilot_value}")
    
    # Full transmission chain
    time_signal = ifft_with_cp(ofdm_symbol)
    rx_signal = time_signal.copy()  # Perfect channel
    
    # Full receiver chain
    rx_time_no_cp = rx_signal[params.CP:]
    rx_freq_natural = np.fft.fft(rx_time_no_cp) / np.sqrt(params.N)
    rx_freq_centered = np.fft.fftshift(rx_freq_natural)
    
    # Extract pilots
    pilot_indices_after_fftshift = (pilot_indices_natural + params.N//2) % params.N
    extracted_pilots = rx_freq_centered[pilot_indices_after_fftshift]
    
    print(f"Extracted pilots: {extracted_pilots}")
    print(f"Expected: {[params.pilot_value]*len(pilot_indices_natural)}")
    
    # Check accuracy
    pilot_errors = np.sum(~np.isclose(extracted_pilots, params.pilot_value, rtol=0.1))
    print(f"Pilot errors: {pilot_errors}/{len(pilot_indices_natural)}")
    print(f"Pilots correct: {'✓' if pilot_errors == 0 else '✗'}")
    
    return pilot_errors == 0

def test_data_only_transmission():
    """Test transmission with data only (no pilots) to isolate data issues."""
    print("\n" + "="*60)
    print("DATA-ONLY TRANSMISSION TEST")
    print("="*60)
    
    # Create test data
    num_data_bits = (params.N - np.sum(params.pilot_pattern)) * params.bits_per_symbol
    test_bits = np.array([i % 2 for i in range(num_data_bits)])
    data_symbols = qam_mod(test_bits)
    
    # Create OFDM symbol with data only (no pilots)
    ofdm_symbol = np.zeros(params.N, dtype=complex)
    data_indices = np.where(~params.pilot_pattern)[0]
    ofdm_symbol[data_indices] = data_symbols
    
    print(f"Transmitting data only at {len(data_indices)} indices")
    print(f"Original data (first 3): {data_symbols[:3]}")
    
    # Full transmission chain
    time_signal = ifft_with_cp(ofdm_symbol)
    rx_signal = time_signal.copy()  # Perfect channel
    
    # Full receiver chain
    rx_time_no_cp = rx_signal[params.CP:]
    rx_freq_natural = np.fft.fft(rx_time_no_cp) / np.sqrt(params.N)
    rx_freq_centered = np.fft.fftshift(rx_freq_natural)
    
    # Extract data (accounting for fftshift)
    pilot_indices_natural = np.where(params.pilot_pattern)[0]
    pilot_indices_after_fftshift = (pilot_indices_natural + params.N//2) % params.N
    data_mask = np.ones(params.N, dtype=bool)
    data_mask[pilot_indices_after_fftshift] = False
    extracted_data = rx_freq_centered[data_mask]
    
    print(f"Extracted data (first 3): {extracted_data[:3]}")
    
    # Check accuracy
    if len(extracted_data) == len(data_symbols):
        data_errors = np.sum(~np.isclose(extracted_data, data_symbols, rtol=0.1))
        print(f"Data errors: {data_errors}/{len(data_symbols)}")
        print(f"Data correct: {'✓' if data_errors == 0 else '✗'}")
        
        # Test demodulation
        from ofdm.transmitter import qam_demod
        demod_bits = qam_demod(extracted_data)
        if len(demod_bits) == len(test_bits):
            bit_errors = np.sum(demod_bits != test_bits)
            ber = bit_errors / len(test_bits)
            print(f"BER: {ber:.6f}")
            print(f"BER acceptable: {'✓' if ber < 0.01 else '✗'}")
            return ber < 0.01
        else:
            print(f"Bit length mismatch: {len(demod_bits)} vs {len(test_bits)}")
            return False
    else:
        print(f"Data length mismatch: {len(extracted_data)} vs {len(data_symbols)}")
        return False

def main():
    """Run pilot value diagnostic."""
    print("PILOT VALUE DIAGNOSTIC")
    print("="*80)
    
    # Trace pilot values through full chain
    pilots_current, pilots_expected, data_rx, data_orig = trace_pilot_values()
    
    # Test pilots only
    pilots_only_ok = test_pilot_only_transmission()
    
    # Test data only
    data_only_ok = test_data_only_transmission()
    
    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)
    print(f"Pilot-only transmission: {'✓ PASS' if pilots_only_ok else '✗ FAIL'}")
    print(f"Data-only transmission: {'✓ PASS' if data_only_ok else '✗ FAIL'}")
    
    if pilots_only_ok and data_only_ok:
        print("✅ Both pilots and data work correctly in isolation!")
        print("❌ Issue is in combined pilot+data transmission")
    elif pilots_only_ok:
        print("✅ Pilots work correctly")
        print("❌ Issue is in data transmission/extraction")
    elif data_only_ok:
        print("✅ Data works correctly")
        print("❌ Issue is in pilot transmission/extraction")
    else:
        print("❌ Issues in both pilot and data transmission")
    
    print("\nNEXT STEPS:")
    if not pilots_only_ok:
        print("- Fix pilot transmission/extraction logic")
    if not data_only_ok:
        print("- Fix data transmission/extraction logic")
    if pilots_only_ok and data_only_ok:
        print("- Investigate pilot-data interference in combined transmission")

if __name__ == "__main__":
    main()
