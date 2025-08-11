"""
Receiver Extraction Fix

This script identifies and fixes the correct pilot/data extraction indices
for the receiver after fftshift.
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
from ofdm.transmitter import qam_mod, qam_demod, insert_pilots, ifft_with_cp

def find_correct_extraction_indices():
    """Find the correct indices for pilot and data extraction after fftshift."""
    print("="*60)
    print("FINDING CORRECT EXTRACTION INDICES")
    print("="*60)
    
    # Create test OFDM symbol with known pilots and data
    num_data_bits = (params.N - np.sum(params.pilot_pattern)) * params.bits_per_symbol
    test_bits = np.array([i % 2 for i in range(num_data_bits)])
    data_symbols = qam_mod(test_bits)
    ofdm_symbol = insert_pilots(data_symbols)
    
    # Transmit and receive
    time_signal = ifft_with_cp(ofdm_symbol)
    rx_time_no_cp = time_signal[params.CP:]
    rx_freq_natural = np.fft.fft(rx_time_no_cp) / np.sqrt(params.N)
    rx_freq_centered = np.fft.fftshift(rx_freq_natural)
    
    print(f"Original OFDM symbol (natural order):")
    pilot_indices_natural = np.where(params.pilot_pattern)[0]
    print(f"  Pilots at indices {pilot_indices_natural}: {ofdm_symbol[pilot_indices_natural]}")
    
    print(f"\nReceived spectrum (centered after fftshift):")
    
    # Method 1: Current approach (WRONG)
    current_pilot_indices = (pilot_indices_natural + params.N//2) % params.N
    current_pilots = rx_freq_centered[current_pilot_indices]
    print(f"  Current method indices {current_pilot_indices}: {current_pilots}")
    
    # Method 2: Search for correct indices
    print(f"\nSearching for correct pilot positions:")
    
    # We know pilots should be at IEEE 802.11 positions [-21, -7, 7, 21]
    expected_pilot_carriers = np.array([-21, -7, 7, 21])
    correct_pilot_indices = expected_pilot_carriers + params.N//2
    correct_pilots = rx_freq_centered[correct_pilot_indices]
    
    print(f"  Correct indices {correct_pilot_indices}: {correct_pilots}")
    print(f"  Expected pilot value: {params.pilot_value}")
    
    # Verify which method gives correct pilots
    current_correct = np.allclose(current_pilots, params.pilot_value, rtol=0.1)
    correct_method_correct = np.allclose(correct_pilots, params.pilot_value, rtol=0.1)
    
    print(f"\nVerification:")
    print(f"  Current method correct: {'✓' if current_correct else '✗'}")
    print(f"  Correct method correct: {'✓' if correct_method_correct else '✗'}")
    
    # Now find correct data extraction
    print(f"\nData extraction:")
    
    # Current method
    current_data_mask = np.ones(params.N, dtype=bool)
    current_data_mask[current_pilot_indices] = False
    current_data = rx_freq_centered[current_data_mask]
    
    # Correct method
    correct_data_mask = np.ones(params.N, dtype=bool)
    correct_data_mask[correct_pilot_indices] = False
    correct_data = rx_freq_centered[correct_data_mask]
    
    print(f"  Original data symbols (first 3): {data_symbols[:3]}")
    print(f"  Current method data (first 3): {current_data[:3]}")
    print(f"  Correct method data (first 3): {correct_data[:3]}")
    
    # Check which gives correct data
    if len(current_data) == len(data_symbols):
        current_data_errors = np.sum(~np.isclose(current_data, data_symbols, rtol=0.1))
        print(f"  Current method data errors: {current_data_errors}/{len(data_symbols)}")
    
    if len(correct_data) == len(data_symbols):
        correct_data_errors = np.sum(~np.isclose(correct_data, data_symbols, rtol=0.1))
        print(f"  Correct method data errors: {correct_data_errors}/{len(data_symbols)}")
    
    return correct_pilot_indices, correct_data_mask

def test_corrected_extraction(correct_pilot_indices, correct_data_mask):
    """Test the corrected extraction method."""
    print("\n" + "="*60)
    print("TESTING CORRECTED EXTRACTION")
    print("="*60)
    
    # Generate test data
    num_data_bits = (params.N - np.sum(params.pilot_pattern)) * params.bits_per_symbol
    original_bits = np.random.randint(0, 2, num_data_bits)
    original_symbols = qam_mod(original_bits)
    
    # Transmitter
    ofdm_symbol = insert_pilots(original_symbols)
    time_signal = ifft_with_cp(ofdm_symbol)
    
    # Channel (minimal noise)
    from ofdm.channel import add_awgn
    rx_signal = add_awgn(time_signal, snr_db=30)
    
    # Receiver
    rx_time_no_cp = rx_signal[params.CP:]
    rx_freq_natural = np.fft.fft(rx_time_no_cp) / np.sqrt(params.N)
    rx_freq_centered = np.fft.fftshift(rx_freq_natural)
    
    # Extract pilots using correct method
    rx_pilots = rx_freq_centered[correct_pilot_indices]
    print(f"Extracted pilots: {rx_pilots}")
    print(f"Expected pilots: {[params.pilot_value]*len(correct_pilot_indices)}")
    pilots_ok = np.allclose(rx_pilots, params.pilot_value, rtol=0.1)
    print(f"Pilots correct: {'✓' if pilots_ok else '✗'}")
    
    # Extract data using correct method
    rx_data = rx_freq_centered[correct_data_mask]
    print(f"Extracted data symbols: {len(rx_data)} (expected: {len(original_symbols)})")
    
    if len(rx_data) == len(original_symbols):
        # Check symbol accuracy
        symbol_errors = np.sum(~np.isclose(rx_data, original_symbols, rtol=0.1))
        print(f"Symbol errors: {symbol_errors}/{len(original_symbols)}")
        
        # Test BER
        demod_bits = qam_demod(rx_data)
        if len(demod_bits) == len(original_bits):
            bit_errors = np.sum(demod_bits != original_bits)
            ber = bit_errors / len(original_bits)
            print(f"Bit errors: {bit_errors}/{len(original_bits)}")
            print(f"BER: {ber:.6f}")
            print(f"BER acceptable: {'✓' if ber < 0.01 else '✗'}")
            
            return ber < 0.01 and pilots_ok
        else:
            print(f"Bit length mismatch: {len(demod_bits)} vs {len(original_bits)}")
            return False
    else:
        print(f"Symbol length mismatch: {len(rx_data)} vs {len(original_symbols)}")
        return False

def generate_corrected_extraction_function():
    """Generate the corrected extraction function code."""
    print("\n" + "="*60)
    print("CORRECTED EXTRACTION FUNCTION")
    print("="*60)
    
    expected_pilot_carriers = np.array([-21, -7, 7, 21])
    correct_pilot_indices = expected_pilot_carriers + params.N//2
    
    print("def extract_pilots_and_data_corrected(rx_freq_centered):")
    print('    """Extract pilots and data using correct indices after fftshift."""')
    print(f"    # IEEE 802.11 pilot positions: {expected_pilot_carriers}")
    print(f"    pilot_indices = np.array({list(correct_pilot_indices)})")
    print("    ")
    print("    # Extract pilots")
    print("    rx_pilots = rx_freq_centered[pilot_indices]")
    print("    ")
    print("    # Extract data")
    print("    data_mask = np.ones(len(rx_freq_centered), dtype=bool)")
    print("    data_mask[pilot_indices] = False")
    print("    rx_data = rx_freq_centered[data_mask]")
    print("    ")
    print("    return rx_pilots, rx_data")
    
    return correct_pilot_indices

def main():
    """Run receiver extraction fix."""
    print("RECEIVER EXTRACTION FIX")
    print("="*80)
    
    # Find correct indices
    correct_pilot_indices, correct_data_mask = find_correct_extraction_indices()
    
    # Test corrected extraction
    success = test_corrected_extraction(correct_pilot_indices, correct_data_mask)
    
    # Generate corrected function
    final_pilot_indices = generate_corrected_extraction_function()
    
    print("\n" + "="*80)
    print("FIX SUMMARY")
    print("="*80)
    print(f"Corrected extraction works: {'✅ YES' if success else '❌ NO'}")
    print(f"Correct pilot indices: {final_pilot_indices}")
    print(f"These correspond to centered positions: {final_pilot_indices - params.N//2}")
    
    if success:
        print("\n✅ SOLUTION FOUND!")
        print("Use the corrected pilot indices in all receiver code:")
        print(f"pilot_indices = np.array({list(final_pilot_indices)})")
        print("This will fix the BER and pilot extraction issues.")
    else:
        print("\n❌ More debugging needed")
    
    return success, final_pilot_indices

if __name__ == "__main__":
    success, indices = main()
    sys.exit(0 if success else 1)
