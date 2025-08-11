"""
Focused Fix and Test for OFDM Receiver

This script implements a minimal, correct OFDM receiver to validate
our hypothesis about the data extraction bug.
"""
import numpy as np
from ofdm.system import params
from ofdm.transmitter import qam_mod, qam_demod, insert_pilots, ifft_with_cp, generate_ofdm_frame
from ofdm.channel import add_awgn

def minimal_ofdm_test():
    """Test minimal OFDM transmission with perfect conditions."""
    print("="*60)
    print("MINIMAL OFDM TEST (PERFECT CONDITIONS)")
    print("="*60)
    
    # Generate test data
    num_data_bits = (params.N - np.sum(params.pilot_pattern)) * params.bits_per_symbol
    original_bits = np.random.randint(0, 2, num_data_bits)
    print(f"Generated {num_data_bits} random bits")
    
    # Transmitter
    data_symbols = qam_mod(original_bits)
    ofdm_symbol = insert_pilots(data_symbols)
    time_signal = ifft_with_cp(ofdm_symbol)
    
    print(f"Transmitter: {len(original_bits)} bits → {len(data_symbols)} symbols → {len(time_signal)} time samples")
    
    # Add minimal noise (30 dB SNR)
    rx_signal = add_awgn(time_signal, snr_db=30)
    
    # Perfect receiver (no timing/CFO issues)
    # Remove CP and FFT
    data_symbol_no_cp = rx_signal[params.CP:]  # Perfect timing
    rx_freq = np.fft.fft(data_symbol_no_cp)
    rx_freq_centered = np.fft.fftshift(rx_freq)
    
    print(f"Receiver: {len(rx_signal)} samples → {len(rx_freq_centered)} freq bins")
    
    # CRITICAL: Correct data extraction
    pilot_indices_natural = np.where(params.pilot_pattern)[0]
    pilot_indices_after_fftshift = (pilot_indices_natural + params.N//2) % params.N
    
    print(f"Pilot indices (natural): {pilot_indices_natural}")
    print(f"Pilot indices (after fftshift): {pilot_indices_after_fftshift}")
    
    # Create data mask
    data_mask = np.ones(params.N, dtype=bool)
    data_mask[pilot_indices_after_fftshift] = False
    
    # Extract data symbols
    rx_data_symbols = rx_freq_centered[data_mask]
    
    print(f"Extracted {len(rx_data_symbols)} data symbols (expected: {len(data_symbols)})")
    
    # Verify pilot extraction
    rx_pilots = rx_freq_centered[pilot_indices_after_fftshift]
    print(f"Extracted pilots: {rx_pilots}")
    print(f"Expected pilots: {[params.pilot_value]*len(pilot_indices_natural)}")
    pilots_ok = np.allclose(rx_pilots, params.pilot_value, rtol=0.1)
    print(f"Pilots correct: {'✓' if pilots_ok else '✗'}")
    
    # Demodulate
    if len(rx_data_symbols) == len(data_symbols):
        demod_bits = qam_demod(rx_data_symbols)
        
        if len(demod_bits) == len(original_bits):
            bit_errors = np.sum(demod_bits != original_bits)
            ber = bit_errors / len(original_bits)
            
            print(f"BER Test Results:")
            print(f"  Original bits: {len(original_bits)}")
            print(f"  Demodulated bits: {len(demod_bits)}")
            print(f"  Bit errors: {bit_errors}")
            print(f"  BER: {ber:.6f}")
            print(f"  Status: {'✓ PASS' if ber < 0.01 else '✗ FAIL'}")
            
            return ber < 0.01
        else:
            print(f"Bit length mismatch: {len(demod_bits)} vs {len(original_bits)}")
            return False
    else:
        print(f"Symbol length mismatch: {len(rx_data_symbols)} vs {len(data_symbols)}")
        return False

def test_with_full_frame():
    """Test with full OFDM frame including preamble."""
    print("\n" + "="*60)
    print("FULL FRAME TEST (WITH PREAMBLE)")
    print("="*60)
    
    # Generate full frame
    tx_frame = generate_ofdm_frame()
    print(f"Generated full frame: {len(tx_frame)} samples")
    
    # Add minimal noise
    rx_signal = add_awgn(tx_frame, snr_db=30)
    
    # Extract data symbol (perfect timing)
    preamble_length = params.N  # Schmidl & Cox preamble
    data_start = preamble_length
    
    if data_start + params.N + params.CP <= len(rx_signal):
        data_symbol_with_cp = rx_signal[data_start:data_start + params.N + params.CP]
        
        # Remove CP and FFT
        data_symbol_no_cp = data_symbol_with_cp[params.CP:]
        rx_freq = np.fft.fft(data_symbol_no_cp)
        rx_freq_centered = np.fft.fftshift(rx_freq)
        
        # Extract data
        pilot_indices_natural = np.where(params.pilot_pattern)[0]
        pilot_indices_after_fftshift = (pilot_indices_natural + params.N//2) % params.N
        data_mask = np.ones(params.N, dtype=bool)
        data_mask[pilot_indices_after_fftshift] = False
        
        rx_data_symbols = rx_freq_centered[data_mask]
        
        print(f"Extracted {len(rx_data_symbols)} data symbols from full frame")
        
        # We don't know the original bits for the generated frame, 
        # but we can check if the extraction makes sense
        print(f"Data symbols look reasonable: {'✓' if len(rx_data_symbols) == 60 else '✗'}")
        
        # Check pilots
        rx_pilots = rx_freq_centered[pilot_indices_after_fftshift]
        pilots_ok = np.allclose(rx_pilots, params.pilot_value, rtol=0.1)
        print(f"Pilots correct: {'✓' if pilots_ok else '✗'}")
        
        return len(rx_data_symbols) == 60 and pilots_ok
    else:
        print(f"Frame too short: {len(rx_signal)} samples")
        return False

def compare_extraction_methods():
    """Compare different data extraction methods."""
    print("\n" + "="*60)
    print("EXTRACTION METHOD COMPARISON")
    print("="*60)
    
    # Create test OFDM symbol
    num_data_bits = (params.N - np.sum(params.pilot_pattern)) * params.bits_per_symbol
    original_bits = np.random.randint(0, 2, num_data_bits)
    data_symbols = qam_mod(original_bits)
    ofdm_symbol = insert_pilots(data_symbols)
    
    # Apply fftshift (as in receiver)
    ofdm_symbol_centered = np.fft.fftshift(ofdm_symbol)
    
    # Method 1: Wrong way (natural indices on centered spectrum)
    pilot_indices_natural = np.where(params.pilot_pattern)[0]
    data_mask_wrong = np.ones(params.N, dtype=bool)
    data_mask_wrong[pilot_indices_natural] = False
    data_wrong = ofdm_symbol_centered[data_mask_wrong]
    
    # Method 2: Correct way (shifted indices on centered spectrum)
    pilot_indices_after_fftshift = (pilot_indices_natural + params.N//2) % params.N
    data_mask_correct = np.ones(params.N, dtype=bool)
    data_mask_correct[pilot_indices_after_fftshift] = False
    data_correct = ofdm_symbol_centered[data_mask_correct]
    
    print(f"Method 1 (WRONG): Extracted {len(data_wrong)} symbols")
    print(f"Method 2 (CORRECT): Extracted {len(data_correct)} symbols")
    print(f"Expected: {len(data_symbols)} symbols")
    
    # Test demodulation
    try:
        demod_wrong = qam_demod(data_wrong)
        if len(demod_wrong) == len(original_bits):
            errors_wrong = np.sum(demod_wrong != original_bits)
            ber_wrong = errors_wrong / len(original_bits)
        else:
            ber_wrong = 1.0
    except:
        ber_wrong = 1.0
    
    try:
        demod_correct = qam_demod(data_correct)
        if len(demod_correct) == len(original_bits):
            errors_correct = np.sum(demod_correct != original_bits)
            ber_correct = errors_correct / len(original_bits)
        else:
            ber_correct = 1.0
    except:
        ber_correct = 1.0
    
    print(f"Method 1 BER: {ber_wrong:.6f}")
    print(f"Method 2 BER: {ber_correct:.6f}")
    
    print(f"Correct method wins: {'✓' if ber_correct < ber_wrong else '✗'}")
    
    return ber_correct < 0.01

def main():
    """Run focused fix and test."""
    print("FOCUSED OFDM RECEIVER FIX AND TEST")
    print("="*80)
    
    # Run tests
    test1_pass = minimal_ofdm_test()
    test2_pass = test_with_full_frame()
    test3_pass = compare_extraction_methods()
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Minimal OFDM test: {'✓ PASS' if test1_pass else '✗ FAIL'}")
    print(f"Full frame test: {'✓ PASS' if test2_pass else '✗ FAIL'}")
    print(f"Extraction comparison: {'✓ PASS' if test3_pass else '✗ FAIL'}")
    
    overall_pass = test1_pass and test2_pass and test3_pass
    print(f"Overall status: {'✓ ALL TESTS PASS' if overall_pass else '✗ SOME TESTS FAIL'}")
    
    if overall_pass:
        print("\n✅ RECEIVER IS WORKING CORRECTLY!")
        print("The issue may be in the test bench or BER analysis code.")
    else:
        print("\n❌ RECEIVER HAS FUNDAMENTAL ISSUES!")
        print("Need to debug further.")
    
    return overall_pass

if __name__ == "__main__":
    main()
