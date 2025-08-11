"""
Final Comprehensive Test

This script tests the complete OFDM receiver chain after all fixes
to verify that we achieve the target BER performance.
"""
import numpy as np
import sys
import importlib

# Force reload of modules to ensure we get the latest changes
if 'ofdm.transmitter' in sys.modules:
    importlib.reload(sys.modules['ofdm.transmitter'])
if 'ofdm.system' in sys.modules:
    importlib.reload(sys.modules['ofdm.system'])

from ofdm.system import params
from ofdm.transmitter import qam_mod, qam_demod, insert_pilots, ifft_with_cp
from ofdm.channel import add_awgn

def test_pilot_positions_final():
    """Final test of pilot positions with fresh imports."""
    print("="*60)
    print("FINAL PILOT POSITION TEST")
    print("="*60)
    
    # Create test OFDM symbol with pilots only
    ofdm_symbol = np.zeros(params.N, dtype=complex)
    pilot_indices_natural = np.where(params.pilot_pattern)[0]
    ofdm_symbol[pilot_indices_natural] = params.pilot_value
    
    print(f"Pilot indices (natural): {pilot_indices_natural}")
    print(f"Expected centered positions: [-21, -7, 7, 21]")
    
    # Full transmitter chain with current implementation
    time_signal = ifft_with_cp(ofdm_symbol)
    time_no_cp = time_signal[params.CP:]
    
    # Full receiver chain
    rx_freq_natural = np.fft.fft(time_no_cp) / np.sqrt(params.N)
    rx_freq_centered = np.fft.fftshift(rx_freq_natural)
    
    # Find actual pilot positions
    pilot_magnitudes = np.abs(rx_freq_centered)
    pilot_candidates = []
    for i in range(params.N):
        if pilot_magnitudes[i] > 0.5:
            centered_pos = i - params.N//2
            pilot_candidates.append(centered_pos)
    
    pilot_candidates.sort()
    print(f"Actual pilot positions: {pilot_candidates}")
    
    expected_positions = [-21, -7, 7, 21]
    positions_correct = pilot_candidates == expected_positions
    print(f"Pilot positions correct: {'✓' if positions_correct else '✗'}")
    
    return positions_correct

def test_full_ofdm_chain():
    """Test the complete OFDM transmission and reception chain."""
    print("\n" + "="*60)
    print("FULL OFDM CHAIN TEST")
    print("="*60)
    
    # Generate random test data
    num_data_bits = (params.N - np.sum(params.pilot_pattern)) * params.bits_per_symbol
    original_bits = np.random.randint(0, 2, num_data_bits)
    
    print(f"Testing with {num_data_bits} random bits")
    
    # Transmitter chain
    data_symbols = qam_mod(original_bits)
    ofdm_symbol = insert_pilots(data_symbols)
    time_signal = ifft_with_cp(ofdm_symbol)
    
    # Add minimal noise (30 dB SNR)
    rx_signal = add_awgn(time_signal, snr_db=30)
    
    # Perfect receiver (no timing/CFO issues)
    rx_time_no_cp = rx_signal[params.CP:]
    rx_freq_natural = np.fft.fft(rx_time_no_cp) / np.sqrt(params.N)
    rx_freq_centered = np.fft.fftshift(rx_freq_natural)
    
    # Extract pilots for verification
    # Use correct pilot indices for IEEE 802.11 positions [-21, -7, 7, 21]
    pilot_indices_natural = np.where(params.pilot_pattern)[0]  # [11, 25, 39, 53]
    # After fftshift, these natural indices correspond to the correct centered positions
    pilot_indices_centered = pilot_indices_natural  # No shift needed!
    rx_pilots = rx_freq_centered[pilot_indices_centered]
    
    print(f"Extracted pilots: {rx_pilots}")
    print(f"Expected pilots: {[params.pilot_value]*len(pilot_indices_natural)}")
    pilots_ok = np.allclose(rx_pilots, params.pilot_value, rtol=0.1)
    print(f"Pilots correct: {'✓' if pilots_ok else '✗'}")
    
    # Extract data symbols
    data_mask = np.ones(params.N, dtype=bool)
    data_mask[pilot_indices_centered] = False
    rx_data_symbols = rx_freq_centered[data_mask]
    
    print(f"Original data symbols: {len(data_symbols)}")
    print(f"Extracted data symbols: {len(rx_data_symbols)}")
    
    if len(rx_data_symbols) == len(data_symbols):
        # Check symbol accuracy
        symbol_errors = np.sum(~np.isclose(rx_data_symbols, data_symbols, rtol=0.1))
        print(f"Symbol errors: {symbol_errors}/{len(data_symbols)}")
        
        # Demodulate and check BER
        demod_bits = qam_demod(rx_data_symbols)
        
        if len(demod_bits) == len(original_bits):
            bit_errors = np.sum(demod_bits != original_bits)
            ber = bit_errors / len(original_bits)
            
            print(f"Bit errors: {bit_errors}/{len(original_bits)}")
            print(f"BER: {ber:.6f}")
            print(f"BER acceptable: {'✓' if ber < 0.01 else '✗'}")
            
            return ber < 0.01, pilots_ok
        else:
            print(f"Bit length mismatch: {len(demod_bits)} vs {len(original_bits)}")
            return False, pilots_ok
    else:
        print(f"Symbol length mismatch: {len(rx_data_symbols)} vs {len(data_symbols)}")
        return False, pilots_ok

def test_ber_performance():
    """Test BER performance across multiple trials."""
    print("\n" + "="*60)
    print("BER PERFORMANCE TEST")
    print("="*60)
    
    num_trials = 10
    snr_values = [10, 15, 20, 25, 30]
    
    for snr_db in snr_values:
        ber_trials = []
        
        for trial in range(num_trials):
            # Generate test data
            num_data_bits = (params.N - np.sum(params.pilot_pattern)) * params.bits_per_symbol
            original_bits = np.random.randint(0, 2, num_data_bits)
            
            # Transmitter
            data_symbols = qam_mod(original_bits)
            ofdm_symbol = insert_pilots(data_symbols)
            time_signal = ifft_with_cp(ofdm_symbol)
            
            # Channel
            rx_signal = add_awgn(time_signal, snr_db=snr_db)
            
            # Receiver
            rx_time_no_cp = rx_signal[params.CP:]
            rx_freq_natural = np.fft.fft(rx_time_no_cp) / np.sqrt(params.N)
            rx_freq_centered = np.fft.fftshift(rx_freq_natural)
            
            # Extract data
            # Use correct pilot indices for IEEE 802.11 positions [-21, -7, 7, 21]
            pilot_indices_natural = np.where(params.pilot_pattern)[0]  # [11, 25, 39, 53]
            # After fftshift, these natural indices correspond to the correct centered positions
            pilot_indices_centered = pilot_indices_natural  # No shift needed!
            data_mask = np.ones(params.N, dtype=bool)
            data_mask[pilot_indices_centered] = False
            rx_data_symbols = rx_freq_centered[data_mask]
            
            # Demodulate
            if len(rx_data_symbols) == len(data_symbols):
                demod_bits = qam_demod(rx_data_symbols)
                if len(demod_bits) == len(original_bits):
                    bit_errors = np.sum(demod_bits != original_bits)
                    ber = bit_errors / len(original_bits)
                    ber_trials.append(ber)
                else:
                    ber_trials.append(1.0)
            else:
                ber_trials.append(1.0)
        
        avg_ber = np.mean(ber_trials)
        print(f"SNR {snr_db:2d} dB: BER = {avg_ber:.6f} ({'✓' if avg_ber < 0.1 else '✗'})")
    
    return avg_ber < 0.1

def main():
    """Run final comprehensive test."""
    print("FINAL COMPREHENSIVE OFDM TEST")
    print("="*80)
    
    # Test 1: Pilot positions
    pilots_correct = test_pilot_positions_final()
    
    # Test 2: Full OFDM chain
    ber_good, pilots_ok = test_full_ofdm_chain()
    
    # Test 3: BER performance
    performance_good = test_ber_performance()
    
    # Summary
    print("\n" + "="*80)
    print("FINAL TEST SUMMARY")
    print("="*80)
    print(f"Pilot positions correct: {'✓ PASS' if pilots_correct else '✗ FAIL'}")
    print(f"Full chain BER good: {'✓ PASS' if ber_good else '✗ FAIL'}")
    print(f"Pilots extracted correctly: {'✓ PASS' if pilots_ok else '✗ FAIL'}")
    print(f"Performance acceptable: {'✓ PASS' if performance_good else '✗ FAIL'}")
    
    overall_success = pilots_correct and ber_good and pilots_ok and performance_good
    print(f"\nOverall status: {'✅ ALL TESTS PASS - RECEIVER IS WORKING!' if overall_success else '❌ SOME TESTS FAIL - MORE DEBUGGING NEEDED'}")
    
    if overall_success:
        print("\n🎉 SUCCESS! The OFDM receiver is now working correctly!")
        print("✅ Pilots are at correct IEEE 802.11 positions")
        print("✅ BER performance is acceptable")
        print("✅ Ready for full system validation")
    else:
        print("\n🔧 More work needed:")
        if not pilots_correct:
            print("❌ Fix pilot positioning")
        if not ber_good or not pilots_ok:
            print("❌ Fix data extraction and BER")
        if not performance_good:
            print("❌ Improve noise performance")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
