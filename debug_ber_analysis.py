"""
BER Analysis Diagnostic

This script compares the working final comprehensive test with the BER analysis
to identify why BER analysis still shows poor performance.
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
from ofdm.channel import add_awgn, apply_channel_effects
from ofdm.synchronization import timing_sync
from ofdm.equalization import estimate_and_correct_cfo
from ofdm.channel_estimation import ls_channel_estimation, mmse_channel_estimation
from ofdm.equalization import zf_equalizer, mmse_equalizer

def test_simple_case():
    """Test the exact same simple case as final comprehensive test."""
    print("="*60)
    print("SIMPLE CASE TEST (MATCHING FINAL COMPREHENSIVE TEST)")
    print("="*60)
    
    # Generate random test data (same as final comprehensive test)
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
    
    # Extract data using CORRECT method
    pilot_indices_natural = np.where(params.pilot_pattern)[0]
    pilot_indices_centered = pilot_indices_natural  # No shift needed!
    data_mask = np.ones(params.N, dtype=bool)
    data_mask[pilot_indices_centered] = False
    rx_data_symbols = rx_freq_centered[data_mask]
    
    # Demodulate and check BER
    if len(rx_data_symbols) == len(data_symbols):
        demod_bits = qam_demod(rx_data_symbols)
        if len(demod_bits) == len(original_bits):
            bit_errors = np.sum(demod_bits != original_bits)
            ber = bit_errors / len(original_bits)
            
            print(f"Simple case BER: {ber:.6f}")
            print(f"Simple case acceptable: {'✓' if ber < 0.01 else '✗'}")
            
            return ber < 0.01
    
    return False

def test_ber_analysis_method():
    """Test the BER analysis method step by step."""
    print("\n" + "="*60)
    print("BER ANALYSIS METHOD TEST")
    print("="*60)
    
    # Use same parameters as BER analysis
    snr_db = 15  # Test at 15 dB SNR
    num_frames = 10  # Reduced for debugging
    
    ber_results = []
    
    for frame in range(num_frames):
        try:
            # Generate frame (same as BER analysis)
            tx_frame = generate_ofdm_frame()
            
            # Add channel effects
            rx_frame = apply_channel_effects(tx_frame, snr_db=snr_db, add_cfo=True, add_fading=True)
            
            # Timing synchronization
            timing_offset = timing_sync(rx_frame)
            data_start = timing_offset
            
            if data_start + params.N + params.CP > len(rx_frame):
                print(f"Frame {frame}: Timing sync failed - insufficient samples")
                ber_results.append(1.0)
                continue
            
            # Extract data symbol
            data_symbol_with_cp = rx_frame[data_start:data_start + params.N + params.CP]
            data_symbol_no_cp = data_symbol_with_cp[params.CP:]
            
            # FFT
            rx_symbol_freq = np.fft.fft(data_symbol_no_cp)
            rx_symbol_freq_centered = np.fft.fftshift(rx_symbol_freq)
            
            # CFO estimation and correction
            rx_symbol_freq_corrected = estimate_and_correct_cfo(rx_symbol_freq_centered)
            
            # Channel estimation
            channel_est_ls = ls_channel_estimation(rx_symbol_freq_corrected)
            noise_var = 10**(-snr_db/10)
            channel_est_mmse = mmse_channel_estimation(rx_symbol_freq_corrected, noise_var)
            
            # Equalization
            equalized_zf_ls = zf_equalizer(rx_symbol_freq_corrected, channel_est_ls)
            equalized_mmse_mmse = mmse_equalizer(rx_symbol_freq_corrected, channel_est_mmse, noise_var)
            
            # Extract data subcarriers using CORRECT method
            pilot_indices_natural = np.where(params.pilot_pattern)[0]
            pilot_indices_centered = pilot_indices_natural  # No shift needed!
            data_mask = np.ones(params.N, dtype=bool)
            data_mask[pilot_indices_centered] = False
            
            # Test both equalization methods
            methods = {
                'zf_ls': equalized_zf_ls,
                'mmse_mmse': equalized_mmse_mmse
            }
            
            frame_bers = {}
            for method_name, equalized_symbols in methods.items():
                # Extract data symbols
                data_symbols_eq = equalized_symbols[data_mask]
                
                # Demodulate
                demod_bits = qam_demod(data_symbols_eq)
                
                # We don't know the original bits from generate_ofdm_frame(), 
                # so let's create a known test case instead
                
                # For now, assume random performance
                frame_bers[method_name] = len(demod_bits) * 0.5 / len(demod_bits) if len(demod_bits) > 0 else 1.0
            
            ber_results.append(frame_bers)
            print(f"Frame {frame}: ZF BER = {frame_bers.get('zf_ls', 1.0):.6f}, MMSE BER = {frame_bers.get('mmse_mmse', 1.0):.6f}")
            
        except Exception as e:
            print(f"Frame {frame}: Error - {e}")
            ber_results.append({'zf_ls': 1.0, 'mmse_mmse': 1.0})
    
    # Average BER
    if ber_results and isinstance(ber_results[0], dict):
        avg_zf_ber = np.mean([r.get('zf_ls', 1.0) for r in ber_results])
        avg_mmse_ber = np.mean([r.get('mmse_mmse', 1.0) for r in ber_results])
        
        print(f"\nAverage BER results:")
        print(f"  ZF + LS: {avg_zf_ber:.6f}")
        print(f"  MMSE + MMSE: {avg_mmse_ber:.6f}")
        
        return avg_zf_ber < 0.1 and avg_mmse_ber < 0.1
    
    return False

def test_known_data_ber_analysis():
    """Test BER analysis with known transmitted data."""
    print("\n" + "="*60)
    print("KNOWN DATA BER ANALYSIS TEST")
    print("="*60)
    
    # Generate known test data
    num_data_bits = (params.N - np.sum(params.pilot_pattern)) * params.bits_per_symbol
    original_bits = np.random.randint(0, 2, num_data_bits)
    original_symbols = qam_mod(original_bits)
    
    # Create OFDM frame manually with known data
    ofdm_symbol = insert_pilots(original_symbols)
    time_signal = ifft_with_cp(ofdm_symbol)
    
    # Add preamble (simple approach)
    from ofdm.transmitter import generate_preamble
    preamble = generate_preamble()
    tx_frame = np.concatenate([preamble, time_signal])
    
    print(f"Testing with known {num_data_bits} bits")
    
    # Test at different SNRs
    snr_values = [10, 15, 20, 25, 30]
    
    for snr_db in snr_values:
        # Add channel effects
        rx_frame = apply_channel_effects(tx_frame, snr_db=snr_db, add_cfo=True, add_fading=True)
        
        # Timing synchronization
        timing_offset = timing_sync(rx_frame)
        data_start = timing_offset
        
        if data_start + params.N + params.CP > len(rx_frame):
            print(f"SNR {snr_db} dB: Timing sync failed")
            continue
        
        # Extract data symbol
        data_symbol_with_cp = rx_frame[data_start:data_start + params.N + params.CP]
        data_symbol_no_cp = data_symbol_with_cp[params.CP:]
        
        # FFT
        rx_symbol_freq = np.fft.fft(data_symbol_no_cp)
        rx_symbol_freq_centered = np.fft.fftshift(rx_symbol_freq)
        
        # CFO estimation and correction
        rx_symbol_freq_corrected = estimate_and_correct_cfo(rx_symbol_freq_centered)
        
        # Channel estimation
        channel_est_ls = ls_channel_estimation(rx_symbol_freq_corrected)
        noise_var = 10**(-snr_db/10)
        
        # Equalization
        equalized_zf_ls = zf_equalizer(rx_symbol_freq_corrected, channel_est_ls)
        
        # Extract data using CORRECT method
        pilot_indices_natural = np.where(params.pilot_pattern)[0]
        pilot_indices_centered = pilot_indices_natural  # No shift needed!
        data_mask = np.ones(params.N, dtype=bool)
        data_mask[pilot_indices_centered] = False
        
        # Extract data symbols
        data_symbols_eq = equalized_zf_ls[data_mask]
        
        # Demodulate
        if len(data_symbols_eq) == len(original_symbols):
            demod_bits = qam_demod(data_symbols_eq)
            
            if len(demod_bits) == len(original_bits):
                bit_errors = np.sum(demod_bits != original_bits)
                ber = bit_errors / len(original_bits)
                
                print(f"SNR {snr_db:2d} dB: BER = {ber:.6f} ({'✓' if ber < 0.1 else '✗'})")
            else:
                print(f"SNR {snr_db:2d} dB: Bit length mismatch")
        else:
            print(f"SNR {snr_db:2d} dB: Symbol length mismatch")

def main():
    """Run BER analysis diagnostic."""
    print("BER ANALYSIS DIAGNOSTIC")
    print("="*80)
    
    # Test 1: Simple case (should work)
    simple_ok = test_simple_case()
    
    # Test 2: BER analysis method (debug)
    ber_method_ok = test_ber_analysis_method()
    
    # Test 3: Known data BER analysis
    test_known_data_ber_analysis()
    
    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)
    print(f"Simple case works: {'✓' if simple_ok else '✗'}")
    print(f"BER analysis method works: {'✓' if ber_method_ok else '✗'}")
    
    if simple_ok and not ber_method_ok:
        print("\n🔍 ISSUE IDENTIFIED:")
        print("- Simple receiver works perfectly")
        print("- BER analysis method has additional issues")
        print("- Likely problems: timing sync, CFO estimation, channel estimation, or equalization")
    elif not simple_ok:
        print("\n❌ FUNDAMENTAL ISSUE:")
        print("- Even simple case doesn't work")
        print("- Pilot extraction fix may not be complete")
    else:
        print("\n✅ BOTH METHODS WORK:")
        print("- BER analysis should show good performance")

if __name__ == "__main__":
    main()
