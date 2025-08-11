"""
Data Symbol Extraction Diagnostic

This script investigates the pilot/data extraction logic in the OFDM receiver
to identify why we're getting 50% BER when QAM itself works perfectly.
"""
import numpy as np
import matplotlib.pyplot as plt
from ofdm.system import params
from ofdm.transmitter import qam_mod, qam_demod, insert_pilots, ifft_with_cp
from ofdm.channel import add_awgn

def test_pilot_data_extraction():
    """Test pilot and data extraction logic."""
    print("="*60)
    print("PILOT/DATA EXTRACTION TEST")
    print("="*60)
    
    # Create known test data
    num_data_bits = (params.N - np.sum(params.pilot_pattern)) * params.bits_per_symbol
    original_bits = np.array([i % 2 for i in range(num_data_bits)])  # Alternating pattern
    print(f"Original data bits ({len(original_bits)}): {original_bits[:12]}...")
    
    # Modulate to symbols
    original_symbols = qam_mod(original_bits)
    print(f"Original data symbols ({len(original_symbols)}): {original_symbols[:2]}")
    
    # Insert pilots
    ofdm_symbol = insert_pilots(original_symbols)
    print(f"OFDM symbol with pilots ({len(ofdm_symbol)}): shape OK")
    
    # Check pilot insertion
    pilot_indices = np.where(params.pilot_pattern)[0]
    data_indices = np.where(~params.pilot_pattern)[0]
    
    print(f"Pilot indices: {pilot_indices}")
    print(f"Data indices: {data_indices[:5]}... (showing first 5)")
    print(f"Number of pilots: {len(pilot_indices)}")
    print(f"Number of data subcarriers: {len(data_indices)}")
    
    # Verify pilot values
    pilot_values = ofdm_symbol[pilot_indices]
    print(f"Pilot values: {pilot_values}")
    print(f"Expected pilot value: {params.pilot_value}")
    pilots_correct = np.allclose(pilot_values, params.pilot_value)
    print(f"Pilots correct: {'✓' if pilots_correct else '✗'}")
    
    # Verify data symbols
    data_symbols_in_ofdm = ofdm_symbol[data_indices]
    print(f"Data symbols in OFDM: {data_symbols_in_ofdm[:2]}")
    print(f"Original data symbols: {original_symbols[:2]}")
    data_match = np.allclose(data_symbols_in_ofdm, original_symbols)
    print(f"Data symbols match: {'✓' if data_match else '✗'}")
    
    return ofdm_symbol, original_bits, original_symbols

def test_fft_shift_effects(ofdm_symbol, original_bits, original_symbols):
    """Test how FFT shift affects pilot/data extraction."""
    print("\n" + "="*60)
    print("FFT SHIFT EFFECTS TEST")
    print("="*60)
    
    # Apply fftshift (as done in receiver)
    ofdm_symbol_centered = np.fft.fftshift(ofdm_symbol)
    
    print("After fftshift:")
    
    # Method 1: Extract using original indices (WRONG - this is likely our bug!)
    pilot_indices_natural = np.where(params.pilot_pattern)[0]
    data_indices_natural = np.where(~params.pilot_pattern)[0]
    
    print(f"Method 1 - Using natural order indices:")
    extracted_pilots_wrong = ofdm_symbol_centered[pilot_indices_natural]
    extracted_data_wrong = ofdm_symbol_centered[data_indices_natural]
    
    print(f"  Extracted pilots: {extracted_pilots_wrong}")
    print(f"  Expected pilots: {[params.pilot_value]*len(pilot_indices_natural)}")
    print(f"  Pilots correct: {'✓' if np.allclose(extracted_pilots_wrong, params.pilot_value) else '✗'}")
    
    print(f"  First few extracted data: {extracted_data_wrong[:2]}")
    print(f"  Original data symbols: {original_symbols[:2]}")
    print(f"  Data match: {'✓' if np.allclose(extracted_data_wrong, original_symbols) else '✗'}")
    
    # Method 2: Extract using shifted indices (CORRECT)
    pilot_indices_after_fftshift = (pilot_indices_natural + params.N//2) % params.N
    data_indices_after_fftshift = (data_indices_natural + params.N//2) % params.N
    
    print(f"\nMethod 2 - Using fftshift-corrected indices:")
    print(f"  Natural pilot indices: {pilot_indices_natural}")
    print(f"  Shifted pilot indices: {pilot_indices_after_fftshift}")
    
    extracted_pilots_correct = ofdm_symbol_centered[pilot_indices_after_fftshift]
    extracted_data_correct = ofdm_symbol_centered[data_indices_after_fftshift]
    
    print(f"  Extracted pilots: {extracted_pilots_correct}")
    print(f"  Expected pilots: {[params.pilot_value]*len(pilot_indices_natural)}")
    print(f"  Pilots correct: {'✓' if np.allclose(extracted_pilots_correct, params.pilot_value) else '✗'}")
    
    print(f"  First few extracted data: {extracted_data_correct[:2]}")
    print(f"  Original data symbols: {original_symbols[:2]}")
    print(f"  Data match: {'✓' if np.allclose(extracted_data_correct, original_symbols) else '✗'}")
    
    return extracted_data_wrong, extracted_data_correct

def test_receiver_data_extraction():
    """Test the exact data extraction used in the receiver."""
    print("\n" + "="*60)
    print("RECEIVER DATA EXTRACTION TEST")
    print("="*60)
    
    # Create test signal through full transmitter chain
    num_data_bits = (params.N - np.sum(params.pilot_pattern)) * params.bits_per_symbol
    original_bits = np.random.randint(0, 2, num_data_bits)
    original_symbols = qam_mod(original_bits)
    
    # Full transmitter chain
    ofdm_symbol = insert_pilots(original_symbols)
    time_signal = ifft_with_cp(ofdm_symbol)
    
    # Add minimal noise
    rx_signal = add_awgn(time_signal, snr_db=30)
    
    # Receiver processing (simplified - no CFO/timing issues)
    # Extract data symbol
    preamble_length = params.N  # Assume perfect timing
    data_start = preamble_length
    data_symbol_with_cp = rx_signal[data_start:data_start + params.N + params.CP]
    
    # Remove CP and FFT
    data_symbol_no_cp = data_symbol_with_cp[params.CP:]
    rx_freq = np.fft.fft(data_symbol_no_cp)
    rx_freq_centered = np.fft.fftshift(rx_freq)
    
    print(f"Received frequency domain symbol shape: {rx_freq_centered.shape}")
    
    # Extract data using the CURRENT receiver method
    pilot_indices_natural = np.where(params.pilot_pattern)[0]
    pilot_indices_after_fftshift = (pilot_indices_natural + params.N//2) % params.N
    data_mask = np.ones(params.N, dtype=bool)
    data_mask[pilot_indices_after_fftshift] = False
    
    extracted_data_symbols = rx_freq_centered[data_mask]
    
    print(f"Original data symbols ({len(original_symbols)}): {original_symbols[:3]}")
    print(f"Extracted data symbols ({len(extracted_data_symbols)}): {extracted_data_symbols[:3]}")
    
    # Check if they match
    if len(extracted_data_symbols) == len(original_symbols):
        symbol_errors = np.sum(~np.isclose(extracted_data_symbols, original_symbols, rtol=1e-2))
        print(f"Symbol errors: {symbol_errors}/{len(original_symbols)}")
        print(f"Symbol match: {'✓' if symbol_errors == 0 else '✗'}")
        
        # Test demodulation
        demod_bits = qam_demod(extracted_data_symbols)
        if len(demod_bits) == len(original_bits):
            bit_errors = np.sum(demod_bits != original_bits)
            ber = bit_errors / len(original_bits)
            print(f"Bit errors: {bit_errors}/{len(original_bits)}")
            print(f"BER: {ber:.6f}")
            print(f"BER acceptable: {'✓' if ber < 0.01 else '✗'}")
        else:
            print(f"Bit length mismatch: {len(demod_bits)} vs {len(original_bits)}")
    else:
        print(f"Symbol length mismatch: {len(extracted_data_symbols)} vs {len(original_symbols)}")
    
    return extracted_data_symbols, original_symbols

def visualize_spectrum_extraction():
    """Visualize the spectrum and extraction process."""
    print("\n" + "="*60)
    print("SPECTRUM VISUALIZATION")
    print("="*60)
    
    # Create test OFDM symbol
    num_data_bits = (params.N - np.sum(params.pilot_pattern)) * params.bits_per_symbol
    original_bits = np.random.randint(0, 2, num_data_bits)
    original_symbols = qam_mod(original_bits)
    ofdm_symbol = insert_pilots(original_symbols)
    
    # Show natural order vs centered
    ofdm_symbol_centered = np.fft.fftshift(ofdm_symbol)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Natural order spectrum
    axes[0,0].stem(np.arange(params.N), np.abs(ofdm_symbol), basefmt=' ')
    axes[0,0].set_title('Natural Order Spectrum')
    axes[0,0].set_xlabel('Subcarrier Index')
    axes[0,0].set_ylabel('Magnitude')
    
    # Mark pilots in natural order
    pilot_indices = np.where(params.pilot_pattern)[0]
    for idx in pilot_indices:
        axes[0,0].axvline(x=idx, color='r', linestyle='--', alpha=0.5)
    
    # Plot 2: Centered spectrum
    centered_indices = np.arange(-params.N//2, params.N//2)
    axes[0,1].stem(centered_indices, np.abs(ofdm_symbol_centered), basefmt=' ')
    axes[0,1].set_title('Centered Spectrum (after fftshift)')
    axes[0,1].set_xlabel('Subcarrier Index (centered)')
    axes[0,1].set_ylabel('Magnitude')
    
    # Mark pilots in centered spectrum
    pilot_carriers = np.array([-21, -7, 7, 21])  # Expected pilot positions
    for pc in pilot_carriers:
        axes[0,1].axvline(x=pc, color='r', linestyle='--', alpha=0.5)
    
    # Plot 3: Wrong extraction (using natural indices on centered spectrum)
    pilot_indices_natural = np.where(params.pilot_pattern)[0]
    data_mask_wrong = np.ones(params.N, dtype=bool)
    data_mask_wrong[pilot_indices_natural] = False
    
    axes[1,0].stem(centered_indices, np.abs(ofdm_symbol_centered), basefmt=' ', alpha=0.3)
    extracted_wrong = ofdm_symbol_centered[data_mask_wrong]
    data_indices_wrong = centered_indices[data_mask_wrong]
    axes[1,0].stem(data_indices_wrong, np.abs(extracted_wrong), basefmt=' ', linefmt='r-', markerfmt='ro')
    axes[1,0].set_title('WRONG: Natural indices on centered spectrum')
    axes[1,0].set_xlabel('Subcarrier Index (centered)')
    axes[1,0].set_ylabel('Magnitude')
    
    # Plot 4: Correct extraction (using shifted indices on centered spectrum)
    pilot_indices_after_fftshift = (pilot_indices_natural + params.N//2) % params.N
    data_mask_correct = np.ones(params.N, dtype=bool)
    data_mask_correct[pilot_indices_after_fftshift] = False
    
    axes[1,1].stem(centered_indices, np.abs(ofdm_symbol_centered), basefmt=' ', alpha=0.3)
    extracted_correct = ofdm_symbol_centered[data_mask_correct]
    data_indices_correct = centered_indices[data_mask_correct]
    axes[1,1].stem(data_indices_correct, np.abs(extracted_correct), basefmt=' ', linefmt='g-', markerfmt='go')
    axes[1,1].set_title('CORRECT: Shifted indices on centered spectrum')
    axes[1,1].set_xlabel('Subcarrier Index (centered)')
    axes[1,1].set_ylabel('Magnitude')
    
    plt.tight_layout()
    plt.savefig('spectrum_extraction_debug.png', dpi=150, bbox_inches='tight')
    print("Spectrum visualization saved to 'spectrum_extraction_debug.png'")

def main():
    """Run all data extraction diagnostic tests."""
    print("DATA EXTRACTION DIAGNOSTIC")
    print("="*80)
    
    # Test pilot/data extraction
    ofdm_symbol, original_bits, original_symbols = test_pilot_data_extraction()
    
    # Test FFT shift effects
    extracted_wrong, extracted_correct = test_fft_shift_effects(ofdm_symbol, original_bits, original_symbols)
    
    # Test receiver data extraction
    test_receiver_data_extraction()
    
    # Visualize the problem
    visualize_spectrum_extraction()
    
    print("\n" + "="*80)
    print("DATA EXTRACTION DIAGNOSTIC COMPLETE")
    print("="*80)
    
    print("\nKEY FINDINGS:")
    print("- QAM modulation/demodulation works perfectly")
    print("- Issue is likely in pilot/data extraction after fftshift")
    print("- Check if receiver uses correct indices for centered spectrum")

if __name__ == "__main__":
    main()
