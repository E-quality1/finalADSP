"""
Pilot Position Diagnostic

This script traces exactly where pilots end up in the frequency domain
after the full transmitter-receiver chain to identify the correct extraction indices.
"""
import numpy as np
import matplotlib.pyplot as plt
from ofdm.system import params
from ofdm.transmitter import qam_mod, insert_pilots, ifft_with_cp

def trace_pilot_positions():
    """Trace pilot positions through the entire chain."""
    print("="*60)
    print("PILOT POSITION TRACING")
    print("="*60)
    
    # Step 1: Create OFDM symbol with known pilots
    print("Step 1: Create OFDM symbol")
    num_data_bits = (params.N - np.sum(params.pilot_pattern)) * params.bits_per_symbol
    test_bits = np.zeros(num_data_bits)  # All zeros for simplicity
    data_symbols = qam_mod(test_bits)
    
    # Insert pilots - this creates natural order
    ofdm_symbol = insert_pilots(data_symbols)
    pilot_indices_natural = np.where(params.pilot_pattern)[0]
    
    print(f"Natural order pilot indices: {pilot_indices_natural}")
    print(f"Pilot values in natural order: {ofdm_symbol[pilot_indices_natural]}")
    
    # Step 2: Transmitter IFFT (now without ifftshift)
    print("\nStep 2: Transmitter IFFT")
    time_signal = ifft_with_cp(ofdm_symbol)
    time_signal_no_cp = time_signal[params.CP:]  # Remove CP for analysis
    
    # Step 3: Receiver FFT
    print("\nStep 3: Receiver FFT")
    rx_freq_natural = np.fft.fft(time_signal_no_cp) / np.sqrt(params.N)
    
    print(f"After receiver FFT (natural order):")
    print(f"Pilot values at natural indices: {rx_freq_natural[pilot_indices_natural]}")
    
    # Step 4: Receiver fftshift
    print("\nStep 4: Receiver fftshift")
    rx_freq_centered = np.fft.fftshift(rx_freq_natural)
    
    # Now where are the pilots in the centered spectrum?
    print(f"After fftshift (centered spectrum):")
    
    # Method A: Use shifted indices (current approach)
    pilot_indices_shifted = (pilot_indices_natural + params.N//2) % params.N
    print(f"Method A - Using shifted indices {pilot_indices_shifted}:")
    print(f"Pilot values: {rx_freq_centered[pilot_indices_shifted]}")
    
    # Method B: Search for actual pilot locations
    print(f"\nMethod B - Search for actual pilot locations:")
    pilot_magnitudes = np.abs(rx_freq_centered)
    
    # Find peaks that might be pilots
    threshold = np.max(pilot_magnitudes) * 0.5
    potential_pilot_indices = np.where(pilot_magnitudes > threshold)[0]
    print(f"Potential pilot indices (magnitude > {threshold:.2f}): {potential_pilot_indices}")
    
    for idx in potential_pilot_indices:
        centered_idx = idx - params.N//2  # Convert to centered index
        print(f"  Index {idx} (centered: {centered_idx}): {rx_freq_centered[idx]}")
    
    # Method C: Check expected pilot positions in centered spectrum
    print(f"\nMethod C - Expected pilot positions in centered spectrum:")
    expected_pilot_carriers = np.array([-21, -7, 7, 21])  # From system params
    expected_pilot_indices = expected_pilot_carriers + params.N//2
    
    print(f"Expected pilot carriers: {expected_pilot_carriers}")
    print(f"Expected pilot indices in centered spectrum: {expected_pilot_indices}")
    
    for i, idx in enumerate(expected_pilot_indices):
        if 0 <= idx < params.N:
            print(f"  Carrier {expected_pilot_carriers[i]} (index {idx}): {rx_freq_centered[idx]}")
        else:
            print(f"  Carrier {expected_pilot_carriers[i]} (index {idx}): OUT OF RANGE")
    
    return rx_freq_centered, pilot_indices_natural, pilot_indices_shifted, expected_pilot_indices

def visualize_spectrum_search(rx_freq_centered, pilot_indices_natural, pilot_indices_shifted, expected_pilot_indices):
    """Visualize the spectrum to identify correct pilot positions."""
    print("\n" + "="*60)
    print("SPECTRUM VISUALIZATION")
    print("="*60)
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot 1: Full spectrum
    centered_indices = np.arange(-params.N//2, params.N//2)
    axes[0].stem(centered_indices, np.abs(rx_freq_centered), basefmt=' ')
    axes[0].set_title('Received Spectrum (Centered)')
    axes[0].set_xlabel('Subcarrier Index (centered)')
    axes[0].set_ylabel('Magnitude')
    axes[0].grid(True, alpha=0.3)
    
    # Mark different pilot extraction attempts
    # Method A: Shifted indices (current)
    for idx in pilot_indices_shifted:
        centered_idx = idx - params.N//2
        axes[0].axvline(x=centered_idx, color='red', linestyle='--', alpha=0.7, label='Method A (shifted)')
    
    # Method C: Expected positions
    expected_pilot_carriers = np.array([-21, -7, 7, 21])
    for pc in expected_pilot_carriers:
        axes[0].axvline(x=pc, color='green', linestyle='-', alpha=0.7, label='Expected')
    
    # Plot 2: Zoom in on pilot region
    pilot_region = slice(params.N//2 - 30, params.N//2 + 30)
    zoom_indices = centered_indices[pilot_region]
    zoom_spectrum = np.abs(rx_freq_centered[pilot_region])
    
    axes[1].stem(zoom_indices, zoom_spectrum, basefmt=' ')
    axes[1].set_title('Pilot Region (Zoomed)')
    axes[1].set_xlabel('Subcarrier Index (centered)')
    axes[1].set_ylabel('Magnitude')
    axes[1].grid(True, alpha=0.3)
    
    # Mark expected pilot positions
    for pc in expected_pilot_carriers:
        axes[1].axvline(x=pc, color='green', linestyle='-', alpha=0.7)
        axes[1].text(pc, np.max(zoom_spectrum)*0.8, f'{pc}', ha='center', color='green')
    
    # Mark current extraction positions
    for idx in pilot_indices_shifted:
        centered_idx = idx - params.N//2
        if -30 <= centered_idx <= 30:
            axes[1].axvline(x=centered_idx, color='red', linestyle='--', alpha=0.7)
            axes[1].text(centered_idx, np.max(zoom_spectrum)*0.6, f'{centered_idx}', ha='center', color='red')
    
    plt.tight_layout()
    plt.savefig('pilot_positions_debug.png', dpi=150, bbox_inches='tight')
    print("Pilot position visualization saved to 'pilot_positions_debug.png'")

def test_correct_extraction():
    """Test extraction using the correct pilot positions."""
    print("\n" + "="*60)
    print("TESTING CORRECT EXTRACTION")
    print("="*60)
    
    # Create test with known data
    num_data_bits = (params.N - np.sum(params.pilot_pattern)) * params.bits_per_symbol
    original_bits = np.random.randint(0, 2, num_data_bits)
    original_symbols = qam_mod(original_bits)
    
    # Full transmitter chain
    ofdm_symbol = insert_pilots(original_symbols)
    time_signal = ifft_with_cp(ofdm_symbol)
    
    # Perfect receiver
    time_signal_no_cp = time_signal[params.CP:]
    rx_freq_natural = np.fft.fft(time_signal_no_cp) / np.sqrt(params.N)
    rx_freq_centered = np.fft.fftshift(rx_freq_natural)
    
    # Try different extraction methods
    methods = []
    
    # Method 1: Current approach (shifted indices)
    pilot_indices_natural = np.where(params.pilot_pattern)[0]
    pilot_indices_shifted = (pilot_indices_natural + params.N//2) % params.N
    data_mask_1 = np.ones(params.N, dtype=bool)
    data_mask_1[pilot_indices_shifted] = False
    data_1 = rx_freq_centered[data_mask_1]
    methods.append(("Current (shifted indices)", data_1))
    
    # Method 2: Expected pilot positions
    expected_pilot_carriers = np.array([-21, -7, 7, 21])
    expected_pilot_indices = expected_pilot_carriers + params.N//2
    data_mask_2 = np.ones(params.N, dtype=bool)
    data_mask_2[expected_pilot_indices] = False
    data_2 = rx_freq_centered[data_mask_2]
    methods.append(("Expected positions", data_2))
    
    # Method 3: Natural order (no fftshift)
    data_mask_3 = np.ones(params.N, dtype=bool)
    data_mask_3[pilot_indices_natural] = False
    data_3 = rx_freq_natural[data_mask_3]
    methods.append(("Natural order (no fftshift)", data_3))
    
    print(f"Original data symbols: {len(original_symbols)}")
    
    for name, extracted_data in methods:
        print(f"\n{name}:")
        print(f"  Extracted symbols: {len(extracted_data)}")
        
        if len(extracted_data) == len(original_symbols):
            # Test symbol accuracy
            symbol_errors = np.sum(~np.isclose(extracted_data, original_symbols, rtol=0.1))
            print(f"  Symbol errors: {symbol_errors}/{len(original_symbols)}")
            
            # Test BER
            try:
                from ofdm.transmitter import qam_demod
                demod_bits = qam_demod(extracted_data)
                if len(demod_bits) == len(original_bits):
                    bit_errors = np.sum(demod_bits != original_bits)
                    ber = bit_errors / len(original_bits)
                    print(f"  BER: {ber:.6f}")
                    print(f"  Status: {'✓ PASS' if ber < 0.01 else '✗ FAIL'}")
                else:
                    print(f"  Bit length mismatch: {len(demod_bits)} vs {len(original_bits)}")
            except Exception as e:
                print(f"  Demodulation error: {e}")
        else:
            print(f"  Length mismatch with original: {len(extracted_data)} vs {len(original_symbols)}")

def main():
    """Run pilot position diagnostic."""
    print("PILOT POSITION DIAGNOSTIC")
    print("="*80)
    
    # Trace pilot positions
    rx_freq_centered, pilot_indices_natural, pilot_indices_shifted, expected_pilot_indices = trace_pilot_positions()
    
    # Visualize spectrum
    visualize_spectrum_search(rx_freq_centered, pilot_indices_natural, pilot_indices_shifted, expected_pilot_indices)
    
    # Test correct extraction
    test_correct_extraction()
    
    print("\n" + "="*80)
    print("PILOT POSITION DIAGNOSTIC COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
