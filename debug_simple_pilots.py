"""
Simple Pilot Diagnostic

This script creates a minimal test to identify where pilots actually end up
in the frequency domain after transmission.
"""
import numpy as np
import matplotlib.pyplot as plt
from ofdm.system import params

def create_test_ofdm_symbol():
    """Create a test OFDM symbol with only pilots (no data)."""
    print("="*60)
    print("CREATING TEST OFDM SYMBOL")
    print("="*60)
    
    # Create OFDM symbol with pilots only
    ofdm_symbol = np.zeros(params.N, dtype=complex)
    
    # Insert pilots at natural order positions
    pilot_indices_natural = np.where(params.pilot_pattern)[0]
    ofdm_symbol[pilot_indices_natural] = params.pilot_value
    
    print(f"Pilot indices (natural order): {pilot_indices_natural}")
    print(f"Pilot value: {params.pilot_value}")
    print(f"OFDM symbol created with {len(pilot_indices_natural)} pilots")
    
    return ofdm_symbol, pilot_indices_natural

def test_transmission_chain(ofdm_symbol, pilot_indices_natural):
    """Test the full transmission chain."""
    print("\n" + "="*60)
    print("TESTING TRANSMISSION CHAIN")
    print("="*60)
    
    # Transmitter: IFFT (no ifftshift after our fix)
    time_signal = np.fft.ifft(ofdm_symbol) * np.sqrt(params.N)
    
    # Add CP
    cp = time_signal[-params.CP:]
    time_signal_with_cp = np.concatenate([cp, time_signal])
    
    print(f"Time signal created: {len(time_signal_with_cp)} samples")
    
    # Receiver: Remove CP and FFT
    time_signal_no_cp = time_signal_with_cp[params.CP:]
    rx_freq_natural = np.fft.fft(time_signal_no_cp) / np.sqrt(params.N)
    
    print(f"After receiver FFT (natural order):")
    print(f"Pilots at original positions: {rx_freq_natural[pilot_indices_natural]}")
    
    # Apply fftshift (as receiver does)
    rx_freq_centered = np.fft.fftshift(rx_freq_natural)
    
    print(f"\nAfter fftshift (centered spectrum):")
    
    # Check where pilots ended up using different methods
    
    # Method 1: Shifted indices (current receiver approach)
    pilot_indices_shifted = (pilot_indices_natural + params.N//2) % params.N
    pilots_method1 = rx_freq_centered[pilot_indices_shifted]
    print(f"Method 1 (shifted indices {pilot_indices_shifted}): {pilots_method1}")
    
    # Method 2: Search all positions for pilot value
    print(f"\nMethod 2 (search for pilot value {params.pilot_value}):")
    for i in range(params.N):
        if np.abs(rx_freq_centered[i] - params.pilot_value) < 0.1:
            centered_pos = i - params.N//2
            print(f"  Found pilot at index {i} (centered position {centered_pos}): {rx_freq_centered[i]}")
    
    # Method 3: Check expected positions based on system design
    expected_pilot_carriers = np.array([-21, -7, 7, 21])  # IEEE 802.11 style
    expected_indices = expected_pilot_carriers + params.N//2
    print(f"\nMethod 3 (expected positions {expected_pilot_carriers}):")
    for i, idx in enumerate(expected_indices):
        if 0 <= idx < params.N:
            print(f"  Carrier {expected_pilot_carriers[i]} (index {idx}): {rx_freq_centered[idx]}")
    
    return rx_freq_centered, pilot_indices_shifted, expected_indices

def visualize_pilot_spectrum(rx_freq_centered, pilot_indices_shifted, expected_indices):
    """Visualize where pilots actually are."""
    print("\n" + "="*60)
    print("VISUALIZING PILOT SPECTRUM")
    print("="*60)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot 1: Full spectrum
    centered_indices = np.arange(-params.N//2, params.N//2)
    ax1.stem(centered_indices, np.abs(rx_freq_centered), basefmt=' ')
    ax1.set_title('Received Spectrum (Centered) - Full View')
    ax1.set_xlabel('Subcarrier Index (centered)')
    ax1.set_ylabel('Magnitude')
    ax1.grid(True, alpha=0.3)
    
    # Mark current extraction positions (red)
    for idx in pilot_indices_shifted:
        centered_pos = idx - params.N//2
        ax1.axvline(x=centered_pos, color='red', linestyle='--', alpha=0.7)
        ax1.text(centered_pos, np.max(np.abs(rx_freq_centered))*0.8, 
                f'{centered_pos}', ha='center', color='red', fontsize=8)
    
    # Mark expected positions (green)
    expected_pilot_carriers = np.array([-21, -7, 7, 21])
    for pc in expected_pilot_carriers:
        ax1.axvline(x=pc, color='green', linestyle='-', alpha=0.7)
        ax1.text(pc, np.max(np.abs(rx_freq_centered))*0.6, 
                f'{pc}', ha='center', color='green', fontsize=8)
    
    # Plot 2: Zoomed view around pilot region
    zoom_start = params.N//2 - 25
    zoom_end = params.N//2 + 25
    zoom_indices = centered_indices[zoom_start:zoom_end]
    zoom_spectrum = np.abs(rx_freq_centered[zoom_start:zoom_end])
    
    ax2.stem(zoom_indices, zoom_spectrum, basefmt=' ')
    ax2.set_title('Pilot Region (Zoomed)')
    ax2.set_xlabel('Subcarrier Index (centered)')
    ax2.set_ylabel('Magnitude')
    ax2.grid(True, alpha=0.3)
    
    # Mark positions in zoom
    for idx in pilot_indices_shifted:
        centered_pos = idx - params.N//2
        if -25 <= centered_pos <= 25:
            ax2.axvline(x=centered_pos, color='red', linestyle='--', alpha=0.7)
            ax2.text(centered_pos, np.max(zoom_spectrum)*0.8, 
                    f'C:{centered_pos}', ha='center', color='red', fontsize=8)
    
    for pc in expected_pilot_carriers:
        if -25 <= pc <= 25:
            ax2.axvline(x=pc, color='green', linestyle='-', alpha=0.7)
            ax2.text(pc, np.max(zoom_spectrum)*0.6, 
                    f'E:{pc}', ha='center', color='green', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('simple_pilot_debug.png', dpi=150, bbox_inches='tight')
    print("Pilot spectrum saved to 'simple_pilot_debug.png'")

def find_correct_pilot_positions(rx_freq_centered):
    """Find where pilots actually are by magnitude."""
    print("\n" + "="*60)
    print("FINDING ACTUAL PILOT POSITIONS")
    print("="*60)
    
    # Find all non-zero positions
    magnitudes = np.abs(rx_freq_centered)
    threshold = 0.5  # Should be close to 1.0 for pilots
    
    pilot_candidates = []
    for i in range(params.N):
        if magnitudes[i] > threshold:
            centered_pos = i - params.N//2
            pilot_candidates.append((i, centered_pos, rx_freq_centered[i]))
    
    print(f"Found {len(pilot_candidates)} pilot candidates (magnitude > {threshold}):")
    for idx, centered_pos, value in pilot_candidates:
        print(f"  Index {idx} (centered: {centered_pos:+3d}): {value:.6f}")
    
    return pilot_candidates

def main():
    """Run simple pilot diagnostic."""
    print("SIMPLE PILOT DIAGNOSTIC")
    print("="*80)
    
    # Create test OFDM symbol with pilots only
    ofdm_symbol, pilot_indices_natural = create_test_ofdm_symbol()
    
    # Test transmission chain
    rx_freq_centered, pilot_indices_shifted, expected_indices = test_transmission_chain(ofdm_symbol, pilot_indices_natural)
    
    # Find actual pilot positions
    pilot_candidates = find_correct_pilot_positions(rx_freq_centered)
    
    # Visualize
    visualize_pilot_spectrum(rx_freq_centered, pilot_indices_shifted, expected_indices)
    
    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)
    
    print("\nSUMMARY:")
    print(f"- Original pilot positions (natural): {pilot_indices_natural}")
    print(f"- Current receiver extraction (shifted): {pilot_indices_shifted}")
    print(f"- Actual pilot locations found: {len(pilot_candidates)} candidates")
    
    if len(pilot_candidates) == 4:  # Expected number of pilots
        actual_positions = [pos[1] for pos in pilot_candidates]  # centered positions
        print(f"- Actual pilot positions (centered): {actual_positions}")
        
        # Check if they match expected IEEE 802.11 positions
        expected = [-21, -7, 7, 21]
        if sorted(actual_positions) == sorted(expected):
            print("✅ Pilots are at expected IEEE 802.11 positions!")
        else:
            print("❌ Pilots are NOT at expected positions")
            print(f"   Expected: {expected}")
            print(f"   Actual: {sorted(actual_positions)}")
    else:
        print("❌ Wrong number of pilots found")

if __name__ == "__main__":
    main()
