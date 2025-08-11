"""
Shift Diagnostic

This script traces exactly where the 4-position shift in pilot positions occurs.
"""
import numpy as np
from ofdm.system import params

def trace_shift_step_by_step():
    """Trace the shift step by step through the transmission chain."""
    print("="*60)
    print("STEP-BY-STEP SHIFT TRACING")
    print("="*60)
    
    N = params.N
    
    # Step 1: Create OFDM symbol with pilots at expected positions
    print("Step 1: Create OFDM symbol")
    ofdm_symbol = np.zeros(N, dtype=complex)
    
    # Use the pilot pattern from params
    pilot_indices_natural = np.where(params.pilot_pattern)[0]
    ofdm_symbol[pilot_indices_natural] = params.pilot_value
    
    print(f"Pilot pattern indices: {pilot_indices_natural}")
    print(f"Expected centered positions: {pilot_indices_natural - N//2}")
    
    # Verify these should be [-21, -7, 7, 21]
    expected_centered = np.array([-21, -7, 7, 21])
    calculated_indices = (expected_centered + N//2) % N
    print(f"For centered positions {expected_centered}, indices should be: {calculated_indices}")
    print(f"Matches pilot pattern: {'✓' if np.array_equal(sorted(pilot_indices_natural), sorted(calculated_indices)) else '✗'}")
    
    # Step 2: Transmitter IFFT (no ifftshift after our fix)
    print(f"\nStep 2: Transmitter IFFT (no ifftshift)")
    time_signal = np.fft.ifft(ofdm_symbol) * np.sqrt(N)
    print(f"Time signal created: {len(time_signal)} samples")
    
    # Step 3: Add and remove CP (perfect timing)
    print(f"\nStep 3: Add/remove CP")
    cp = time_signal[-params.CP:]
    time_with_cp = np.concatenate([cp, time_signal])
    time_no_cp = time_with_cp[params.CP:]
    print(f"CP operations: {len(time_signal)} → {len(time_with_cp)} → {len(time_no_cp)}")
    
    # Step 4: Receiver FFT
    print(f"\nStep 4: Receiver FFT")
    rx_freq_natural = np.fft.fft(time_no_cp) / np.sqrt(N)
    
    # Check where pilots are in natural order
    pilot_magnitudes_natural = np.abs(rx_freq_natural)
    pilot_positions_natural = np.where(pilot_magnitudes_natural > 0.5)[0]
    print(f"Pilots found at natural indices: {pilot_positions_natural}")
    print(f"Expected at natural indices: {pilot_indices_natural}")
    print(f"Shift from expected: {pilot_positions_natural - pilot_indices_natural}")
    
    # Step 5: Receiver fftshift
    print(f"\nStep 5: Receiver fftshift")
    rx_freq_centered = np.fft.fftshift(rx_freq_natural)
    
    # Check where pilots are in centered spectrum
    pilot_magnitudes_centered = np.abs(rx_freq_centered)
    pilot_positions_centered_indices = np.where(pilot_magnitudes_centered > 0.5)[0]
    pilot_positions_centered = pilot_positions_centered_indices - N//2
    
    print(f"Pilots found at centered positions: {pilot_positions_centered}")
    print(f"Expected at centered positions: {expected_centered}")
    print(f"Shift from expected: {pilot_positions_centered - expected_centered}")
    
    return pilot_positions_natural, pilot_positions_centered

def test_different_ifft_approaches():
    """Test different IFFT approaches to see which one gives correct results."""
    print("\n" + "="*60)
    print("TESTING DIFFERENT IFFT APPROACHES")
    print("="*60)
    
    N = params.N
    
    # Create OFDM symbol with pilots
    ofdm_symbol = np.zeros(N, dtype=complex)
    pilot_indices_natural = np.where(params.pilot_pattern)[0]
    ofdm_symbol[pilot_indices_natural] = params.pilot_value
    
    approaches = [
        ("Current: IFFT only", lambda x: np.fft.ifft(x) * np.sqrt(N)),
        ("Alternative: ifftshift + IFFT", lambda x: np.fft.ifft(np.fft.ifftshift(x)) * np.sqrt(N)),
        ("Test: IFFT + fftshift", lambda x: np.fft.fftshift(np.fft.ifft(x)) * np.sqrt(N)),
    ]
    
    for name, ifft_func in approaches:
        print(f"\n{name}:")
        
        # Transmitter
        time_signal = ifft_func(ofdm_symbol)
        
        # Add/remove CP
        cp = time_signal[-params.CP:]
        time_with_cp = np.concatenate([cp, time_signal])
        time_no_cp = time_with_cp[params.CP:]
        
        # Receiver
        rx_freq_natural = np.fft.fft(time_no_cp) / np.sqrt(N)
        rx_freq_centered = np.fft.fftshift(rx_freq_natural)
        
        # Find pilots
        pilot_positions = np.where(np.abs(rx_freq_centered) > 0.5)[0] - N//2
        print(f"  Pilots at centered positions: {pilot_positions}")
        print(f"  Expected: [-21, -7, 7, 21]")
        print(f"  Correct: {'✓' if np.array_equal(sorted(pilot_positions), sorted([-21, -7, 7, 21])) else '✗'}")

def identify_root_cause():
    """Try to identify the exact root cause of the shift."""
    print("\n" + "="*60)
    print("ROOT CAUSE ANALYSIS")
    print("="*60)
    
    N = params.N
    
    # Test hypothesis: The issue might be in how we create the pilot pattern
    print("Hypothesis 1: Pilot pattern creation issue")
    
    # What we want: pilots at centered positions [-21, -7, 7, 21]
    desired_centered = np.array([-21, -7, 7, 21])
    
    # What indices should these map to in natural order?
    # After fftshift, natural index i maps to centered position (i - N//2)
    # So for centered position p, we need natural index i = p + N//2
    correct_natural_indices = (desired_centered + N//2) % N
    print(f"For centered positions {desired_centered}")
    print(f"Natural indices should be: {correct_natural_indices}")
    
    # What does our pilot pattern give us?
    actual_natural_indices = np.where(params.pilot_pattern)[0]
    print(f"Our pilot pattern gives: {actual_natural_indices}")
    print(f"Match: {'✓' if np.array_equal(sorted(correct_natural_indices), sorted(actual_natural_indices)) else '✗'}")
    
    # Test hypothesis: The issue is in the IFFT/FFT chain
    print(f"\nHypothesis 2: IFFT/FFT chain issue")
    
    # Create a simple test: put a single pilot at natural index 11 (should be centered position -21)
    test_signal = np.zeros(N, dtype=complex)
    test_signal[11] = 1.0
    
    print(f"Test: Put pilot at natural index 11")
    
    # Our current transmitter chain
    time_signal = np.fft.ifft(test_signal) * np.sqrt(N)
    cp = time_signal[-params.CP:]
    time_with_cp = np.concatenate([cp, time_signal])
    time_no_cp = time_with_cp[params.CP:]
    
    # Our current receiver chain
    rx_freq_natural = np.fft.fft(time_no_cp) / np.sqrt(N)
    rx_freq_centered = np.fft.fftshift(rx_freq_natural)
    
    # Where did it end up?
    pilot_idx = np.argmax(np.abs(rx_freq_centered))
    pilot_centered_pos = pilot_idx - N//2
    
    print(f"Pilot ended up at centered position: {pilot_centered_pos}")
    print(f"Expected centered position: -21")
    print(f"Shift: {pilot_centered_pos - (-21)}")
    
    # The shift should tell us what's wrong
    shift = pilot_centered_pos - (-21)
    if shift == 0:
        print("✓ No shift - system is working correctly")
    elif shift == -4:
        print("❌ 4-position left shift detected")
    elif shift == 4:
        print("❌ 4-position right shift detected")
    else:
        print(f"❌ Unexpected shift of {shift} positions")

def main():
    """Run shift diagnostic."""
    print("SHIFT DIAGNOSTIC")
    print("="*80)
    
    # Trace shift step by step
    natural_positions, centered_positions = trace_shift_step_by_step()
    
    # Test different approaches
    test_different_ifft_approaches()
    
    # Identify root cause
    identify_root_cause()
    
    print("\n" + "="*80)
    print("SHIFT DIAGNOSTIC COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
