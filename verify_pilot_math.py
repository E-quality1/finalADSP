"""
Pilot Math Verification

This script verifies the mathematical relationship between pilot positions
and indices to ensure we have the correct mapping.
"""
import numpy as np

def verify_pilot_math():
    """Verify the pilot position to index conversion."""
    print("="*60)
    print("PILOT MATH VERIFICATION")
    print("="*60)
    
    N = 64  # OFDM size
    
    # Desired pilot positions in centered spectrum
    desired_pilot_carriers = np.array([-21, -7, 7, 21])
    
    print(f"N = {N}")
    print(f"N//2 = {N//2}")
    print(f"Desired pilot positions (centered): {desired_pilot_carriers}")
    
    # Current conversion formula
    pilot_indices_current = (desired_pilot_carriers + N // 2) % N
    print(f"Current formula (p + N//2) % N: {pilot_indices_current}")
    
    # Verify: what centered positions do these indices map to?
    # After fftshift, natural index i maps to centered position (i - N//2)
    actual_centered_positions = pilot_indices_current - N//2
    print(f"These indices map to centered positions: {actual_centered_positions}")
    
    # Check if they match
    match = np.array_equal(actual_centered_positions, desired_pilot_carriers)
    print(f"Positions match desired: {'✓' if match else '✗'}")
    
    if not match:
        print(f"ERROR: Mismatch!")
        print(f"  Desired: {desired_pilot_carriers}")
        print(f"  Actual:  {actual_centered_positions}")
        
        # What should the correct indices be?
        # If we want centered position p, and fftshift maps index i to position (i - N//2),
        # then we need i such that (i - N//2) = p, so i = p + N//2
        correct_indices = desired_pilot_carriers + N//2
        print(f"  Correct indices should be: {correct_indices}")
        
        # But we need to handle negative indices
        correct_indices_wrapped = correct_indices % N
        print(f"  Wrapped to [0, N-1]: {correct_indices_wrapped}")
        
        # Verify these
        verify_positions = correct_indices_wrapped - N//2
        print(f"  These would give centered positions: {verify_positions}")
    
    return pilot_indices_current, actual_centered_positions

def test_with_simple_example():
    """Test with a simple example to understand the mapping."""
    print("\n" + "="*60)
    print("SIMPLE EXAMPLE TEST")
    print("="*60)
    
    N = 8  # Small example
    print(f"Simple example with N = {N}")
    
    # Create a test signal with a single pilot at index 2
    signal_natural = np.zeros(N, dtype=complex)
    signal_natural[2] = 1.0
    
    print(f"Natural order signal: {signal_natural}")
    
    # Apply fftshift (as receiver does)
    signal_centered = np.fft.fftshift(signal_natural)
    print(f"After fftshift: {signal_centered}")
    
    # Find where the pilot ended up
    pilot_pos = np.where(np.abs(signal_centered) > 0.5)[0][0]
    centered_position = pilot_pos - N//2
    
    print(f"Pilot at natural index 2 maps to centered position {centered_position}")
    print(f"Formula check: 2 - {N//2} = {2 - N//2}")
    
    # Now reverse: if we want pilot at centered position -1, what natural index?
    desired_centered = -1
    required_natural = (desired_centered + N//2) % N
    print(f"To get centered position {desired_centered}, use natural index {required_natural}")
    
    # Verify
    test_signal = np.zeros(N, dtype=complex)
    test_signal[required_natural] = 1.0
    test_centered = np.fft.fftshift(test_signal)
    actual_pos = np.where(np.abs(test_centered) > 0.5)[0][0] - N//2
    print(f"Verification: natural index {required_natural} gives centered position {actual_pos}")

def main():
    """Run pilot math verification."""
    print("PILOT MATH VERIFICATION")
    print("="*80)
    
    # Verify current math
    pilot_indices, actual_positions = verify_pilot_math()
    
    # Test with simple example
    test_with_simple_example()
    
    print("\n" + "="*80)
    print("VERIFICATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
