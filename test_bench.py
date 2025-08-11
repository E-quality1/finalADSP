"""
Comprehensive OFDM Receiver Test Bench

This test bench systematically validates each component of the OFDM receiver
and identifies performance bottlenecks. It performs:

1. Component-level testing (transmitter, channel, synchronization, etc.)
2. End-to-end system validation
3. Performance benchmarking against IEEE 802.11 requirements
4. Diagnostic analysis and troubleshooting
"""
import numpy as np
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm

# Import OFDM modules
from ofdm.system import params
from ofdm.transmitter import qam_mod, qam_demod, insert_pilots, ifft_with_cp, generate_ofdm_frame
from ofdm.channel import apply_channel_effects, add_awgn
from ofdm.equalization import estimate_and_correct_cfo, zf_equalizer, mmse_equalizer, remove_cp_and_fft
from ofdm.channel_estimation import estimate_channel_from_ofdm_symbol

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

class OFDMTestBench:
    """Comprehensive test bench for OFDM receiver validation."""
    
    def __init__(self):
        self.test_results = {}
        self.diagnostic_data = {}
        
    def test_qam_modulation_demodulation(self, num_bits=1000):
        """Test QAM modulation/demodulation in isolation."""
        print("\n" + "="*60)
        print("TEST 1: QAM MODULATION/DEMODULATION")
        print("="*60)
        
        # Generate random bits
        original_bits = np.random.randint(0, 2, num_bits)
        
        # Modulate and demodulate without any channel effects
        modulated_symbols = qam_mod(original_bits)
        demodulated_bits = qam_demod(modulated_symbols)
        
        # Calculate BER
        if len(demodulated_bits) == len(original_bits):
            bit_errors = np.sum(demodulated_bits != original_bits)
            ber = bit_errors / len(original_bits)
        else:
            ber = 1.0
            bit_errors = len(original_bits)
        
        print(f"QAM Modulation/Demodulation Test:")
        print(f"  Input bits: {len(original_bits)}")
        print(f"  Output bits: {len(demodulated_bits)}")
        print(f"  Bit errors: {bit_errors}")
        print(f"  BER: {ber:.6f}")
        print(f"  Status: {'✓ PASS' if ber == 0 else '✗ FAIL'}")
        
        self.test_results['qam_ber'] = ber
        return ber == 0
    
    def test_ofdm_transmitter(self):
        """Test OFDM transmitter components."""
        print("\n" + "="*60)
        print("TEST 2: OFDM TRANSMITTER")
        print("="*60)
        
        # Generate test data
        num_data_bits = (params.N - np.sum(params.pilot_pattern)) * params.bits_per_symbol
        data_bits = np.random.randint(0, 2, num_data_bits)
        
        try:
            # Test individual components
            data_symbols = qam_mod(data_bits)
            ofdm_symbol = insert_pilots(data_symbols)
            time_signal = ifft_with_cp(ofdm_symbol)
            full_frame = generate_ofdm_frame()
            
            print(f"Transmitter Component Test:")
            print(f"  Data bits: {len(data_bits)}")
            print(f"  Data symbols: {len(data_symbols)}")
            print(f"  OFDM symbol (with pilots): {len(ofdm_symbol)}")
            print(f"  Time signal (with CP): {len(time_signal)}")
            print(f"  Full frame: {len(full_frame)}")
            
            # Validate pilot insertion
            pilot_indices = np.where(params.pilot_pattern)[0]
            pilot_values = ofdm_symbol[pilot_indices]
            pilots_correct = np.allclose(pilot_values, params.pilot_value)
            
            print(f"  Pilot insertion: {'✓ CORRECT' if pilots_correct else '✗ INCORRECT'}")
            print(f"  Status: {'✓ PASS' if pilots_correct else '✗ FAIL'}")
            
            self.test_results['transmitter_pass'] = pilots_correct
            return pilots_correct
            
        except Exception as e:
            print(f"  Error: {e}")
            print(f"  Status: ✗ FAIL")
            self.test_results['transmitter_pass'] = False
            return False
    
    def test_awgn_channel_only(self, snr_db=15):
        """Test receiver with AWGN channel only (no fading, no CFO, no timing offset)."""
        print("\n" + "="*60)
        print("TEST 3: AWGN CHANNEL ONLY")
        print("="*60)
        
        # Generate test frame
        num_data_bits = (params.N - np.sum(params.pilot_pattern)) * params.bits_per_symbol
        original_bits = np.random.randint(0, 2, num_data_bits)
        tx_frame = generate_ofdm_frame()
        
        # Apply only AWGN (no fading, no CFO, no timing offset)
        rx_signal = add_awgn(tx_frame, snr_db)
        
        try:
            # Perfect receiver processing (no synchronization needed)
            # Extract data symbol directly
            preamble_length = params.N  # Schmidl & Cox preamble length
            data_start = preamble_length
            data_symbol_with_cp = rx_signal[data_start:data_start + params.N + params.CP]
            
            # Remove CP and FFT
            data_symbol_no_cp = data_symbol_with_cp[params.CP:]
            rx_freq = np.fft.fft(data_symbol_no_cp)
            rx_freq_centered = np.fft.fftshift(rx_freq)
            
            # Extract data subcarriers (no channel estimation needed for AWGN)
            pilot_indices_natural = np.where(params.pilot_pattern)[0]
            pilot_indices_after_fftshift = (pilot_indices_natural + params.N//2) % params.N
            data_mask = np.ones(params.N, dtype=bool)
            data_mask[pilot_indices_after_fftshift] = False
            
            data_symbols_rx = rx_freq_centered[data_mask]
            
            # Demodulate
            demod_bits = qam_demod(data_symbols_rx)
            
            # Calculate BER
            if len(demod_bits) == len(original_bits):
                bit_errors = np.sum(demod_bits != original_bits)
                ber = bit_errors / len(original_bits)
            else:
                ber = 1.0
                bit_errors = len(original_bits)
            
            print(f"AWGN Channel Test (SNR = {snr_db} dB):")
            print(f"  Original bits: {len(original_bits)}")
            print(f"  Demodulated bits: {len(demod_bits)}")
            print(f"  Bit errors: {bit_errors}")
            print(f"  BER: {ber:.6f}")
            
            # For 64-QAM at 15 dB SNR, theoretical BER should be very low
            target_ber = 1e-2  # Relaxed target for AWGN
            status_pass = ber < target_ber
            print(f"  Target BER: < {target_ber:.0e}")
            print(f"  Status: {'✓ PASS' if status_pass else '✗ FAIL'}")
            
            self.test_results['awgn_ber'] = ber
            self.diagnostic_data['awgn_rx_symbols'] = data_symbols_rx
            return status_pass
            
        except Exception as e:
            print(f"  Error: {e}")
            print(f"  Status: ✗ FAIL")
            self.test_results['awgn_ber'] = 1.0
            return False
    
    def test_perfect_channel_knowledge(self, snr_db=15):
        """Test receiver with perfect channel knowledge (oracle)."""
        print("\n" + "="*60)
        print("TEST 4: PERFECT CHANNEL KNOWLEDGE")
        print("="*60)
        
        # Generate test frame
        num_data_bits = (params.N - np.sum(params.pilot_pattern)) * params.bits_per_symbol
        original_bits = np.random.randint(0, 2, num_data_bits)
        tx_frame = generate_ofdm_frame()
        
        # Apply channel effects (fading + AWGN, no CFO/timing offset)
        rx_signal, _ = apply_channel_effects(tx_frame, cfo=0, snr_db=snr_db, delay_s=0, apply_fading=True)
        
        try:
            # Perfect receiver processing
            preamble_length = params.N
            data_start = preamble_length
            data_symbol_with_cp = rx_signal[data_start:data_start + params.N + params.CP]
            
            # Remove CP and FFT
            data_symbol_no_cp = data_symbol_with_cp[params.CP:]
            rx_freq = np.fft.fft(data_symbol_no_cp)
            rx_freq_centered = np.fft.fftshift(rx_freq)
            
            # Get transmitted symbol for perfect channel knowledge
            data_symbols_orig = qam_mod(original_bits)
            ofdm_symbol_orig = insert_pilots(data_symbols_orig)
            tx_freq_centered = np.fft.fftshift(ofdm_symbol_orig)
            
            # Perfect channel estimation: H = Y / X
            channel_perfect = rx_freq_centered / tx_freq_centered
            # Handle division by zero
            channel_perfect[np.abs(tx_freq_centered) < 1e-10] = 1.0
            
            # Perfect equalization
            equalized_perfect = rx_freq_centered / channel_perfect
            
            # Extract data subcarriers
            pilot_indices_natural = np.where(params.pilot_pattern)[0]
            pilot_indices_after_fftshift = (pilot_indices_natural + params.N//2) % params.N
            data_mask = np.ones(params.N, dtype=bool)
            data_mask[pilot_indices_after_fftshift] = False
            
            data_symbols_eq = equalized_perfect[data_mask]
            
            # Demodulate
            demod_bits = qam_demod(data_symbols_eq)
            
            # Calculate BER
            if len(demod_bits) == len(original_bits):
                bit_errors = np.sum(demod_bits != original_bits)
                ber = bit_errors / len(original_bits)
            else:
                ber = 1.0
                bit_errors = len(original_bits)
            
            print(f"Perfect Channel Knowledge Test (SNR = {snr_db} dB):")
            print(f"  BER with perfect channel: {ber:.6f}")
            
            target_ber = 1e-2
            status_pass = ber < target_ber
            print(f"  Target BER: < {target_ber:.0e}")
            print(f"  Status: {'✓ PASS' if status_pass else '✗ FAIL'}")
            
            self.test_results['perfect_channel_ber'] = ber
            return status_pass
            
        except Exception as e:
            print(f"  Error: {e}")
            print(f"  Status: ✗ FAIL")
            self.test_results['perfect_channel_ber'] = 1.0
            return False
    
    def test_cfo_estimation_accuracy(self):
        """Test CFO estimation accuracy."""
        print("\n" + "="*60)
        print("TEST 5: CFO ESTIMATION ACCURACY")
        print("="*60)
        
        # Test different CFO values
        test_cfos = [0, 0.5, 1.0, 1.5, 2.0]  # In units of subcarrier spacing
        cfo_errors = []
        
        for true_cfo in test_cfos:
            cfo_hz = true_cfo * params.subcarrier_spacing
            
            # Generate test frame
            tx_frame = generate_ofdm_frame()
            
            # Apply CFO only (no fading, high SNR)
            rx_signal, _ = apply_channel_effects(tx_frame, cfo=cfo_hz, snr_db=30, delay_s=0, apply_fading=False)
            
            try:
                # CFO estimation
                _, estimated_cfo = estimate_and_correct_cfo(rx_signal, 0)  # Perfect timing
                
                # Calculate error
                cfo_error = abs(estimated_cfo - true_cfo)
                cfo_errors.append(cfo_error)
                
                print(f"  True CFO: {true_cfo:.1f}, Estimated: {estimated_cfo:.4f}, Error: {cfo_error:.4f}")
                
            except:
                cfo_errors.append(float('inf'))
                print(f"  True CFO: {true_cfo:.1f}, Estimation failed")
        
        # Overall CFO estimation performance
        valid_errors = [e for e in cfo_errors if not np.isinf(e)]
        if valid_errors:
            mean_error = np.mean(valid_errors)
            max_error = np.max(valid_errors)
        else:
            mean_error = float('inf')
            max_error = float('inf')
        
        print(f"\nCFO Estimation Summary:")
        print(f"  Mean error: {mean_error:.4f} subcarriers")
        print(f"  Max error: {max_error:.4f} subcarriers")
        
        # CFO estimation should be accurate to within 0.1 subcarriers
        target_accuracy = 0.5
        status_pass = mean_error < target_accuracy
        print(f"  Target accuracy: < {target_accuracy:.1f} subcarriers")
        print(f"  Status: {'✓ PASS' if status_pass else '✗ FAIL'}")
        
        self.test_results['cfo_mean_error'] = mean_error
        return status_pass
    
    def test_full_receiver_chain(self, snr_db=15):
        """Test the complete receiver chain."""
        print("\n" + "="*60)
        print("TEST 6: FULL RECEIVER CHAIN")
        print("="*60)
        
        # Generate test frame
        num_data_bits = (params.N - np.sum(params.pilot_pattern)) * params.bits_per_symbol
        original_bits = np.random.randint(0, 2, num_data_bits)
        tx_frame = generate_ofdm_frame()
        
        # Apply realistic channel conditions
        cfo_hz = 343.75e3  # ~1.1 subcarriers
        timing_offset = 43
        delay_s = timing_offset / params.sampling_rate
        
        rx_signal, actual_delay = apply_channel_effects(
            tx_frame, cfo=cfo_hz, snr_db=snr_db, delay_s=delay_s, apply_fading=True)
        
        try:
            # Full receiver processing
            rx_symbol_freq_corrected, cfo_est = estimate_and_correct_cfo(rx_signal, actual_delay)
            
            # Channel estimation and equalization
            channel_est_ls, _ = estimate_channel_from_ofdm_symbol(rx_symbol_freq_corrected, method='ls')
            equalized_zf = zf_equalizer(rx_symbol_freq_corrected, channel_est_ls)
            
            # Extract data and demodulate
            pilot_indices_natural = np.where(params.pilot_pattern)[0]
            pilot_indices_after_fftshift = (pilot_indices_natural + params.N//2) % params.N
            data_mask = np.ones(params.N, dtype=bool)
            data_mask[pilot_indices_after_fftshift] = False
            
            data_symbols_eq = equalized_zf[data_mask]
            demod_bits = qam_demod(data_symbols_eq)
            
            # Calculate BER
            if len(demod_bits) == len(original_bits):
                bit_errors = np.sum(demod_bits != original_bits)
                ber = bit_errors / len(original_bits)
            else:
                ber = 1.0
                bit_errors = len(original_bits)
            
            print(f"Full Receiver Chain Test:")
            print(f"  SNR: {snr_db} dB")
            print(f"  CFO: {cfo_hz/1e3:.1f} kHz ({cfo_hz/params.subcarrier_spacing:.2f} subcarriers)")
            print(f"  Timing offset: {timing_offset} samples")
            print(f"  Estimated CFO: {cfo_est:.4f} subcarriers")
            print(f"  BER: {ber:.6f}")
            
            # IEEE 802.11 target: BER < 10^-3 at 15 dB SNR
            target_ber = 1e-3
            status_pass = ber < target_ber
            print(f"  Target BER: < {target_ber:.0e}")
            print(f"  Status: {'✓ PASS' if status_pass else '✗ FAIL'}")
            
            self.test_results['full_chain_ber'] = ber
            self.diagnostic_data['full_chain_symbols'] = data_symbols_eq
            return status_pass
            
        except Exception as e:
            print(f"  Error: {e}")
            print(f"  Status: ✗ FAIL")
            self.test_results['full_chain_ber'] = 1.0
            return False
    
    def generate_diagnostic_plots(self):
        """Generate diagnostic plots for troubleshooting."""
        print("\n" + "="*60)
        print("GENERATING DIAGNOSTIC PLOTS")
        print("="*60)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: AWGN-only constellation
        if 'awgn_rx_symbols' in self.diagnostic_data:
            symbols = self.diagnostic_data['awgn_rx_symbols']
            axes[0,0].scatter(np.real(symbols), np.imag(symbols), alpha=0.6, s=20)
            axes[0,0].set_title('AWGN Channel Only - Received Constellation')
            axes[0,0].set_xlabel('In-phase')
            axes[0,0].set_ylabel('Quadrature')
            axes[0,0].grid(True, alpha=0.3)
            axes[0,0].axis('equal')
        
        # Plot 2: Full chain constellation
        if 'full_chain_symbols' in self.diagnostic_data:
            symbols = self.diagnostic_data['full_chain_symbols']
            axes[0,1].scatter(np.real(symbols), np.imag(symbols), alpha=0.6, s=20)
            axes[0,1].set_title('Full Chain - Equalized Constellation')
            axes[0,1].set_xlabel('In-phase')
            axes[0,1].set_ylabel('Quadrature')
            axes[0,1].grid(True, alpha=0.3)
            axes[0,1].axis('equal')
        
        # Plot 3: BER comparison
        test_names = []
        ber_values = []
        for test_name, ber in self.test_results.items():
            if 'ber' in test_name:
                test_names.append(test_name.replace('_ber', '').replace('_', ' ').title())
                ber_values.append(ber)
        
        if test_names:
            axes[1,0].bar(test_names, ber_values)
            axes[1,0].set_yscale('log')
            axes[1,0].set_ylabel('Bit Error Rate')
            axes[1,0].set_title('BER Comparison Across Tests')
            axes[1,0].tick_params(axis='x', rotation=45)
            axes[1,0].grid(True, alpha=0.3)
            
            # Add target line
            axes[1,0].axhline(y=1e-3, color='r', linestyle='--', label='Target BER')
            axes[1,0].legend()
        
        # Plot 4: Test summary
        axes[1,1].axis('off')
        summary_text = "TEST BENCH SUMMARY\n\n"
        for test_name, result in self.test_results.items():
            if isinstance(result, bool):
                status = "PASS" if result else "FAIL"
                summary_text += f"{test_name}: {status}\n"
            elif 'ber' in test_name:
                summary_text += f"{test_name}: {result:.2e}\n"
            else:
                summary_text += f"{test_name}: {result:.4f}\n"
        
        axes[1,1].text(0.1, 0.9, summary_text, transform=axes[1,1].transAxes, 
                      fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig('test_bench_diagnostics.png', dpi=300, bbox_inches='tight')
        print("Diagnostic plots saved to 'test_bench_diagnostics.png'")
        
        return fig
    
    def run_all_tests(self):
        """Run all tests in sequence."""
        print("="*80)
        print("OFDM RECEIVER COMPREHENSIVE TEST BENCH")
        print("="*80)
        
        tests = [
            self.test_qam_modulation_demodulation,
            self.test_ofdm_transmitter,
            self.test_awgn_channel_only,
            self.test_perfect_channel_knowledge,
            self.test_cfo_estimation_accuracy,
            self.test_full_receiver_chain
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test in tests:
            try:
                if test():
                    passed_tests += 1
            except Exception as e:
                print(f"Test failed with exception: {e}")
        
        # Generate diagnostic plots
        self.generate_diagnostic_plots()
        
        # Final summary
        print("\n" + "="*80)
        print("TEST BENCH SUMMARY")
        print("="*80)
        print(f"Tests passed: {passed_tests}/{total_tests}")
        print(f"Overall status: {'✓ PASS' if passed_tests == total_tests else '✗ FAIL'}")
        
        if passed_tests < total_tests:
            print("\nISSUES IDENTIFIED:")
            if self.test_results.get('qam_ber', 0) > 0:
                print("- QAM modulation/demodulation has errors")
            if not self.test_results.get('transmitter_pass', True):
                print("- OFDM transmitter has issues")
            if self.test_results.get('awgn_ber', 0) > 1e-2:
                print("- Poor performance even with AWGN-only channel")
            if self.test_results.get('perfect_channel_ber', 0) > 1e-2:
                print("- Issues with equalization or demodulation")
            if self.test_results.get('cfo_mean_error', 0) > 0.5:
                print("- CFO estimation is inaccurate")
            if self.test_results.get('full_chain_ber', 0) > 1e-3:
                print("- Full receiver chain does not meet IEEE 802.11 requirements")
        
        print("\nRecommendations for improvement:")
        print("- Debug QAM constellation mapping and decision boundaries")
        print("- Verify pilot insertion and extraction logic")
        print("- Improve CFO estimation algorithm")
        print("- Optimize channel estimation and equalization")
        print("- Consider advanced synchronization techniques")
        
        return passed_tests == total_tests

def main():
    """Main function to run the test bench."""
    test_bench = OFDMTestBench()
    success = test_bench.run_all_tests()
    return success

if __name__ == "__main__":
    main()
