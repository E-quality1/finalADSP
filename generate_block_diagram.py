"""
Generate OFDM Receiver Block Diagram

This script creates a professional block diagram showing the OFDM receiver chain
architecture for inclusion in the technical documentation.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

def create_ofdm_receiver_block_diagram():
    """Create a professional OFDM receiver block diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Define colors
    colors = {
        'input': '#E8F4FD',      # Light blue
        'sync': '#FFE6CC',       # Light orange
        'cfo': '#E6F3E6',        # Light green
        'channel': '#F0E6FF',    # Light purple
        'equalize': '#FFE6F0',   # Light pink
        'output': '#FFF2E6',     # Light yellow
        'arrow': '#4A4A4A'       # Dark gray
    }
    
    # Block dimensions
    block_width = 1.8
    block_height = 0.8
    spacing_x = 2.2
    spacing_y = 1.5
    
    # Define blocks with positions and labels
    blocks = [
        # Row 1 (top)
        {'pos': (0, 3), 'label': 'Received\nSignal\nr(t)', 'color': colors['input'], 'type': 'input'},
        {'pos': (spacing_x, 3), 'label': 'Timing\nSynchronization\n(Schmidl & Cox)', 'color': colors['sync'], 'type': 'process'},
        {'pos': (2*spacing_x, 3), 'label': 'Remove\nCyclic Prefix', 'color': colors['sync'], 'type': 'process'},
        {'pos': (3*spacing_x, 3), 'label': 'FFT\n(Time → Freq)', 'color': colors['sync'], 'type': 'process'},
        
        # Row 2 (middle)
        {'pos': (4*spacing_x, 3), 'label': 'FFT Shift\n(Center Spectrum)', 'color': colors['cfo'], 'type': 'process'},
        {'pos': (5*spacing_x, 3), 'label': 'Integer CFO\nEstimation\n(Pilot Correlation)', 'color': colors['cfo'], 'type': 'process'},
        
        # Row 3 (lower middle)
        {'pos': (5*spacing_x, 1.5), 'label': 'Fractional CFO\nEstimation\n(Pilot Phase)', 'color': colors['cfo'], 'type': 'process'},
        {'pos': (4*spacing_x, 1.5), 'label': 'CFO\nCorrection', 'color': colors['cfo'], 'type': 'process'},
        
        # Row 4 (bottom)
        {'pos': (3*spacing_x, 1.5), 'label': 'Channel\nEstimation\n(LS/MMSE)', 'color': colors['channel'], 'type': 'process'},
        {'pos': (2*spacing_x, 1.5), 'label': 'Equalization\n(ZF/MMSE)', 'color': colors['equalize'], 'type': 'process'},
        {'pos': (spacing_x, 1.5), 'label': 'Extract Data\nSubcarriers', 'color': colors['equalize'], 'type': 'process'},
        {'pos': (0, 1.5), 'label': '64-QAM\nDemodulation', 'color': colors['output'], 'type': 'process'},
        
        # Final output
        {'pos': (0, 0), 'label': 'Decoded\nBits', 'color': colors['output'], 'type': 'output'},
    ]
    
    # Draw blocks
    for block in blocks:
        x, y = block['pos']
        
        # Create fancy box
        if block['type'] == 'input' or block['type'] == 'output':
            # Rounded rectangle for input/output
            fancy_box = FancyBboxPatch(
                (x - block_width/2, y - block_height/2),
                block_width, block_height,
                boxstyle="round,pad=0.1",
                facecolor=block['color'],
                edgecolor='black',
                linewidth=1.5
            )
        else:
            # Regular rectangle for processing blocks
            fancy_box = FancyBboxPatch(
                (x - block_width/2, y - block_height/2),
                block_width, block_height,
                boxstyle="square,pad=0.05",
                facecolor=block['color'],
                edgecolor='black',
                linewidth=1
            )
        
        ax.add_patch(fancy_box)
        
        # Add text
        fontsize = 9 if '\n' in block['label'] else 10
        fontweight = 'bold' if block['type'] in ['input', 'output'] else 'normal'
        ax.text(x, y, block['label'], ha='center', va='center', 
                fontsize=fontsize, fontweight=fontweight, wrap=True)
    
    # Define arrows (from_block_index, to_block_index)
    arrows = [
        # Main forward path
        (0, 1),   # Input to Timing Sync
        (1, 2),   # Timing Sync to Remove CP
        (2, 3),   # Remove CP to FFT
        (3, 4),   # FFT to FFT Shift
        (4, 5),   # FFT Shift to Integer CFO
        
        # CFO estimation and correction loop
        (5, 6),   # Integer CFO to Fractional CFO
        (6, 7),   # Fractional CFO to CFO Correction
        (7, 8),   # CFO Correction to Channel Estimation
        
        # Final processing chain
        (8, 9),   # Channel Estimation to Equalization
        (9, 10),  # Equalization to Extract Data
        (10, 11), # Extract Data to QAM Demod
        (11, 12), # QAM Demod to Output
    ]
    
    # Draw arrows
    for from_idx, to_idx in arrows:
        from_pos = blocks[from_idx]['pos']
        to_pos = blocks[to_idx]['pos']
        
        # Calculate arrow start and end points
        if from_pos[1] == to_pos[1]:  # Same row - horizontal arrow
            if from_pos[0] < to_pos[0]:  # Left to right
                start_x = from_pos[0] + block_width/2
                end_x = to_pos[0] - block_width/2
            else:  # Right to left
                start_x = from_pos[0] - block_width/2
                end_x = to_pos[0] + block_width/2
            start_y = end_y = from_pos[1]
        else:  # Different rows - vertical arrow
            start_x = end_x = from_pos[0]
            if from_pos[1] > to_pos[1]:  # Top to bottom
                start_y = from_pos[1] - block_height/2
                end_y = to_pos[1] + block_height/2
            else:  # Bottom to top
                start_y = from_pos[1] + block_height/2
                end_y = to_pos[1] - block_height/2
        
        # Draw arrow
        ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color=colors['arrow']))
    
    # Add feedback arrow for CFO correction
    feedback_start = (5*spacing_x - block_width/2, 1.5 + block_height/2)
    feedback_end = (4*spacing_x + block_width/2, 3 - block_height/2)
    
    # Create curved feedback arrow
    ax.annotate('', xy=feedback_end, xytext=feedback_start,
               arrowprops=dict(arrowstyle='->', lw=1.5, color=colors['arrow'],
                             connectionstyle="arc3,rad=0.3"))
    
    # Add labels for major sections
    section_labels = [
        {'pos': (spacing_x, 4), 'label': 'Synchronization', 'color': colors['sync']},
        {'pos': (4.5*spacing_x, 4), 'label': 'CFO Estimation & Correction', 'color': colors['cfo']},
        {'pos': (2.5*spacing_x, 0.5), 'label': 'Channel Estimation & Equalization', 'color': colors['channel']},
    ]
    
    for section in section_labels:
        x, y = section['pos']
        ax.text(x, y, section['label'], ha='center', va='center',
                fontsize=12, fontweight='bold', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor=section['color'], alpha=0.7))
    
    # Set axis properties
    ax.set_xlim(-1, 12)
    ax.set_ylim(-0.5, 4.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Add title
    plt.title('64-QAM OFDM Receiver Architecture', fontsize=16, fontweight='bold', pad=20)
    
    # Add legend
    legend_elements = [
        patches.Patch(color=colors['input'], label='Input/Output'),
        patches.Patch(color=colors['sync'], label='Synchronization'),
        patches.Patch(color=colors['cfo'], label='CFO Processing'),
        patches.Patch(color=colors['channel'], label='Channel Processing'),
        patches.Patch(color=colors['equalize'], label='Equalization'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    plt.savefig('ofdm_receiver_block_diagram.png', dpi=300, bbox_inches='tight')
    plt.savefig('ofdm_receiver_block_diagram.pdf', bbox_inches='tight')
    print("OFDM receiver block diagram saved to 'ofdm_receiver_block_diagram.png' and '.pdf'")
    
    return fig

def main():
    """Generate the OFDM receiver block diagram."""
    print("Generating OFDM Receiver Block Diagram...")
    fig = create_ofdm_receiver_block_diagram()
    plt.show()

if __name__ == "__main__":
    main()
