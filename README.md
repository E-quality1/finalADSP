# 64-QAM OFDM Receiver Simulation (IEEE 802.11-class)

This project simulates a baseband receiver chain for a 64-QAM, 20 MHz, 64-subcarrier OFDM link, including time-frequency synchronization and equalisation.

## Structure
- `ofdm/` – Core OFDM modules (system, transmitter, channel, synchronization, equalization, utils)
- `scripts/` – Simulation entry point

## Getting Started
1. Install dependencies: `pip install -r requirements.txt`
2. Run the simulation: `python scripts/simulate.py`

## Features
- Schmidl & Cox timing sync
- Two-stage CFO estimation/correction
- Pilot-assisted LS and MMSE channel estimation
- ZF and MMSE equalisation
- BER and ICI analysis

## References
- Schmidl & Cox (1997), van de Beek et al. (1997), Coleri et al. (2002)

---

Work in progress. See code comments for details.
# finalADSP
