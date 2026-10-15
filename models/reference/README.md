# Phase 0 Reference Pretrained Models

This directory contains reference models evaluated during Phase 0 proof-of-concept testing.

## Files
- `isl_model.h5`: Reference LSTM model from aju22/Real-Time-ISL-Translation.
  - Architecture: Sequential LSTM (64 units -> 128 units -> 64 units -> Dense 64 -> Dense 32 -> Dense 11)
  - Input Contract: (1, 30, 150) — sequence of 30 frames with 150 legacy features
  - Target Classes (11): ['Hello', 'I am', 'Deaf', 'and', 'Mute', 'Father', 'Mother', 'Happy', 'Sad', 'Food', 'Water']
  - Status: Read-only reference for Phase 0 offline validation; kept strictly separate from SETU's 158-feature recognition pipeline.
  - License/Provenance Note: Reference model provided by aju22 under MIT license for research evaluation. Not for direct production deployment in SETU without retrained 158-feature architecture.
