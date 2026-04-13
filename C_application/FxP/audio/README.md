# Audio FxP Module (Planned)

This directory is intentionally minimal for now.

When audio fixed-point porting starts, keep only audio-domain adapters here:

- FFT/scaling adapters around KissFFT
- periodogram and mel scaling glue
- audio-domain kernel wrappers

Generic math/types/converters should remain in `FxP/core/`.
