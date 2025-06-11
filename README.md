# rsvm - Rust Support Vector Machine

A pure Rust implementation of Support Vector Machine (SVM) based on the SVMlight paper "Making Large-Scale SVM Learning Practical" by Thorsten Joachims.

## Features

- Efficient sparse vector operations
- SMO (Sequential Minimal Optimization) solver
- Linear kernel (RBF and polynomial kernels planned)
- Support for libsvm and CSV data formats
- Parallel processing with rayon
- Memory-efficient kernel caching

## Status

This project is under active development. See [DESIGN.md](DESIGN.md) for technical details.

## License

This project aims to be licensed under MIT or BSD-3-Clause (TBD).