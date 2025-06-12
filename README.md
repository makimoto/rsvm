# rsvm - Rust Support Vector Machine

A pure Rust implementation of Support Vector Machine (SVM) based on the SVMlight paper "Making Large-Scale SVM Learning Practical" by Thorsten Joachims.

## Features

- **High-level API**: User-friendly interface with builder pattern
- **Command-line Interface**: Complete CLI for training, prediction, and evaluation
- **Efficient Implementation**: Sparse vector operations and SMO solver
- **Multiple Data Formats**: LibSVM and CSV format support with auto-detection
- **Linear Kernel**: Production-ready linear SVM implementation
- **Model Persistence**: Save and load trained models in JSON format
- **Comprehensive Testing**: 88 tests with 90%+ code coverage
- **Memory Efficient**: LRU kernel caching and sparse data structures

## Status

This project is under active development. See [DESIGN.md](DESIGN.md) for technical details.

## License

This project is licensed under the [MIT License](LICENSE).