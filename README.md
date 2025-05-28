# ai_copper

`ai_copper` is a Rust library that provides bindings to the C++ `libtorch` library from PyTorch, allowing the use of tensors and machine learning operations directly in Rust. This library is designed to facilitate the integration between Rust and C++ for projects that utilize `libtorch` for machine learning implementations.

## Features

- Create tensors directly in Rust using `libtorch`.
- Support for basic tensor manipulation operations.
- Easy integration with PyTorch for running machine learning models in Rust.

## Requirements

- **Rust**: The library is designed to work with the latest version of Rust.
- **C++**: A C++ compiler must be installed on your system.
- **libtorch**: The PyTorch C++ library (CPU version) must be installed on your system.

## Installation

To add `ai_copper` to your Rust project:

Include the following line in your `Cargo.toml`:

```toml
[dependencies]
ai_copper = { git = "https://github.com/CopperRS/ai_copper.git" }
```
