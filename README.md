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

## Environment variables

`Windows`

Extract libtorch to `C:\libtorch`, in **PATH** add the variables **C:\libtorch** and **C:\libtorch\bin**

---

`Linux`

Extract libtorch into the directory you want.
Add **libtorch/lib** and **libtorch/bin** to LD_LIBRARY_PATH

```
export LIBTORCH_PATH=/home/yourname/libtorch
export LD_LIBRARY_PATH=$LIBTORCH_PATH/lib:$LD_LIBRARY_PATH
```

## Common Errors and Fixes

| Error                                 | Cause                                      | Fix                                                           |
| ------------------------------------- | ------------------------------------------ | ------------------------------------------------------------- |
| `STATUS_DLL_NOT_FOUND (0xc0000135)`   | Windows can't find `torch.dll`, etc.       | Add `libtorch/bin` to `PATH` or copy DLLs to `target/debug/`. |
| `undefined symbol` (Linux)            | Shared library not found                   | Export `LD_LIBRARY_PATH` correctly.                           |
| Linker errors (MSVCRT/Debug mismatch) | libtorch compiled in Release vs Rust Debug | Use same build mode (Debug ↔ Debug, Release ↔ Release).       |
| `ldd` shows `not found`               | Linux can't locate `.so` files             | Fix `LD_LIBRARY_PATH` and restart terminal.                   |

## How to Run on Linux

After adding `ai_copper` to your project and building with `cargo build`, you need to ensure that Rust can find the shared library (`libai_copper.so`) generated during the build process.

By default, the `.so` file is generated in the `cpp/build` directory inside the project. Before running your binary, set the `LD_LIBRARY_PATH` environment variable to this directory:

```bash
export LD_LIBRARY_PATH=$(pwd)/cpp/build:$LD_LIBRARY_PATH
cargo run
```

Or, if you want to run the compiled binary directly:

```bash
export LD_LIBRARY_PATH=$(pwd)/cpp/build:$LD_LIBRARY_PATH
./target/debug/<your_binary_name>
```

**Tip:** You can create a simple `run.sh` script to automate this:

```bash
#!/bin/bash
export LD_LIBRARY_PATH=$(pwd)/cpp/build:$LD_LIBRARY_PATH
cargo run
```

Make it executable:

```bash
chmod +x run.sh
./run.sh
```

This ensures your Rust application can locate the `.so` file at runtime, avoiding errors related to missing shared libraries.


## Installation of lib

To add `ai_copper` to your Rust project:

Include the following line in your `Cargo.toml`:

```toml
[dependencies]
ai_copper = { git = "https://github.com/CopperRS/ai_copper.git" }
```