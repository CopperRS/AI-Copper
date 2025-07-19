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

Extract libtorch to `C:\libtorch`(CPU), in **PATH** add the variables **C:\libtorch** and **C:\libtorch\bin**

---

`Linux`

Extract libtorch(CPU) into the directory you want.
Add **libtorch/lib** and **libtorch/bin** to LD_LIBRARY_PATH

```
export LIBTORCH_PATH=/home/yourname/libtorch
export LD_LIBRARY_PATH=$LIBTORCH_PATH/lib:$LD_LIBRARY_PATH
```

To make these changes permanent, add them to your `~/.bashrc`or `~/.zshrc:`

```
echo 'export LIBTORCH=/home/yourname/libtorch' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```
## Installation

**Windows**

To add `ai_copper` to your Rust project, simply include the following line in your `Cargo.toml:`

```toml
[dependencies]
ai_copper = { git = "https://github.com/CopperRS/ai_copper.git" }
```

Then, run:

```
cargo build
cargo run
```
---

**Linux**

To use `ai_copper` on Linux, you need to clone the repository and build it locally to generate the shared library (`libai_copper.so`). Follow these steps`

1. Clone the Repository:

```bash
git clone https://github.com/CopperRS/ai_copper.git
cd ai_copper
```

2. Add as a Local Dependency: In your project's `Cargo.toml`, add `ai_copper` as a path dependency, pointing to the cloned repository:

```toml
[dependencies]
ai_copper = { path = "/path/to/ai_copper" }
```
Replace /path/to/ai_copper with the actual path where you cloned the repository

3. Build the Project: Run the following command in your project directory to build the project and generate the `libai_copper.so` file`

```bash
cargo build
```

This will create the shared library in `/path/to/ai_copper/cpp/build`.

4. Run the Project: Before running your project, set the `LD_LIBRARY_PATH` to include the directory containing `libai_copper.so`: 

```bash
export LD_LIBRARY_PATH=/path/to/ai_copper/cpp/build:$LD_LIBRARY_PATH
cargo run
```
