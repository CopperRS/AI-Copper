# ai_copper

**`ai_copper`** is a library developed for the **`Copper language`**, written in Rust with revamped C++ functions, which provides connections to **`PyTorch's`** libtorch C++ library and **`TensorFlow's`** C++ library. It allows you to create and manipulate tensors, perform machine learning operations, and use pre-trained TensorFlow models directly in Copper. The library is designed to facilitate integration between Copper, Rust, and C++ in projects that use libtorch and TensorFlow for machine learning implementations.

## Features

- Create tensors directly in Rust using `libtorch`.
- Support for basic tensor manipulation operations.
- Easy integration with PyTorch for running machine learning models in Rust.
- Support for TensorFlow C++ library integration.

## Requirements

- **Rust**: The library is designed to work with the latest version of Rust.
- **C++**: A C++ compiler (such as g++ or Clang) must be installed on the system.
- **CMake**: You must have [CMake](https://cmake.org/download/) installed and available in your PATH.
- **MSVC/Build Tools (Windows)**: On Windows, you need the [Microsoft Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (MSVC) with the C++ workload enabled.
- **libtorch**: The PyTorch C++ library (CPU version) must be installed on your system.
- **TensorFlow C++**: The TensorFlow C++ library (CPU version) must be installed on your system (optional, for TensorFlow integration)(https://www.tensorflow.org/install/lang_c?hl=pt-br).
- **Copper**: The Copper language must be configured in the environment to use the library.

## Environment variables

### Windows

You must set the environment variables correctly for the build to work:

Before building, you need to manually adjust the library names in the libtorch\lib directory to match the expected format by the linker. Follow these steps:

1. Navigate to the libtorch\lib directory (e.g., C:\libtorch\lib).
2. Rename the following files:
    
    - Change **libittnotify.lib** to ***ittnotify.lib.***
    - Change **libprotobuf.lib** to ***protobuf.lib.***
    - Change **libprotobuf-lite.lib** to ***protobuf-lite.lib.***
    - Change **libprotoc.lib** to ***protoc.lib.***


3. Ensure these renamed files are present before running cargo build.


> The build script automatically copies the necessary DLLs from libtorch and TensorFlow folders to your target directory.
You do not need to manually copy DLLs or adjust the PATH after setup.


- `LIBTORCH` — Path to the root of libtorch (e.g., `C:\libtorch`)
- `TENSORFLOW_ROOT` — Path to the root of TensorFlow C++ (e.g., `C:\libtensorflow`)

How to set (temporary, for the current terminal only):

```powershell
$env:LIBTORCH = "C:\libtorch"
$env:TENSORFLOW_ROOT = "C:\libtensorflow"
```

Or set permanently via Control Panel → System → Advanced → Environment Variables.

> The build.rs will automatically copy the required DLLs from `libtorch/lib` and `libtensorflow/lib` to the executable directory.

### Linux

Set the environment variables:

```bash
export LIBTORCH=/home/youruser/libtorch
export TENSORFLOW_ROOT=/home/youruser/libtensorflow
```

For temporary use, run the commands above in the terminal before building.

To make it permanent, add to the end of your `~/.bashrc` or `~/.zshrc`:

```bash
export LIBTORCH=/home/youruser/libtorch
export TENSORFLOW_ROOT=/home/youruser/libtensorflow
```

> The build.rs will automatically copy the required .so libraries from `libtorch/lib` and `libtensorflow/lib` to the executable directory.

**Attention:** If the variables are not set correctly, the build will fail or the executable will not find the required libraries at runtime.

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

> The build script automatically copies the necessary DLLs from libtorch and TensorFlow folders to your target directory.  
> You do **not** need to manually copy DLLs or adjust the PATH after setup.

---

**Linux**

To use `ai_copper` on Linux, you need to clone the repository and build it locally to generate the shared library (`libai_copper.so`). Follow these steps:

1. Clone the Repository:

```bash
git clone https://github.com/CopperRS/ai_copper.git
cd ai_copper
```

2. Install to cmake and g++ or clang

**`Terminal`**

```
#g++
sudo apt-get install build-essential

#clang
sudo apt-get install clang

#cmake
sudo apt-get install cmake
```

3. Create the .so file to use the lib.

**`Terminal`**

```
cd cpp
mkdir build && cd build
cmake ..
cmake --build .
cd ../..
cargo build
```

4. Add as a Local Dependency: In your project's `Cargo.toml`, add `ai_copper` as a path dependency, pointing to the cloned repository:

```toml
[dependencies]
ai_copper = { path = "/path/to/ai_copper" }
```

Replace `/path/to/ai_copper` with the actual path where you cloned the repository

5. Build the Project: Run the following command in your project directory to build the project and generate the `libai_copper.so` file

```bash
cargo build
```

This will create the shared library in `/path/to/ai_copper/cpp/build`.

6. Run the Project: Before running your project, set the `LD_LIBRARY_PATH` to include the directory containing `libai_copper.so`, libtorch, and TensorFlow libs:

**_If you haven't defined the variables permanently, you can temporarily set them to run at runtime._**

```bash
export LIBTORCH_PATH=/home/yourname/libtorch
export TENSORFLOW_ROOT=/home/yourname/libtensorflow
export LD_LIBRARY_PATH=/path/to/ai_copper/cpp/build:$LIBTORCH_PATH/lib:$TENSORFLOW_ROOT/lib:$LD_LIBRARY_PATH
cargo run
```

---

**Notes**

- Ensure that the TensorFlow C++ package you use contains the DLLs (`.dll`) on Windows or shared objects (`.so`) on Linux inside the `lib` folder.
- The build script of `ai_copper` will handle copying DLLs automatically on Windows.
- On Linux, it is essential to have your `LD_LIBRARY_PATH` correctly set so that runtime linking works.
- This setup allows you to use both PyTorch (`libtorch`) and TensorFlow C++ APIs seamlessly within Rust using `ai_copper`.

