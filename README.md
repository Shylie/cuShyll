# cuShyll
cuShyll is a pseudocode to C++ transpiler to aid in writing object-oriented code for [CUDA](https://en.wikipedia.org/wiki/CUDA) programs.
## Why it exists
Virtual methods exist only on the side (CPU or GPU) that the object was created on, so passing an object made on the CPU to the GPU doesn't work, as the vtable holds incorrect addresses. The workaround to this is very convoluted and is more work than just writing code without virtual methods.
## How cuShyll solves this problem
cuShyll outputs a tagged union inside a struct to indicate the type of the object, and upon calling a "virtual method", actually calls a function based on the type held in the tag. by calling the function this way, there are no vtables.
