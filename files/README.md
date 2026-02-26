Use a setup_files.sh script to download the following into the RL env sandbox:
* [ROCm](https://github.com/ROCm/) (code, curated subset of relevant ones)
  * [rocm-libraries](https://github.com/ROCm/rocm-libraries) — monorepo (hipBLAS, rocBLAS, hipFFT, hipSPARSE, rocSPARSE, rocPRIM, rocsolver, MIOpen, and more)
  * [hipBLASLt](https://github.com/ROCm/hipBLASLt) — flexible GEMM API
  * [hipRAND](https://github.com/ROCm/rocRAND) — random number generation
  * [hipSOLVER](https://github.com/ROCm/hipSOLVER) — LAPACK-style solvers
  * [hipSPARSELt](https://github.com/ROCm/hipSPARSELt) — lightweight sparse ops
  * [hipCUB](https://github.com/ROCm/hipCUB) — parallel primitives (scan, sort, reduce)
  * [hipTensor](https://github.com/ROCm/hipTensor) — tensor operations
  * [rocPRIM](https://github.com/ROCm/rocPRIM) — primitive parallel algorithms
  * [rocThrust](https://github.com/ROCm/rocThrust) — STL-like parallel templates
  * [rocWMMA](https://github.com/ROCm/rocWMMA) — matrix multiply-accumulate primitives
  * [composable_kernel](https://github.com/ROCm/composable_kernel) — portable performance-critical kernels (GEMM, reductions)
  * [AMDMIGraphX](https://github.com/ROCm/AMDMIGraphX) — graph optimization for neural network inference
  * [RCCL](https://github.com/ROCm/rccl) — multi-GPU collectives (all-reduce, broadcast)
  * [rocSHMEM](https://github.com/ROCm/rocSHMEM) — PGAS communication library
  * [Magpie](https://github.com/AMD-AGI/Magpie) — GPU kernel correctness and performance evaluation
* [AITER](https://github.com/ROCm/aiter) (code)
* [Triton](https://github.com/triton-lang/triton) (code)
* [SGLang](https://github.com/sgl-project/sglang) and [vLLM](https://github.com/vllm-project/vllm) (code)

Look for documentation and download them, about:
* ROCm docs[https://rocm.docs.amd.com/]
* HIP and CK docs
* Tutorials on HIP/ROCm kernel optimization
* AMD Instinct GPUs
* MI355X architecture whitepapers, technical documentation, instruction set information
* Triton docs

Create markdown files with:
* Best practices for writing kernels in HIP
* Best practices for writing kernels in Triton