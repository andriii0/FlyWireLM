# Full FlyWire v783 benchmark

Environment: RTX 5070 12 GB, Windows WDDM, PyTorch 2.11.0+cu128. The desktop
used about 2.2 GB VRAM outside PyTorch during the run. Timings use 10 CUDA
warm-up iterations, 30 synchronized measurements, and report the median.

The preprocessing reproduced 139,255 neurons, 15,091,983 aggregated directed
pairs before thresholding, and 2,700,513 connections with `syn_count >= 5`.
There are 19,265 afferent and 1,489 efferent neurons. The graph stores
`A[post, pre]`; each weight is the pair's synapse count divided by the total
incoming synapse count of its postsynaptic neuron.

| Backend | Batch | `A @ h` forward | Recurrent forward | Recurrent forward + backward | PyTorch peak allocation |
|---|---:|---:|---:|---:|---:|
| CSR | 1 | 0.107 ms | 0.124 ms | 2.924 ms | 253 MiB |
| CSR | 32 | 0.452 ms | 0.663 ms | 4.187 ms | 369 MiB |
| CSR | 128 | 0.805 ms | 2.600 ms | 9.233 ms | 748 MiB |
| CSR | 256 | 1.595 ms | 5.331 ms | 15.882 ms | 1,360 MiB |
| CSR | 512 | 3.159 ms | 10.016 ms | 27.835 ms | 2,584 MiB |
| CSR | 1,024 | 6.335 ms | 20.095 ms | 53.990 ms | 5,032 MiB |
| edge `index_add_` | 1 | 0.170 ms | 0.191 ms | 0.466 ms | 106 MiB |
| edge `index_add_` | 32 | 4.868 ms | 5.175 ms | 10.731 ms | 861 MiB |
| edge `index_add_` | 128 | 14.725 ms | 16.529 ms | 34.614 ms | 3,195 MiB |
| edge `index_add_` | 192 | 22.463 ms | 24.937 ms | 48.644 ms | 4,752 MiB |

CSR is the practical training backend. At batch 1-2, `index_add_` has a faster
backward, but from batch 4 onward CSR is both faster and substantially more
memory-efficient. CSR recurrent forward + backward reaches about 51.2 billion
connection-sample updates per second at batch 1,024.

The largest non-pathological one-step batch was 1,024 for CSR and 192 for
`index_add_`. CSR batch 2,048 slowed from 53.99 ms to 3.98 s and reached
9,928 MiB of PyTorch allocation. Edge batch 224 slowed from 48.64 ms at batch
192 to 715.51 ms. These are WDDM memory-paging cliffs rather than useful batch
sizes. A sequence-training batch and BPTT length must therefore be selected in
the next milestone; the single-step maximum cannot be reused directly.

Synthetic dense/CSR/edge checks passed. On the full graph, CSR and edge outputs
had maximum absolute difference `4.77e-7` and mean difference `1.39e-8`.
