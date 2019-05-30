# Data-Driven Design for Fourier Ptychographic Microscopy

This library provides an implementation of physics-based learned design for Fourier Ptychographic Microscopy (FPM). Reconstructions are implemented in Pytorch, learning is accomplished using backpropagation, and all is fully supported to run on GPU.

Simulated datasets and experimental data is [here]().

## Demonstration

![](https://people.eecs.berkeley.edu/~kellman/topics/iccp_fpm_2019/iccp_fpm_2019-03.png)

## Requirements

* [Pytorch](https://pytorch.org/)
* [SigPy](https://pypi.org/project/sigpy/)

## References

[1] Michael Kellman, Emrah Bostan, Michael Chen, Laura Waller. "Data-driven Design for Fourier Ptychographic Microscopy." International Conference for Computational Photography. IEEE, 2019.

[2] Michael Kellman, Emrah Bostan, Nicole Repina, Laura Waller. "Physics-based Learned Design: Optimized Coded-Illumination for Quantitative Phase Imaging." Transactions on Computational Imaging. IEEE, 2019.

If you found this library/demonstration useful in your research, please consider citing

```
@article{kellman2019data,
    title={Data-Driven Design for Fourier Ptychographic Microscopy},
    author={Kellman, Michael and Bostan, Emrah and Chen, Michael and Waller, Laura},
    journal={international conference for computational photography},
    year={2019}
}
```




Put gif of learning LED patterns by iterations
put low res/single LED/heuristic/learned result
