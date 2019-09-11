# Data-Driven Design for Fourier Ptychographic Microscopy

![](https://people.eecs.berkeley.edu/~kellman/topics/iccp_fpm_2019/iccp_fpm_2019-03.png)

Fourier Ptychographic Microscopy (FPM) is a computational imaging method that is able to super-resolve features beyond the diffraction-limit set by the objective lens of a traditional microscope. This is accomplished by using synthetic aperture and phase retrieval algorithms to combine many measurements captured by an LED array microscope with programmable source patterns. FPM provides simultaneous large field-of-view and high resolution imaging, but at the cost of reduced temporal resolution, thereby limiting live cell applications. In this work, we learn LED source pattern designs that compress the many required measurements into only a few, with negligible loss in reconstruction quality or resolution. This is accomplished by recasting the super-resolution reconstruction as a Physics-based Network and learning the experimental design to optimize the network's overall performance. Specifically, we learn LED patterns for different applications ({\it e.g.} amplitude contrast and quantitative phase imaging) and show that the designs we learn through simulation generalize well in the experimental setting.

This library provides an implementation of physics-based learned design for Fourier Ptychographic Microscopy (FPM). Reconstructions are implemented in Pytorch, learning is accomplished using backpropagation, and all is fully supported to run on GPU. We provide synthetic training data and experimental testing data used in the paper's experiments [here]().

## Demonstration


## Requirements

* [pytorch](https://pytorch.org/)
* [tensorboard](https://pypi.org/project/tensorboard/)
* [SigPy](https://pypi.org/project/sigpy/)

## References

[1] Michael Kellman, Emrah Bostan, Michael Chen, Laura Waller. "Data-driven Design for Fourier Ptychographic Microscopy." International Conference for Computational Photography. IEEE, 2019.

[2] Michael Kellman, Emrah Bostan, Nicole Repina, Laura Waller. "Physics-based Learned Design: Optimized Coded-Illumination for Quantitative Phase Imaging." Transactions on Computational Imaging. IEEE, 2019.

If you found this library/demonstration useful in your research, please consider citing

```
@article{kellman2019data,
    title={Data-Driven Design for Fourier Ptychographic Microscopy},
    author={Kellman, Michael and Bostan, Emrah and Chen, Michael and Waller, Laura},
    journal={International Conference for Computational Photography},
    year={2019}
}
```

## TO BE DONE

Put gif of learning LED patterns by iterations
put low res/single LED/heuristic/learned result
