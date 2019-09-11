# Data-Driven Design for Fourier Ptychographic Microscopy

![](https://people.eecs.berkeley.edu/~kellman/topics/iccp_fpm_2019/iccp_fpm_2019-03.png)

Fourier Ptychographic Microscopy (FPM) is a computational imaging method that is able to super-resolve features beyond the diffraction-limit set by the objective lens of a traditional microscope. This is accomplished by using synthetic aperture and phase retrieval algorithms to combine many measurements captured by an LED array microscope with programmable source patterns. FPM provides simultaneous large field-of-view and high resolution imaging, but at the cost of reduced temporal resolution, thereby limiting live cell applications. In this work, we learn LED source pattern designs that compress the many required measurements into only a few, with negligible loss in reconstruction quality or resolution. This is accomplished by recasting the super-resolution reconstruction as a Physics-based Network and learning the experimental design to optimize the network's overall performance. Specifically, we learn LED patterns for different applications (_e.g._ amplitude contrast and quantitative phase imaging) and show that the designs we learn through simulation generalize well in the experimental setting.

This library provides an implementation of physics-based learned design for Fourier Ptychographic Microscopy (FPM). Reconstructions are implemented in Pytorch, learning is accomplished using backpropagation, and all is fully supported to run on GPU. We provide synthetic training data and experimental testing data used in the paper's experiments [here]([data]).

## Demonstration

1. To run the code, please download all of the training and experimental data from [here]([data]). We provide synthetic training and experimental testing data for both amplitude and quantitative phase imaging applications.

2. Training (make sure to set path argument to the desired training data). We have also included an ipython notebook that has similar functionality.
``` python train.py --verbose True --training_iter 100 --batch_size 5 --test_freq 1 --step_size 0.01 --num_unrolls 75 --alpha 0.1 --num_bf 1 --num_df 9 --loss abs --tensorboard True --path=./training_data_amplitude.mat```
    
3. Monitor training by starting tensorboard
```tensorboard --logdir runs```

4. Inference can be run using the ipython notebook Inference (experimental data).


**Arguments:**

    * path (string) - _training dataset path_
    * training_iter (int) - _number of iterations for training_
    * step_size (float) - _step size for training_
    * batch_size (int) - _batch size per training iteration_
    * num_batch (int) - _number of batches_
    * loss (string) - _loss function for training (mse on the complex value, mse on the amplitude, mse on the phase)_
    * test_freq (int) - _test dataset evaluated every number of training iterations_
    * optim (string) - _optimizer for training (_e.g._ adam, sgd)_
    * gpu (int) - _GPU device number used for training (-1 for cpu)_
    * verbose (bool) - _prints extra outputs_
    * tensorboard (bool) - _writes out intermediate training information to a tensorboard_
    * alpha (float) - _step size for physics-based network_
    * num_meas (int) - _number of measurements for the learned design_
    * num_bf (int) - _number of bright-field images for learned design constraint_
    * num_df (int) - _number of dark-field images for learned design constraint_
    * num_unrolls (int) - _number of layers for physics-based network_


## Requirements

* [pytorch](https://pytorch.org/)
* [tensorboard](https://pypi.org/project/tensorboard/)

## References

[1] Michael Kellman, Emrah Bostan, Michael Chen, Laura Waller. "Data-driven Design for Fourier Ptychographic Microscopy." International Conference for Computational Photography. IEEE, 2019.

[2] Michael Kellman, Emrah Bostan, Nicole Repina, Laura Waller. "Physics-based Learned Design: Optimized Coded-Illumination for Quantitative Phase Imaging." Transactions on Computational Imaging. IEEE, 2019.

If you found this library/demonstration useful in your research, please consider citing and checkout my other [research](mrkellman.com)

```
@article{kellman2019data,
    title={Data-Driven Design for Fourier Ptychographic Microscopy},
    author={Kellman, Michael and Bostan, Emrah and Chen, Michael and Waller, Laura},
    journal={International Conference for Computational Photography},
    year={2019}
}
```

[data]:https://drive.google.com/open?id=1vfvI_AqS5XGdLabum4dB0jOl2u0AAFVy
