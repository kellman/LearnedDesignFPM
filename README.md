# Data-Driven Design for Fourier Ptychographic Microscopy

![](https://people.eecs.berkeley.edu/~kellman/topics/iccp_fpm_2019/iccp_fpm_2019-03.png)

Fourier Ptychographic Microscopy (FPM) is a computational imaging method that is able to super-resolve features beyond the diffraction-limit set by the objective lens of a traditional microscope. This is accomplished by using synthetic aperture and phase retrieval algorithms to combine many measurements captured by an LED array microscope with programmable source patterns. FPM provides simultaneous large field-of-view and high resolution imaging, but at the cost of reduced temporal resolution, thereby limiting live cell applications. In this work, we learn LED source pattern designs that compress the many required measurements into only a few, with negligible loss in reconstruction quality or resolution. This is accomplished by recasting the super-resolution reconstruction as a Physics-based Network and learning the experimental design to optimize the network's overall performance. Specifically, we learn LED patterns for different applications (_e.g._ amplitude contrast and quantitative phase imaging) and show that the designs we learn through simulation generalize well in the experimental setting.

This library provides an implementation of physics-based learned design for Fourier Ptychographic Microscopy (FPM). Reconstructions are implemented in Pytorch, learning is accomplished using backpropagation, and all is fully supported to run on GPU. We provide synthetic training data and experimental testing data used in this work's experiments [here]([data]).

## Demonstration

1. Download all of the training and experimental data below. We provide synthetic training and experimental testing data for both amplitude and quantitative phase imaging applications.
    * [Training data for quantitative phase imaging](https://drive.google.com/file/d/1oQ-K53pDB4ilDVFOLxTe8r3xjR3NBViM/view)
    * [Training data for amplitude imaging](https://drive.google.com/file/d/1kM_ub7yTCrh68ADykjiCUHCXAI6ou7w4/view)
    * [Experimental testing data for amplitude imaging](https://drive.google.com/a/berkeley.edu/file/d/1U41k1hFSJ3FS6rh6etAAdX9AkTCUXRsK/view?usp=sharing)
    * [Experimental testing data for quantitative phase imaging](https://drive.google.com/a/berkeley.edu/file/d/1vfvI_AqS5XGdLabum4dB0jOl2u0AAFVy/view?usp=sharing)

2. Training (make sure to set path argument to the desired training data). We have also included an ipython notebook that has similar functionality.
``` python train.py --verbose True --training_iter 100 --batch_size 5 --test_freq 1 --step_size 0.01 --num_unrolls 75 --alpha 0.1 --num_bf 1 --num_df 9 --loss abs --tensorboard True --path=./training_data_amplitude.mat```
    
3. Monitor training by starting tensorboard
```tensorboard --logdir runs```

4. Inference can be run using the ipython notebook Inference (experimental data).


**Arguments:**

    * path (string) - training dataset path
    * training_iter (int) - number of iterations for training
    * step_size (float) - step size for training
    * batch_size (int) - batch size per training iteration
    * num_batch (int) - number of batches
    * loss (string) - loss function for training (mse on the complex value, mse on the amplitude, mse on the phase)
    * test_freq (int) - test dataset evaluated every number of training iterations
    * optim (string) - optimizer for training (_e.g._ adam, sgd)
    * gpu (int) - GPU device number used for training (-1 for cpu)
    * verbose (bool) - prints extra outputs
    * tensorboard (bool) - writes out intermediate training information to a tensorboard
    * alpha (float) - step size for physics-based network
    * num_meas (int) - number of measurements for the learned design
    * num_bf (int) - number of bright-field images for learned design constraint
    * num_df (int) - number of dark-field images for learned design constraint
    * num_unrolls (int) - number of layers for physics-based network

## Requirements

* [pytorch](https://pytorch.org/)
* [tensorboard](https://pypi.org/project/tensorboard/)

## References

[1] **Michael Kellman**, Emrah Bostan, Michael Chen, Laura Waller. _"Data-driven Design for Fourier Ptychographic Microscopy." International Conference for Computational Photography_. IEEE, 2019.

[2] **Michael Kellman**, Emrah Bostan, Nicole Repina, Laura Waller. _"Physics-based Learned Design: Optimized Coded-Illumination for Quantitative Phase Imaging."_ Transactions on Computational Imaging. IEEE, 2019.

If you found this library/demonstration useful for your research, please consider citing this work and checking out my other [research]:

```
@article{kellman2019data,
    title={Data-Driven Design for Fourier Ptychographic Microscopy},
    author={Kellman, Michael and Bostan, Emrah and Chen, Michael and Waller, Laura},
    journal={International Conference for Computational Photography},
    year={2019}
}
```
[data]:https://drive.google.com/open?id=1vfvI_AqS5XGdLabum4dB0jOl2u0AAFVy
[research]:https://people.eecs.berkeley.edu/~kellman/
