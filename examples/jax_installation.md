# Set up environment for JAX sampling with GPU supports in PyMC v4

This week, I have spent sometimes to re-install my dev environment, as I need to change to a new hard-drive. So I make a note on the steps I have done, hope that it will be useful for others, who want to run PyMC v4 with GPU support for Jax sampling. The step-by-step as follow:

## 1. Install Ubuntu 20.04.4 LTS (Focal Fossa)

The latest Ubuntu version is 22.04, but I'm a little bit conservative, so decided to install version 20.04. I download the `64-bit PC (AMD64) desktop image` from [here](https://releases.ubuntu.com/20.04/)

I made a Bootable USB using Rufus with the above ubuntu desktop .iso image. You can check this video [How to Make Ubuntu 20.04 Bootable USB Drive](https://www.youtube.com/watch?v=X_fDdUgqIUQ)


## 2. Install NVIDIA Driver, CUDA 11.4, cuDNN v8.2.4

According to Jax's [guidelines](https://github.com/google/jax#pip-installation-gpu-cuda), To install GPU support for Jax, first we need to install CUDA and CuDNN.

To do that, I follow the Installation of NVIDIA Drivers, CUDA and cuDNN from [this guide](https://github.com/ashutoshIITK/install_cuda_cudnn_ubuntu_20).

After confirming the suitable version, go to this page and download appropriate driver for your GPU: https://www.nvidia.com/download/index.aspx?lang=en-us


One note is that we may not be able to find a specific version of nvidia driver by using the previous website. Instead, we can go to this url: https://download.nvidia.com/XFree86/Linux-x86_64/ to download our specific driver version. For my case, I download the file `NVIDIA-Linux-x86_64-470.82.01.run` at this link: https://download.nvidia.com/XFree86/Linux-x86_64/470.82.01/


After successfull these steps, we can run `nvidia-smi` and `nvcc --version` commands to verify the installation.

In my case, it will be somethings like this:


## 3. Install Jax with GPU supports

Following the Jax's [guidelines](https://github.com/google/jax#pip-installation-gpu-cuda), after installing CUDA and CuDNN, we can using pip to install Jax with GPU support.

pip install --upgrade pip
# Installs the wheel compatible with CUDA 11 and cuDNN 8.2 or newer.
# Note: wheels only available on linux.
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html


### Check if GPU device is available in Jax

We can run `Ipython` or `python` and using these following commands to check.

In [1]: import jax
In [2]: jax.default_backend()
Out[2]: 'gpu'
In [3]: jax.devices()
Out[3]: [GpuDevice(id=0, process_index=0)]
In [4]: jax.devices()[0]
Out[4]: GpuDevice(id=0, process_index=0)

That's it. We have successfully installed Jax with GPU supports. Now, we can run Jax-based sampling `pm.sampling_jax.sample_numpyro_nuts(...)` in PyMC v4 with the GPU capability.

Feel free to ask any questions here if you face any difficulty in these steps.

Cheers!


