# Hyperspherical
Pytorch Implementation of the Hyperspherical Distribution, including Uniform, von Mises Fisher, Power Spherical.

The code in this repository **references** [power_spherical](https://github.com/nicola-decao/power_spherical) and [s-vae-pytorch](https://github.com/nicola-decao/s-vae-pytorch), which implement the aforementioned spherical distributions but contain some non-critical bugs. I spent a significant amount of effort studying `torch.distributions` and eventually made the implementation of these distributions **conform to the programming logic of `torch.distributions`**, making it **more robust**. Compared to the code in the original projects, this repository has **the following advantages**:

- Supports `batch_shape` and `event_shape`, meaning you can now create **multiple instances** of one distribution in batch;
- The `constraints` are more accurate, and the `log_prob` can be accurately computed in the **`Transform` mode**, which is something the original projects cannot do;
- Supports **broadcasting of `mu` and `kappa`**, meaning that when `mu.shape=(3, 8)` and `kappa=()` or `kappa=(1,)`, it can still create three distributions with the same `kappa` value, and vice versa;
- Supports **optimization** of `mu` and `kappa` if their `require_grad=True`.

The distributions in package `hyperspherical` are implemented by inheriting `distributions.Distribution`, while the distributions in package `sphericaltrans` are implemented by combining simple distributions with `distributions.Transform` and `distributions.TransformedDistribution`.
