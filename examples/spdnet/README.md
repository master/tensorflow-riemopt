# SPDNet in TensorFlow

Implementation of SPDNet [1], a Riemannian network for SPD matrix learning.

<img align="center" width="800" src="https://github.com/master/tensorflow-riemopt/blob/master/examples/spdnet/spdnet.png?raw=true">

## Requirements

 * Python 3.6+
 * SciPy
 * NumPy
 * TensorFlow 2.0+
 * TensorFlow RiemOpt

## Training

Configure `gcloud` to use Python 3:

```bash
gcloud config set ml_engine/local_python /usr/bin/python3
```

Train SPDNet locally on the Acted Facial Expression in Wild [2] dataset:

```bash
gcloud ai-platform local train \
       --module-name spdnet.task \
       --package-path . \
       -- \
       --data-dir data
       --job-dir ckpt
```

## References

 1. Huang, Zhiwu, and Luc Van Gool. "A riemannian network for SPD matrix
 learning." Proceedings of the Thirty-First AAAI Conference on Artificial
 Intelligence. AAAI Press, 2017.
 2. Dhall, Abhinav, et al. "Acted facial expressions in the wild database."
 Australian National University, Canberra, Australia, Technical Report
 TR-CS-11 2 (2011): 1.
