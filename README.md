# postfilt_gan
This is an implementation of "Generative adversarial network-based postfilter for statistical parametric speech synthesis"

## PREPARE DATA

Need parallel data including natural mgc - synthesized mgc. 
The MGC files are 32 bit float binary files.

Need to create:

* The `ref_files.list` contains absolute paths of natural mgc
* The `gen_files.list` contains absolute paths of synthesized mgc

by running `mkList.py`

## TRAINING DATA

* The `models.py` contains the structures of Generator and Discriminator
* The `main.py` contains the training process for GAN 
* The `main_circle.py` contains the training process for GAN circle (Work in progress)
* The `main_wgan.py` contains the training process for Wasserstein GAN (Work in progress)

## SYNTHESIS

* Synthesize using WORLD vocoder synthesizer in `synthesize.py`
* Run `synthesize.py`
* Need SPTK library: `SOPR, X2X, MGC2SP` for running `prepare.sh`

## REQUIREMENTS

python==3.6.8
numpy==1.16.4
scipy>=1.2.1
matplotlib==3.0.2
pytorch==1.1.0
scikit-learn==0.21.2
