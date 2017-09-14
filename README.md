# SIC-Gen: Synthetic Iris-Code generator
This repository provides a reference implementation of the synthetic Iris-Code generation method proposed by [Drozdowski](https://www.dasec.h-da.de/staff/pawel-drozdowski/ "Pawel Drozdowski's website") *et al.* [1].

## License
This work is licensed under license agreement provided by Hochschule Darmstadt ([h_da-License](/hda-license.pdf)).

## Attribution
Any publications using SIC-Gen must cite and reference the conference paper [1], in which the method was proposed.

## Instructions
### Dependencies
* [Python3.5+](https://www.python.org/ "Python")
* [Matplotlib](https://matplotlib.org/ "Matplotlib")
* [NumPy](http://www.numpy.org "NumPy")
* [SciPy](https://scipy.org/ "SciPy")

### Usage
sic-gen.py [-h] [-n [SUBJECTS]] [-d [DIRECTORY]] [-p [PROCESSES]] [-v] [-l] [--version]

*optional arguments:*
* **-n** [SUBJECTS], **--subjects** [SUBJECTS] : number of subjects (default: 1)
* **-d** [DIRECTORY], **--directory** [DIRECTORY] : relative path to directory where the generated Iris-Codes will be stored (default: generated)
* **-p** [PROCESSES], **--processes** [PROCESSES] : number of CPU processes to use (default: 1)
* **-v**, **--validate** : run statistical validation after generation of templates
* **-l**, **--logging** : logging verbosity level (default is warning, -l for info and -ll for debug)
* **-h**, **--help** : show this help message and exit
* **--version** : show program's version number and exit

## References
* [1] Pawel Drozdowski, Christian Rathgeb, Christoph Busch, "SIC-Gen: A Synthetic Iris-Code Generator", in Proc. Int. Conf. of the Biometrics Special Interest Group (BIOSIG), Darmstadt, Germany, September 2017.

© [Hochschule Darmstadt](https://www.h-da.de/ "Hochschule Darmstadt website")