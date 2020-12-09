# CCB-ID

CCB-ID is the Stanford Center for Conservation Biology's imaging-spectroscopy-based species classification approach.

This work is described in *Anderson, 2018*, [The CCB-ID approach to tree species mapping with airborne imaging spectroscopy](https://peerj.com/articles/5666/). It was developed as part of the NEON-NIST [ECODSE](http://www.ecodse.org/) data science evaluation competition.

All (c) 2018+ Christopher B. Anderson
- [E-mail](mailto:cbanders@stanford.edu)
- [Google Scholar](https://scholar.google.com/citations?user=LoGxS40AAAAJ&hl=en)
- [Personal website](https://cbanderson.info/)
 
## Functionality

`CCB-ID` can be used in two ways. First, you can run the scripts for training and applying species classification models (under `bin/train` and `bin/apply` respectively). Second, you could import the underlying python functions used in these scripts using `import ccbid` (based on the functions in the `ccbid/` directory.

If you install this package using Singularity (e.g., following the [Singularity install instructions](#singularity)), you could train and apply the models using the following commands.

```sh
ccb-id train -i /path/to/training_data -o /path/to/ccbid_model
ccb-id apply -i /path/to/testing_data -m /path/to/ccbid_model -o /path/to/predictions
```

You could also import the functions from `ccbid.py` in the singularity shell environment. 

```sh
ccb-id ipython
import ccbid
ccbid.read.bands('ccbid/suport_files/neon-bands.csv')
# etc.
```

Run `ccb-id train -h` and `ccb-id apply -h` to review command line options. 

These scripts are intended to work with csv and raster data inputs. HDF support is planned. However, support for raster-based data is currently limited (hdf support is even more so). Please let me know if this is something you would use and I can get my `[redacted]` together.

## ECODSE results

You can reproduce the results submitted to the ECODSE competition by using the `-e` flag in `ccb-id train` and `ccb-id apply`. To do this, run the following commands for a Singularity install.

```sh
ccb-id train -o ecodse-model -e -v
ccb-id apply -m ecodse-model -o ecodse-results.csv -e -v
```

Or from the ccbid [conda](#conda) environment.

```sh
train -o ecodse-model -e -v
apply -m ecodse-model -o ecodse-results.csv -e -v
```

Where the output file `ecodse-results.csv` will have the output species prediction probabilities. The `-e` flag ensure the ECODSE data will be used, and the `-v` flag sets the module to verbose mode to report classification metrics. 

Due to some versioning issues, the results are not exactly the same as what was submitted. If you *really* want to find the original results, see the original [scrappy code](https://github.com/christobal54/aei-grad-school/blob/master/bin/neon-classification.py).

## Using other data

The CCB-ID scripts allow using custom data as inputs to model building. These custom data should share the same formats as the data in `support_files/`. Other modifications can be made to the CCB-ID approach, such as using a custom data reducer or custom classification models. This is done by saving these custom objects to a python `pickle` file, then using `ccb-id train` options like `--reducer /path/to/reducer.pck` or `--models /path/to/model1.pck /path/to/model2.pck`. The idea here was to allow you to bring your own data to run new models. Currently, the defaults set to use the NEON/ECODSE data.
 
## Install options

Users have several options for installing CCB-ID. I originally developed the package using [Singularity](#singularity), but [conda](#conda) and [pip](#pip) installs are supported and are much less burdensome to set up.

### conda

Additionally, users can install a custom conda environment to run the CCB-ID module. You can run:

```sh
git clone https://github.com/stanford-ccb/ccb-id.git
cd ccb-id
conda env update
source activate ccbid
pip install -r requirements.txt
python setup.py install
```

Then you should have a conda environment you can actiave with `conda activate ccbid`. You can then run the executable `train -h`, or `import ccb` in python from this environment. 

### pip

You could also install the package via pip. This won't install the binary packages that are necessary to run some of the commands (e.g., `gdal`), but will install the ccb-id package into your python environment.

```sh
git clone https://github.com/stanford-ccb/ccb-id.git
cd ccb-id
pip install -r requirements.txt
python setup.py install
```

If you want to make sure you have all the binary requirements, you could follow the same commands from `singularity.build` a la:

```sh
sudo apt-get install -y python-gdal gdal-bin libgdal20 ipython python-setuptools python-dev python-pip python-tk build-essential libfontconfig1 mesa-common-dev python-numpy python-scipy python-pandas python-geopandas python-qt4 python-sip python-pyside gcc gfortran qt5.1 git vim
git clone https://github.com/stanford-ccb/ccb-id.git
cd ccb-id/
sudo pip install -r requirements.txt
sudo python setup.py install
```

But at that point I think you're better of running the Singularity [install](#singularity) since custom `gdal` installs tends to wreak havoc.

### singularity

[Singularity](http://singularity.lbl.gov/) containers can be used to package workflows, software, libraries, and data, and can be transferred across machines. The CCB-ID package comes with a Singularity build script to run the module, and contains the full CCB-ID workflow. To use it, you must have [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) and Singularity installed (instructions for [Linux](http://singularity.lbl.gov/install-linux), [Mac](http://singularity.lbl.gov/install-mac), or [Windows](http://singularity.lbl.gov/install-windows)). You can then run:

```sh
# clone the repo then build the singularity image
git clone https://github.com/stanford-ccb/ccb-id.git
cd ccb-id
sudo singularity build ccb-id singularity.build
```

Building the container will take a while. Once built, you can run:
```sh
./ccbid train -h
./ccbid python -c "import ccbid; print(dir(ccbid))"
```

These will verify the package installed correctly, and list out the command line options and the package functions.

The CCB-ID module is localized inside the `ccb-id` container, so you can move this container to any directory (e.g., if you store all your containers in one place like `~/singularity/`. Or, you could add the path with the `ccb-id` container to `$PATH`. 

## Additional information

That's all, folks

You've surely read enough. Go reward yourself by grabbing a warm beverage and perusing [any](http://70sscifiart.tumblr.com/) of [these](https://wearethemutants.com/) cool and good  [sites](http://www.iamag.co/features/the-art-of-moebius/).
