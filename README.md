# BPDL
Binders Polymer Dimers &amp; Loop extrusion

Python scripts to implement Loop Extrusion mechanism and RNAPolII-driven transcription condensates formation.

The framework relies on the API of EspressoMD to perform Molecular Dynamics simulations.

### Requirements
scripts have been tested for:
- EspressoMD 4.1.4
- pandas 1.3.4, numpy 1.21.3, scipy 1.8.0
- python 3.9.0
- Unix environment
- Libre office in order to read/edit .ods files

### Run scripts
When launched from within the downloading folder, the following command can be used to run the script:

pypresso bpdl.py -model deg2 -modelName bpdl -procid 1 -maxt 100000 -tit 100 -tsam lin -param deg2e2 -c c0 -reg chr2s1s1z1m1 -dopt dCHp2

3D configurations will be saved in a new folder within the downloading folder, according to the name given to the model.

In the folder ./obs additional files will contain additional observables:
- obs : Gyration radius of the different model's types
- obsLope: Which cohesin molecule is attached to which polymer site for each time
- obsReel: Which RNAPolII molecule is attached to which polymer site for each time
- obsSpec: Which CTCF molecule is attached to which polymer site for each time

Here is the description of the options given as input of the script:
- -procid : int : id of simulation
- -maxt : int : total number of time steps
- -tit  : int : total number of time semplings
- -model  : str : id of the model type
- -tsam  : str : time sampling type: lin or log
- -param : str : name of parameters set selected (to be found in file degron_model.ods, sheets= param and paramtype)
- -c : str : name of concentration set selected (to be found in file degron_model.ods, sheet=concentrations)
- -reg : str : name of the polymer system (to be found as sheet in file degron_sys.ods)
- -dopt : str : options to swith between 90/10% scenario of cohesin loading at promoter/enhancer (dCHp2) and cohesin loading with same probability across the polymer (dCHp1)

In file degron_model.ods -> paramtype are selected the epsilon of the Lennard-Jones potential in the column 'val'. They are in $K_BT$ units.

In file degron_model.ods -> concentrations the concentrations reported are in nanomoles per liter.
