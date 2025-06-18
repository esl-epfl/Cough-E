# Embedded application

This folder contains the code for the embedded application of Cough-E, together with the necessary scripts to compile and run it.

The `Inc` and `Src` folders contain the header and source files, respectively.
The `kiss_fftr` folder instead contains both header and source files for the computation of the FFT (taken from [here](https://github.com/mborgerding/kissfft)).


#### Settings
The `main.h` file contains the main parameters of the multimodal execution.
In particular, it is possible to modify the period (in seconds) of the post-processing execution (`TIME_DEADLINE_OUTPUT` macro). This parameter specifies how often the model will provide an estimation of the number of coughs.
Moreover, it is possible to specify if to use only one of the two modalities (`RUN_ONLY_AUD` or `RUN_ONLY_IMU`) or if to use the efficient cooperative scheme (`RUN_MIXED`).

More settings related to the features to extract can be found in the `audio_features.h` and `imu_features.h`.


#### Compile and execute
Compilation and execution are triggerred by means of a Makefile through the following commands:

- `make`: to build the application. This will generate a `cough-e` executable file.
- `make run`: to build and launch the execution


#### Input data
The input data are supposed to be stored in three separate header files, for audio, kinematic, and bio signals.
These three files should be included in the `main.h` file.
An example containing 9 windows of data can be found under `./input_data/`




