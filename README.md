# ARoboCarController
Controller for ARoboCar simulator
## Setup notes
Please set OS environment variable called DATA_DIR to a directory where the controller will write the training and model hdf5 files. It needs to be fully qualified and not a relative path as h5py isn't able to deal with relative paths on Mac. 

## Running it on Windows
- Install Unreal engine if you haven't.
- Install Cygwin (Cygwin.com) - default installation worked for me. We need this for some code to work.
- Install Visual Studio 2017 if you haven't. (This is the environment I tried with and you need it to rebuild the project)
- Clone git repos - ARoboCar and ARoboCarController (you will need git) - https://github.com/eastbayML/
- Open ARoboCar directory in file explorer, choose ARoboCar.uproject and open with a text editor. Under plugins, set UnrealEnginePython plugin to false. Now, select ARoboCar.uproject, click right mouse button and you should see the option to generate Visual Studio Project files.
- Open the Visual studio solution file and build the solution. It also shows the path to python interactive environment. You should note that in case you have to install hdpy.
- Install UnrealEnginePython from https://github.com/20tab/UnrealEnginePython/. Just place the directory under Plugins directory of your project. 
- Now open Epic Games, launch the Unreal Engine and open the project just to make sure you can. Re-enable UnrealEnginePython plugin and quit the engine.
- Now go to a Cygwin terminal and set the required python environment e.g. "source activate root"
- Open the project from commandline.  
  "$PATH_TO_ENGINE/UE4Editor.exe path_to_uproject_file_in_windows_convention"  
  Unreal engine doesn't seem to understand cygwin version of path so just use windows path
- Next set your DATA_DIR environment variable, open your favorite python editor from commandline (in the cygwin environment). This is important for the communication to work between the two processes.

