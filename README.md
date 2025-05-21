# Python scripts
This is a collection of python scripts for eCAL and camera utilities.

### Setup steps for eCAL:
  * Get Anaconda or miniconda: https://www.anaconda.com/products/individual#windows
  * Create Environment (do this once):
      ```shell
        conda create -n eCAL_SKY360 python=3.11.7
      ```
  * Activate Environment (do this when opening new terminal):
      ```shell
        conda activate eCAL_SKY360
      ```

  * This setup is assuming that the release 5.12 of eCAL is installed: https://eclipse-ecal.github.io/ecal/releases/
  * Download the .whl file (eCAL dependencies) that matches both eCAL and python installation.
  * For example, "ecal5-5.12.1-cp311-cp311-win_amd64.whl" for eCAL 5.12.1 and python 3.11.
  * To install the .whl file, copy it in this folder and run (do this once):
      ```shell
        pip install ecal5-****-win_amd64.whl
      ```

  * Install the rest of the dependencies (do this once):
      ```shell
        pip install -r requirements.txt
      ```

  * Clone the repo and create proto bindings (do this once):
      ```shell
        sh proto_files/create_proto_bindings.sh
      ```

  NB: Although the creation of the proto bindings is needed only once, keep in mind that the command needs to be run every time a change occurs in the PROTO files or when checking out a branch that has a new PROTO definition.

  Standalone functions to test the setup can be found in ECAL_functions: [ECAL_functions readme](ECAL_functions/README.md)

### Workflow:

  * Normal Workflow to record from webcam
  * camera_id is the ID used to select the camera using opencv
      ```shell
        start /b python webcam2ecal.py --camera_id 1 --channel_name camera_1
      ```
      ```shell
        start /b python ecal_recorder.py --channel_name camera_1
      ```

  * Press [Esc] on the capture window to disconnect the camera. The recording stops if it has no signal.

  * Play and view the recording
      ```shell
        start /b python view_ecal_video.py --channel_name camera_1
      ```
      ```shell
        python ecal_player.py
      ```

  * For more info on the arguments
      ```shell
        python ecal_player.py -h
      ```

  * ecal_player and ecal_recorder can work with any type of data by setting the correct arguments.
