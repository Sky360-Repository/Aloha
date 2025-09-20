# Software Architecture

## Folder structure

```
+---doc
+---scripts
+---src
¦   +---contrib
¦   ¦   +---ecal_whl
¦   ¦   ¦   +---ecal5-5.13.3-cp312-cp312-linux_aarch64.whl
¦   ¦   +---QHY183c_sdk
¦   ¦   ¦   +---libqhyccd.so
¦   ¦   +---Bobtracker
¦   +---background_sub
¦   +---camera_calibration
¦   +---ecal
¦   ¦   +---proto_files
¦   ¦   ¦   +---ADSB.proto
¦   ¦   +---ecal
¦   +---img_processing
¦   +---ros
¦   +---utils
¦   +---Camera controller
¦   ¦   +---QHY_controller
¦   ¦   +---Opecv_controller
¦   ¦   +---PTF_controller
¦   +---ADSB
+---platform_opi5
¦   +---nodes
¦   +---CameraNode
¦   +---ADSBNode
+---platform_pc
¦   +---nodes
+---platform_web
+---tests
+---tools
    +---camera_tools
    +---Multicast_Debug_Tools
```

## Implementation of Object Detection

Please follow this documentation
- [Object Detection Architecture](ObjectDetectionArchitecture.md)
- [ViBe](ViBe.md)
- [Gaussian Mixture of Models](GMM.md)
