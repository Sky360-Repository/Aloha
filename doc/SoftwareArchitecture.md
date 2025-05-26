# Software Architecture


## Folder structure

```
+---doc
+---proto_files
¦   +---ADSB.proto
+---scripts
+---src
¦   +---contrib
¦   ¦   +---QHY183c_sdk
¦   ¦   +---Bobtracker
¦   +---core
¦   ¦   +---background_sub
¦   ¦   +---camera_calibration
¦   ¦   +---ecal
¦   ¦   +---img_processing
¦   ¦   +---ros
¦   ¦   +---utils
¦   ¦   +---Camera controller 
¦   ¦   ¦   +---QHY_controller
¦   ¦   ¦   +---Opecv_controller
¦   ¦   ¦   +---PTF_controller
¦   ¦   +---ADSB
¦   +---platform_opi5
¦   ¦   +---nodes
¦   ¦   +---CameraNode
¦   ¦   +---ADSBNode
¦   +---platform_pc
¦   ¦   +---nodes
¦   +---platform_web
+---tests
+---tools
    +---camera_tools
    +---Multicast_Debug_Tools
```