#!/bin/bash
echo wait for device
adb wait-for-device
adb shell "su -c 'mkdir -p /data/local/NBUnity'"
adb push bin/Debug/NBBehaviour00067.dll /data/local/NBUnity/NBBehaviour00067.dll
adb push NBBehaviour.cfg /data/local/NBUnity/NBBehaviour.cfg
