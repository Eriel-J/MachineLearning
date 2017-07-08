#!/bin/bash
echo "wait for device"
adb wait-for-device
adb shell "su -c 'mkdir /data/local/NBUnity'"
adb shell "su -c 'chmod 777 /data/local/NBUnity'"

adb push runtime/inject /data/local/tmp/inject
adb push runtime/libloader.so /data/local/NBUnity/libloader.so
adb push runtime/nbloader.apk /data/local/NBUnity/nbloader.apk
