#!/bin/bash
sh runtime/release2device.sh
sh release2device.sh

adb shell "su -c 'chmod 777 /data/local/NBUnity/*'"
adb shell "su -c 'chmod 777 /data/local/tmp/inject'"
adb shell "su -c 'rm /data/local/NBUnity/notify'"

while true; do 
    adb shell "ps" |dos2unix|egrep 'com.tencent.tmgp.snk$'
    if [ $? -eq 0 ]; then
        break
    fi
    echo "slep 4s"
    sleep 4
done

sleep 4
adb shell "su -c '/data/local/tmp/inject  com.tencent.tmgp.snk /data/local/NBUnity/libloader.so'"
