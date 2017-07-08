@echo off
call runtime\release2device.bat
call release2device.bat

adb shell "su -c 'chmod 777 /data/local/NBUnity/*'"
adb shell "su -c 'chmod 777 /data/local/tmp/inject'"
adb shell "su -c 'rm /data/local/NBUnity/notify'"
:testpps 
adb shell "ps" |findstr -E com.tencent.tmgp.snk
if %errorlevel% == 0 (goto runed ) else (timeout /t 4 && goto testpps)

:runed
timeout /t 4
adb shell "su -c '/data/local/tmp/inject  com.tencent.tmgp.snk /data/local/NBUnity/libloader.so'"
