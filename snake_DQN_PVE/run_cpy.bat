@echo off
:testpps 
adb shell "ps" |findstr -E com.tencent.tmgp.snk
if %errorlevel% == 0 (goto runed ) else (timeout /t 4 && goto testpps)

:runed
timeout /t 4
adb shell "su -c '/data/local/tmp/inject  com.tencent.tmgp.snk /data/local/NBUnity/libloader.so'"
