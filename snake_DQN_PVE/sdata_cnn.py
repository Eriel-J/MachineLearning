#-*- coding:utf-8 -* 
import socket
import json
import struct
import os
import array
import time
import sys
Protocal = {
            "GET_CURENERGY":1,
            "GET_ISDEAD":2,
            "MOVE":3,
            "RESTART":4,
            "GET_MAPINFO":5,
            "GET_ALLSNAKEINFO":6,
            "GET_ALLDEADFOODS":7,
            "GET_ALLACCFOODS":8,
            "GET_ALLNOTMOVABLEFOODS":9,
            "GET_NEARBYFOODS":10,
            "GET_ALLMOVABLEFOODS":11,
            "ACC":12,
            "BACK":13,
            "SINGLE":14,
            "ALL":15
            }
#FOOD type 
#Basic,     0
#Growing,   1
#Movable,   2
#AfterAcc,  3
#AfterDead  4
def inject():
    os.system("inject.bat")
    time.sleep(35)
    
class SNK(object):
    s = None
    
    def RecvAll(self):
        ret = ""
        bytes_num=1024
        recvlen = struct.unpack('i',self.s.recv(4))[0]
        while(len(ret) < recvlen):
            ret =ret + bytes.decode(self.s.recv(bytes_num))
        return ret
    def __init__(self):
        os.system("adb forward tcp:2989 tcp:2989")
        self.s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.s.connect(('127.0.0.1',2989))
    def ToJson(self,js):
        ret = None
        try:
            ret = json.loads(js)
        except Exception:
            ret = {'result':'recv error'}
        return ret
    def Align(slef,buf,toNum):
        return array.array('b',buf+str.encode('\x00'*(toNum-len(buf))))
    def GetCurrentEnergy(self):
        buf = struct.pack('i',Protocal['GET_CURENERGY'])
        self.s.send(self.Align(buf,50))
        return self.ToJson(self.RecvAll()) 
    
    def GetIsDead(self):
        buf = struct.pack('i',Protocal['GET_ISDEAD'])
        self.s.send(self.Align(buf,50))
        return self.ToJson(self.RecvAll()) 

 
    def Move(self,x,y):
        buf = struct.pack('iff',Protocal['MOVE'],x,y)
        # print (self.Align(buf,50))
        self.s.send(self.Align(buf,50))
        return self.ToJson(self.RecvAll()) 
		
    def Restart(self):
        buf = struct.pack('i',Protocal['RESTART'])
        # print (self.Align(buf,50))
        self.s.send(self.Align(buf,50))
        return self.ToJson(self.RecvAll()) 
        
    def AllSnakeInfo(self):
        buf = struct.pack('i',Protocal['GET_ALLSNAKEINFO'])

        self.s.send(self.Align(buf,50))
        return self.ToJson(self.RecvAll())
    def GetMapInfo(self):
        buf = struct.pack('i',Protocal['GET_MAPINFO'])
        self.s.send(self.Align(buf,50))
        return self.ToJson(self.RecvAll())    
    def GetAllDeadFoods(self):
        buf = struct.pack('i',Protocal['GET_ALLDEADFOODS'])
        self.s.send(self.Align(buf,50))
        return self.ToJson(self.RecvAll()) 
    def GetAccFoods(self):
        buf = struct.pack('i',Protocal['GET_ALLACCFOODS'])
        self.s.send(self.Align(buf,50))
        return self.ToJson(self.RecvAll()) 
    def GetAllMovableFoods(self):
        buf = struct.pack('i',Protocal['GET_ALLMOVABLEFOODS'])
        self.s.send(self.Align(buf,50))
        return self.ToJson(self.RecvAll()) 
    def GetNearByFoods(self,x,y,r):
        buf = struct.pack('ifff',Protocal['GET_NEARBYFOODS'],x,y,r)
        self.s.send(self.Align(buf,50))
        return self.ToJson(self.RecvAll()) 
    def GetAllNotMovableFoods(self):
        buf = struct.pack('i',Protocal['GET_ALLNOTMOVABLEFOODS'])
        self.s.send(self.Align(buf,50))
        return self.ToJson(self.RecvAll())
    def Acc(self,acc):
        buf = struct.pack('ii',Protocal['ACC'],acc)
        self.s.send(self.Align(buf,50))
        return self.ToJson(self.RecvAll())
    def Back(self):
        buf = struct.pack('i', Protocal['BACK'])
        self.s.send(self.Align(buf, 50))
        return self.ToJson(self.RecvAll())
    def SinglePlay(self):
        buf = struct.pack('i', Protocal['SINGLE'])
        self.s.send(self.Align(buf, 50))
        return self.ToJson(self.RecvAll())
    def GetAll(self, r):
        buf = struct.pack('if', Protocal['ALL'], r)
        self.s.send(self.Align(buf, 50))
        return self.RecvAll()
    def Close(self):
        self.s.close()

def test():
    mysnk = SNK()
    #print 'GetCurrentEnergy'
    #print mysnk.GetCurrentEnergy()['result'],'\n'
    #print 'GetIsDead'
    #print mysnk.GetIsDead()['result'],'\n'
    #mysnk.Move(0.5,0.5)

    while(True):
        switch = input("1\tGetCurrentEnergy\n2\tGetIsDead\n3\tMove\n4\tRestart\n5\tAllSnakeInfo\n6\tGetMapInfo\n7\tGetAllDeadFoods\n8\tGetAccFoods\n9\tGetAllMovableFoods\n10\tGetAllNotMovableFoods\n11\tGetNearByFoods\n12\tACC\n"
                       "13\tBack\n14\tSinglePlay\n15\tGetAll\n")
        if switch == "1":
            print (mysnk.GetCurrentEnergy()['result'])
        if switch == "2":
            print (mysnk.GetIsDead()['result'])
        if switch == '3':
            x = input("x:")
            y = input("y:")
            print (mysnk.Move(float(x),float(y))['result'])
        if switch == '4':
            print (mysnk.Restart()['result'])
        if switch == '5':
            print (mysnk.AllSnakeInfo()['result'])
        if switch == '6':
            print (mysnk.GetMapInfo()['result'])
        if switch == '7':
            print (mysnk.GetAllDeadFoods()['result'])
        if switch == '8':
            print (mysnk.GetAccFoods()['result'])
        if switch == '9':
            print (mysnk.GetAllMovableFoods()['result'])
        if switch == '10':
            print (mysnk.GetAllNotMovableFoods()['result'])
        if switch == '11':
            x = input("x:")
            y = input("y:")
            r = input("r:")
            print (mysnk.GetNearByFoods(float(x),float(y),float(r))['result'])
        if switch == '12':
            acc = int(input("acc:"))
            print (mysnk.Acc(acc)['result'])
        if switch == '13':
            print(mysnk.Back()['result']['status'])
        if switch == '14':
            print(mysnk.SinglePlay()['result']['status'])
        if switch == '15':
            print(mysnk.GetAll(1000))
if __name__ == '__main__':
    test()