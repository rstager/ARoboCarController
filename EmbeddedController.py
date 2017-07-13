from unreal_engine import FVector,FTransform,FRotator
from math import sqrt
import numpy as np


def calc_speedlimit(path):
    limits=[]
    npoints=path.component.GetNumberOfSplinePoints()
    print(npoints)
    for n in range(npoints):
        p=path.component.GetDirectionAtSplinePoint(n)
        d0=path.component.GetDistanceAlongSplineAtSplinePoint(n)
        A=100
        if n!= 0:
            speedlimit=sqrt(abs(A*(d1-d0)/FVector.cross(p,p1).z))
            print("n {} d {} sl {} {}".format(n,d0,speedlimit,FVector.cross(p,p1).z))
            limits.append(min(speedlimit,1400))
            print("limits {} = {}".format(n, speedlimit))
        p1=p
        d1=d0

    return limits
speedlimit=None
throttle=0
speed_integral = 0
speed_delta = 0
last_speed_error=0

def drive(state,path,driver,pawn):
    global speedlimit,throttle,speed_integral,last_speed_error,didx,didxmax,diag
    if speedlimit==None:
        speedlimit=calc_speedlimit(path)

    #print("state {} path {} pawn {} {}".format(state["delta_time"],path,driver.location,pawn))

    l0=driver.location
    forward=pawn.get_actor_forward()
    closest=path.component.FindLocationClosestToWorldLocation(l0)
    offpath=(closest-l0)
    pathoffset = offpath.length()

    key=path.component.FindInputKeyClosestToWorldLocation(l0)
    frac=key%1
    key =int(key+1)%len(speedlimit)
    key1 = int(key+1)%len(speedlimit)

    d0=path.component.GetDistanceAlongSplineAtSplinePoint(key)
    d1=path.component.GetDistanceAlongSplineAtSplinePoint(key1)

    d=d0+frac*(d1-d0)
    sd0=path.component.GetDirectionAtSplinePoint(key)
    sd1=path.component.GetDirectionAtSplinePoint(key1)
    sd=sd1*frac+sd0*(1-frac)

    angle=-FVector.cross(forward, sd).z
    adjust=FVector.cross(forward,offpath).z*.001
    adjust = min(0.2,max(-0.2,adjust))
    angle -= adjust
    #a0=-FVector.cross(forward, sd0).z
    #a1=-FVector.cross(forward, sd1).z
    if key<len(speedlimit):
        goal_speed=speedlimit[key]*(1-frac)+speedlimit[key1]*frac
    else:
        goal_speed=300
    speed=driver.speed
    speed_error=goal_speed - speed
    speed_integral = speed_integral + speed_error
    speed_delta = speed_error - last_speed_error
    throttle = speed_error * 0.002 + speed_delta * 0.009 + speed_integral * 0.00002
    print("key {:3.0f} angle {:-1.3f}  throttle {:-0.3f} goal {:4.0f} speed {:4.0f} offpath {:3.0f}".format(key,angle,throttle,goal_speed,speed,pathoffset))
    last_speed_error = speed_error

    state.update({"ec.goal_speed":goal_speed,"ec.speed_delta":speed_delta,"ec.speed_integral":speed_integral})
    return throttle,angle

def reward(state,path,driver,pawn):
    pathdistance, pathoffset = path.closest(driver.location)
    state.update({"pathoffset":pathoffset,"pathdistance":pathdistance})
    ret=driver.speed
    return ret
