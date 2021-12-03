#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d
import pylab
import sys
import csv
import pandas as pd
import random

def compute_chisqr(ydata,yfit,yerr,dof):
    my_chisq=0
    for i in range(len(ydata)):
         my_chisq += ( ( (ydata[i]-yfit[i])/yerr[i] ) **2 )

    my_chisq /=dof
    return my_chisq

def decimation(del_r,Lcorr,CErr,dp):
    rdec=np.array([])
    Ldec=np.array([])
    Edec=np.array([])

    i=0
    decimation_precision=dp
    while(i<len(del_r)):
        x=del_r[i]
        y=Lcorr[i]
        #print(i)
        for j in np.arange(1,1000,1):
            if i+j < len(del_r):
                if ( (del_r[i+j]-del_r[i]) < decimation_precision):
                    x += del_r[i+j]
                    y += Lcorr[i+j]
                    #print(i,j,x,y)
                else:
                    break
            else:
                break

        #print(j)
        x /=(j)
        y /=(j)
        z = np.sqrt( np.max(CErr[i:i+j] )**2+np.std(Lcorr[i:i+j])**2 )
        i += j
        rdec=np.append(rdec,x)
        Ldec=np.append(Ldec,y)
        Edec=np.append(Edec,z)


    return rdec,Ldec,Edec

###################################

def decimation2(del_r,Lcorr,dp):
    rdec=np.array([])
    Ldec=np.array([])
    Edec=np.array([])

    i=0
    decimation_precision=dp
    while(i<len(del_r)):
        x=del_r[i]
        y=Lcorr[i]
        #print(i)
        for j in np.arange(1,1000,1):
            if i+j < len(del_r):
                if ( (del_r[i+j]-del_r[i]) < decimation_precision):
                    x += del_r[i+j]
                    y += Lcorr[i+j]
                    #print(i,j,x,y)
                else:
                    break
            else:
                break

        #print(j)
        x /=(j)
        y /=(j)
        z = np.std(Lcorr[i:i+j])
        i += j
        rdec=np.append(rdec,x)
        Ldec=np.append(Ldec,y)
        Edec=np.append(Edec,z)


    return rdec,Ldec,Edec

##########################################################
def decimation3(dec_ind,average_data):
    avg_dec=np.array([])
    xbinerr=np.array([])


    for i in range(len(dec_ind)-2):
        ind1=int(dec_ind[i])
        ind2=int(dec_ind[i+1])

        avg_dec=np.append(avg_dec,np.mean(average_data[:,ind1:ind2],axis=1)  ) #axis 1 for rowwise mean
        xbinerr=np.append(xbinerr,np.std( average_data[:,ind1:ind2],axis=1) )


    avg_dec=avg_dec.reshape(len(dec_ind)-1,len(average_data)).T


    xbinerr=xbinerr.reshape(len(dec_ind)-1,len(average_data)).T

    xbinerr=np.max(xbinerr,axis=0) # 0 for columnwise
    xbinerr.shape
    return avg_dec,xbinerr


##############################################################

def decimation3(dec_ind,average_data):
    avg_dec=np.array([])
    xbinerr=np.array([])


    for i in range(len(dec_ind)-1):
        ind1=int(dec_ind[i])
        ind2=int(dec_ind[i+1])

        avg_dec=np.append(avg_dec,np.mean(average_data[:,ind1:ind2],axis=1)  ) #axis 1 for rowwise mean
        xbinerr=np.append(xbinerr,np.std( average_data[:,ind1:ind2],axis=1) )


    avg_dec=avg_dec.reshape(len(dec_ind)-1,len(average_data)).T


    xbinerr=xbinerr.reshape(len(dec_ind)-1,len(average_data)).T

    xbinerr=np.max(xbinerr,axis=0) # 0 for columnwise
    xbinerr.shape
    return avg_dec,xbinerr

####################################


def decimation4(x,B):
    C=len(x)
    L=len(x[0])
    Lnew=L//B
    #print(B,C,L,Lnew,B*Lnew)

    x=x[:,0:B*Lnew]
    #print(x)
    x=(x.reshape(C*B*Lnew)).reshape(C,Lnew,B)
    #print(x)
    np.sum(x,axis=2)
    xfinal=np.mean(x,axis=2)
    #y=np.std(x,axis=2)*np.sqrt(B-1)

    #print(xfinal.shape,y.shape)
    return xfinal

def decimation5(x,B):
    C=len(x)
    L=len(x[0])
    Lnew=L//B
    #print(B,C,L,Lnew,B*Lnew)

    x=x[:,0:B*Lnew]
    #print(x)
    x=(x.reshape(C*B*Lnew)).reshape(C,Lnew,B)
    #print(x)
    np.sum(x,axis=2)
    x=np.std(x,axis=2)*np.sqrt(B-1)
    #print(x)
    return x

def numerical_derivative(y,x,yerr):
    dery=np.zeros(len(y))
    L=len(y)
    der_err=np.zeros(L)

    dery[0]=( y[1]-y[0] )/ (x[1] -x[0] )
    dery[L-1]=(y[L-1]-y[L-2])/ (x[L-1] -x[L-2] )

    der_err[0]=( yerr[0]+yerr[1] )/ (x[1] -x[0] )
    der_err[L-1]=(yerr[L-1]+yerr[L-2] )/ (x[L-1] -x[L-2] )

    for i in range(1,L-1):
        dery[i]=(y[i+1]-y[i-1])/ (x[i+1] -x[i-1] )
        der_err[i]=(yerr[i-1]+yerr[i+1]) /(x[i+1]-x[i-1])


    return dery,der_err

###########################################################


###################################

def qintegrand(y,dof):
    return y**(dof/2 -1 ) *np.exp(-y)

def compute_Q(chisqr,dof):
    x=0
    qint,qinterr=quad(qintegrand,chisqr/2,np.inf,args=(dof) )
    x=1/gamma(dof/2) *qint
    return x
