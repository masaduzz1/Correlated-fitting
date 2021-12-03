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
import jackknife_utilities
import numerical_utilities
from scipy.stats import chi2
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


def func1(x,A,B,Delta):
    return A*x**(-Delta)+B

def func3(x,A,B,Delta):
    return A*np.exp(-Delta*np.sqrt(x) ) +B


######## main parameters ##########################
def main():
    VOL=sys.argv[1]
    T=sys.argv[2]

    p01=float(sys.argv[3])
    p02=float(sys.argv[4])
    p03=float(sys.argv[5])

    str3="../data/corr.Lbv"+VOL+".T"+T  # i/p file: two column format: ln(\sigma)    ln(P)
    X=np.array(pd.read_csv(str3,index_col=False,header=None,sep='\s+') )
    A=X[:,0]
    rc=X[:,1]

    fit_start=int(sys.argv[6])
    fit_end=int(sys.argv[7])
    dp=int(sys.argv[8])
    polyfit=1
    expfit=0
    r_length=int(int(VOL)/2) +1
    omit_data_from_end=0
    cutoff=r_length-omit_data_from_end
    print("check rlength at config file: rlength=",r_length)
    del_r=A[0:cutoff].astype(np.int32)
    del_r= 1 - np.cos(del_r/float(VOL)*2*np.pi)
    #print(del_rUB)
    #quit()
    config_length=len(rc)//r_length
    print("saved from how many config?", config_length)
    rc=rc.reshape(config_length,r_length)
    rc=rc[:,0:cutoff]
    rc[:,0] *=2
    #print(rc[10,:])

    #rc= rc/(float(VOL)/2)

    length_of_block=200
    bin=len(rc[:,0])//length_of_block
    rc=rc[0:bin*length_of_block,:]

    print("Best bin=%d, block length/data in each bin=%d" %(length_of_block,bin))



    ############### binned data jackknife #############################

    print("dimension good?",del_r.shape,rc.shape)

    binned_data=np.apply_along_axis(jackknife_utilities.binner ,0,rc,length_of_block)
    average_data=np.apply_along_axis(jackknife_utilities.partial_average,0,binned_data)
    #average_data.shape

    ############ downsampling

    dp2=len(del_r)//dp
    del_r_dec=del_r[0:dp2*dp]
    del_r_dec=np.mean( del_r_dec.reshape( (dp2,dp) ), axis=1) # axis 1 sum over column data: rowwise sum: final dim=#of row
    #del_r_dec
    avg_dec=numerical_utilities.decimation4(average_data,dp)
    jkavg=np.apply_along_axis(np.mean,0,avg_dec) # 0 for column-wise
    jkerr=np.apply_along_axis(jackknife_utilities.jack_error,0,avg_dec)

    ## unbinned data jackknife_utilities
    UBavg_data=np.apply_along_axis(jackknife_utilities.partial_average,0,rc) # columnwise operation
    UBavg_dec=numerical_utilities.decimation4(UBavg_data,dp)

    UBjkavg=np.apply_along_axis(np.mean,0,UBavg_dec)
    UBjkerr=np.apply_along_axis(jackknife_utilities.jack_error,0,UBavg_dec)

    ##############Un-jackknife data and inflation factor for correlated fitting
    UBavg_dec_unjacked=np.apply_along_axis(jackknife_utilities.unjack,0,UBavg_dec)
    inflation_factor=jkerr/UBjkerr
    print("inflation factor=",inflation_factor)
    cov_unbinned=np.cov(UBavg_dec,rowvar=0,bias=True)*(len(UBavg_dec)-1)
    print("covariance unbinned shape=",cov_unbinned.shape)
    inflate_matrix=np.diag(inflation_factor)
    print("inflate matrix shape=",inflate_matrix.shape)
    inflated_cov=inflate_matrix @ cov_unbinned @inflate_matrix
    print("inflated cov shape=",inflated_cov.shape)

    ################### fit range information ###########################


    x_min=del_r_dec[fit_start]
    x_max=del_r_dec[fit_end]
    print("fit_min=%f, fit_max=%f" %(x_min,x_max ))

    r_min=float(VOL)/(2*np.pi) *np.arccos(1-x_min)
    r_max=float(VOL)/(2*np.pi) *np.arccos(1-x_max)
    print("fit_min=%f, fit_max=%f" %(r_min,r_max ))
    #######################################################

    pops=np.array([])
    chisqr_arr=np.array([])
    pops3=np.array([])
    chisqr_arr3=np.array([])
    plt.rc('text', usetex=True)

    for i in np.arange(0,config_length,1):  #config_length
        del_one_config=np.delete(UBavg_dec_unjacked,i,axis=0) # delete one config data for all steps/ delete row of rc
        #del_one_config.shape
        UBavg_dec2=np.apply_along_axis(jackknife_utilities.partial_average,0,del_one_config)

        #UBjkavg=np.apply_along_axis(np.mean,0,UBavg_dec)

        cov_unbinned2=np.cov(UBavg_dec2,rowvar=0,bias=True)*(len(UBavg_dec2)-1) ############### CHECK NORMALIZATION IS IT CORRECT????????
        #print("cov unbinned2 shape", cov_unbinned2.shape)
        inflated_cov2=inflate_matrix @ cov_unbinned2 @ inflate_matrix



        xtest=del_r_dec[fit_start:fit_end]
        ytest=UBavg_dec[i,fit_start:fit_end]
        ycovtest=inflated_cov2[fit_start:fit_end,fit_start:fit_end]



        if polyfit:
            pop,poc=curve_fit(func1,xtest,ytest,sigma=ycovtest, absolute_sigma=True,p0=(p01,p02,p03))

            mychisqr=np.dot( (ytest-func1(xtest,*pop)).T,  np.dot( np.linalg.inv(ycovtest), (ytest-func1(xtest,*pop) ) ) )
            mychisqr /= (len(xtest)-3)
            chisqr_arr=np.append(chisqr_arr,mychisqr)
            pops=np.append(pops,pop)
            if i%1000==0:
                print(i,pop[0],pop[1],pop[2],mychisqr)
            #print(i)


        if expfit:
            pop3,poc3=curve_fit(func3,xtest,ytest,sigma=ycovtest, absolute_sigma=True,p0=(3,190,p03))

            mychisqr3=np.dot( (ytest-func3(xtest,*pop3)).T,  np.dot( np.linalg.inv(ycovtest), (ytest-func3(xtest,*pop3) ) ) )
            mychisqr3 /= (len(xtest)-3)
            chisqr_arr3=np.append(chisqr_arr3,mychisqr3)
            pops3=np.append(pops3,pop3)
            #print(i,pop3[0],pop3[1],pop3[2],mychisqr3)





    ######### polynomial fitting final info ###########################

    if polyfit:
        popsR=pops.reshape(len(pops)//3,3)

        ## write individual fit info of the loop of correlated fit
        df3=pd.DataFrame(popsR)
        df3.to_csv("indiv_polyfit_T"+T,header=False,index=False,sep='\t')



        ###final fit value computation
        final_pops=np.mean(popsR,axis=0) # 0 for columnwise mean
        print("fits poly",final_pops)
        final_pops_err=np.apply_along_axis(jackknife_utilities.jack_error,0,popsR) # you need jackknife error as the
                                           # data here coming from jackknife
        print("errors", final_pops_err)
        chisqr_final=np.mean(chisqr_arr)
        print("chisqr", chisqr_final)

        ## pvalue
        fit_param_total=3
        dof=fit_end-fit_start-fit_param_total
        chisqr_final*dof
        pval1=1-chi2.cdf(chisqr_final*dof,dof)
        pval2=chi2.sf(chisqr_final*dof,dof,loc=0, scale=1)

        print("pval",pval1,pval2)

        #### plot distributions of fit parameter
        plt.hist(popsR[:,2],bins=40)
        plt.xlabel(r'$\Delta$',fontsize=16)
        plt.savefig("poly.T"+T+".delta_hist.png",bbox_inches='tight')
        #plt.show()
        plt.close()
        # plot chisqr distributions
        plt.hist(chisqr_arr,bins=40)
        plt.xlabel(r'$\chi^2$',fontsize=16)
        plt.savefig("poly.T"+T+".chisqr_hist.png",bbox_inches='tight')
        #plt.show()
        plt.close()

        df=pd.DataFrame([[float(T),*final_pops,*final_pops_err,chisqr_final,x_min,x_max,int(r_min),int(r_max),pval1  ]])
        df.to_csv("fit_info.vol"+VOL,header=False,index=False,sep='\t',mode='a')
        df2=pd.DataFrame([[float(T),final_pops[2],final_pops_err[2],chisqr_final,int(r_min),int(r_max),fit_start,fit_end,dp,pval1]])
        df2.to_csv("fit.T"+T,header=False,index=False,sep='\t',mode='a')


    ###############################################################################
    if expfit:
        pops3R=pops3.reshape(len(pops3)//3,3)

        df4=pd.DataFrame(pops3R)
        df4.to_csv("indiv_expfit_T"+T,header=False,index=False,sep='\t')



        final_pops3=np.mean(pops3R,axis=0) # 0 for columnwise mean
        print("fits exp",final_pops3)

        final_pops3_err=np.apply_along_axis(jackknife_utilities.jack_error,0,pops3R) # you need jackknife error as the
                                           # data here coming from jackknife
        print("errors", final_pops3_err)
        chisqr3_final=np.mean(chisqr_arr3)
        print("chisqr", chisqr3_final)
        plt.hist(pops3R[:,2],bins=20)
        plt.xlabel(r'$m$',fontsize=16)
        plt.savefig("exp.T"+T+".m_hist.png",bbox_inches='tight')
        #plt.show()
        plt.close()

        plt.hist(chisqr_arr3,bins=20)
        plt.xlabel(r'$\chi^2$',fontsize=16)
        plt.savefig("exp.T"+T+".chisqr_hist.png",bbox_inches='tight')
        #plt.show()
        plt.close()
        df=pd.DataFrame([[float(T),*final_pops3,*final_pops3_err,chisqr3_final,x_min,x_max,int(r_min),int(r_max)]])
        df.to_csv("fit_info.vol"+VOL,header=False,index=False,sep='\t',mode='a')
    #######################################################################





    #### plots

    ##fit-1: only fitted part regular scale
    #rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rc('text',usetex=True)
    fig, ax = plt.subplots(facecolor="w")
    ax.tick_params(axis='both', which='major', labelsize=25)

    ax.errorbar(del_r_dec[fit_start:fit_end],jkavg[fit_start:fit_end],yerr=jkerr[fit_start:fit_end],marker='x',c='b',ls='none',capsize=5)
    if polyfit:
        ax.plot(del_r_dec[fit_start:fit_end],func1(del_r_dec[fit_start:fit_end],*final_pops),'k',label='poly')
    if expfit:
        ax.plot(del_r_dec[fit_start:fit_end],func3(del_r_dec[fit_start:fit_end],*final_pops3),'r',label='exp')
    #plt.legend(loc='best')
    plt.xlabel(r'$(1-\cos(\theta) ) $', fontsize=30)
    plt.ticklabel_format(useOffset=False)
    #plt.ylabel(r'$ \log C(\theta)$', fontsize=16)
    plt.annotate(r'$T$=%s' %(T), xy=(0.4,0.9),xycoords='axes fraction',fontsize=30 )
    plt.annotate(r'$\Delta$=%0.2f $\pm$ %0.2f' %(final_pops[2],final_pops_err[2]),xy=(0.4,0.8),xycoords='axes fraction',fontsize=25)
    plt.annotate(r'$\chi^2/\mathrm{dof}$=%0.2f, $p$=%0.2f' %(chisqr_final,pval2),xy=(0.3,0.7),xycoords='axes fraction',fontsize=25)
    plt.ylabel(r'$ C(\theta)$', fontsize=30)
    #plt.xlim([0.2,1])
    #plt.ylim([0.05,0.08])
    plt.savefig("fit1_regularscale.T."+T+".png",bbox_inches='tight')
    #plt.savefig("fit1_regularscale.T."+T+".pdf",bbox_inches='tight')
    plt.show()
    plt.close()

    ##fit-2: only fitted part log-scale
    fig = plt.figure(facecolor="w")
    plt.errorbar(del_r_dec[fit_start:fit_end],jkavg[fit_start:fit_end],yerr=jkerr[fit_start:fit_end],marker='x',c='b',ls='none',capsize=5)
    if polyfit:
        plt.plot(del_r_dec[fit_start:fit_end],func1(del_r_dec[fit_start:fit_end],*final_pops),'k',label='poly')
    if expfit:
        plt.plot(del_r_dec[fit_start:fit_end],func3(del_r_dec[fit_start:fit_end],*final_pops3),'r',label='exp')
    plt.legend(loc='best')
    plt.xlabel(r'$(1-\cos(\theta) ) $', fontsize=30)
    plt.ticklabel_format(useOffset=False)
    #plt.ylabel(r'$ \log C(\theta)$', fontsize=16)
    plt.annotate(r'$\mathrm{T}$=%s' %(T), xy=(0.4,0.9),xycoords='axes fraction',fontsize=30 )
    plt.ylabel(r'$ C(\theta)$', fontsize=30)
    plt.xscale('log')
    plt.yscale('log')
    #plt.xlim([0.2,1])
    #plt.ylim([0.05,0.08])
    plt.savefig("fit2_logscale.T."+T+".png",bbox_inches='tight')
    #plt.savefig("fit2_logscale.T."+T+".pdf",bbox_inches='tight')
    #plt.show()
    plt.close()

    #fit3: complete range--regular scale

    plt.errorbar(del_r_dec,jkavg,yerr=jkerr,marker='x',c='b',ls='none',capsize=5)
    if polyfit:
        plt.plot(del_r_dec,func1(del_r_dec,*final_pops),'k',label='poly')
    if expfit:
        plt.plot(del_r_dec,func3(del_r_dec,*final_pops3),'r',label='exp')
    plt.legend(loc='best')
    plt.xlabel(r'$(1-\cos(\theta) ) $', fontsize=30)
    plt.ticklabel_format(useOffset=False)
    #plt.ylabel(r'$ \log C(\theta)$', fontsize=16)
    plt.annotate(r'$\mathrm{T}$=%s' %(T), xy=(0.4,0.9),xycoords='axes fraction',fontsize=30 )
    plt.ylabel(r'$ C(\theta)$', fontsize=30)
    #plt.xscale('log')
    #plt.yscale('log')
    plt.savefig("fit3_regularscale.T."+T+".png",bbox_inches='tight')
    #plt.savefig("fit3_regularscale.T."+T+".pdf",bbox_inches='tight')
    #plt.show()
    plt.close()

    #fit 4: complete range---log scale

    plt.errorbar(del_r_dec,jkavg,yerr=jkerr,marker='x',c='b',ls='none',capsize=5)
    if polyfit:
        plt.plot(del_r_dec,func1(del_r_dec,*final_pops),'k',label='poly')
    if expfit:
        plt.plot(del_r_dec,func3(del_r_dec,*final_pops3),'r',label='exp')
    plt.legend(loc='best')
    plt.xlabel(r'$(1-\cos(\theta) ) $', fontsize=30)
    plt.ticklabel_format(useOffset=False)
    #plt.ylabel(r'$ \log C(\theta)$', fontsize=16)
    plt.annotate(r'$\mathrm{T}$=%s' %(T), xy=(0.4,0.9),xycoords='axes fraction',fontsize=30 )
    plt.ylabel(r'$ C(\theta)$', fontsize=30)
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig("fit4_logscale.T."+T+".png",bbox_inches='tight')
    plt.savefig("fit4_logscale.T."+T+".pdf",bbox_inches='tight')
    #plt.show()
    plt.close()

    #fit5: complete range with fit only to the fitted part

    fig, ax = plt.subplots(facecolor="w")
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.minorticks_on()
    ax.errorbar(del_r_dec[1:len(del_r_dec)],jkavg[1:len(del_r_dec)],yerr=jkerr[1:len(del_r_dec)],marker='x',c='b',ls='none',capsize=5)
    if polyfit:
        ax.plot(del_r_dec[fit_start:fit_end],func1(del_r_dec[fit_start:fit_end],*final_pops),'k',lw=2)
    if expfit:
        ax.plot(del_r_dec[fit_start:fit_end],func3(del_r_dec[fit_start:fit_end],*final_pops3),'r',lw=2,label='exp')
    #plt.legend(loc='best')
    plt.xlabel(r'$1-\cos(\theta)$', fontsize=30)
    plt.ticklabel_format(useOffset=False)
    plt.locator_params(axis="x", nbins=5)
    #plt.ylabel(r'$ \log C(\theta)$', fontsize=16)
    plt.annotate(r'$T$=%s' %(T), xy=(0.4,0.9),xycoords='axes fraction',fontsize=30 )
    plt.annotate(r'$\Delta$=%0.2f $\pm$ %0.2f' %(final_pops[2],final_pops_err[2]),xy=(0.4,0.8),xycoords='axes fraction',fontsize=25)
    plt.annotate(r'$\chi^2/\mathrm{dof}$=%0.2f, $p$=%0.2f' %(chisqr_final,pval2),xy=(0.3,0.7),xycoords='axes fraction',fontsize=25)
    plt.ylabel(r'$ C(\theta)$', fontsize=30)
    #plt.xscale('log')
    #plt.yscale('log')
    plt.savefig("fit5_regular.T."+T+".pdf",bbox_inches='tight')
    #plt.savefig("fit5_regular.T."+T+".pdf",bbox_inches='tight')
    plt.show()
    plt.close()


    #fit6
    fig = plt.figure(facecolor="w")
    plt.errorbar(del_r_dec,jkavg,yerr=jkerr,marker='x',c='b',ls='none',capsize=5)
    if polyfit:
        plt.plot(del_r_dec[fit_start:fit_end],func1(del_r_dec[fit_start:fit_end],*final_pops),'k',lw=2,label='poly')
    if expfit:
        plt.plot(del_r_dec[fit_start:fit_end],func3(del_r_dec[fit_start:fit_end],*final_pops3),'r',lw=2,label='exp')
    plt.legend(loc='best')
    plt.xlabel(r'$(1-\cos(\theta) ) $', fontsize=30)
    plt.ticklabel_format(useOffset=False)
    #plt.ylabel(r'$ \log C(\theta)$', fontsize=16)
    plt.annotate(r'$\mathrm{T}$=%s' %(T), xy=(0.4,0.9),xycoords='axes fraction',fontsize=30 )
    plt.ylabel(r'$ C(\theta)$', fontsize=30)
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig("fit6_logscale.T."+T+".png",bbox_inches='tight')
    #plt.savefig("fit6_logscale.T."+T+".pdf",bbox_inches='tight')
    #plt.show()
    plt.close()

if __name__ == "__main__":
    main()
