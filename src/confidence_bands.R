# install.packages("matlab")

# install.packages("locpol")
# install.packages("KernSmooth")
library("matlab")
library("locpol")
library("KernSmooth")
library("data.table")
library("plyr")

# set working directory for data loading
# setwd("/Users/...")


#######################################################################
########################### SUBROUTINES ###############################
#######################################################################

SPDlp2 = function(RawData, xGrid, locband, metric){
 
  p            = 2 # polynomial order
  Scorrected   = as.matrix(RawData[,1]*exp(-RawData[,3]*RawData[,4]))
  Moneyness    = Scorrected/RawData[,2]
  Maturity     = RawData[,4]
  ivola        = RawData[,5]
 
################################################################################################
# The following case corresponds to: 
# RawData entails data of one smile!
  if (metric==0){
    xGridTemp   = as.matrix(xGrid)#as.matrix(Scorrected[1]/xGrid)
  }else{
    if (metric==1){
      xGridTemp = as.matrix(Scorrected[1]/(xGrid*RawData[1,1]))
    }
  }
  hGrid         = c(seq(0.02,by=0.05,length=10),seq(0.5,by=0.2,length=15))
          
# In the next lines, we first estimate the smile with an automatic selection of the local bandwidths.
# Then, we reuse the same bandwiths to estimate the first and second derivative of the smile!
# In a last step, we use these estimations to compute the spd using the result of Rookley (see spdbl.xpl)
  MoneynessSTD  = (Moneyness-matrix(mean(Moneyness),nrow(Moneyness),ncol(Moneyness)))/matrix(sqrt(var(Moneyness)),nrow(Moneyness),ncol(Moneyness))
  xGridTempSTD  = (xGridTemp[,1]-mean(Moneyness))/sqrt(var(Moneyness))
  dataf         = data.frame(tmp1=MoneynessSTD,tmp2=ivola) 
  smile         = locpoly(x=MoneynessSTD,y=ivola,gridsize=length(xGridTempSTD),range.x=c(min(xGridTempSTD),max(xGridTempSTD)),bandwidth=locband)$y
         
  FirstDerSmile = locpoly(x=MoneynessSTD,y=ivola,gridsize=length(xGridTempSTD),range.x=c(min(xGridTempSTD),max(xGridTempSTD)),bandwidth=locband,drv=1)$y/c(sqrt(var(Moneyness)))
          
  SecondDerSmile= locpoly(x=MoneynessSTD,y=ivola,gridsize=length(xGridTempSTD),range.x=c(min(xGridTempSTD),max(xGridTempSTD)),bandwidth=locband,drv=2)$y/c(var(Moneyness))

  # The result is scaled with factor 10000        
  result        = spdbl(flipud(xGridTemp),smile,FirstDerSmile,SecondDerSmile,Scorrected[1,1],RawData[1,3],RawData[1,4])
  
  if (metric==0){
    fstar=result$fstar
  }else{
    fstar=RawData[1,1]*result$fstar
  }
  band=locband
return(list(fstar=fstar,band=band))
}



spdbl = function(m, sigma, sigma1, sigma2, s, r, tau){ # (fstar,delta, gamma)
#       spdbl uses the Breeden and Litzenberger (1978) method and a
#		semiparametric specification of the Black-Scholes
#		option pricing function to calculate the empirical State 
#		Price Density. The analytic formula uses an estimate of 
#	 	the volatility smile and its first and second derivative to
#		calculate the State-price density, as well as Delta and 
#		Gamma of the option. This method can only be applied to
#		European options (due to the assumptions).   


  rm   = length(m)
  ones = matrix(1,rm,1)
  st   = sqrt(tau)
  ert  = exp(r*tau)
  rt   = r*tau
  
#Modified Black-Scholes scaled by S-div instead of F
  d1   = (log(m)+tau*(r+0.5*(sigma^2)))/(sigma*st)
  
  d2   = d1-sigma*st
 

  f    = pnorm(d1)-pnorm(d2)/(ert*m)
 

#first derivative of d1 term
  d11  = (1/(m*sigma*st))-(1/(st*(sigma^2)))*((log(m)+tau*r)*sigma1)+0.5*st*sigma1


#first derivative of d2 term
  d21  = d11-st*sigma1


#second derivative of d1 term
  d12  = -(1/(st*(m^2)*sigma))-sigma1/(st*m*(sigma^2))+sigma2*(0.5*st-(log(m)+rt)/(st*(sigma^2)))+sigma1*(2*sigma1*(log(m)+rt)/(st*sigma^3)-1/(st*m*sigma^2))



#second derivative of d2 term
  d22  = d12-st*sigma2


#Please refer to either Rookley (1997) or the XploRe Finance Guide for derivations
  f1   = dnorm(d1)*d11+(1/ert)*((-dnorm(d2)*d21)/m+pnorm(d2)/(m^2))

  f2   = dnorm(d1)*d12-d1*dnorm(d1)*(d11^2)-(1/(ert*m)*dnorm(d2)*d22)+ ((dnorm(d2)*d21)/(ert*m^2))+(1/(ert*m)*d2*dnorm(d2)*(d21^2))-(2*pnorm(d2)/(ert*(m^3)))+(1/(ert*(m^2))*dnorm(d2)*d21) 


#recover strike price
  x    = s/m  
  	
  c1   = -(m^2)*f1

  c2   = s*((1/x^2)*((m^2)*f2+2*m*f1))

#calculate the quantities of interest
  cdf  = ert*c1+1

  fstar= ert*c2

  delta= f + s* f1/x

  gamma= 2*f1/x+s*f2/(x^2)

return(list(fstar=fstar,delta=delta, gamma=gamma))
}


bootstrap = function(x,nb,opt){
  n = length(x)
  if(opt=="wild"){
    sq5 = sqrt(5)
    aa  = (1-sq5)/2  		# define golden section bootstrap
    bb  = aa+sq5
    cc  = (5+sq5)/10  
    mult= matrix(runif(n*nb),n,nb)
    mult= aa*(mult<cc) + bb*(mult>=cc) 
    bx  = (mult * c(x)) 
    
   }
   return(bx)
}



spdci = function(RawData,coef,xGrid,metric,locband){
  
  IR          = RawData[,3]                               # interest rate
  mat         = RawData[,4]                               # time to maturity
  
  #( S - exp(-i * maturity) )/ k
  moneyness   = (RawData[,1]*exp(-IR*mat))/RawData[,2]    # moneyness

  IVpoints    = cbind(moneyness,RawData[,4],RawData[,5])  # implied volatility
#IVpoints=sort(IVpoints)

  bn          = 100        # number of elements selected for bootstrap

  IVpointsSTD = cbind((IVpoints[,1]-mean(IVpoints[,1]))/sqrt(var(IVpoints[,1])),(IVpoints[,2]-mean(IVpoints[,2]))/sqrt(var(IVpoints[,2]))) # standardized IV points
  SPD1stEstimation = SPDlp2(RawData,xGrid,locband,metric)  
  IVs          = data.frame(IvpSTD=IVpointsSTD[,1],Ivp=IVpoints[,3])     
  IVsurf       = locpol(Ivp~IvpSTD,data=IVs,bw=locband,xevalLen=length(IVpointsSTD[,1]))
  IVsurfg      = locpol(Ivp~IvpSTD,data=IVs,bw=(coef*locband),xevalLen=length(IVpointsSTD[,1]))

# estimation of SPD for original smoothed vola
  f            = flipud(SPD1stEstimation$fstar)
#######################################################################################
#As the local bandwidths are chosen using standardized data, we
#have to standardize them before applying lplocband with local
#bandwidths equal to coef.*SPD1stEstimation.band

  eps          = (flipud(IVsurf$lpFit[,2])-IVpoints[,3])
  eps          = eps-mean(eps)
  beps         = bootstrap(eps,bn,"wild")
  
#######################################################################################
  i=1
  while(i< bn+1){
    IVpointstar  = cbind(RawData[,1:6],(flipud(IVsurfg$lpFit[,2]) + beps[,i]))
    SPDEstimates = SPDlp2(IVpointstar,xGrid,locband,metric)
    if (i==1){
      fci=(SPDEstimates$fstar)
    }else{
      fci=cbind(fci,SPDEstimates$fstar)
    }
    i=i+1
  }
return(list(f=f,fci=fci))
}



caltm = function(ospd, bspd, alpha,CBorCI, p){
  m = ncol(bspd)
  i = 1
  p_scaled = p$gstar # inflation factor
  if(CBorCI=="CB"){
    while(i <= m){
      # This should be where we divide by the physical density p_hat (below)
      z = max(abs(bspd[,i]-ospd)/p_scaled) # bootstrapped spd - original spd
      if (i == 1){
        y = z
      }else{
        y = c(y,z)
      }
      i = i+1
    }
    tmlow = quantile(y, alpha/2)
    tmup  = quantile(y, 1-alpha/2)
  }else{
    while(i<=nrow(bspd)){
      tmlowTemp = quantile(c(bspd[i,]), alpha/2)
      tmupTemp  = quantile(c(bspd[i,]), 1-alpha/2)
      if(i==1){
        tmlow = tmlowTemp
        tmup  = tmupTemp
      }else{
        tmlow = c(tmlow,tmlowTemp)
        tmup  = c(tmup,tmupTemp)
      }
      i=i+1
    }
  }
return(list(tmup=tmup, tmlow=tmlow))
}


#######################################################################
########################### Main Calculation ##########################
#######################################################################

calc_confidence_bands = function(fname, pd_fname, do_plot = TRUE){
    # pd_fname = 'Data/2020-03-04_0.0055_PD_.csv'
    # fname = 'confidence_band_input_2.csv'
    #R object with classes: ('character',) mapped to:
    #['Data/2021-04-06_0.0274_PD_.csv']
    #fname = 'confidence_band_input_10.csv'
    # pd_fname = 'Data/2021-04-06_0.0274_PD_.csv'

    # Looks better
    #fname = 'confidence_band_input_80.csv'
    #pd_fname = "Data/2021-04-06_0.2192_PD_.csv"

    coef=1.3
    set.seed(1)
    dataloaded   = read.csv(fname) # "confidence_band3.csv"#read.table('/Users/julian/src/XFG/XFGSPDcb2/XFGData9701.dat')
    data         = subset(dataloaded, dataloaded[,10]>0)
    data         = subset(data,(data[,4]==1)) # &(data[,5]<30) # 43 #  & (data[,5] == 1)
    data = na.omit(data)
   
    if(nrow(data) < 30){
      print("data too short")
      return(list(NULL, NULL, NULL, NULL, NULL, NULL))
    }else if(nrow(data) > 2000){
      print("data is long, subsetting")
    data = data[sample(2000),]
    } 

    mat          = data[,5]/365 # time to maturity
    IR           = data[,9]     # risk free interest rate
    ForwardPrice = data[,8]*exp(IR*mat)
    RawData      = cbind(ForwardPrice,data[,6],IR,mat,data[,7],data[,4],data[,10])



    ###

 
    # Merge physical density data to the SPD on the xGrid
    pd = read.csv(pd_fname)
    names(pd) = c('gdom', 'moneyness', 'gstar')
    setDT(pd)
    physical_density = pd[moneyness >= 0.7 & moneyness <= 1.3]


    # Todo
    #https://stats.stackexchange.com/questions/143571/nadaraya-watson-optimal-bandwidth
    warning('maybe the spd is oversmoothed??! that could explain
    the flat shape sometimes!')

    # Maybe just use the same xgrid as the physical density has
    locband      = 0.75		# nrow(data)^(-1/5)#
    xGrid        = physical_density$moneyness #seq(0.7, 1.3, 0.005)# seq(0.9,by=0.005,length=40)#seq(0.9, 1.1, by = 0.005) #seq(0.9,by=0.005,length=400)


    xdf = data.frame('m' = xGrid)
    p = physical_density[moneyness %in% xGrid]
    setDF(p)
    warning('changed metric to 0 in order to prevent scaling with forward price!!!')
    metric       = 1
    temp         = spdci(RawData, coef, xGrid, metric,locband)
    f            = temp$f
    fci          = temp$fci 
    result       = caltm(f, fci, 0.1, "CB", p)

    clo          = f - result$tmup
    cup          = f + result$tmup
    fh           = cbind(xGrid,f)   
    clo          = cbind(xGrid,clo) 
    cup          = cbind(xGrid,cup) 




    if(do_plot){
        

        fname_without_csv = gsub('.csv', '', fname)
        png(paste0('pricingkernel/plots/', fname_without_csv, '.png'))
        par(mfrow = c(2,1))

        # Plot pricing Kernel with Bootstrapped CBs 
        plot(fh[,1], fh[,2]/(p$gstar),  col = 'blue', ylim = c(-5, 10), 
            xlab = 'Moneyness', ylab = 'Pricing Kernel', type = 'l')
          lines(fh[,1], clo[,2], lty="dashed", col = 'pink') # /(p$gstar*10000)
          lines(fh[,1], cup[,2], lty="dashed", col = 'orange') # /(p$gstar*10000)
        
        # Plot SPD vs PD
        plot(fh[,1], fh[,2], xlab = 'Moneyness', type = 'l',ylab = 'Density * 10^4', col = 'black')
        lines(fh[,1],p$gstar, col = 'green')

        dev.off()

      }
    #inflation_factor = 10000
    #out = list(xGrid, fh[,'f'], clo[,'clo'], cup[,'cup'])
    out = list(xGrid,  fh[,2]/(p$gstar), clo[,2]/(p$gstar), cup[,2]/(p$gstar), p$gstar, fh[,2])
    return(out)
  }

#calc_confidence_bands('confidence_band_input_1.csv')