import mypkg.my2DPlotB
import numpy as np
import math
from numpy.linalg import inv 

def line_eval(apts,bpts,xloc):
  x1 = apts[0]
  fx1 = apts[1]

  x2 = bpts[0]
  fx2 = bpts[1]

  m = (fx2 - fx1) / (x2 - x1)
  line = lambda x: fx1 + m*(x - x1)

  return line(xloc)



def driver():
    
    f = lambda x: math.exp(x)
    a = 0
    b = 1
    
    ''' create points you want to evaluate at'''
    Neval = 100
    xeval =  np.linspace(a,b,Neval)
    
    ''' number of intervals'''
    Nint = 10
    
    '''evaluate the linear spline'''
    yeval = eval_lin_spline(xeval,a,b,f,Nint)
    
    ''' evaluate f at the evaluation points'''
    fex = np.zeros(Neval)
    for j in range(Neval):
      fex[j] = f(xeval(j)) 
      
    plt = mypkg.my2DPlotB(xeval,fex)
    plt.addPlot(xeval,yeval)
    plt.show()   
     
     
    err = abs(yeval-fex)
    plt2 = mypkg.my2DPlotB(xeval,err)
    plt2.show()       

    
    
def  eval_lin_spline(xeval,Neval,a,b,f,Nint):

    '''create the intervals for piecewise approximations'''
    xint = np.linspace(a,b,Nint+1)
   
    '''create vector to store the evaluation of the linear splines'''
    yeval = np.zeros(Neval) 
    
    for j in range(Nint):
      '''find indices of xeval in interval (xint(jint),xint(jint+1))'''
      '''let ind denote the indices in the intervals'''
      atmp = xint[j]
      btmp= xint[j+1]
      # find indices of values of xeval in the interval
      ind = np.where((xeval >= atmp) & (xeval <= btmp))
      xloc = xeval[ind]
      n = len(xloc)
      '''temporarily store your info for creating a line in the interval of
      interest'''
      fa = f(atmp)
      fb = f(btmp)
      yloc = np.zeros(len(xloc))

      for kk in range(n):
      #use your line evaluator to evaluate the spline at each location
        yloc[kk] = line_eval((atmp,fa),(btmp,fb),xloc) #Call your line evaluator with points (atmp,fa) and (btmp,fb)

      # Copy yloc into the final vector
      yeval[ind] = yloc

    return yeval
           
           
if __name__ == '__main__':
      # run the drivers only if this is called from the command line
      driver()               
