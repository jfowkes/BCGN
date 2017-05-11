"""

**More-Garbow-Hillstrom test functions with first derivatives 
(PyOPUS subsystem name: MGH)**

Translated from MATLAB implementation (with bugs omitted). 

All test functions in this module are maps from :math:`R` to :math:`R^n`. 
Every function is comprised of *m* auxiliary functions 

.. math::
  f_1(x) ... f_m(x)
  
where *x* is a *n*-dimensional vector. 

The actual test function is then constructed as 

.. math::
  f(x) = \\sum_{i=1}^m \\left( f_i(x) \\right)^2

The *i*-th component of the function's gradient can be expressed as

.. math::
  (\\nabla f(x))_i = 2 \\sum_{k=1}^{m} J_{ki}(x) f_{k}(x)
  
where *J(x)* is the Jacobian matrix of the auxiliary functions at *x* where the 
first index corresponds to the auxiliary function and the second index 
corresponds to the component of auxiliary function's gradient. 

An exception is the McKinnon function which is not part of the original test 
suite but is included here because it is a well known counterexample for the 
Nelder-Mead simplex algorithm. Another exception is the Gao-Han almost 
quadratic function. 

The functions were first published as a test suite in [mgh]_.  
Later bounds were added to most test functions in [gay]_. 

If a function's documentation does not say anything about bounds, the bounds 
are defined for all allowed values of *n*. 

This module is independent of PyOPUS, meaning that it can be taken as is 
and used as a module in some other package.

.. [mgh] More J.J, Garbow B. S., Hillstrom K. E.: Testing Unconstrained 
         Optimization Software. ACM Transactions on Mathematical Software, 
         vol. 7, pp. 17-41, 1981. 

.. [gay] Gay D. M., A trust-region approach to linearly constrained 
         optimization. Numerical Analysis (Griffiths, D.F., ed.), Lecture 
         Notes in Mathematics 1066, pp. 72-105, Springer, Berlin, 1984. 
"""

# TODO: speedup powell singular, kowalik, brown almost, gaussian, brown and dennis, penalty II, biggs, rosex, ext pow sing, watson
#       reduced gradient computation

from numpy import array, dot, zeros, ones, arange, where, indices, exp, log, abs, sin, cos, arctan, pi, sqrt, sign, diagflat, triu
from numpy.lib.polynomial import poly1d
from scipy.special import chebyt
import numpy as np

def DbgMsg(x, y):
    return x+": "+y

__all__ = [ 'MGH', 'MGHsuite', 
    'Rosenbrock',
    'FreudensteinAndRoth',
    'PowellBadlyScaled',
    'BrownBadlyScaled',
    'Beale',
    'JennrichAndSampson',
    'HelicalValley',
    'Bard',
    'Gaussian',
    'Meyer',
    'GulfResearchAndDevelopement',
    'Box3D',
    'PowellSingular',
    'Wood',
    'KowalikAndOsborne',
    'BrownAndDennis',
    'Osborne1',
    'BiggsEXP6',
    'Osborne2',
    'Watson',
    'ExtendedRosenbrock',
    'ExtendedPowellSingular',
    'PenaltyI',
    'PenaltyII',
    'VariablyDimensioned',
    'Trigonometric',
    'BrownAlmostLinear',
    'DiscreteBoundaryValue',
    'DiscreteIntegralEquation',
    'BroydenTridiagonal',
    'BroydenBanded',
    'LinearFullRank',
    'LinearRank1',
    'LinearRank1ZeroColumnsAndRows',
    'Chebyquad',
    'Quadratic',
    'Dixon',
    'OrenPower'
]

class MGH():
    """
    Base class for test functions

    The initial point can be obtained from the :attr:`initial` member. The full
    name of the problem is in the :attr:`name` member.

    The :attr:`xl` and :attr:`xh` members specify the lower and the upper bound
    given by D. M. Gay in his paper. If they are ``None`` no known lower or
    upper bound is available in the literature.

    Objects of this class are callable. The calling convention is

    ``r(x)``

    for the auxiliary functions

    ``jacobian(x)``

    for the Jacobian where *x* is the input values vector.

    To create an instance of the extended Rosenbrock function with n=m=4 and
    evaluate the function and the gradient at the initial point one should::

      from MGH import ExtendedRosenbrock
      rb = ExtendedRosenbrock(n=4, m=4)
      r = rb.r(rb.initial)
      J = rb.jacobian(rb.initial)
    """
    name=None
    "The name of the test function"

    def __init__(self, n=2, m=2):
        self.n=n
        self.m=m
        self.initial=None
        self.xl=None
        self.xh=None
        self.fi=zeros(m)
        self.J=zeros([m,n])

    def r(self, x):
        """
        Returns the value of the auxiliary functions at *x*.
        """
        self.func(x)
        return self.fi

    def jacobian(self, x):
        """
        Returns the value of the Jacobian at *x*.
        """
        self.jac(x)
        return self.J

    # Below are the fxopt from the original MGH paper
    # N.B. some of these are parameter dependent
    # WARNING: these are for ||r||^2 without the 1/2
    """
    Rosenbrock                      0
    FreudensteinAndRoth             4.89843e+01
    PowellBadlyScaled               0
    BrownBadlyScaled                0
    Beale                           0
    JennrichAndSampson              1.24362e+02
    HelicalValley                   0
    Bard                            8.21487e-03
    Gaussian                        1.12793e-08
    Meyer                           8.79458e+01
    GulfResearchAndDevelopement     0
    Box3D                           0
    PowellSingular                  0
    Wood                            0
    KowalikAndOsborne               3.07506e-04
    BrownAndDennis                  8.58222e+04     #m=20
    Osborne1                        5.46489e-05
    BiggsEXP6                       5.65565e-03
    Osborne2                        4.01377e-02
    Watson                          2.28767e-3      #n=6
    ExtendedRosenbrock              0
    ExtendedPowellSingular          0
    PenaltyI                        2.24998e-05     #n=4
    PenaltyII                       9.37629e-06     #n=4
    VariablyDimensioned             0
    Trigonometric                   0
    BrownAlmostLinear               0
    DiscreteBoundaryValue           0
    DiscreteIntegralEquation        0
    BroydenTridiagonal              0
    BroydenBanded                   0
    LinearFullRank                  0               #m=n
    LinearRank1                     0.90909090909   #m=n=5 
    LinearRank1ZeroColumnsAndRows   2.42857142857   #m=n=5 
    Chebyquad                       0               #m=n, 1<=n<=7, n=9
    """

class Rosenbrock(MGH):
    """
    Rosenbrock test function (n=m=2).

    See the :class:`MGH` class for more details.
    """
    name="Rosenbrock function"

    def __init__(self, n=2, m=2):
        MGH.__init__(self, n, m)

        self.initial=array([-1.2, 1])

        self.xl=array([-50.0,   0.0])
        self.xh=array([  0.5, 100.0])

        if self.n!=2 or self.m!=2:
            raise Exception, DbgMsg("MGH", "Bad n or m.")

    def func(self, x):
        self.fi[0]=10.0*(x[1]-x[0]**2)
        self.fi[1]=1.0-x[0]

    def jac(self, x):
        # One row contains derivatives wrt n dimensions (has n columns)
        # There is one row per component (m in total)
        self.J[0,0]=-20.0*x[0]
        self.J[0,1]=10.0

        self.J[1,0]=-1.0
        self.J[1,1]=0.0

class FreudensteinAndRoth(MGH):
    """
    Freudenstein and Roth test function (n=m=2).

    See the :class:`MGH` class for more details.
    """
    name="Freudenstein and Roth"

    def __init__(self, n=2, m=2):
        MGH.__init__(self, n, m)

        self.initial=array([0.5, -2.0])

        self.xl=array([ 0.0, -30.0])
        self.xh=array([20.0,  -0.9])

        if self.n!=2 or self.m!=2:
            raise Exception, DbgMsg("MGH", "Bad n or m.")

    def func(self, x):
        self.fi[0]=-13.0+x[0]+((5.0-x[1])*x[1]-2.0)*x[1]
        self.fi[1]=-29.0+x[0]+((x[1]+1.0)*x[1]-14.0)*x[1]

    def jac(self, x):
        self.J[0,0]=1.0
        self.J[0,1]=10*x[1]-3*x[1]**2-2.0

        self.J[1,0]=1.0
        self.J[1,1]=3*x[1]**2+2*x[1]-14.0

class PowellBadlyScaled(MGH):
    """
    Powell badly scaled test function (n=m=2).

    See the :class:`MGH` class for more details.
    """
    name="Powell badly scaled"


    def __init__(self, n=2, m=2):
        MGH.__init__(self, n, m)

        self.initial=array([0.0, 1.0])

        self.xl=array([0.0, 1.0])
        self.xh=array([1.0, 9.0])

        if self.n!=2 or self.m!=2:
            raise Exception, DbgMsg("MGH", "Bad n or m.")

    def func(self, x):
        self.fi[0]=10000*x[0]*x[1]-1.0
        self.fi[1]=exp(-x[0])+exp(-x[1])-1.0001

    def jac(self, x):
        self.J[0,0]=10000*x[1]
        self.J[0,1]=10000*x[0]

        self.J[1,0]=-exp(-x[0])
        self.J[1,1]=-exp(-x[1])

class BrownBadlyScaled(MGH):
    """
    Brown test function (n=2, m=3).

    See the :class:`MGH` class for more details.
    """
    name="Brown badly scaled"

    def __init__(self, n=2, m=3):
        MGH.__init__(self, n, m)

        self.initial=array([1.0, 1.0])

        self.xl=array([0.0, 3e-5])
        self.xh=array([1e6, 100.0])

        if self.n!=2 or self.m!=3:
            raise Exception, DbgMsg("MGH", "Bad n or m.")

    def func(self, x):
        self.fi[0]=x[0]-1e6
        self.fi[1]=x[1]-(2e-6)
        self.fi[2]=x[0]*x[1]-2.0

    def jac(self, x):
        self.J[0,0]=1.0
        self.J[0,1]=0.0

        self.J[1,0]=0.0
        self.J[1,1]=1.0

        self.J[2,0]=x[1]
        self.J[2,1]=x[0]

class Beale(MGH):
    """
    Beale test function (n=2, m=3).

    See the :class:`MGH` class for more details.
    """
    name="Beale"

    def __init__(self, n=2, m=3):
        MGH.__init__(self, n, m)

        self.initial=array([1.0, 1.0])

        self.xl=array([ 0.6,   0.5])
        self.xh=array([10.0, 100.0])

        if self.n!=2 or self.m!=3:
            raise Exception, DbgMsg("MGH", "Bad n or m.")

    def func(self, x):
        self.fi[0]=1.5-x[0]*(1.0-x[1])
        self.fi[1]=2.25-x[0]*(1.0-x[1]**2)
        self.fi[2]=2.625-x[0]*(1.0-x[1]**3)

    def jac(self, x):
        self.J[0,0]=-(1.0-x[1])
        self.J[0,1]=x[0]

        self.J[1,0]=-(1.0-x[1]**2)
        self.J[1,1]=x[0]*2*x[1]

        self.J[2,0]=-(1.0-x[1]**3)
        self.J[2,1]=x[0]*3*x[1]**2

class JennrichAndSampson(MGH):
    """
    Jennrich and Sampson test function (n=2, m>=n).

    See the :class:`MGH` class for more details.
    """
    name="Jennrich and Sampson"

    def __init__(self, n=2, m=10):
        MGH.__init__(self, n, m)

        self.initial=array([0.3, 0.4])

        self.xl=array([ 0.26, 0.0])
        self.xh=array([10.0, 20.0])

        if self.n!=2 or self.m<self.n:
            raise Exception, DbgMsg("MGH", "Bad n or m.")

    def func(self, x):
        for i in range(0, self.m):
            self.fi[i]=2.0+2*(i+1)-(exp((i+1)*x[0])+exp((i+1)*x[1]))

    def jac(self, x):
        for i in range(0, self.m):
            self.J[i,0]=(-(i+1)*exp((i+1)*x[0]))
            self.J[i,1]=(-(i+1)*exp((i+1)*x[1]))

class HelicalValley(MGH):
    """
    Helical valley test function (n=m=3).

    See the :class:`MGH` class for more details.
    """
    name="Helical valley"

    def __init__(self, n=3, m=3):
        MGH.__init__(self, n, m)

        self.initial=array([-1.0, 0.0, 0.0])

        self.xl=array([-100.0, -1.0, -1.0])
        self.xh=array([   0.8,  1.0,  1.0])

        if self.n!=3 or self.m!=3:
            raise Exception, DbgMsg("MGH", "Bad n or m.")

    def func(self, x):
        if x[0]>0.0:
            self.fi[0]=10*(x[2]-10*((1.0/(2*pi))*arctan(x[1]/x[0])))
        elif x[0]<0.0:
            self.fi[0]=10*(x[2]-10*((1.0/(2*pi))*arctan(x[1]/x[0])+0.5))
        else:
            self.fi[0]=0.0
        self.fi[1]=10*((x[0]**2+x[1]**2)**0.5-1.0)
        self.fi[2]=x[2]

    def jac(self, x):
        self.J[0,0]=(50.0/pi)*(x[1]/x[0]**2)*(1.0/(1.0+(x[1]/x[0])**2))
        self.J[0,1]=(-50.0/pi)*(1.0/x[0])*(1.0/(1.0+(x[1]/x[0])**2))
        self.J[0,2]=10.0

        self.J[1,0]=(10*x[0])/sqrt(x[0]**2+x[1]**2)
        self.J[1,1]=(10*x[1])/sqrt(x[0]**2+x[1]**2)
        self.J[1,2]=0.0

        self.J[2,0]=0.0
        self.J[2,1]=0.0
        self.J[2,2]=1.0

class Bard(MGH):
    """
    Bard test function (n=3, m=15).

    See the :class:`MGH` class for more details.
    """
    name="Bard"

    y=array([.14,  .18,  .22,  .25,  .29,  .32,  .35,  .39,  .37,  .58,
                .73,  .96,  1.34, 2.10, 4.39,   0,    0,    0,    0,    0 ])

    def __init__(self, n=3, m=15):
        MGH.__init__(self, n, m)

        self.initial=array([1.0, 1.0, 1.0])

        self.xl=array([ 0.1,   0.0,  0.0])
        self.xh=array([50.0, 100.0, 50.0])

        if self.n!=3 or self.m!=15:
            raise Exception, DbgMsg("MGH", "Bad n or m.")

    def func(self, x):
        for i in range(0, self.m):
            u=i+1
            v=16-(i+1)
            w=float(min(u, v))

            self.fi[i]=self.y[i]-(x[0]+(1.0*u/(v*x[1]+w*x[2])))

    def jac(self, x):
        for i in range(0, self.m):
            u=i+1
            v=16-(i+1)
            w=float(min(u, v))

            self.J[i,0]=-1.0
            self.J[i,1]=1.0*u*v/((v*x[1]+w*x[2])**2)
            self.J[i,2]=1.0*u*w/((v*x[1]+w*x[2])**2)

class Gaussian(MGH):
    """
    Gaussian test function (n=3, m=15).

    See the :class:`MGH` class for more details.
    """
    name="Gaussian"

    y=array([.0009,  .0044,  .0175,  .0540,  .1295,  .2420,  .3521,  .3989,
        .3521,  .2420,  .1295,  .0540,  .0175,  .0044,  .0009,   0 ])

    def __init__(self, n=3, m=15):
        MGH.__init__(self, n, m)

        self.initial=array([0.4, 1.0, 0.0])

        self.xl=array([0.398, 1.0, -0.5])
        self.xh=array([4.2,   2.0,  0.1])

        if self.n!=3 or self.m!=15:
            raise Exception, DbgMsg("MGH", "Bad n or m.")

    def func(self, x):
        for i in range(0, self.m):
            t=(8.0-(i+1))/2.0

            self.fi[i]=x[0]*exp((-x[1]*((t-x[2])**2))/2.0)-self.y[i]

    def jac(self, x):
        for i in range(0, self.m):
            t=(8.0-(i+1))/2.0

            self.J[i,0]=exp((-x[1]*((t-x[2])**2))/2.0)
            self.J[i,1]=x[0]*((-((t-x[2])**2))/2.0)*exp((-x[1]*((t-x[2])**2))/2.0)
            self.J[i,2]=x[0]*x[1]*(t-x[2])*exp((-x[1]*((t-x[2])**2))/2.0)

class Meyer(MGH):
    """
    Meyer test function (n=3, m=16).

    See the :class:`MGH` class for more details.
    """
    name="Meyer"

    y=array([ 34780.0, 28610.0, 23650.0, 19630.0, 16370.0, 13720.0, 11540.0, 9744.0,
                8261.0, 7030.0, 6005.0, 5147.0, 4427.0, 3820.0, 3307.0, 2872.0 ])

    def __init__(self, n=3, m=16):
        MGH.__init__(self, n, m)

        self.initial=array([0.02, 4000.0, 250.0])

        self.xl=array([0.006, 0.0, 0.0])
        self.xh=array([2.0,   3e5, 3e4])

        if self.n!=3 or self.m!=16:
            raise Exception, DbgMsg("MGH", "Bad n or m.")

    def func(self, x):
        for i in range(0, self.m):
            ti = (45.0+5*(i+1))
            di = ti + x[2]
            qi = 1.0 / di
            ei = exp(x[1]*qi)
            si = x[0]*qi*ei

            self.fi[i]=(x[0]*ei)-self.y[i]

    def jac(self, x):
        for i in range(0, self.m):
            ti = (45.0+5*(i+1))
            di = ti + x[2]
            qi = 1.0 / di
            ei = exp(x[1]*qi)
            si = x[0]*qi*ei

            self.J[i,0]=ei
            self.J[i,1]=si
            self.J[i,2]=-x[1]*qi*si


class GulfResearchAndDevelopement(MGH):
    """
    Gulf research and developement test function (n=3, n<=m<=100).

    See the :class:`MGH` class for more details.
    """
    name="Gulf research and developement"

    def __init__(self, n=3, m=3):
        MGH.__init__(self, n, m)

        self.initial=array([5.0, 2.5, 0.15])

        self.xl=array([ 0.0,  0.0,  0.0])
        self.xh=array([10.0, 10.0, 10.0])

        if self.n!=3 or self.m>100 or self.m<self.n:
            raise Exception, DbgMsg("MGH", "Bad n or m.")

    def func(self, x):
        #if x[0] == 0.0:
        #	raise Warning, DbgMsg("MGH", "+++ singularity in gulf function evaluation")

        ti=arange(1.0, self.m+1)/100.0
        yi=25.0+(-50.0*log(ti))**(2.0/3.0)
        yimix1=yi-x[1]
        ex=exp(-abs(yimix1)**x[2]/x[0])

        self.fi=ex-ti

        x1inv=1.0/x[0]

    def jac(self, x):
        ti = arange(1.0, self.m + 1) / 100.0
        yi = 25.0 + (-50.0 * log(ti)) ** (2.0 / 3.0)
        yimix1 = yi - x[1]
        ex = exp(-abs(yimix1) ** x[2] / x[0])
        self.J[:,0]=abs(yimix1)**x[2]/x[0]**2*ex
        self.J[:,1]=x[2]*abs(yimix1)**(x[2]-1.0)*sign(yimix1)/x[0]*ex
        self.J[:,2]=-log(abs(yimix1))*abs(yimix1)**x[2]/x[0]*ex

class Box3D(MGH):
    """
    Box 3D test function (n=3, m>=n).

    See the :class:`MGH` class for more details.
    """
    name="Box 3D"

    def __init__(self, n=3, m=10):
        MGH.__init__(self, n, m)

        self.initial=array([0.0, 10.0, 20.0])

        self.xl=array([0.0, 5.0,  0.0])
        self.xh=array([2.0, 9.5, 20.0])

        if self.n!=3 or self.m<self.n:
            raise Exception, DbgMsg("MGH", "Bad n or m.")

    def func(self, x):
        for i in range(0, self.m):
            t = 0.1*(i+1)
            self.fi[i]=exp(-t*x[0])-exp(-t*x[1])-x[2]*(exp(-t)-exp(-10*t))

    def jac(self, x):
        for i in range(0, self.m):
            t = 0.1*(i+1)
            self.J[i,0]=-t*exp(-t*x[0])
            self.J[i,1]=t*exp(-t*x[1])
            self.J[i,2]=-(exp(-t)-exp(-10*t))

class PowellSingular(MGH):
    """
    Powell singular test function (n=m=4).

    See the :class:`MGH` class for more details.
    """
    name="Powell singular"

    def __init__(self, n=4, m=4):
        MGH.__init__(self, n, m)

        self.initial=array([3.0, -1.0, 0.0, 1.0])

        self.xl=array([  0.1, -20.0, -1.0, -1.0])
        self.xh=array([100.0,  20.0,  1.0, 50.0])

        if self.n!=4 or self.m!=4:
            raise Exception, DbgMsg("MGH", "Bad n or m.")

    def func(self, x):
        self.fi[0]=x[0]+10*x[1]
        self.fi[1]=sqrt(5.0)*(x[2]-x[3])
        self.fi[2]=(x[1]-2*x[2])**2
        self.fi[3]=sqrt(10.0)*((x[0]-x[3])**2)

    def jac(self, x):
        self.J[0,0]=1.0
        self.J[0,1]=10.0
        self.J[0,2]=0.0
        self.J[0,3]=0.0

        self.J[1,0]=0.0
        self.J[1,1]=0.0
        self.J[1,2]=sqrt(5.0)
        self.J[1,3]=-sqrt(5.0)

        self.J[2,0]=0.0
        self.J[2,1]=2*(x[1]-2*x[2])
        self.J[2,2]=-4*(x[1]-2*x[2])
        self.J[2,3]=0.0

        self.J[3,0]=2*sqrt(10.0)*(x[0]-x[3])
        self.J[3,1]=0.0
        self.J[3,2]=0.0
        self.J[3,3]=-2*sqrt(10.0)*(x[0]-x[3])

class Wood(MGH):
    """
    Wood test function (n=4, m=6).

    See the :class:`MGH` class for more details.
    """
    name="Wood"

    def __init__(self, n=4, m=6):
        MGH.__init__(self, n, m)

        self.initial=array([-3.0, -1.0, -3.0, -1.0])

        self.xl=array([-100.0, -100.0, -100.0, -100.0])
        self.xh=array([   0.0,   10.0,  100.0,  100.0])

        if self.n!=4 or self.m!=6:
            raise Exception, DbgMsg("MGH", "Bad n or m.")

    def func(self, x):
        self.fi[0]=10.0*(x[1]-x[0]**2)
        self.fi[1]=1.0-x[0]
        self.fi[2]=sqrt(90.0)*(x[3]-x[2]**2)
        self.fi[3]=1.0-x[2]
        self.fi[4]=sqrt(10.0)*(x[1]+x[3]-2.0)
        self.fi[5]=(1.0/sqrt(10.0))*(x[1]-x[3])

    def jac(self, x):
        self.J[0,0]=-20.0*x[0]
        self.J[0,1]=10.0
        self.J[0,2]=0.0
        self.J[0,3]=0.0

        self.J[1,0]=-1.0
        self.J[1,1]=0.0
        self.J[1,2]=0.0
        self.J[1,3]=0.0

        self.J[2,0]=0.0
        self.J[2,1]=0.0
        self.J[2,2]=-2*sqrt(90.0)*x[2]
        self.J[2,3]=sqrt(90.0)

        self.J[3,0]=0.0
        self.J[3,1]=0.0
        self.J[3,2]=-1.0
        self.J[3,3]=0.0

        self.J[4,0]=0.0
        self.J[4,1]=sqrt(10.0)
        self.J[4,2]=0.0
        self.J[4,3]=sqrt(10.0)

        self.J[5,0]=0.0
        self.J[5,1]=1.0/sqrt(10.0)
        self.J[5,2]=0.0
        self.J[5,3]=-1.0/sqrt(10.0)

class KowalikAndOsborne(MGH):
    """
    Kowalik and Osborne test function (n=4, m=11).

    See the :class:`MGH` class for more details.
    """
    name="Kowalik and Osborne"

    y=array([.1957,  .1947,  .1735,  .1600,  .0844,  .0627,
                .0456,  .0342,  .0323,  .0235,  .0246,      0.0 ])

    u=array([4.0000,  2.0000,  1.0000,  0.5000,  0.2500,  0.1670,
                0.1250,  0.1000,  0.0833,  0.0714,  0.0625,       0.0 ])

    def __init__(self, n=4, m=11):
        MGH.__init__(self, n, m)

        self.initial=array([0.25, 0.39, 0.415, 0.39])

        self.xl=array([ 0.0, -1.0,  0.13, 0.12])
        self.xh=array([10.0, 12.0, 13.0, 12.0])

        if self.n!=4 or self.m!=11:
            raise Exception, DbgMsg("MGH", "Bad n or m.")

    def func(self, x):
        for i in range(0, self.m):
            c1 = self.u[i]**2 + self.u[i]*x[1]
            c2 = self.u[i]**2 + self.u[i]*x[2] + x[3]

            self.fi[i]=self.y[i]-(x[0]*c1)/c2

    def jac(self, x):
        for i in range(0, self.m):
            c1 = self.u[i]**2 + self.u[i]*x[1]
            c2 = self.u[i]**2 + self.u[i]*x[2] + x[3]

            self.J[i,0]=-c1/c2
            self.J[i,1]=-x[0]*self.u[i]/c2
            self.J[i,2]=x[0]*c1*(c2**(-2))*self.u[i]
            self.J[i,3]=x[0]*c1*(c2**(-2))

class BrownAndDennis(MGH):
    """
    Brown and Dennis test function (n=4, m=20).

    See the :class:`MGH` class for more details.
    """
    name="Brown and Dennis"

    def __init__(self, n=4, m=20):
        MGH.__init__(self, n, m)

        self.initial=array([25.0, 5.0, -5.0, -1.0])

        self.xl=array([-10.0,  0.0, -100.0, -20.0])
        self.xh=array([100.0, 15.0,    0.0,   0.2])

        if self.n!=4 or self.m<self.n:
            raise Exception, DbgMsg("MGH", "Bad n or m.")

    def func(self, x):
        for i in range(0, self.m):
            ti = (i+1)*0.2
            ei = exp(ti)
            si = sin(ti)
            ci = cos(ti)

            self.fi[i]=(x[0] + ti*x[1] - ei)**2 + (x[2] + x[3]*si - ci)**2

    def jac(self, x):
        for i in range(0, self.m):
            ti = (i+1)*0.2
            ei = exp(ti)
            si = sin(ti)
            ci = cos(ti)
            f1 = 2.0*(x[0] + ti*x[1] - ei)
            f3 = 2.0*(x[2] + x[3]*si - ci)

            self.J[i,0]=f1
            self.J[i,1]=f1*ti
            self.J[i,2]=f3
            self.J[i,3]=f3*si

class Osborne1(MGH):
    """
    Osborner 1 test function (n=5, m=33).

    See the :class:`MGH` class for more details.
    """
    name="Osborne 1"

    y=array([0.844, 0.908, 0.932, 0.936, 0.925,	0.908, 0.881, 0.850, 0.818, 0.784,
                0.751, 0.718, 0.685, 0.658, 0.628, 0.603, 0.580, 0.558, 0.538, 0.522,
                0.506, 0.490, 0.478, 0.467, 0.457, 0.448, 0.438, 0.431, 0.424, 0.420,
                0.414, 0.411, 0.406 ])

    def __init__(self, n=5, m=33):
        MGH.__init__(self, n, m)

        self.initial=array([0.5, 1.5, -1.0, 0.01, 0.02])

        self.xl=array([ 0.0, 0.0, -50.0,  0.0,  0.0])
        self.xh=array([50.0, 1.9,  -0.1, 10.0, 10.0])

        if self.n!=5 or self.m!=33:
            raise Exception, DbgMsg("MGH", "Bad n or m.")

    def func(self, x):
        for i in range(0, self.m):
            ti = ((i+1)-1)*10.0
            e4 = exp(-ti*x[3])
            e5 = exp(-ti*x[4])
            t2 = x[1]*e4
            t3 = x[2]*e5
            self.fi[i] = self.y[i] - (x[0] + t2 + t3)

    def jac(self, x):
        for i in range(0, self.m):
            ti = ((i+1)-1)*10.0
            e4 = exp(-ti*x[3])
            e5 = exp(-ti*x[4])
            t2 = x[1]*e4
            t3 = x[2]*e5

            self.J[i,0]=-1.0
            self.J[i,1]=-e4
            self.J[i,2]=-e5
            self.J[i,3]=ti*t2
            self.J[i,4]=ti*t3

class BiggsEXP6(MGH):
    """
    Biggs EXP6 test function (n=6, m>=n).

    See the :class:`MGH` class for more details.
    """
    name="Biggs EXP6"

    def __init__(self, n=6, m=13):
        MGH.__init__(self, n, m)

        self.initial=array([1.0, 2.0, 1.0, 1.0, 1.0, 1.0])

        self.xl=array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        self.xh=array([2.0, 8.0, 1.0, 7.0, 5.0, 5.0])

        if self.n!=6 or self.m<self.n:
            raise Exception, DbgMsg("MGH", "Bad n or m.")

    def func(self, x):
        for i in range(0, self.m):
            ti = 0.1*(i+1)
            yi = exp(-ti)-5*exp(-10*ti)+3*exp(-4*ti)

            self.fi[i]=x[2]*exp(-ti*x[0])-x[3]*exp(-ti*x[1])+x[5]*exp(-ti*x[4])-yi

    def jac(self, x):
        for i in range(0, self.m):
            ti = 0.1*(i+1)
            yi = exp(-ti)-5*exp(-10*ti)+3*exp(-4*ti)

            self.J[i,0]=-ti*x[2]*exp(-ti*x[0])
            self.J[i,1]=ti*(x[3])*exp(-ti*x[1])
            self.J[i,2]=exp(-ti*x[0])
            self.J[i,3]=-exp(-ti*x[1])
            self.J[i,4]=x[5]*(-ti)*exp(-ti*x[4])
            self.J[i,5]=exp(-ti*x[4])

class Osborne2(MGH):
    """
    Osborner 2 test function (n=11, m=65).

    See the :class:`MGH` class for more details.
    """
    name="Osborne 2"

    y=array([  1.366, 1.191, 1.112, 1.013,  .991,
                .885,  .831,  .847,  .786,  .725,
                .746,  .679,  .608,  .655,  .616,
                .606,  .602,  .626,  .651,  .724,
                .649,  .649,  .694,  .644,  .624,
                .661,  .612,  .558,  .533,  .495,
                .500,  .423,  .395,  .375,  .372,
                .391,  .396,  .405,  .428,  .429,
                .523,  .562,  .607,  .653,  .672,
                .708,  .633,  .668,  .645,  .632,
                .591,  .559,  .597,  .625,  .739,
                .710,  .729,  .720,  .636,  .581,
                .428,  .292,  .162,  .098,  .054 ])

    def __init__(self, n=11, m=65):
        MGH.__init__(self, n, m)

        self.initial=array([1.3, 0.65, 0.65, 0.7, 0.6, 3.0, 5.0, 7.0, 2.0, 4.5, 5.5])

        # NOTE: Modified last component upper limit to 10 (in paper it is 0 resulting in [0,0] which makes no sense)
        self.xl=array([  1.0,   0.5,   0.0,   0.6,   0.0,   0.0,   0.0,   0.0,   0.0,  0.0,  0.0])
        self.xh=array([150.0, 100.0, 100.0, 100.0, 100.0, 500.0, 500.0, 500.0, 100.0, 10.00, 10.0])

        if self.n!=11 or self.m!=65:
            raise Exception, DbgMsg("MGH", "Bad n or m.")

    def func(self, x):
        ti=arange(0.0, self.m)/10.0
        self.fi=self.y-(
            x[0]*exp(-ti*x[4])+
            x[1]*exp(-(ti-x[8])**2*x[5])+
            x[2]*exp(-(ti-x[9])**2*x[6])+
            x[3]*exp(-(ti-x[10])**2*x[7])
        )
        #for i in range(0, self.m):	# Matlab implementation has opposite sign for fi
        #	ti =((i+1)-1.0)/10
        #	t09=ti-x[8]
        #	t10=ti-x[9]
        #	t11=ti-x[10]
        #	s09=t09**2
        #	s10=t10**2
        #	s11=t11**2
        #	e1= exp(-ti*x[4])
        #	e2= exp(-s09*x[5])
        #	e3= exp(-s10*x[6])
        #	e4= exp(-s11*x[7])
        #
        #	self.fi[i] = (x[0]*e1 + x[1]*e2 + x[2]*e3 + x[3]*e4) - self.y[i]

    def jac(self, x):
        ti = arange(0.0, self.m) / 10.0
        self.J[:,0]=-exp(-ti*x[4])
        self.J[:,1]=-exp(-(ti-x[8])**2*x[5])
        self.J[:,2]=-exp(-(ti-x[9])**2*x[6])
        self.J[:,3]=-exp(-(ti-x[10])**2*x[7])
        self.J[:,4]=x[0]*ti*exp(-ti*x[4])
        self.J[:,5]=x[1]*(ti-x[8])**2*exp(-(ti-x[8])**2*x[5])
        self.J[:,6]=x[2]*(ti-x[9])**2*exp(-(ti-x[9])**2*x[6])
        self.J[:,7]=x[3]*(ti-x[10])**2*exp(-(ti-x[10])**2*x[7])
        self.J[:,8]=-x[1]*x[5]*2.0*(ti-x[8])*exp(-(ti-x[8])**2*x[5])
        self.J[:,9]=-x[2]*x[6]*2.0*(ti-x[9])*exp(-(ti-x[9])**2*x[6])
        self.J[:,10]=-x[3]*x[7]*2.0*(ti-x[10])*exp(-(ti-x[10])**2*x[7])
        #for i in range(0, self.m):
        #	ti =((i+1)-1.0)/10
        #	t09=ti-x[8]
        #	t10=ti-x[9]
        #	t11=ti-x[10]
        #	s09=t09**2
        #	s10=t10**2
        #	s11=t11**2
        #	e1= exp(-ti*x[4])
        #	e2= exp(-s09*x[5])
        #	e3= exp(-s10*x[6])
        #	e4= exp(-s11*x[7])
        #	r2=x[1]*e2
        #	r3=x[2]*e3
        #	r4=x[3]*e4
        #
        #	self.J[i,0]=e1
        #	self.J[i,1]=e2
        #	self.J[i,2]=e3
        #	self.J[i,3]=e4
        #	self.J[i,4]=-ti*x[0]*e1
        #	self.J[i,5]=-s09*r2
        #	self.J[i,6]=-s10*r3
        #	self.J[i,7]=-s11*r4
        #	self.J[i,8]=2*t09*x[5]*r2
        #	self.J[i,9]=2*t10*x[6]*r3
        #	self.J[i,10]=2*t11*x[7]*r4

class Watson(MGH):
    """
    Watson test function (2<=n<=31, m=31).

    Bounds defined for n=2, 9, and 12.

    See the :class:`MGH` class for more details.
    """
    name="Watson"

    def __init__(self, n=6, m=31):
        MGH.__init__(self, n, m)
        if self.n<2 or self.n>31 or self.m!=31:
            raise Exception, DbgMsg("MGH", "Bad n or m.")

        self.initial=zeros(self.n)

        if self.n==6:
            self.xl=array([-0.015, -10.0, -10.0, -10.0, -10.0, -10.0])
            self.xh=array([10.0,   100.0, 100.0, 100.0, 100.0,   0.99])
        elif self.n==9:
            self.xl=array([-1e-5, 0.0, 0.0, 0.0, 0.0, -3.0, 0.0, -3.0, 0.0])
            self.xh=array([ 1e-5, 0.9, 0.1, 1.0, 1.0,  0.0, 4.0,  0.0, 2.0])
        elif self.n==12:
            self.xl=array([-1.0, 0.0, -1.0, -1.0, -1.0, 0.0, -3.0,  0.0, -10.0,  0.0, -5.0, 0.0])
            self.xh=array([ 0.0, 0.9,  0.0,  0.3,  0.0, 1.0,  0.0, 10.0,   0.0, 10.0,  0.0, 1.0])

    def func(self, x):
        for i in range(0, 29):
            ti = (i+1) / 29.0

            j=arange(2, self.n+1)
            sum1=sum((j-1)*x[j-1]*(ti**(j-2))).sum()

            j=arange(1, self.n+1)
            sum2=(x[j-1]*(ti**(j-1))).sum()

            self.fi[i]=sum1-(sum2**2)-1.0

        self.fi[29] = x[0]
        self.fi[30] = x[1]-((x[0])**2)-1.0

    def jac(self, x):
        for i in range(0, self.m):
            ti = (i+1) / 29.0

            j=arange(2, self.n+1)
            sum1=sum((j-1)*x[j-1]*(ti**(j-2))).sum()

            j=arange(1, self.n+1)
            sum2=(x[j-1]*(ti**(j-1))).sum()

            self.J[i,0]=-(2*sum2)

            for j in range(1, self.n):
                self.J[i,j]=j*((ti)**(j-1))-2*sum2*(ti)**j

        self.J[29,0]=1.0
        self.J[29,1:]=0.0

        self.J[30,0]=-2*x[0]
        self.J[30,1]=1.0
        self.J[30,2:]=0.0

class ExtendedRosenbrock(MGH):
    """
    Extended Rosenbrock test function (n=2k>=2, m=n).

    See the :class:`MGH` class for more details.
    """
    name="Extended Rosenbrock"

    def __init__(self, n=2, m=None):
        if m is None:
            m=n
        MGH.__init__(self, n, m)
        if self.n<2 or self.m!=self.n or (self.n % 2)!=0:
            raise Exception, DbgMsg("MGH", "Bad n or m.")

        self.initial=ones(self.n)
        self.initial[0:self.n:2]=-1.2

        self.xl=zeros(self.n)
        self.xh=zeros(self.n)
        self.xl[0:self.n:2]=-50.0
        self.xl[1:self.n:2]=  0.0
        self.xh[0:self.n:2]=  0.5
        self.xh[1:self.n:2]=100.0

    def func(self, x):
        for i in range(0, self.m/2):
            self.fi[2*i]=10*(x[2*i+1]-x[2*i]**2)
            self.fi[2*i+1]=1.0-x[2*i]

    def jac(self, x):
        for i in range(0, self.m/2):
            self.J[2*i, 2*i]  =-20*x[2*i]
            self.J[2*i, 2*i+1]=10.0
            self.J[2*i+1, 2*i]=-1.0
            self.J[2*i+1, 2*i+1]=0.0

class ExtendedPowellSingular(MGH):
    """
    Extended Powell singular test function (n=4k>=4, m=n).

    See the :class:`MGH` class for more details.
    """
    name="Extended Powell Singular"

    def __init__(self, n=4, m=None):
        if m is None:
            m=n
        MGH.__init__(self, n, m)
        if self.n<4 or self.m!=self.n or (self.n % 4)!=0:
            raise Exception, DbgMsg("MGH", "Bad n or m.")

        self.initial=ones(self.n)
        self.initial[0:self.n:4]=3.0
        self.initial[1:self.n:4]=-1.0
        self.initial[2:self.n:4]=0.0

        self.xl=zeros(self.n)
        self.xh=zeros(self.n)
        self.xl[0:self.n:4]=  0.1
        self.xl[1:self.n:4]=-20.0
        self.xl[2:self.n:4]= -1.0
        self.xl[3:self.n:4]= -1.0
        self.xh[0:self.n:4]=100.0
        self.xh[1:self.n:4]= 20.0
        self.xh[2:self.n:4]=  1.0
        self.xh[3:self.n:4]= 50.0

    def func(self, x):
        for i in range(0, self.m/4):
            self.fi[4*i]=x[4*i]+10*x[4*i+1]
            self.fi[4*i+1]=sqrt(5.0)*(x[4*i+2]-x[4*i+3])
            self.fi[4*i+2]=(x[4*i+1]-2*(x[4*i+2]))**2
            self.fi[4*i+3]=sqrt(10.0)*(x[4*i]-x[4*i+3])**2

    def jac(self, x):
        for i in range(0, self.m/4):
            self.J[4*i, 4*i]  =1.0
            self.J[4*i, 4*i+1]=10.0
            self.J[4*i, 4*i+2]=0.0
            self.J[4*i, 4*i+3]=0.0

            self.J[4*i+1, 4*i]  =0.0
            self.J[4*i+1, 4*i+1]=0.0
            self.J[4*i+1, 4*i+2]=sqrt(5.0)
            self.J[4*i+1, 4*i+3]=-sqrt(5.0)

            self.J[4*i+2, 4*i]  =0.0
            self.J[4*i+2, 4*i+1]=2*x[4*i+1]-4*x[4*i+2]
            self.J[4*i+2, 4*i+2]=8*x[4*i+2]-4*x[4*i+1]
            self.J[4*i+2, 4*i+3]=0.0

            self.J[4*i+3, 4*i]  =2*sqrt(10.0)*(x[4*i]-x[4*i+3])
            self.J[4*i+3, 4*i+1]=0.0
            self.J[4*i+3, 4*i+2]=0.0
            self.J[4*i+3, 4*i+3]=2*sqrt(10)*(x[4*i+3]-x[4*i])

class PenaltyI(MGH):
    """
    Penalty I test function (m=n+1).

    Bounds defined for n=10.

    See the :class:`MGH` class for more details.
    """
    name="Penalty I"

    def __init__(self, n=4, m=None):
        if m is None:
            m=n+1
        MGH.__init__(self, n, m)
        if self.m!=(self.n+1):
            raise Exception, DbgMsg("MGH", "Bad n or m.")

        self.initial=arange(1.0, self.n+1)

        if self.n==10:
            self.xl=array([  0.0,   1.0,   0.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   1.0])
            self.xh=array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])

    def func(self, x):
        self.fi[0:self.n]=sqrt(1e-5)*(x-1.0)
        self.fi[self.n]=(x**2).sum()-0.25

    def jac(self, x):
        self.J[arange(0,self.n), arange(0,self.n)]=sqrt(1e-5)
        self.J[self.n,:]=2*x

class PenaltyII(MGH):
    """
    Penalty II test function (m=2n).

    Bounds defined for n=1, 4, and 10.

    See the :class:`MGH` class for more details.
    """
    name="Penalty II"

    def __init__(self, n=4, m=None):
        if m is None:
            m=2*n
        MGH.__init__(self, n, m)
        if self.m!=(2*self.n):
            raise Exception, DbgMsg("MGH", "Bad n or m.")

        self.initial=ones(self.n)*0.5

        if self.n==1:
            self.xl=array([-1.0])
            self.xh=array([ 1.0])
        elif self.n==4:
            self.xl=array([-10.0,  0.3,  0.0, -1.0])
            self.xh=array([ 50.0, 50.0, 50.0,  0.5])
        elif self.n==10:
            self.xl=array([-10.0,  0.1,  0.0,  0.05, 0.0, -10.0,  0.0,  0.2,  0.0, 0.0])
            self.xh=array([ 50.0, 50.0, 50.0, 50.0, 50.0,  50.0, 50.0, 50.0, 50.0, 0.5])

    def func(self, x):
        self.fi[0]=x[0]-0.2

        if self.n>1:
            yi=exp(arange(2,self.n+1)/10.0)+exp(arange(1,self.n)/10.0)

            self.fi[1:self.n]=sqrt(1e-5)*(exp(x[1:]/10.0)+exp(x[0:-1]/10.0)-yi)
            self.fi[self.n:(self.m-1)]=sqrt(1e-5)*(exp(x[1:]/10.0)-exp(-0.1))

        self.fi[self.m-1]=((self.n-arange(1, self.n+1)+1)*(x**2)).sum()-1.0

    def jac(self, x):
        self.J[0,0]=1.0

        if self.n>1:
            self.J[arange(1,self.n), arange(1,self.n)]=sqrt(1e-5)*exp(x[1:self.n]/10.0)*(1.0/10.0)
            self.J[arange(1,self.n), arange(0,self.n-1)]=sqrt(1e-5)*exp(x[0:self.n-1]/10.0)*(1.0/10.0)

            self.J[arange(self.n,self.m-1), arange(1,self.n)]=sqrt(1e-5)*exp(x[1:]/10.0)*(1.0/10.0)

        self.J[self.m-1,:]=(self.n-arange(1, self.n+1)+1.0)*2*x


class VariablyDimensioned(MGH):
    """
    Variably dimensional test function (m=n+2).

    Bounds defined for n=10.

    See the :class:`MGH` class for more details.
    """

    name="Variably dimensioned"

    def __init__(self, n=8, m=None):
        if m is None:
            m=n+2
        MGH.__init__(self, n, m)
        if self.m!=(self.n+2):
            raise Exception, DbgMsg("MGH", "Bad n or m.")

        self.initial=1.0-1.0*arange(1,self.n+1)/self.n

        if self.n==10:
            self.xl=array([ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 0.0])
            self.xh=array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 0.5])

    def func(self, x):
        s=(arange(1,self.n+1)*(x-1.0)).sum()

        self.fi[0:self.n]=x-1.0
        self.fi[self.n]=s
        self.fi[self.n+1]=s**2

    def jac(self, x):
        s=(arange(1,self.n+1)*(x-1.0)).sum()

        self.J[arange(1, self.n), arange(1, self.n)]=1.0
        self.J[self.n, :]=arange(1, self.n+1)
        self.J[self.n+1, :]=2*s*arange(1, self.n+1)

class Trigonometric(MGH):
    """
    Trigonometric test function (m=n).

    Bounds defined for n=10.

    See the :class:`MGH` class for more details.
    """

    name="Trigonometric"

    def __init__(self, n=10, m=None):
        if m is None:
            m=n
        MGH.__init__(self, n, m)
        if self.m!=self.n:
            raise Exception, DbgMsg("MGH", "Bad n or m.")

        self.initial=1.0*ones(self.n)/self.n

        if self.n==10:
            self.xl=array([ 0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0,  90.0])
            self.xh=array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])

    def func(self, x):
        self.fi[0:self.n]=self.n-cos(x).sum()+arange(1, self.n+1)*(1-cos(x))-sin(x)

    def jac(self, x):
        self.J[arange(0,self.n),:]=sin(x)
        self.J[arange(0,self.n), arange(0,self.n)]+=arange(1, self.n+1)*sin(x)-cos(x)

class BrownAlmostLinear(MGH):
    """
    Brown almost linear test function (m=n).

    Bounds defined for n=10k.

    See the :class:`MGH` class for more details.
    """

    name="Brown almost-linear"

    def __init__(self, n=5, m=None):
        if m is None:
            m=n
        MGH.__init__(self, n, m)
        if self.m!=self.n:
            raise Exception, DbgMsg("MGH", "Bad n or m.")

        self.initial=ones(self.n)*0.5

        if self.n % 10 == 0:
            self.xl=zeros(self.n)*0.0
            self.xh=ones(self.n)*100.0
            self.xl[1:self.n:10]=1.0
            self.xh[2:self.n:10]=0.9

    def func(self, x):
        self.fi[0:self.n-1]=x[0:self.n-1]+x.sum()-(self.n+1.0)
        self.fi[self.n-1]=x.prod()-1.0

    def jac(self, x):
        self.J[arange(0,self.n-1), :]=ones(self.n)
        self.J[arange(0,self.n-1), arange(0,self.n-1)]+=ones(self.n-1)

        for i in range(0, self.n):
            pr=1.0
            if i>0:
                pr*=x[:i].prod()
            if i+1<self.n:
                pr*=x[i+1:].prod()
            self.J[self.n-1, i]=pr

# No bounds				
class DiscreteBoundaryValue(MGH):
    """
    Discrete boundary value test function (m=n).

    No bounds defined.

    See the :class:`MGH` class for more details.
    """

    name="Discrete boundary value"

    def __init__(self, n=5, m=None):
        if m is None:
            m=n
        MGH.__init__(self, n, m)
        if self.m!=self.n:
            raise Exception, DbgMsg("MGH", "Bad n or m.")

        self.initial=arange(1, self.n+1)/(self.n+1.0)*(arange(1, self.n+1)/(self.n+1.0)-1.0)

    def func(self, x):
        xeff=zeros(self.n+1)
        xeff[0:self.n]=x

        h=1.0/(self.n+1.0)
        ti=arange(1, self.n+1)*h

        self.fi[0]=2*x[0]-xeff[1]+h**2*(x[0]+ti[0]+1)**3/2.0
        if self.n>1:
            self.fi[1:self.n]=2*x[1:self.n]-x[0:self.n-1]-xeff[2:self.n+1]+h**2*(x[1:self.n]+ti[1:self.n]+1.0)**3/2.0

    def jac(self, x):
        h = 1.0 / (self.n + 1.0)
        ti = arange(1, self.n + 1) * h
        self.J[arange(0,self.n), arange(0,self.n)]=2.0+h**2*3.0*(x+ti+1.0)**2/2.0
        if self.n>1:
            self.J[arange(0,self.n-1), arange(1,self.n)]=-1.0
            self.J[arange(1,self.n), arange(0,self.n-1)]=-1.0

# No bounds
class DiscreteIntegralEquation(MGH):
    """
    Discrete integral equation test function (m=n).

    No boundas defined.

    See the :class:`MGH` class for more details.
    """

    name="Discrete integral equation"

    def __init__(self, n=5, m=None):
        if m is None:
            m=n
        MGH.__init__(self, n, m)
        if self.m!=self.n:
            raise Exception, DbgMsg("MGH", "Bad n or m.")

        self.initial=arange(1, self.n+1)/(self.n+1.0)*(arange(1, self.n+1)/(self.n+1.0)-1.0)

    def func(self, x):
        h=1.0/(self.n+1.0)
        ti=arange(1, self.n+1)*h

        for i in range(0, self.n):
            sum1=(ti[:i]*(x[:i]+ti[:i]+1.0)**3).sum()
            if self.n>1:
                sum2=((1.0-ti[i:])*(x[i:]+ti[i:]+1.0)**3).sum()
            else:
                sum2=0.0

            self.fi[i]=x[i]+h*((1.0-ti[i])*sum1+ti[i]*sum2)/2.0

    def jac(self, x):
        h = 1.0 / (self.n + 1.0)
        ti = arange(1, self.n + 1) * h
        for i in range(0, self.n):
            for j in range(0, self.n):
                if (j<i):
                    self.J[i,j]=3*h/2*(1.0-ti[i])*ti[j]*(x[j]+ti[j]+1.0)**2
                elif (j>i):
                    self.J[i,j]=3*h/2*ti[i]*(1.0-ti[j])*(x[j]+ti[j]+1.0)**2
                else:
                    self.J[i,i]=1.0+3*h/2*(1.0-ti[i])*ti[i]*(x[i]+ti[i]+1.0)**2

# No bounds
class BroydenTridiagonal(MGH):
    """
    Broyden tridiagonal test function (m=n).

    No bounds defined.

    See the :class:`MGH` class for more details.
    """

    name="Broyden tridiagonal"

    def __init__(self, n=5, m=None):
        if m is None:
            m=n
        MGH.__init__(self, n, m)
        if self.m!=self.n:
            raise Exception, DbgMsg("MGH", "Bad n or m.")

        self.initial=ones(self.n)*(-1.0)

    def func(self, x):
        xeff=zeros(self.n+1)
        xeff[0:self.n]=x

        self.fi[0]=(3.0-2.0*x[0])*x[0]-2*xeff[1]+1.0
        if self.n>1:
            self.fi[1:self.n]=(3.0-2.0*x[1:self.n])*x[1:self.n]-x[0:self.n-1]-2*xeff[2:self.n+1]+1.0

    def jac(self, x):
        self.J[arange(0,self.n), arange(0,self.n)]=3-4*x
        if self.n>1:
            self.J[arange(0,self.n-1), arange(1,self.n)]=-2.0
            self.J[arange(1,self.n), arange(0,self.n-1)]=-1.0

# No bounds
class BroydenBanded(MGH):
    """
    Broyden banded test function (m=n).

    No bounds defined.

    See the :class:`MGH` class for more details.
    """

    name="Broyden banded"

    def __init__(self, n=5, m=None):
        if m is None:
            m=n
        MGH.__init__(self, n, m)
        if self.m!=self.n:
            raise Exception, DbgMsg("MGH", "Bad n or m.")

        self.initial=ones(self.n)*(-1.0)

    def func(self, x):
        ml=5
        mu=1

        ndx=arange(0, self.n)
        for i in range(0, self.n):
            lb=max(0, i-ml)
            ub=min(self.n-1, i+mu)
            j=where((ndx!=i) & (ndx>=lb) & (ndx<=ub))

            self.fi[i]=x[i]*(2.0+5.0*x[i]**2)+1.0-(x*(1.0+x))[j].sum()

    def jac(self, x):
        ml = 5
        mu = 1

        self.J[arange(0, self.n), arange(0, self.n)]=2.0+15.0*x**2

        ndx = arange(0, self.n)
        for i in range(0, self.n):
            lb=max(0, i-ml)
            ub=min(self.n-1, i+mu)
            j=where((ndx!=i) & (ndx>=lb) & (ndx<=ub))

            xg=-(1.0+2.0*x)
            self.J[i,j[0]]=xg[j]

class LinearFullRank(MGH):
    """
    Linear full rank test function (m>=n).
    Default m=n.

    Bounds defined for n=5.

    See the :class:`MGH` class for more details.
    """

    name="Linear (full rank)"

    def __init__(self, n=5, m=None):
        if m is None:
            m=n
        MGH.__init__(self, n, m)
        if self.m<self.n:
            raise Exception, DbgMsg("MGH", "Bad n or m.")

        self.initial=ones(self.n)

        if self.n==5:
            self.xl=array([ -0.5,  -2.0,  -2.0,  -2.0,  -2.0])
            self.xh=array([100.0, 100.0, 100.0, 100.0, 100.0])

    def func(self, x):
        xsum=x.sum()
        self.fi[0:self.n]=x-2.0/self.m*xsum-1.0
        if self.m> self.n:
            self.fi[self.n:self.m]=-2.0/self.m*xsum-1.0

    def jac(self, x):
        self.J[:,:]=0.0
        self.J[arange(0,self.n), arange(0,self.n)]=1.0
        self.J+=ones([self.m, self.n])*(-2.0)/self.m

class LinearRank1(MGH):
    """
    Linear rank 1 test function (m>=n).
    Default m=n.

    Bounds defined for n=5.

    See the :class:`MGH` class for more details.
    """

    name="Linear (rank 1)"

    def __init__(self, n=5, m=None):
        if m is None:
            m=n
        MGH.__init__(self, n, m)
        if self.m<self.n:
            raise Exception, DbgMsg("MGH", "Bad n or m.")

        self.initial=ones(self.n)

        if self.n==5:
            self.xl=array([ -0.9,  -2.0,  -2.0,  -2.0,  -2.0])
            self.xh=array([100.0, 100.0, 100.0, 100.0, 100.0])

    def func(self, x):
        xsum=(arange(1,self.n+1)*x).sum()
        self.fi=arange(1,self.m+1)*xsum-1.0

    def jac(self, x):
        ndx=indices([self.m, self.n])
        i=ndx[0]
        j=ndx[1]

        self.J=(i+1.0)*(j+1.0)

class LinearRank1ZeroColumnsAndRows(MGH):
    """
    Linear rank 1 zero columns and rows test function (m>=n).
    Default m=n.

    Bounds defined for n=5.

    See the :class:`MGH` class for more details.
    """

    name="Linear (rank 1) with zero columns and rows"

    def __init__(self, n=5, m=None):
        if m is None:
            m=n
        MGH.__init__(self, n, m)
        if self.m<self.n:
            raise Exception, DbgMsg("MGH", "Bad n or m.")

        self.initial=ones(self.n)

        if self.n==5:
            self.xl=array([  0.0,  -0.4,   0.3,   0.0,   0.0])
            self.xh=array([100.0, 100.0, 100.0, 100.0, 100.0])

    def func(self, x):
        self.fi[0]=-1.0
        self.fi[self.m-1]=-1.0

        self.fi[1:self.m-1]=arange(1,self.m-1)*(arange(2,self.n)*x[1:self.n-1]).sum()-1.0

    def jac(self, x):
        ndx=indices([self.m, self.n])
        i=ndx[0][1:self.m-1,1:self.n-1:]
        j=ndx[1][1:self.m-1,1:self.n-1]

        self.J[1:self.m-1,1:self.n-1]=i*(j+1.0)

class Chebyquad(MGH):
    """
    Chebyquad test function (m>=n).
    Default m=n.

    Bounds defined for n=1, 7, 8, 9, and 10.

    See the :class:`MGH` class for more details.
    """

    name="Chebyquad"

    def __init__(self, n=7, m=None):
        if m is None:
            m=n
        MGH.__init__(self, n, m)
        if self.m<self.n:
            raise Exception, DbgMsg("MGH", "Bad n or m.")

        # chebyt objects are not picklable, poly1d are
        self.T=[]
        self.dT=[]
        for i in range(1,m+1):
            Ti=poly1d(chebyt(i).coeffs)
            self.T.append(Ti)
            self.dT.append(poly1d(Ti.deriv(1).coeffs))

        self.initial=arange(1.0, self.n+1)/(self.n+1)

        if self.n==1:
            self.xl=array([  0.5])
            self.xh=array([100.0])
        elif self.n==7:
            self.xl=array([ 0.0,  0.0, 0.0,   0.0, 0.0, 0.0, 0.0])
            self.xh=array([0.05, 0.23, 0.333, 1.0, 1.0, 1.0, 1.0])
        elif self.n==8:
            self.xl=array([0.0,  0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
            self.xh=array([0.04, 0.2, 0.3, 1.0, 1.0, 1.0, 1.0, 1.0])
        elif self.n==9:
            self.xl=array([0.0, 0.0, 0.1,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            self.xh=array([1.0, 0.2, 0.23, 0.4, 1.0, 1.0, 1.0, 1.0, 1.0])
        elif self.n==10:
            self.xl=array([0.0, 0.1, 0.2, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5])
            self.xh=array([1.0, 0.2, 0.3, 0.4, 0.4, 1.0, 1.0, 1.0, 1.0, 1.0])

    def func(self, x):
        # The interval where the Chebyshev polynomials are observed is [-1,1].
        # We must scale this interval to to [0,1] so x -> 2x-1.
        for i in range(0, self.m):
            self.fi[i]=self.T[i](2.0*x-1.0).sum()*1.0/self.n
            if (i+1) % 2 == 0:
                self.fi[i]+=1.0/((i+1.0)**2-1.0)

    def jac(self, x):
        for i in range(0, self.m):
            self.J[i, :]=2.0*self.dT[i](2.0*x-1.0)*1.0/self.n

class Quadratic(MGH):
    """
    Quadratic test function (m=n).

    See the :class:`MGH` class for more details.
    """
    name="Quadratic"

    def __init__(self, n=2, m=None):
        if m is None:
            m=n
        MGH.__init__(self, n, m)
        if self.m!=self.n:
            raise Exception, DbgMsg("MGH", "Bad n or m.")

        self.initial=ones(self.n)
        self.initial[0]=2.0

        self.xl=0.1*ones(self.n)
        self.xl[0]=1.0
        self.xh=100.0*ones(self.n)

    def func(self, x):
        self.fi=x

    def jac(self, x):
        self.J[range(0, self.n), range(0,self.n)]=ones(self.n)

class Dixon(MGH):
    """
    Dixon function. Minimum is at [1, 1, ..., 1].

    M.A. Wolfe, C. Viazminsky, Supermemory descent methods for unconstrained
    minimization, Journal of Optimization Theory and Applications 18 (1976)
    455-469.

    See the :class:`MGH` class for more details.
    """
    name="Dixon"

    def __init__(self, n=10, m=11):
        MGH.__init__(self, n, m)

        self.initial=zeros(10)-2.0

        self.xl=-5.0*ones(10)
        self.xh=5.0*ones(10)

        if self.n!=10 or self.m!=11:
            raise Exception, DbgMsg("MGH", "Bad n or m.")

    def func(self, x):
        self.fi[0]=1-x[0]
        self.fi[1]=1-x[9]
        self.fi[2:11]=x[0:9]**2-x[1:10]

    # One row contains derivatives wrt n dimensions (has n columns)
    # There is one row per component (m in total)
    def jac(self, x):
        self.J[0,0]=-1.0
        self.J[1,9]=-1.0
        ii=arange(0,9)
        self.J[2+ii,ii]=2*x[ii]
        self.J[2+ii,ii+1]=-1.0


class OrenPower(MGH):
    """
    Oren's power function. Minimum is at [0, 0, ..., 0].

    E. Spedicato, Computational experience with quasi-Newton algorithms for
    minimization problems of moderately large size, Report CISE-N-175,
    CISE Documentation Series, Segrato, 1975.

    Definition from
    D. F. Shanno, Kang-Hoh Phua, Matrix conditioning and nonlinear optimization,
    Mathematical Programming, Volume 14, Issue 1, 1978, pp 149-160.

    See the :class:`MGH` class for more details.
    """
    name="OrenPower"

    def __init__(self, n=2, m=None):
        if m is None:
            m=n
        MGH.__init__(self, n, m)
        self.initial=zeros(n)+1.0
        self.xl=-5.0*ones(n)
        self.xh=5.0*ones(n)

    def func(self, x):
        self.fi=x*x*arange(1,self.n+1)

    # One row contains derivatives wrt n dimensions (has n columns)
    # There is one row per component (m in total)
    def jac(self, x):
        self.J[:,:]=np.diag(2*x*arange(1,self.n+1))


MGHsuite=[
    Rosenbrock,
    FreudensteinAndRoth,
    PowellBadlyScaled,
    BrownBadlyScaled,
    Beale,
    JennrichAndSampson,
    HelicalValley,
    Bard,
    Gaussian,
    Meyer,
    GulfResearchAndDevelopement,
    Box3D,
    PowellSingular,
    Wood,
    KowalikAndOsborne,
    BrownAndDennis,
    Osborne1,
    BiggsEXP6,
    Osborne2,
    Watson,
    ExtendedRosenbrock,
    ExtendedPowellSingular,
    PenaltyI,
    PenaltyII,
    VariablyDimensioned,
    Trigonometric,
    BrownAlmostLinear,
    DiscreteBoundaryValue, 			# No bounds
    DiscreteIntegralEquation, 		# No bounds
    BroydenTridiagonal, 			# No bounds
    BroydenBanded, 				# No bounds
    LinearFullRank,
    LinearRank1,
    LinearRank1ZeroColumnsAndRows,
    Chebyquad,	# Agrees with Mathematica for n=m=9, but not with Matlab for n=m=7. Probably Matlab implementation is broken.
    Quadratic,
    Dixon,
    OrenPower
]
"""
A list holding references to all function classes in this module. 
"""



































