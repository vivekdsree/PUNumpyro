from jax import jit
from jax import numpy  as jnp
from jax.experimental.ode import odeint

## Auxiliary Functions ##
@jit
def Hill(X, X0, nX):
    return X**nX/(X0**nX+X**nX)
@jit
def Hev(X, X0, k):
    return 1.0/(1.0+jnp.exp(-2.0*k*(X-X0)))
@jit 
def Hsu1(X, X0, nX, lamb):    
    return lamb + (1.0-lamb)/(1.0 + (X/X0)**nX)
@jit
def Gh(X, X0, nX, lamb):
    return (lamb-1.0)*(1.0-Hill(X, X0, nX))+1.0

## Dynamics ##
@jit
def rhok_dot(X, param):
    """
    
    """
    rhok = X[0]; rhon = X[1]; rhom1 = X[2]; rhom2 = X[3]; O2 = X[4];
    Ta = X[5]; I1b = X[6]; Tb = X[7]; XO = X[8]; ROS = X[9]; Da = X[10]; KC = X[11]

    p_k = param[0]
    d_k = param[1]
    lkTa = param[2]
    lkI1b = param[3]
    lkR = param[4]
    lkTb = param[5]

    '''Hill functions Tb=10,il=15,Ta=20'''
#    return p_k*Hill(O2,0.2,4.0)*Hsu1(Tb,10,3,lkTb)*rhok*(1.0-rhok/2.0)-d_k*(1+lkTa*Hill(Ta,20.0,3)+lkI1b*Hill(I1b,10,3)+lkR*Hill(ROS,0.5,3))*rhok
    return p_k*Hill(O2,0.2,4.0)*rhok*(1.0-rhok/2.0)-d_k*(1+lkTa*Hill(Ta,20.0,3)+lkI1b*Hill(I1b,10,3)+lkR*Hill(ROS,0.5,3))*rhok

@jit
def rhon_dot(X, param):
    rhok = X[0]; rhon = X[1]; rhom1 = X[2]; rhom2 = X[3]; O2 = X[4];
    Ta = X[5]; I1b = X[6]; Tb = X[7]; XO = X[8]; ROS = X[9]; Da = X[10]; KC = X[11]

    p_n = param[0]
    lnDa = param[1]
    lnKC = param[2]
    d_n = param[3]
    alpha_m1n = param[4]
    '''Changed rho_n,max to 80'''
    return p_n*(1+lnDa*Hill(Da,0.3,3)+lnKC*Hill(KC,0.5,3))*rhon*(1.0-rhon/80.)-rhon*(d_n+alpha_m1n*rhom1)

@jit
def rhom1_dot(X, param):
    rhok = X[0]; rhon = X[1]; rhom1 = X[2]; rhom2 = X[3]; O2 = X[4];
    Ta = X[5]; I1b = X[6]; Tb = X[7]; XO = X[8]; ROS = X[9]; Da = X[10]; KC = X[11]

    p_m = param[0]
    Tahf = param[1]
    lmTa = param[2]
    Ilhf = param[3]
    lmI1b = param[4]
    lmKC = param[5]
    Tbhf1 = param[6]
    lm1Tb = param[7]
    d_m1 = param[8]
    alpha = param[9]
    beta = param[10]
    '''Tahf = 20, Ilhf=15,Tbhf1=10'''
    return p_m*(1+lmTa*Hill(Ta,Tahf,3)+lmI1b*Hill(I1b,Ilhf,3)+lmKC*Hill(KC,0.5,3))*Hsu1(Tb,Tbhf1,3,lm1Tb)*rhom1*(1.0-rhom1/60)-d_m1*rhom1-alpha*rhom1*rhon+beta*rhom2

@jit
def rhom2_dot(X, param):
    rhok = X[0]; rhon = X[1]; rhom1 = X[2]; rhom2 = X[3]; O2 = X[4];
    Ta = X[5]; I1b = X[6]; Tb = X[7]; XO = X[8]; ROS = X[9]; Da = X[10]; KC = X[11]

    p_m = param[0]
    Ilhf = param[1]
    lmI1b = param[2]
    Tbhf = param[3]
    lmTb = param[4]
    d_m2 = param[5]
    alpha = param[6]
    beta = param[7]

    '''Ilhf=15,Tbhf=10'''
    return p_m*(1+lmI1b*Hill(I1b,Ilhf,3)+lmTb*Hill(Tb,Tbhf,3))*rhom2*(1.0-rhom2/20)+alpha*rhom1*rhon-d_m2*rhom2-beta*rhom2

## CHEMICALS
@jit
def O2_dot(X, param):
    rhok = X[0]; rhon = X[1]; rhom1 = X[2]; rhom2 = X[3]; O2 = X[4];
    Ta = X[5]; I1b = X[6]; Tb = X[7]; XO = X[8]; ROS = X[9]; Da = X[10]; KC = X[11]

    c_k = param[0]
    D_O2 = param[1]
    O2d = param[2]

    return -c_k*O2*rhok + D_O2*(O2d-O2)

@jit
def Ta_dot(X, param):
    rhok = X[0]; rhon = X[1]; rhom1 = X[2]; rhom2 = X[3]; O2 = X[4];
    Ta = X[5]; I1b = X[6]; Tb = X[7]; XO = X[8]; ROS = X[9]; Da = X[10]; KC = X[11]

    p_Ta = param[0]
    d_Ta = param[1]

    return p_Ta*(rhon+0.1*rhom1)-d_Ta*Ta


@jit
def I1b_dot(X, param):
    rhok = X[0]; rhon = X[1]; rhom1 = X[2]; rhom2 = X[3]; O2 = X[4];
    Ta = X[5]; I1b = X[6]; Tb = X[7]; XO = X[8]; ROS = X[9]; Da = X[10]; KC = X[11]

    p_I1b = param[0]
    d_I1b = param[1]

    return p_I1b*rhom1-d_I1b*I1b

@jit
def Tb_dot(X, param):
    rhok = X[0]; rhon = X[1]; rhom1 = X[2]; rhom2 = X[3]; O2 = X[4];
    Ta = X[5]; I1b = X[6]; Tb = X[7]; XO = X[8]; ROS = X[9]; Da = X[10]; KC = X[11]

    p_Tb = param[0]
    d_Tb = param[1]

    return p_Tb*rhom2-d_Tb*Tb

@jit
def XO_dot(X, param):
    rhok = X[0]; rhon = X[1]; rhom1 = X[2]; rhom2 = X[3]; O2 = X[4];
    Ta = X[5]; I1b = X[6]; Tb = X[7]; XO = X[8]; ROS = X[9]; Da = X[10]; KC = X[11]

    Tau_XO = param[0]
    d_XO = param[1]

    return (1.0/Tau_XO)*(1.0-Hill(O2,0.25,3))*(1.0-XO)-d_XO*XO

@jit
def ROS_dot(X, param):
    rhok = X[0]; rhon = X[1]; rhom1 = X[2]; rhom2 = X[3]; O2 = X[4];
    Ta = X[5]; I1b = X[6]; Tb = X[7]; XO = X[8]; ROS = X[9]; Da = X[10]; KC = X[11]

    p_RXO = param[0]
    p_Rn = param[1]
    d_R = param[2]

    return p_RXO*Hill(XO,0.25,2)*O2 + p_Rn*Hill(O2,0.5,4)*rhon - d_R*ROS

@jit
def Da_dot(X, param):
    rhok = X[0]; rhon = X[1]; rhom1 = X[2]; rhom2 = X[3]; O2 = X[4];
    Ta = X[5]; I1b = X[6]; Tb = X[7]; XO = X[8]; ROS = X[9]; Da = X[10]; KC = X[11]

    p_a = param[0]
    d_a = param[1]
    lkTa = param[2]
    lkI1b = param[3]
    lkR = param[4]
    '''The hillufnction values at 20 and 15 respectively for Ta and Il1b'''
    return p_a*(lkTa*Hill(Ta,20,3,)+lkI1b*Hill(I1b,15,3,)+lkR*Hill(ROS,0.6,3,))*rhok - d_a*Da


@jit
def KC_dot(X, param):
    rhok = X[0]; rhon = X[1]; rhom1 = X[2]; rhom2 = X[3]; O2 = X[4];
    Ta = X[5]; I1b = X[6]; Tb = X[7]; XO = X[8]; ROS = X[9]; Da = X[10]; KC = X[11]

    p_kc = param[0]
    d_kc = param[1]
    return p_kc*(Hill(1-O2,0.4,3))*rhok - d_kc*KC


def f(vals,t,par):
    """"This is the ODE system.
    Function takes jnp or numpy arrays as input variables
    Args:
        vals (vector): system state
        t (float): current time (optional)
        par (vector): parameters of the ODEs
    Returns:
        dydt (vector): result of the ODEs
    """
    lkTa = par[0]
    lkI1b = par[1]
    lkR = par[2]
    lkTb = par[3]
    ## neutrophils
    lnKC = par[4]
    lnDa = par[5]
    ## macrophages
    Tahf = par[6]
    lm1Ta = par[7]
    I1bhfm1 = par[8]
    lm1I1b = par[9]
    lmKcm1 = par[10]
    Tbhfm1 = par[11]
    lm1Tb = par[12]
    I1bhfm2 = par[13]
    lm2I1b = par[14]
    Tbhfm2 = par[15]
    lm2Tb = par[16]
    alpha_m1n = par[17]  #Efferocytosis
    alpha_m1m2 = par[18]

    #Keratinocytes
    p_k =  0.028 # 1/hr (ref: Takei et al)
    d_k = p_k/2.0 # Based on the steady state equirement

    ## neutrophils
    # 1/h # See PLOS_Notes
    d_n = 0.141 # 1/hr
    p_n = (d_n+alpha_m1n*1.)/(1.-1./80.0)    # For steady state :See PLOS_Notes
    ## macrophages

    ## CHECK
    d_m1 = 0.009*4     # 0.1/hr
    d_m2 = 0.009*10   # 1/hr

    beta_m1m2 =0.0


    p_m1 = (d_m1+alpha_m1m2*1.*1)/(1.-1/60.0)  # For steady state :See PLOS_Notes
    p_m2 = (-alpha_m1m2*1.*1.+(d_m2+beta_m1m2)*1.0)/(1.0*(1.0-1.0/60.)) # For steady state :See PLOS_Notes

    ## O2
    c_O2k = 20.00 # 1/hr   #
    D_O2 = 22.68 # 1/hr
    O2d = 1.88
    ## XO
    Tau_XO = 4.0 # hr
    d_XO = 0.021 # 0.078 #1/hr
    p_RXO = 0.048 # 1/hr
    ## ROS
    pRn =  4.2e-4 # 1/hr
    dR = 0.084 # 1/hr
    ## Da
    d_a = 0.144 # 1/hr
    p_a = 0.23 # 1/hr


    ##----------------------------------------------------##
    ## Assign parameters for each eqn
    ##----------------------------------------------------##
    param_rhok = [p_k,d_k,lkTa,lkI1b, lkR, lkTb]
    param_rhon= [p_n,lnDa,lnKC,d_n,alpha_m1n]
    param_m1 = [p_m1,Tahf,lm1Ta,I1bhfm1,lm1I1b,lmKcm1,Tbhfm1,lm1Tb,d_m1,alpha_m1m2,beta_m1m2]
    param_m2 = [p_m2,I1bhfm2,lm2I1b,Tbhfm2,lm2Tb,d_m2,alpha_m1m2,beta_m1m2]
    param_Ox = [c_O2k,D_O2,O2d]
    param_Ta = [0.144,0.144]
    param_Il = [0.144,0.144]
    param_Tb = [0.144,0.144]
    param_Xo = [Tau_XO,d_XO]
    param_Ro = [p_RXO,pRn,dR]
    param_Da = [p_a,d_a,lkTa,lkI1b,lkR]
    param_Kc = [0.144,0.144]

    X = vals
    
    ydot = [rhok_dot(X,param_rhok),rhon_dot(X,param_rhon),rhom1_dot(X,param_m1),\
    rhom2_dot(X,param_m2),0.0,Ta_dot(X,param_Ta),I1b_dot(X,param_Il),Tb_dot(X,param_Tb),\
    XO_dot(X,param_Xo),ROS_dot(X,param_Ro),Da_dot(X,param_Da),KC_dot(X,param_Kc)]

    return jnp.stack(ydot)


@jit
def acute_solver(par,time):
    '''
    Solve the ODE system for a given set of parameters and return the observed quantities
    '''
    v0 = jnp.array([0.18,10.0,5.0,2.0,1.0,10.0,5.0,2.0,0.15,0.2,1.0,1.0]) # initial condition
    odesol = odeint(f,v0,time,par)
    [rhok_sol,rhon_sol,rhom_sol,ta_sol,tb_sol,kc_sol] = [odesol[:,0],odesol[:,1],odesol[:,2]+odesol[:,3],odesol[:,5],odesol[:,7],odesol[:,-1]]
    return jnp.array([rhok_sol,rhon_sol,rhom_sol,ta_sol,tb_sol,kc_sol])