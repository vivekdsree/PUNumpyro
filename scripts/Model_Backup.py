def model(id=ids, data1 =None, data2=None, data3=None, data4=None):
    """
    :id tuple(jnp.ndarray)  : Observed time indices of keratinocytes, neutrophils, macrophages and TNFa
    :data tuple(jnp.ndarray): Observed data points of keratinocytes, neutrophils, macrophages and TNFa
    """
    # v0 = numpyro.deterministic("v0",jnp.array([0.18,10.0,5.0,2.0,1.0,10.0,5.0,2.0,0.15,0.2,1.0,1.0]))
    v0 = numpyro.sample(
        "v0",
        dist.Uniform(
            low =jnp.array([0.1, 5.0,  2.0, 1.0, 0.99, 5.0,  4.0, 1.0, 0.149, 0.199, 0.99, 0.99]),
            high=jnp.array([0.2, 15.0, 8.0, 8.0, 1.01, 15.0, 8.0, 4.0, 0.151, 0.201, 1.01, 1.01]),
        ),
    )
    lkTa =numpyro.sample("lkTa", dist.Uniform(low=0.4, high=0.6))
    lkI1b =numpyro.deterministic("lkI1b", 0.07)
    lkR =numpyro.deterministic("lkR", 0.07)
    lkTb =numpyro.deterministic("lkTb", 0.2)

    lnKC =numpyro.deterministic("lnKC", 2.0)
    lnDa =numpyro.sample("lnDa", dist.Uniform(low=1.8, high=2.1))

    Tahf =numpyro.sample("Tahf", dist.Uniform(low=8.0, high=16.0))
    lm1Ta =numpyro.sample("lm1Ta", dist.Uniform(low=.75, high=1.25))
    I1bhfm1 =numpyro.sample("I1bhfm1", dist.Uniform(low=10.0, high=12.0))
    lm1I1b =numpyro.sample("lm1I1b", dist.Uniform(low=0.8, high=1.6))
    lmKcm1 =numpyro.deterministic("lmKcm1", 1.0)
    Tbhfm1 =numpyro.sample("Tbhfm1", dist.Uniform(low=7.0, high=8.5))
    lm1Tb =numpyro.deterministic("lm1Tb", 0.1)
    I1bhfm2 =numpyro.sample("I1bhfm2", dist.Uniform(low=7.0, high=8.0))
    lm2I1b =numpyro.sample("lm2I1b", dist.Uniform(low=1.1, high=1.2))
    Tbhfm2 =numpyro.deterministic("Tbhfm2", 10.0)
    lm2Tb =numpyro.deterministic("lm2Tb", 0.2)
    alpha_m1n =numpyro.deterministic("alpha_m1n", 0.0001)
    alpha_m1m2 =numpyro.deterministic("alpha_m1m2", 0.0001)

    sig1 =numpyro.sample("sig1", dist.HalfCauchy(scale=0.01))
    sig2 =numpyro.sample("sig2", dist.HalfCauchy(scale=0.1))
    sig3 =numpyro.sample("sig3", dist.HalfCauchy(scale=0.1))
    sig4 =numpyro.sample("sig4", dist.HalfCauchy(scale=0.1))

    # sig1 =numpyro.deterministic("sig1", 0.2)
    # sig2 =numpyro.deterministic("sig2", 3.0)
    # sig3 =numpyro.deterministic("sig3", 2.0)
    # sig4 =numpyro.deterministic("sig4", 3.0)

    par = jnp.array([lkTa, lkI1b,lkR, lkTb,lnKC, lnDa,Tahf, lm1Ta, I1bhfm1,lm1I1b,
                        lmKcm1, Tbhfm1, lm1Tb,I1bhfm2,lm2I1b, Tbhfm2,lm2Tb,alpha_m1n,
                        alpha_m1m2])
    times = jnp.linspace(0,480,480)

    ode_solution = acute_solver(par,times,v0)
    #rhok_sol,rhon_sol,rhom_sol,ta_sol
    sol = numpyro.deterministic("sol",ode_solution)
    
    y_rhok = numpyro.sample('y_rhok', dist.TruncatedNormal(ode_solution[0, ids[0]], sig1, low=0,high=1.0), obs=data1)
    y_rhon = numpyro.sample('y_rhon', dist.TruncatedNormal(ode_solution[1, ids[1]], sig2,low=0.0), obs=data2)
    y_rhom = numpyro.sample('y_rhom', dist.TruncatedNormal(ode_solution[2, ids[2]], sig3,low=0.0), obs=data3)
    y_ta   = numpyro.sample('y_ta', dist.TruncatedNormal(ode_solution[3, ids[3]], sig4,low=0.0), obs=data4)

    # #For sampling posterior dynamics
    # rhok_dyn = numpyro.sample('rhok_dyn', dist.Normal(ode_solution[0,:], sig1))
    # rhon_dyn = numpyro.sample('rhon_dyn', dist.Normal(ode_solution[1,:], sig2))
    # rhom_dyn = numpyro.sample('rhom_dyn', dist.Normal(ode_solution[2,:], sig3))
    # ta_dyn = numpyro.sample('ta_dyn', dist.Normal(ode_solution[3,:], sig4))

    
