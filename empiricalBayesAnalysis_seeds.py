import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, BayesianRidge
from sklearn.datasets import make_low_rank_matrix
import statsmodels.formula.api as smf
import torch
import pyro
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.optim import ClippedAdam, Adam
import pyro.infer.autoguide as autoguide
import pyro.poutine as poutine
from bayesian_lasso2 import BayesianLasso

pyro.enable_validation(True)
torch.autograd.set_detect_anomaly(True)

def final_elbo(nsamples=1000):    
    closs=0
    for i in range(nsamples):
        closs+=svi.evaluate_loss(tX, ty)
    return closs/nsamples

nullmodel = False
testtwosided=True

X=pd.read_csv('data2.csv').drop('CODE',1).apply(stats.zscore)
y=pd.DataFrame(X['BTG_T1'])
X.drop(['BTG_T1','DTS_T1'],1,inplace=True)

reslist=[]
lpelbo=[]
vbelbo=[]

for s in range(10):
    
    torch.manual_seed(s)
    np.random.seed(s)
    pyro.clear_param_store()
    
    ridgebayes=BayesianRidge().fit(X,y)
    linreg=LinearRegression().fit(X,y)
    postsamples=np.random.multivariate_normal(ridgebayes.coef_,ridgebayes.sigma_, 10000)
    bayesridgeres=pd.DataFrame(np.vstack((np.quantile(postsamples,0.025,0),np.quantile(postsamples,0.05,0),postsamples.mean(0),np.quantile(postsamples,0.95,0),np.quantile(postsamples,0.975,0))).T).set_index(X.columns)
    
    tX=torch.as_tensor(X.to_numpy()).float()
    ty=torch.as_tensor(y.to_numpy().squeeze()).float()
    
    posteriorguide='laplaceapprox'
    
    model = BayesianLasso(nullmodel)
    if nullmodel:
        guide=autoguide.AutoDelta(model)
    else:
        guide=autoguide.AutoGuideList(model)
        if posteriorguide=='multivargauss':
            guide.append(autoguide.AutoMultivariateNormal(poutine.block(model, expose=["weights"])))
        elif posteriorguide=='independentgauss':
            guide.append(autoguide.AutoNormal(poutine.block(model, expose=["weights"])))
        else:
            guide.append(autoguide.AutoLaplaceApproximation(poutine.block(model, expose=["weights"])))        
        guide.append(autoguide.AutoDelta(poutine.block(model, expose=['sigma','alpha'])))
    
    
    n_steps = 1500
    initial_lr = 0.01
    gamma = 0.1
    lrd = gamma ** (1 / n_steps)
    optimizer = ClippedAdam({'lr': initial_lr, 'lrd': lrd})
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO(num_particles=5))
    print('Performing SVI\n')
    for step in range(n_steps):
        elbo=svi.step(tX, ty)
        if step % 50 == 0:
            print(step,elbo)
    
    if posteriorguide=='multivargauss':
        postsamples=np.array(guide[0].get_posterior().sample((10000,)))
    elif posteriorguide=='independentgauss':
        pass
    else:
        map_estimates=guide[0].loc
        map_model = pyro.condition(model, data={"weights":map_estimates, "alpha":guide[1].median()['alpha'], "sigma":guide[1].median()['sigma']})
        trace = poutine.trace(map_model).get_trace(tX,ty)
        loss = -trace.log_prob_sum()
        dys = torch.autograd.grad(loss, map_estimates, create_graph=True)
        flat_dy = torch.cat([dy.reshape(-1) for dy in dys])
        H = []
        for dyi in flat_dy:
            Hi = torch.cat([Hij.reshape(-1) for Hij in torch.autograd.grad(dyi, map_estimates, retain_graph=True)])
            H.append(Hi)
        H = torch.stack(H)
        covariance = H.inverse().numpy()
        postsamples=np.random.multivariate_normal(map_estimates.detach().numpy(),covariance, 10000)
    
    
    bayeslassores_lp=pd.DataFrame(np.vstack((np.quantile(postsamples,0.025,0),np.quantile(postsamples,0.05,0),postsamples.mean(0),np.quantile(postsamples,0.95,0),np.quantile(postsamples,0.975,0))).T).set_index(X.columns)
    lpelbo.append(final_elbo(500))
    
    pyro.clear_param_store()
    posteriorguide='multivargauss'
    
    model = BayesianLasso(nullmodel)
    if nullmodel:
        guide=autoguide.AutoDelta(model)
    else:
        guide=autoguide.AutoGuideList(model)
        if posteriorguide=='multivargauss':
            guide.append(autoguide.AutoMultivariateNormal(poutine.block(model, expose=["weights"])))
        elif posteriorguide=='independentgauss':
            guide.append(autoguide.AutoNormal(poutine.block(model, expose=["weights"])))
        else:
            guide.append(autoguide.AutoLaplaceApproximation(poutine.block(model, expose=["weights"])))        
        guide.append(autoguide.AutoDelta(poutine.block(model, expose=['sigma','alpha'])))
    
    
    n_steps = 2500
    initial_lr = 0.01
    gamma = 0.1
    lrd = gamma ** (1 / n_steps)
    optimizer = ClippedAdam({'lr': initial_lr, 'lrd': lrd})
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO(num_particles=5))

    print('Performing SVI\n')
    for step in range(n_steps):
        elbo=svi.step(tX, ty)
        if step % 50 == 0:
            print(step,elbo)
    
    if posteriorguide=='multivargauss':
        postsamples=np.array(guide[0].get_posterior().sample((10000,)))
    elif posteriorguide=='independentgauss':
        pass
    else:
        map_estimates=guide[0].loc
        map_model = pyro.condition(model, data={"weights":map_estimates, "alpha":guide[1].median()['alpha'], "sigma":guide[1].median()['sigma']})
        trace = poutine.trace(map_model).get_trace(tX,ty)
        loss = -trace.log_prob_sum()
        dys = torch.autograd.grad(loss, map_estimates, create_graph=True)
        flat_dy = torch.cat([dy.reshape(-1) for dy in dys])
        H = []
        for dyi in flat_dy:
            Hi = torch.cat([Hij.reshape(-1) for Hij in torch.autograd.grad(dyi, map_estimates, retain_graph=True)])
            H.append(Hi)
        H = torch.stack(H)
        covariance = H.inverse().numpy()
        postsamples=np.random.multivariate_normal(map_estimates.detach().numpy(),covariance, 10000)
    
    
    bayeslassores_mg=pd.DataFrame(np.vstack((np.quantile(postsamples,0.025,0),np.quantile(postsamples,0.05,0),postsamples.mean(0),np.quantile(postsamples,0.95,0),np.quantile(postsamples,0.975,0))).T).set_index(X.columns)
    vbelbo.append(final_elbo(500))
       
    predlist=''
    predlist=predlist.join([x+'+' for x in X.columns])[:-1]
    ols=smf.ols(formula=y.columns.values[0]+'~'+predlist,data=y.join(X)).fit()
    ols.summary()
    
    res = pd.DataFrame()
    
    res['Unregularized Mean'] = ols.params[1:].round(3).astype(str)
    olsconfint=ols.conf_int(0.05)
    res['Unregularized 95% CI'] = ['[' + str(olsconfint.loc[x,0].round(3)) + ' ' + str(olsconfint.loc[x,1].round(3)) + ']' for x in olsconfint.index[1:]]
    res['BayesRidge Mean'] = bayesridgeres.iloc[:,2].round(3).astype(str)
    res['BayesRidge 95% CI'] = ['[' + str(bayesridgeres.loc[x,0].round(3)) + ' ' + str(bayesridgeres.loc[x,4].round(3)) + ']' for x in bayesridgeres.index]
    res['BayesLasso local Mean'] = bayeslassores_lp.iloc[:,2].round(3).astype(str)
    res['BayesLasso local 95% CI'] = ['[' + str(bayeslassores_lp.loc[x,0].round(3)) + ' ' + str(bayeslassores_lp.loc[x,4].round(3)) + ']' for x in bayeslassores_lp.index]
    res['BayesLasso global Mean'] = bayeslassores_mg.iloc[:,2].round(3).astype(str)
    res['BayesLasso global 95% CI'] = ['[' + str(bayeslassores_mg.loc[x,0].round(3)) + ' ' + str(bayeslassores_mg.loc[x,4].round(3)) + ']' for x in bayeslassores_mg.index]
    reslist.append(res)

vblassomeans=np.array([x['BayesLasso global Mean'] for x in reslist])
lplassomeans=np.array([x['BayesLasso local Mean'] for x in reslist])
ridgemeans=np.array([x['BayesRidge Mean'] for x in reslist])