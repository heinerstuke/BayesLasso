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
import pickle

pyro.enable_validation(True)
torch.autograd.set_detect_anomaly(True)

def final_elbo(nsamples=1000):    
    closs=0
    for i in range(nsamples):
        closs+=svi.evaluate_loss(tX, ty)
    return closs/nsamples

testtwosided=True

X=pd.read_csv('data2.csv').drop('CODE',1).apply(stats.zscore)
y=pd.DataFrame(X['BTG_T1'])
X.drop(['BTG_T1','DTS_T1'],1,inplace=True)

reslist=[]
lpelbo=[]
vbelbo=[]

for s in range(10):
    # Set random set and clear parameter store
    torch.manual_seed(s)
    np.random.seed(s)
    pyro.clear_param_store()
    
    # ----- Bayesian Ridge ------
    
    ridgebayes=BayesianRidge().fit(X,y)
    linreg=LinearRegression().fit(X,y)

    # Bayesian Ridge: Calculate credible interval and probability mass in null region per sampling   
    postsamples=np.random.multivariate_normal(ridgebayes.coef_,ridgebayes.sigma_, 10000)
    bayesridgeres=pd.DataFrame(np.vstack((np.quantile(postsamples,0.025,0),np.quantile(postsamples,0.05,0),postsamples.mean(0),np.quantile(postsamples,0.95,0),np.quantile(postsamples,0.975,0),np.sum((postsamples<0.1) & (postsamples > -0.1),0)/100)).T).set_index(X.columns)
    
    # Bayesian Ridge: Calculate credible interval and probability mass in null region explicitly
    explci=[]
    explnullmass=[]
    for v in range(ridgebayes.n_features_in_):
        marginaldist=stats.norm(ridgebayes.coef_[v],np.sqrt(ridgebayes.sigma_[v,v]))
        explci.append(marginaldist.interval(0.95))
        explnullmass.append((marginaldist.cdf(0.1)-marginaldist.cdf(-0.1))*100)
    explres=pd.DataFrame([np.array(explci)[:,0],np.array(explci)[:,1],np.array(explnullmass)]).T.set_index(X.columns)
    
    # Bayesian Ridge: Join results    
    bayesridgeres=pd.concat([bayesridgeres,explres],axis=1)
    bayesridgeres.columns=np.arange(len(bayesridgeres.columns))     

 
    # ----- Bayesian Lasso with Laplace approximation ------
   
    # Bayesian Lasso with Laplace approximation: setup
    tX=torch.as_tensor(X.to_numpy()).float()
    ty=torch.as_tensor(y.to_numpy().squeeze()).float()
        
    model = BayesianLasso(False)
    guide=autoguide.AutoGuideList(model)
    guide.append(autoguide.AutoLaplaceApproximation(poutine.block(model, expose=["weights"])))        
    guide.append(autoguide.AutoDelta(poutine.block(model, expose=['sigma','alpha'])))
    
    # Bayesian Lasso with Laplace approximation: ELBO maximization for MAP model
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
        
    # Bayesian Lasso with Laplace approximation: Perform laplace approximation at MAP estimates
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
 
    # Bayesian Lasso with Laplace approximation: Calculate credible interval and probability mass in null region per sampling   
    postsamples=np.random.multivariate_normal(map_estimates.detach().numpy(),covariance, 10000)
    bayeslassores_lp=pd.DataFrame(np.vstack((np.quantile(postsamples,0.025,0),np.quantile(postsamples,0.05,0),postsamples.mean(0),np.quantile(postsamples,0.95,0),np.quantile(postsamples,0.975,0),np.sum((postsamples<0.1) & (postsamples > -0.1),0)/100)).T).set_index(X.columns)

    # Bayesian Lasso with Laplace approximation: Calculate credible interval and probability mass in null explicitly      
    explci=[]
    explnullmass=[]
    for v in range(ridgebayes.n_features_in_):
        marginaldist=stats.norm(map_estimates.detach().numpy()[v],np.sqrt(covariance[v,v]))
        explci.append(marginaldist.interval(0.95))
        explnullmass.append((marginaldist.cdf(0.1)-marginaldist.cdf(-0.1))*100)
    explres=pd.DataFrame([np.array(explci)[:,0],np.array(explci)[:,1],np.array(explnullmass)]).T.set_index(X.columns)
    
    # Bayesian Lasso with Laplace approximation: Join results
    bayeslassores_lp=pd.concat([bayeslassores_lp,explres],axis=1)   
    bayeslassores_lp.columns=np.arange(len(bayeslassores_lp.columns))     
    lpelbo.append(final_elbo(500))
    
    # ----- Bayesian Lasso with global appproximation ------
    
    
    # Bayesian Lasso with global appproximation: setup
    pyro.clear_param_store()    
    model = BayesianLasso(False)
    guide=autoguide.AutoGuideList(model)
    guide.append(autoguide.AutoMultivariateNormal(poutine.block(model, expose=["weights"])))      
    guide.append(autoguide.AutoDelta(poutine.block(model, expose=['sigma','alpha'])))
    
    # Bayesian Lasso with global approximation: ELBO maximization
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
    
    # Bayesian Lasso with global approximation: Calculate credible interval and probability mass in null region per sampling   
    postsamples=np.array(guide[0].get_posterior().sample((10000,)))
    bayeslassores_mg=pd.DataFrame(np.vstack((np.quantile(postsamples,0.025,0),np.quantile(postsamples,0.05,0),postsamples.mean(0),np.quantile(postsamples,0.95,0),np.quantile(postsamples,0.975,0),np.sum((postsamples<0.1) & (postsamples > -0.1),0)/100)).T).set_index(X.columns)
    
    # Bayesian Lasso with global approximation: Calculate credible interval and probability mass in null explicitly      
    explci=[]
    explnullmass=[]
    for v in range(ridgebayes.n_features_in_):
        marginaldist=stats.norm(guide[0].get_posterior().loc.detach().numpy()[v],guide[0].get_posterior().scale_tril.detach().numpy()[v,v])
        explci.append(marginaldist.interval(0.95))
        explnullmass.append((marginaldist.cdf(0.1)-marginaldist.cdf(-0.1))*100)
    explres=pd.DataFrame([np.array(explci)[:,0],np.array(explci)[:,1],np.array(explnullmass)]).T.set_index(X.columns)
 
    # Bayesian Lasso with global approximation: Join results
    bayeslassores_mg=pd.concat([bayeslassores_mg,explres],axis=1)
    bayeslassores_mg.columns=np.arange(len(bayeslassores_mg.columns))     
    vbelbo.append(final_elbo(500))
       
    
    # Create results table
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
    res['BayesRidge neglectible prob'] = bayesridgeres.iloc[:,5].round(3).astype(str) + '%'
    res['BayesLasso local Mean'] = bayeslassores_lp.iloc[:,2].round(3).astype(str)
    res['BayesLasso local 95% CI'] = ['[' + str(bayeslassores_lp.loc[x,0].round(3)) + ' ' + str(bayeslassores_lp.loc[x,4].round(3)) + ']' for x in bayeslassores_lp.index]
    res['BayesLasso local neglectible prob'] = bayeslassores_lp.iloc[:,5].round(3).astype(str) + '%'
    res['BayesLasso global Mean'] = bayeslassores_mg.iloc[:,2].round(3).astype(str)
    res['BayesLasso global 95% CI'] = ['[' + str(bayeslassores_mg.loc[x,0].round(3)) + ' ' + str(bayeslassores_mg.loc[x,4].round(3)) + ']' for x in bayeslassores_mg.index]
    res['BayesLasso global neglectible prob'] = bayeslassores_mg.iloc[:,5].round(3).astype(str) + '%'
    reslist.append(res)

vblassomeans=np.array([x['BayesLasso global Mean'] for x in reslist])
lplassomeans=np.array([x['BayesLasso local Mean'] for x in reslist])
ridgemeans=np.array([x['BayesRidge Mean'] for x in reslist])
pickle.dump(reslist,open( "ebseedsres.p", "wb" ))