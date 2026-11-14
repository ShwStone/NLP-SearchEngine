"""Train/export the pre-CE same-cost Router with train-only grouped OOF."""
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold
from evaluation.prece_router import FEATURE_NAMES,extract_features
from evaluation.dureader_policy_features import rerank
from evaluation.prepare_bm25_router import metrics

def read_qrels(path):
 out={}
 for line in Path(path).read_text().splitlines():
  q,_,d,g=line.split();out.setdefault(q,{})[d]=float(g)
 return out

def target(row,labels,cheap=20,heavy=40):
 rr=dict(row);rr[f'ce{cheap}']={'scores':(row['ce28']['scores'] if cheap<=28 else row['ce40']['scores'])[:cheap]}
 return metrics(rerank(row,heavy),labels[row['query_id']])['ndcg']-metrics(rerank(rr,cheap),labels[row['query_id']])['ndcg']

def export_model(model,threshold,out,cheap,heavy,training_queries):
 trees=[]
 for est in model.estimators_:
  t=est.tree_; trees.append({'feature':t.feature.astype(int).tolist(),'threshold':t.threshold.astype(float).tolist(),'left':t.children_left.astype(int).tolist(),'right':t.children_right.astype(int).tolist(),'value':t.value[:,0,0].astype(float).tolist()})
 spec={'schema_version':1,'kind':'pre-ce-extra-trees-regressor','feature_names':list(FEATURE_NAMES),'cheap_budget':cheap,'heavy_budget':heavy,'threshold':float(threshold),'trees':trees,'training_seed':17,'training_queries':int(training_queries),'model_selection':'fixed hyperparameters selected before holdout'}
 Path(out).write_text(json.dumps(spec,ensure_ascii=False)+'\n')

def run(train_scores,train_qrels,out,cheap=20,heavy=40):
 rows=[json.loads(line) for line in Path(train_scores).open()]; labels=read_qrels(train_qrels)
 rows=[r for r in rows if r['query_id'] in labels]; X=np.stack([extract_features(r) for r in rows]); y=np.asarray([target(r,labels,cheap,heavy) for r in rows]); groups=np.asarray([r['query_id'] for r in rows]); oof=np.zeros(len(rows)); folds=[]
 for fold,(fit,hold) in enumerate(GroupKFold(5).split(X,y,groups)):
  m=ExtraTreesRegressor(n_estimators=800,min_samples_leaf=5,max_features=.7,random_state=17,n_jobs=-1).fit(X[fit],y[fit]);oof[hold]=m.predict(X[hold]);folds.append({'fold':fold,'train_queries':len(fit),'holdout_queries':len(hold)})
 eligible=np.asarray([not r['explicit_constraints'] for r in rows]); frac=(33-cheap)/(heavy-cheap); threshold=float(np.quantile(oof[eligible],1-frac)); model=ExtraTreesRegressor(n_estimators=800,min_samples_leaf=5,max_features=.7,random_state=17,n_jobs=-1).fit(X,y)
 export_model(model,threshold,out,cheap,heavy,len(rows))
 Path(str(out)+'.oof.json').write_text(json.dumps({'schema_version':1,'cheap_budget':cheap,'heavy_budget':heavy,'queries':len(rows),'eligible_queries':int(eligible.sum()),'heavy_fraction_target':frac,'threshold_from_oof':threshold,'folds':folds,'oof_gain_mean':float(oof[eligible].mean())},indent=2)+'\n')
 print(json.dumps({'model':str(out),'queries':len(rows),'threshold':threshold,'target_heavy_fraction':frac},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--train-scores',type=Path,required=True);p.add_argument('--train-qrels',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--cheap',type=int,default=20);p.add_argument('--heavy',type=int,default=40);a=p.parse_args();run(a.train_scores,a.train_qrels,a.output,a.cheap,a.heavy)
