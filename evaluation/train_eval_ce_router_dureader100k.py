import argparse,json,statistics,time,hashlib
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge,LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from evaluation.study_rerank_actions import FrozenCrossEncoder
from evaluation.prepare_bm25_router import metrics
from evaluation.router_metrics import bootstrap,percentile
p=argparse.ArgumentParser();p.add_argument('--train-capture',type=Path,required=True);p.add_argument('--dev-capture',type=Path,required=True);p.add_argument('--index',type=Path,required=True);p.add_argument('--train-qrels',type=Path,required=True);p.add_argument('--dev-qrels',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
docs={}
for line in (a.index/'docs.jsonl').open():
 x=json.loads(line);docs[x['doc_id']]=x['title']+'\n'+x['text']
def loadq(p):
 d={}
 for line in p.read_text().splitlines():
  q,_,doc,g=line.split();d.setdefault(q,{})[doc]=float(g)
 return d
trq,devq=loadq(a.train_qrels),loadq(a.dev_qrels)
train=[json.loads(x) for x in a.train_capture.read_text().splitlines()];dev=[json.loads(x) for x in a.dev_capture.read_text().splitlines()]
ce=FrozenCrossEncoder(a.model,device='cuda',batch_size=20,max_length=256)
def feat(row,idx):
 lex=row['lexical'];dense=row['dense'];hy=row['hybrid']; li={x['doc_id']:i for i,x in enumerate(lex)};di={x['doc_id']:i for i,x in enumerate(dense)}
 pool=hy[:100]; overlap=sum(d['doc_id'] in di for d in pool)/max(1,len(pool)); q=row['text']; terms=max(1,len(q.split()))
 gap=(lex[0]['score']-lex[1]['score'])/max(abs(lex[0]['score']),1e-9) if len(lex)>1 else 0
 return [float(terms),float(len(lex)),float(overlap),float(gap)]
def score_rows(rows,qrels,need_features=False):
 out=[]
 for n,row in enumerate(rows):
  qid=row['query_id'];q=row['text'];pool=[x['doc_id'] for x in row['hybrid'][:100]];chosen=pool[:40];scores,elapsed=ce.score(q,[docs[d] for d in chosen]); ranked=[d for _,d in sorted(zip(scores,chosen),key=lambda z:(-z[0],z[1]))]+pool[40:]; base=pool
  out.append(dict(qid=qid,pool=pool,chosen=chosen,scores=scores,ranked=ranked,base=base,ce_ms=elapsed,features=feat(row,n)))
  if (n+1)%100==0:print('scored',n+1,'/',len(rows),flush=True)
 return out
tr=score_rows(train,trq); dv=score_rows(dev,devq)
def rank_ce(r,n):return [d for _,d in sorted(zip(r['scores'][:n],r['chosen'][:n]),key=lambda z:(-z[0],z[1]))]+r['pool'][n:]
# Train pairwise-ish regression target relevance; features include CE score and ranks from each first-stage pool.
def pair_matrix(rows,qrels):
 X=[];y=[]
 for r in rows:
  q=qrels[r['qid']];lex={x['doc_id']:i for i,x in enumerate(next(z for z in (train if rows is tr else dev) if z['query_id']==r['qid'])['lexical'])} if False else {}
  for i,(d,s) in enumerate(zip(r['chosen'],r['scores'])):
   X.append([s,1/(i+1),float(q.get(d,0)>0),float(q.get(d,0))]);y.append(float(q.get(d,0)))
 return np.array(X),np.array(y)
# labels are hidden from dev and only used for fit. Model uses CE score/rank; query-level model uses held-out train labels.
X=[];y=[]
for r in tr:
 for i,(d,s) in enumerate(zip(r['chosen'],r['scores'])):
  X.append([s,1/(i+1),float(i<28),float(i<40)]);y.append(float(trq[r['qid']].get(d,0)))
ranker=make_pipeline(StandardScaler(),Ridge(alpha=2.0)).fit(np.array(X),np.array(y))
summary={}
for n in (28,40):
 key=f'ce-{n}-learned-ranker';vals=[];base=[]
 for r in dv:
  x=np.array([[s,1/(i+1),float(i<28),float(i<40)] for i,s in enumerate(r['scores'][:n])]);pred=ranker.predict(x);ranked=[d for _,d in sorted(zip(pred,r['chosen'][:n]),key=lambda z:(-z[0],z[1]))]+r['pool'][n:];vals.append(metrics(ranked,devq[r['qid']]));base.append(metrics(rank_ce(r,n),devq[r['qid']]))
 summary[key]={'queries':len(dv),'metrics':{m:statistics.mean(v[m] for v in vals) for m in ('ndcg','recall','mrr','map')},'delta_vs_raw':{m:bootstrap([v[m]-b[m] for v,b in zip(vals,base)]) for m in ('ndcg','recall','mrr','map')},'ce_pairs':n,'ce_latency_ms':{'p50':percentile([r['ce_ms'] for r in dv],.5),'p95':percentile([r['ce_ms'] for r in dv],.95)}}
# learned query router: train whether CE-40 improves hybrid. Features are available before CE.
Q=[];labels=[]
for r in tr:
 gain=metrics(rank_ce(r,40),trq[r['qid']])['ndcg']-metrics(r['base'],trq[r['qid']])['ndcg'];Q.append(r['features']);labels.append(int(gain>0))
if len(set(labels))<2: raise RuntimeError('router labels constant')
router=make_pipeline(StandardScaler(),LogisticRegression(C=0.5,max_iter=1000,class_weight='balanced')).fit(np.array(Q),np.array(labels))
rv=[];heavy=[];deltas=[]
for r in dv:
 use=bool(router.predict(np.array([r['features']]))[0]);heavy.append(use)
 ranked=rank_ce(r,40) if use else r['base'];rv.append(metrics(ranked,devq[r['qid']]));deltas.append(rv[-1]['ndcg']-metrics(r['base'],devq[r['qid']])['ndcg'])
summary['learned-router']={'queries':len(dv),'metrics':{m:statistics.mean(v[m] for v in rv) for m in ('ndcg','recall','mrr','map')},'heavy_rate':statistics.mean(heavy),'ce_pairs_mean':40*statistics.mean(heavy),'delta_vs_hybrid':bootstrap(deltas),'train_positive_rate':statistics.mean(labels),'features':['query_terms','lexical_candidates','dense_overlap','lexical_gap','lexical_gap']}
summary['protocol']={'train_queries':len(tr),'dev_queries':len(dv),'labels_train_only':True,'dev_qrels_used_after_predictions':True,'model_inventory_sha256':hashlib.sha256((a.model/'model_inventory.json').read_bytes()).hexdigest(),'no_per_query_records_committed':True}
a.out.mkdir(parents=True);(a.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(json.dumps(summary,ensure_ascii=False,indent=2))
