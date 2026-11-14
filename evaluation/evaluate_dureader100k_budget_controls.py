import argparse,json,statistics,random
from pathlib import Path
from evaluation.study_rerank_actions import FrozenCrossEncoder
from evaluation.prepare_bm25_router import metrics
from evaluation.router_metrics import percentile,bootstrap
p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--index',type=Path,required=True);p.add_argument('--qrels',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
docs={}
for line in (a.index/'docs.jsonl').open():
 x=json.loads(line);docs[x['doc_id']]=x['title']+'\n'+x['text']
qrels={}
for line in a.qrels.read_text().splitlines():
 q,_,d,g=line.split();qrels.setdefault(q,{})[d]=float(g)
rows=[json.loads(x) for x in a.capture.read_text().splitlines()];ce=FrozenCrossEncoder(a.model,device='cuda',batch_size=20,max_length=256)
vals={k:[] for k in ('hybrid','fixed-ce28','fixed-ce40','random-budget','random-budget2')}; heavy={k:[] for k in vals}; rng=random.Random('dureader100k-random-budget-v1'); rng2=random.Random('dureader100k-random-budget-v2'); pctl=.6068627450980392
for i,row in enumerate(rows):
 qid=row['query_id'];pool=[x['doc_id'] for x in row['hybrid'][:100]];chosen=pool[:40];scores,_=ce.score(row['text'],[docs[d] for d in chosen]); ranked28=[d for _,d in sorted(zip(scores[:28],chosen[:28]),key=lambda z:(-z[0],z[1]))]+pool[28:];ranked40=[d for _,d in sorted(zip(scores,chosen),key=lambda z:(-z[0],z[1]))]+pool[40:]
 actions={'hybrid':pool,'fixed-ce28':ranked28,'fixed-ce40':ranked40}
 # fixed random policy has exact same expected heavy budget; deterministic per qid
 actions['random-budget']=ranked40 if rng.random()<pctl else pool
 actions['random-budget2']=ranked40 if rng2.random()<pctl else pool
 for k,r in actions.items(): vals[k].append(metrics(r,qrels[qid]));heavy[k].append(k.startswith('fixed') or (k.startswith('random') and r is ranked40))
 if (i+1)%100==0:print('controls',i+1,'/',len(rows),flush=True)
summary={}
for k in vals:
 summary[k]={'queries':len(rows),'metrics':{m:statistics.mean(v[m] for v in vals[k]) for m in ('ndcg','recall','mrr','map')},'heavy_rate':statistics.mean(heavy[k])}
for k in ('fixed-ce28','fixed-ce40','random-budget','random-budget2'):
 summary[k]['paired_ndcg_vs_hybrid']=bootstrap([a['ndcg']-b['ndcg'] for a,b in zip(vals[k],vals['hybrid'])])
summary['protocol']={'queries':len(rows),'budget_heavy_rate_target':pctl,'random_seeds':['dureader100k-random-budget-v1','dureader100k-random-budget-v2'],'qrels_read_after_actions':True,'no_tuning_on_dev':True}
a.out.mkdir();(a.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(json.dumps(summary,ensure_ascii=False,indent=2))
