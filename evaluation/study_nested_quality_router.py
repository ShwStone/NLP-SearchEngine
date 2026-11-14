"""Outer-isolated cheap/heavy action fitting and quality-gain routing.

Raw frozen feature datasets are the inputs. Globally OOF supervised action
predictions are deliberately not accepted: they can indirectly leak outer
validation labels into a stacked Router. All family, model and policy choices
here use only the current outer-training groups.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .prepare_bm25_router import metrics, sha
from .ranking_dataset import load_dataset
from .study_candidate_ranking import RANKING_SPECS, RICH_RANKING_SPECS, RECALL_SPECS, fit
from .train_bm25_router import group_bootstrap


THRESHOLDS = (-.20, -.10, -.05, -.02, 0., .01, .02, .05, .10, .20)
GATE_SPECS = tuple(dict(kind='ridge', alpha=a) for a in (1., 10., 100.)) + (
    dict(kind='hist', max_leaf_nodes=7, min_samples_leaf=40, l2_regularization=10.),)


def aligned(datasets):
    reference = datasets[0]['metadata']
    for dataset in datasets[1:]:
        actual = dataset['metadata']
        for key in ('query_ids','groups','docs','relevance','identities','baselines','bm25_features'):
            if actual[key]!=reference[key]:
                raise ValueError('Router action dataset alignment mismatch: '+key)


def action_confidence(values,x,names,indices):
    top=np.asarray([values[j] for j in indices[:11]],dtype=float)
    if not len(top):raise ValueError('empty action confidence')
    exp=np.exp(top[:10]-np.max(top[:10]));probabilities=exp/exp.sum()
    confidence=[top[0],top[0]-top[min(1,len(top)-1)],
        top[min(9,len(top)-1)]-top[min(10,len(top)-1)],
        float(top[:10].mean()),float(top[:10].std()),
        float(-np.sum(probabilities*np.log(np.maximum(probabilities,1e-30))))]
    for name in ('lexical_rank_reciprocal','dense_rank_reciprocal','term_coverage',
                 'm3_dense','m3_sparse','m3_colbert_int8'):
        if name in names:
            column=names.index(name)
            confidence.extend((float(x[indices[0],column]),float(np.mean(x[indices[:10],column]))))
    return confidence


def predict_action(model, spec, dataset, selected):
    data = dataset['metadata']
    output = {}
    for i in selected:
        x = dataset['x'][dataset['offsets'][i]:dataset['offsets'][i+1]]
        if spec.get('kind')=='fixed-late':
            values = x[:,data['feature_names'].index('m3_colbert_int8')]
        elif spec.get('kind')=='fixed-native-dense':
            values=np.where(x[:,data['feature_names'].index('dense_rank_reciprocal')]>0,
                x[:,data['feature_names'].index('dense_score')],-np.inf)
        elif spec.get('kind')=='fixed-teacher':
            values=np.where(x[:,data['feature_names'].index('teacher_scored')]>0,
                x[:,data['feature_names'].index('teacher_logit')],-np.inf)
        else:
            values = model.predict(x)
        available=np.flatnonzero(np.isfinite(values))
        indices = sorted(available, key=lambda j:(-values[j],data['identities'][data['docs'][i][j]]))
        ranking = [data['docs'][i][j] for j in indices]
        quality = metrics(ranking,data['relevance'][data['query_ids'][i]])
        # All confidence features are derived from this CHEAP action alone.
        confidence=action_confidence(values,x,data['feature_names'],indices)
        output[int(i)] = dict(metrics=quality,ranked=ranking[:100],confidence=confidence)
    return output


def fit_action(spec, dataset, selected):
    if spec.get('kind') in ('fixed-late','fixed-teacher','fixed-native-dense'):return None
    totals=None
    if spec.get('kind')=='recall-ranker':
        totals=np.array([sum(float(v)>0 for v in dataset['metadata']['relevance'][q].values())
                         for q in dataset['metadata']['query_ids']],dtype=float)
    return fit(spec,dataset['x'],dataset['y'],dataset['sizes'],selected,dataset['offsets'],relevance_totals=totals)


def quality_objective(predictions,baseline=None):
    means=np.mean([[r['metrics'][m] for m in ('ndcg','recall')] for r in predictions.values()],axis=0)
    return float(np.min(means/baseline-1)) if baseline is not None else float(np.mean(means))


def select_action(datasets, training, groups, include_fixed=False,selection='mean-quality',include_teacher=False,include_native=False,include_recall=False,include_rich=False):
    trials, best = [], None
    for family, dataset in enumerate(datasets):
        specs = list(RANKING_SPECS)
        if include_rich: specs += list(RICH_RANKING_SPECS)
        if include_recall: specs += list(RECALL_SPECS)
        if include_fixed and 'm3_colbert_int8' in dataset['metadata']['feature_names']:
            specs += [dict(kind='fixed-late')]
        if include_teacher and 'teacher_logit' in dataset['metadata']['feature_names']:
            specs += [dict(kind='fixed-teacher')]
        if include_native and 'dense_score' in dataset['metadata']['feature_names']:
            specs += [dict(kind='fixed-native-dense')]
        for spec in specs:
            predictions, folds = {}, []
            for train_rel, valid_rel in GroupKFold(3).split(training,groups=groups[training]):
                train, valid = training[train_rel],training[valid_rel]
                if set(groups[train]) & set(groups[valid]):raise ValueError('inner group leakage')
                model = fit_action(spec,dataset,train)
                predictions.update(predict_action(model,spec,dataset,valid))
                folds.append(dict(training_ids=[dataset['metadata']['query_ids'][i] for i in train],
                                  heldout_ids=[dataset['metadata']['query_ids'][i] for i in valid]))
            baseline=None
            if selection=='min-lift':
                baseline=np.mean([[dataset['metadata']['baselines']['hybrid'][i][m] for m in ('ndcg','recall')] for i in training],axis=0)
            objective = quality_objective(predictions,baseline)
            print('inner action family',family,'spec',spec,'objective',round(objective,6),flush=True)
            trial = dict(family=family,spec=spec,inner_objective=objective,inner_folds=folds)
            trials.append(trial)
            if best is None or objective>best[0]:best=(objective,trial,predictions)
    return best[1],best[2],trials


def gate_matrix(metadata, selected, predictions):
    names = metadata['bm25_feature_names']
    return np.array([[metadata['bm25_features'][i][name] for name in names]+predictions[int(i)]['confidence']
                     for i in selected],dtype=float)


def fit_gate(spec,x,y):
    if spec['kind']=='ridge':
        return make_pipeline(StandardScaler(),Ridge(alpha=spec['alpha'])).fit(x,y)
    settings = {k:v for k,v in spec.items() if k!='kind'}
    return [HistGradientBoostingRegressor(**settings,max_iter=60,learning_rate=.05,
            early_stopping=False,random_state=20261028).fit(x,y[:,j]) for j in range(2)]


def predict_gate(model,x):
    return np.column_stack([m.predict(x) for m in model]) if isinstance(model,list) else model.predict(x)


def export_gate(model):
    if not isinstance(model,list):
        scaler,regressor=model.steps[0][1],model.steps[1][1]
        return dict(kind='ridge',mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),
                    coef=regressor.coef_.tolist(),intercept=regressor.intercept_.tolist())
    heads=[]
    for head in model:
        trees=[]
        for iteration in head._predictors:
            nodes=iteration[0].nodes
            trees.append([dict(value=float(n['value']),leaf=bool(n['is_leaf']),
                feature=int(n['feature_idx']),threshold=float(n['num_threshold']),
                left=int(n['left']),right=int(n['right']),missing_left=bool(n['missing_go_to_left'])) for n in nodes])
        heads.append(dict(bias=float(head._baseline_prediction[0,0]),trees=trees))
    return dict(kind='hist',heads=heads)


def replay_gate(artifact,x):
    x=np.asarray(x,dtype=float)
    if artifact['kind']=='ridge':
        return ((x-np.array(artifact['mean']))/np.array(artifact['scale']))@np.array(artifact['coef']).T+artifact['intercept']
    result=np.zeros((len(x),len(artifact['heads'])))
    for j,head in enumerate(artifact['heads']):
        result[:,j]=head['bias']
        for tree in head['trees']:
            for i,row in enumerate(x):
                node=tree[0]
                while not node['leaf']:
                    value=row[node['feature']]
                    left=node['missing_left'] if np.isnan(value) else value<=node['threshold']
                    node=tree[node['left'] if left else node['right']]
                result[i,j]+=node['value']
    return result


def decision(predicted,threshold,constant=None):
    if constant is not None:return np.full(len(predicted),constant,dtype=bool)
    return predicted.mean(axis=1)>threshold


def gate_selection(x,y,groups,cheap_quality,heavy_quality,baseline,require_nonconstant=False,max_heavy_rate=None):
    trials, best = [],None
    policies=[dict(constant=False),dict(constant=True)]
    for spec in GATE_SPECS:
        oof=np.zeros_like(y)
        for train,valid in GroupKFold(3).split(x,groups=groups):
            oof[valid]=predict_gate(fit_gate(spec,x[train],y[train]),x[valid])
        for threshold in THRESHOLDS:
            policies.append(dict(spec=spec,threshold=threshold,prediction=oof))
    for policy in policies:
        choice=decision(policy.get('prediction',np.zeros_like(y)),policy.get('threshold',0.),policy.get('constant'))
        quality=np.where(choice[:,None],heavy_quality,cheap_quality)
        lifts=quality.mean(axis=0)/baseline.mean(axis=0)-1
        # Optimize the weaker of the two required quality metrics, using only
        # outer-training labels. Constants remain explicit controls/fallbacks.
        heavy_rate=float(np.mean(choice))
        if require_nonconstant and not (0.0 < heavy_rate < 1.0):
            continue
        if max_heavy_rate is not None and heavy_rate > max_heavy_rate:
            continue
        key=(float(min(lifts)),float(np.mean(lifts)),-heavy_rate)
        record={k:v for k,v in policy.items() if k!='prediction'}
        record.update(ndcg=float(quality[:,0].mean()),recall=float(quality[:,1].mean()),
                      min_relative_lift=key[0],heavy_rate=float(np.mean(choice)))
        trials.append(record)
        if best is None or key>best[0]:best=key,record
    if best is None:
        raise ValueError('no nonconstant gate policy survived inner selection')
    return best[1],trials


def save_action(model,spec,out,prefix):
    if spec.get('kind') in ('fixed-late','fixed-teacher','fixed-native-dense'):return dict(kind=spec['kind'])
    name=prefix+'-action.txt';model.booster_.save_model(str(out/name))
    return dict(model_file=name,model_sha256=sha(out/name))


def run(args):
    start=time.perf_counter();args.out.mkdir(parents=True,exist_ok=False)
    cheap=[load_dataset(p) for p in args.cheap_dataset]
    heavy=[load_dataset(p) for p in args.heavy_dataset]
    aligned(cheap+heavy)
    gate_dataset=load_dataset(args.gate_features_dataset) if args.gate_features_dataset else None
    if gate_dataset is not None:aligned(cheap+[gate_dataset])
    # Gate inputs must be computable before the expensive scorer is called.
    for dataset in ([gate_dataset] if gate_dataset is not None else cheap):
        if any('teacher' in name or 'secondary' in name for name in dataset['metadata']['feature_names']):
            raise ValueError('cheap action contains expensive scorer features')
        if any(name.startswith('dense_top') and name.endswith('cosine') for name in dataset['metadata']['feature_names']):
            raise ValueError('unsupported cheap support provenance')
    metadata=cheap[0]['metadata'];groups=cheap[0]['groups'];ids=metadata['query_ids']
    baseline=np.array([[r[m] for m in ('ndcg','recall')] for r in metadata['baselines']['hybrid']])
    protocol=dict(kind='outer-isolated-quality-router-v1',queries=len(ids),outer_folds=5,inner_folds=3,
        inputs={str(p/'dataset.json'):sha(p/'dataset.json') for p in args.cheap_dataset+args.heavy_dataset},
        cheap_families=[str(p) for p in args.cheap_dataset],heavy_families=[str(p) for p in args.heavy_dataset],
        family_and_action_selection='within each outer-training fold; grouped 3-fold action OOF',
        action_selection_objective=args.action_selection,
        include_recall_ranking=args.include_recall_ranking,
        include_rich_ranking=args.include_rich_ranking,
        include_fixed_teacher=args.include_fixed_teacher,
        include_fixed_native_dense=args.include_fixed_native_dense,
        gate_selection='grouped 3-fold gate OOF on action OOF data, entirely within outer training; max weaker relative quality lift',
        require_nonconstant_gate=args.require_nonconstant_gate,
        max_heavy_rate=args.max_heavy_rate,
        gate_label='heavy minus cheap nDCG and Recall divided by outer-training hybrid means',
        gate_features='BM25 visible features and cheap-action score/native-feature confidence; no heavy features',
        gate_features_dataset=str(args.gate_features_dataset) if args.gate_features_dataset else None,
        gate_native_proxy='fixed native dense, computed before either CE action' if gate_dataset is not None else None,
        nested_scope='outer honest; internal gate CV reuses outer-training action OOF; not an independent quality guarantee',
        confirmation_read=False,http_evaluated=False,goal_complete=False,
        code_sha256={str(p):sha(p) for p in (Path(__file__),Path(__file__).with_name('study_candidate_ranking.py'),Path(__file__).with_name('ranking_dataset.py'))})
    if gate_dataset is not None:
        protocol['inputs'][str(args.gate_features_dataset/'dataset.json')]=gate_dataset['manifest_sha256']
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (args.out/'code').mkdir()
    for path in protocol['code_sha256']:
        (args.out/'code'/Path(path).name).write_bytes(Path(path).read_bytes())
    outer={};folds=[];controls={'cheap':{},'heavy':{}};control_predictions={}
    for fold,(training,validation) in enumerate(GroupKFold(5).split(ids,groups=groups)):
        if set(groups[training]) & set(groups[validation]):raise ValueError('outer group leakage')
        print('outer',fold,'selecting cheap action',flush=True)
        cs,cp,ct=select_action(cheap,training,groups,include_fixed=True,selection=args.action_selection,
                             include_native=args.include_fixed_native_dense,include_teacher=args.include_fixed_teacher if gate_dataset is not None else False,
                             include_recall=args.include_recall_ranking,include_rich=args.include_rich_ranking)
        print('outer',fold,'selecting heavy action',flush=True)
        hs,hp,ht=select_action(heavy,training,groups,selection=args.action_selection,include_teacher=args.include_fixed_teacher,
                             include_recall=args.include_recall_ranking,include_rich=args.include_rich_ranking)
        cq=np.array([[cp[int(i)]['metrics'][m] for m in ('ndcg','recall')] for i in training])
        hq=np.array([[hp[int(i)]['metrics'][m] for m in ('ndcg','recall')] for i in training])
        proxy_training=(predict_action(None,dict(kind='fixed-native-dense'),gate_dataset,training)
                        if gate_dataset is not None else cp)
        gx=gate_matrix(metadata,training,proxy_training);scale=baseline[training].mean(axis=0)
        gy=(hq-cq)/scale
        selection,trials=gate_selection(gx,gy,groups[training],cq,hq,baseline[training],args.require_nonconstant_gate,args.max_heavy_rate)
        cm=fit_action(cs['spec'],cheap[cs['family']],training)
        cv=predict_action(cm,cs['spec'],cheap[cs['family']],validation)
        hm=fit_action(hs['spec'],heavy[hs['family']],training)
        hv=predict_action(hm,hs['spec'],heavy[hs['family']],validation)
        gate=None
        if 'spec' in selection:
            model=fit_gate(selection['spec'],gx,gy)
            gate=export_gate(model)
            proxy_validation=(predict_action(None,dict(kind='fixed-native-dense'),gate_dataset,validation)
                              if gate_dataset is not None else cv)
            vx=gate_matrix(metadata,validation,proxy_validation)
            predicted=predict_gate(model,vx);replayed=replay_gate(gate,vx)
            if not np.allclose(predicted,replayed,rtol=0,atol=1e-10):raise ValueError('Router export mismatch')
            choice=decision(replayed,selection['threshold'])
        else:choice=np.full(len(validation),selection['constant'],dtype=bool)
        for i,uses_heavy in zip(validation,choice):
            i=int(i);actual=hv[i] if uses_heavy else cv[i]
            outer[i]=dict(query_id=ids[i],group=str(groups[i]),action='heavy' if uses_heavy else 'cheap',
                          metrics=actual['metrics'],ranked=actual['ranked'])
            controls['cheap'][i]=cv[i]['metrics'];controls['heavy'][i]=hv[i]['metrics']
            control_predictions[i]=dict(query_id=ids[i],group=str(groups[i]),
                cheap={k:v for k,v in cv[i].items() if k!='confidence'},
                heavy={k:v for k,v in hv[i].items() if k!='confidence'})
        gate_name=f'fold-{fold}-gate.json'
        (args.out/gate_name).write_text(json.dumps(dict(selection=selection,model=gate),indent=2)+'\n')
        folds.append(dict(fold=fold,training_ids=[ids[i] for i in training],heldout_ids=[ids[i] for i in validation],
            cheap_selection=cs,heavy_selection=hs,cheap_trials=ct,heavy_trials=ht,
            gate_selection=selection,gate_trials=trials,gate_file=gate_name,gate_sha256=sha(args.out/gate_name),
            cheap_model=save_action(cm,cs['spec'],args.out,f'fold-{fold}-cheap'),
            heavy_model=save_action(hm,hs['spec'],args.out,f'fold-{fold}-heavy'),
            gate_crossfit=[dict(training_ids=[ids[training[i]] for i in t],heldout_ids=[ids[training[i]] for i in v])
                for t,v in GroupKFold(3).split(training,groups=groups[training])],
            action_counts={name:int(np.sum(choice==value)) for name,value in [('cheap',False),('heavy',True)]}))
        (args.out/'fold-progress.json').write_text(json.dumps(folds,indent=2)+'\n')
        print('outer',fold,'gate',selection,'actions',folds[-1]['action_counts'],flush=True)
    summary={}
    for m,column in [('ndcg',0),('recall',1)]:
        values=np.array([outer[i]['metrics'][m] for i in range(len(ids))])
        ci=group_bootstrap(values-baseline[:,column],groups)
        summary[m]=dict(value=float(values.mean()),baseline=float(baseline[:,column].mean()),
            relative_lift=float(values.mean()/baseline[:,column].mean()-1),delta_ci=ci,
            passes_20_percent=bool(values.mean()>1.2*baseline[:,column].mean()))
    counts={name:sum(r['action']==name for r in outer.values()) for name in ('cheap','heavy')}
    result=dict(protocol=protocol,folds=folds,outer=summary,action_counts=counts,nonconstant=all(counts.values()),
        fixed_action_controls={name:{m:float(np.mean([r[m] for r in rows.values()])) for m in ('ndcg','recall')}
                               for name,rows in controls.items()},
        quality_gate_passed=bool(all(counts.values()) and all(r['passes_20_percent'] and r['delta_ci']['ci95'][0]>0 for r in summary.values())),
        http_evaluated=False,independent_confirmation_evaluated=False,goal_complete=False,
        elapsed_seconds=time.perf_counter()-start)
    oracle_choice=np.array([np.mean([controls['heavy'][i][m]/baseline[:,j].mean()-controls['cheap'][i][m]/baseline[:,j].mean()
        for j,m in enumerate(('ndcg','recall'))])>0 for i in range(len(ids))])
    result['diagnostic_outer_action_oracle']=dict(
        label_used=True,deployable=False,
        metrics={m:float(np.mean([controls['heavy' if oracle_choice[i] else 'cheap'][i][m] for i in range(len(ids))]))
                 for m in ('ndcg','recall')},
        heavy_rate=float(oracle_choice.mean()))
    (args.out/'outer-predictions.jsonl').write_text(''.join(json.dumps(outer[i])+'\n' for i in range(len(ids))))
    result['predictions_sha256']=sha(args.out/'outer-predictions.jsonl')
    (args.out/'action-controls.jsonl').write_text(''.join(json.dumps(control_predictions[i])+'\n' for i in range(len(ids))))
    result['controls_sha256']=sha(args.out/'action-controls.jsonl')
    (args.out/'study.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('folds','protocol')},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--cheap-dataset',type=Path,action='append',required=True)
    parser.add_argument('--heavy-dataset',type=Path,action='append',required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--action-selection',choices=('mean-quality','min-lift'),default='mean-quality')
    parser.add_argument('--include-fixed-teacher',action='store_true')
    parser.add_argument('--include-fixed-native-dense',action='store_true')
    parser.add_argument('--require-nonconstant-gate',action='store_true')
    parser.add_argument('--max-heavy-rate',type=float)
    parser.add_argument('--gate-features-dataset',type=Path,help='optional CE-free native proxy for gate decisions before either action')
    parser.add_argument('--include-recall-ranking',action='store_true',help='development-only Recall-weighted ranking candidates')
    parser.add_argument('--include-rich-ranking',action='store_true',help='development-only high-capacity ranking candidates')
    run(parser.parse_args())
