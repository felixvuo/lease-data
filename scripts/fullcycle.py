#!/usr/bin/env python3
# coding: utf-8

# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>.

import argparse
import itertools as it
import numpy as np
import os
import pandas as pd
from ruamel.yaml import YAML

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance as pimp

import xgboost as xgb

def _setupArgs():
    ap = argparse.ArgumentParser(
        description="Train classifiers on given dataset, output predictions"
    )
    ap.add_argument(
        'config_file', type=str, 
        help="Configuration file (modified version of fullcycle.yaml)"
    )
    ap.add_argument(
        '--do-all', '-a', action="store_true",
        help="Perform all steps"
    )
    ap.add_argument(
        '--do-prep-timings', '-p', action="store_true",
        help="Prepare the PAR data"
    )
    ap.add_argument(
        '--do-train-predict', '-t', action="store_true",
        help="Train models, predict configs"
    )
    ap.add_argument(
        '--do-evaluate', '-e', action="store_true",
        help="Run evaluation on predictions"
    )
    ap.add_argument(
        '--quick-mode', '-q', action="store_true",
        help="For debugging, do a stripped down version"
    )
    ap.add_argument(
        '--single', '-s', type=str, default='',
        help="Just do a single run.  Pass in: cycle,featureset,split"
    )
    return ap.parse_args()


def main():
    args = _setupArgs()    
    yaml = YAML()
    conf = yaml.load(open(args.config_file))
    if not conf['READY']:
        raise Exception("You're not ready, tweak the config")
    if args.quick_mode:
        conf['portfolio_size'] = 3
        conf['ml_iterations'] = 2
        conf['hp_tune_iterations'] = 16

    # calculate and store (or load in) the median penalised timings
    print("Calculating penalised average runtimes...")
    timings_path = os.path.join(
        conf['timings_dir_path'],
        f"times_par{conf['penalty']}_to{conf['total_timeout']}.csv.gz"
    )
    if args.do_prep_timings or args.do_all:
        timings = createTimings(
            pd.read_csv(conf['algo_runs_path']),
            conf['penalty'],
            conf['total_timeout']
        )
        timings.to_csv(timings_path, index=False)
    else:
        timings = pd.read_csv(timings_path)

    # prepare features for all featuresets
    print("Dropping unmatched instances from feature data...")
    feat_dfs = prepareFeatureFrames(conf['featuresets'], conf['feature_dir_path'], timings)

    # this is the heart of the thing - train pairwise models and predict encodings
    if args.do_train_predict or args.do_all:
        single_dic = {}
        if args.single:
            (_c, _f, _s) = tuple(args.single.split(','))
            single_dic = {'cycle':int(_c), 'featureset':_f, 'split':_s}
        engines = {
            'pairwise-combined'     : trainPredictPairwiseCombined,
            'pairwise-separate'     : trainPredictPairwiseSeparate,
            'single-combined'       : trainPredictSingleCombined,
            'pairwise-combined-xgb' : trainPredictPairwiseCombinedXGB,
        }
        trainPredict = engines[conf.get('train_predict_engine','pairwise-combined')]
        diagnostics_dir = conf.get('ml_diagnostics_dir',None)
        results = trainPredict(
            timings, feat_dfs,
            start_seed = conf['seed'], test_frac = conf['test_set_proportion'],
            portfolio_size = conf['portfolio_size'], cycles = conf['ml_iterations'],
            hp_iterations = conf['hp_tune_iterations'], cpus = conf['cpus'],
            cust_score = conf['use_custom_scorer'], use_weight = conf['use_sample_weight'],
            single = single_dic, diagnostics = True if diagnostics_dir else False
        )
        predictions = results['predictions']
        predictions.to_csv(
            args.single+'.csv' if args.single else conf['predictions_path'],
            index=False)
        if diagnostics_dir:
            _f = f'bp-{args.single}.csv' if args.single else 'bestparams.csv'
            results['best_hyperparameters'].to_csv(os.path.join(diagnostics_dir,_f),index=False)
            _f = f'fi-{args.single}.csv' if args.single else 'feature-importances.csv.gz'
            results['feature_importances'].to_csv(os.path.join(diagnostics_dir,_f),index=False)
            _f = f'acc-{args.single}.csv' if args.single else 'accuracy.csv.gz'
            results['accuracy'].to_csv(os.path.join(diagnostics_dir,_f),index=False)
    elif args.do_evaluate:
        predictions = pd.read_csv(conf['predictions_path'])

    # how did the predictions compare with vb, sb, def, etc?
    if args.do_evaluate or args.do_all:
        evaluatePredictions(
            predictions, timings, costs_path=conf.get('feature_costs_path',None),
            out_dir=conf['eval_output_dir'], prefix=conf['eval_prefix'],
            featuresets=conf['featuresets']
        )


def createTimings(infos:pd.DataFrame, pen:float, timeout:float):
    """Create a DataFrame with instance timings
    
    The penalty multiplier is applied to runs which timed out; the median time is returned across
    all runs.

    """
    partime = pen*timeout
    ID = ['ProblemDir','ParamFile']
    infos['OverallTime'] = infos['SavileRowTotalTime'] + infos['SolverTotalTime']
    timeout_conditions = (
        (infos.SolverTimeOut > 0), (infos.SavileRowTimeOut > 0),
        (infos.SolverTotalTime.isna()), (infos.SavileRowTotalTime.isna()),
        (infos.OverallTime.isna()), (infos.OverallTime > timeout),
        (infos.SolverSatisfiable.isna()),
    )
    infos['OverallTime'] = np.where(
        np.logical_or.reduce(timeout_conditions), partime, infos.OverallTime
    )
    meds = infos.groupby(
        ['ProblemDir', 'ParamFile', 'Encs']
    )['OverallTime'].median().reset_index()
    
    # it's possible that some configs are missing; if so, assign them the penalty time
    conf_counts = meds.groupby(ID)['Encs'].count().reset_index(name="conf_count")
    configs = meds.Encs.unique().tolist()
    for _,row in conf_counts.iterrows():
        if int(row['conf_count']) < len(configs):
            got_configs = meds.loc[
                (meds.ProblemDir==row['ProblemDir']) & (meds.ParamFile==row['ParamFile'])
            ].Encs.unique().tolist()
            missing_encs = [c for c in configs if c not in got_configs]
            for e in missing_encs:
                meds.loc[len(meds)] = [row['ProblemDir'], row['ParamFile'], e, partime]

    # drop instances where all the configs led to a timeout
    meds = meds.groupby(ID).filter( lambda x:x['OverallTime'].min() < timeout )
                
    # put in the single instance identifier to match up more easily later
    meds['ID'] = meds['ProblemDir'] + '/' + meds['ParamFile']
    return meds


def prepareFeatureFrames(featuresets, feat_files_dir, timings):
    """Open feature files, trim to ensure we have same instances

    Ensures that we only keep instances for which we have all features from all featuresets.  Also
    drops any instances for which we have no timings.

    Returns:
      {featureset_A : features_df_A, ... }

    """
    frames_dict = {
        f : pd.read_csv(os.path.join(feat_files_dir,f"features-{f}.csv.gz"))
        for f in featuresets
    }
    
    for f,df in frames_dict.items():
        df['ID'] = df.ProblemDir + "/" + df.ParamFile
    common = None
    for df in [frames_dict[k] for k in sorted(frames_dict.keys())]:
        if common is None:
            common = pd.DataFrame(df['ID'])
        else:
            common = common.merge(df['ID'], on=['ID'])
    for df in frames_dict.values():      
        # drop rows which don't show up in other feature frames
        df.drop(df.loc[~df['ID'].isin(common['ID'])].index,inplace=True)

        # drop rows for which we have no timings
        df.drop(df.loc[~df['ID'].isin(timings['ID'])].index,inplace=True)

        # let's drop any columns that have no data at all
        df.dropna(axis='columns', how='all', inplace=True)

    return frames_dict


def trainPredictPairwiseCombined(
        timings, feat_sets_frames, start_seed = 0, test_frac = 0.25,
        portfolio_size = 2, cycles = 1, hp_iterations = 10, cpus = -1,
        cust_score=False, use_weight=False, single={}, diagnostics=False):
    """"Repeat split/train/predict a few times

    Pass in single to just do the split/train/pred once, e.g.
    single = {'cycle' : 3, 'split' :'class', 'featureset' : 'f2f'}
    """

    if diagnostics:
        id_cols = ['split','featureset','cycle','confA','confB']
        df_best_hp = pd.DataFrame(
            columns = id_cols + [
                'n_estimators','max_features','max_depth', 'max_samples','criterion'
            ]
        )
        df_fimps = pd.DataFrame(
            columns = id_cols + ['feature','source','importance']
        )
        df_accuracy = pd.DataFrame(columns = id_cols[:3] + ['train_accuracy', 'test_accuracy'])
    if single:
        iterations = [(single['cycle'], single['featureset'], single['split'])]
    else:
        c_range = range(1,cycles+1)
        f_names = feat_sets_frames.keys()
        splits  = ('class', 'instance')
        iterations = it.product(c_range, f_names, splits)
    pred_frames = []
    for (cycle, featureset, split) in iterations:
        seed = start_seed + cycle
        print(f"Start cycle {cycle:02d}: {featureset:>6}, split by {split:>8}... ",
              end="",flush=True)
        feats = feat_sets_frames[featureset]
        trainIDs, testIDs = _tt_split(feats, split, test_frac, seed)
        portfolio = _makePortfolio(portfolio_size, timings.loc[timings.ID.isin(trainIDs.ID)])
        print("T/T: {}, {}; portf: {}".format( len(trainIDs),len(testIDs),str(portfolio) ) )

        # set up the hyperparameter space
        len_f_col = len(feats.columns) - 3 # without the instance stuff
        hp = _hpRanges(len_f_col)

        # prepare tiebreaker for equal votes - take the config with best mean over training set
        _t = timings.loc[timings.Encs.isin(portfolio)&timings.ID.isin(trainIDs.ID)]
        config_mean = _t.groupby(['Encs'])['OverallTime'].mean().to_dict()
        # pair up the configs
        pairwise_models = {}

        # train the classifiers
        for A,B in it.combinations(portfolio,2):
            print(f"{A}/{B},",end="",flush=True)
            trainX, trainY, trainIDs = _makeXY(feats, timings, trainIDs, [A,B])
            model = RandomizedSearchCV(
                RandomForestClassifier(n_jobs = cpus, random_state = seed), hp,
                n_iter = hp_iterations, n_jobs = cpus, random_state = seed,
                scoring = make_scorer(
                    _vb_loss, greater_is_better=False,IDs=trainIDs, timings=timings
                ) if cust_score else None,
            )
            _sample_weight = _sampleWeights(trainIDs,timings) if use_weight else None
            model.fit(trainX, trainY, sample_weight = _sample_weight)
            pairwise_models[(A,B)] = model
            if diagnostics:
                context = {'split':split,'featureset':featureset,'cycle':cycle,'confA':A,'confB':B}

                # record the best hyper-parameters
                row = model.best_params_
                row.update(context)
                df_best_hp = df_best_hp.append(row, ignore_index=True)

                # record the feature importance measures
                mdi_fimps = getattr(model.best_estimator_,'feature_importances_',None)
                if mdi_fimps is not None:
                    context['source']='mdi'
                    df_mdi = pd.DataFrame({
                        'importance':mdi_fimps,
                        'feature': [
                            c for c in feats.columns if c not in ['ID','ProblemDir','ParamFile']]
                    })
                    for k,v in context.items():
                        df_mdi[k]=v
                    df_fimps = pd.concat([df_fimps,df_mdi], ignore_index=True)
                    
        print("TRAINED")

        df_pred = _pairwisePredict(pairwise_models,feats,testIDs,timings,config_mean)
        df_pred['split'] = split
        df_pred['featureset'] = featureset
        df_pred['cycle'] = cycle
        pred_frames.append(df_pred)

        if diagnostics:
            # record the training and validation classification accuracy
            row = {'split' : split, 'featureset' : featureset, 'cycle' : cycle}
            trnX, trnY, _ = _makeXY(feats, timings, trainIDs, portfolio)
            tstX, tstY, _ = _makeXY(feats, timings, testIDs, portfolio)            
            row['train_accuracy'] = accuracy_score(
                _pairwisePredict(pairwise_models,feats,trainIDs,timings,config_mean)['pred_encs'],
                trnY
            )
            row['test_accuracy'] = accuracy_score(
                _pairwisePredict(pairwise_models,feats,testIDs,timings,config_mean)['pred_encs'], 
                tstY
            )
            df_accuracy = df_accuracy.append(row, ignore_index=True)

            df_pfi = _permutationImportance(pairwise_models,feats,testIDs,timings,config_mean,
                                            repeat=5, random_state=seed)
            for k,v in dict(split=split,featureset=featureset,cycle=cycle,source='perm').items():
                df_pfi[k] = v
            df_fimps = pd.concat([df_fimps,df_pfi],ignore_index=True)
    results = {
        'predictions':pd.concat(pred_frames, ignore_index=True)
    }
    if diagnostics:
        results['best_hyperparameters'] = df_best_hp
        results['feature_importances'] = df_fimps
        results['accuracy'] = df_accuracy
    return results


def trainPredictPairwiseCombinedXGB(
        timings, feat_sets_frames, start_seed = 0, test_frac = 0.25,
        portfolio_size = 2, cycles = 1, hp_iterations = 10, cpus = -1,
        cust_score=False, use_weight=False, single={}, diagnostics=False):
    """"Repeat split/train/predict a few times, using XGBoost

    Pass in single to just do the split/train/pred once, e.g.
    single = {'cycle' : 3, 'split' :'class', 'featureset' : 'f2f'}
    """

    if diagnostics:
        assert False, "Sorry, diagnostics not implemented for xgb"
    if single:
        iterations = [(single['cycle'], single['featureset'], single['split'])]
    else:
        c_range = range(1,cycles+1)
        f_names = feat_sets_frames.keys()
        splits  = ('class', 'instance')
        iterations = it.product(c_range, f_names, splits)
    pred_frames = []
    for (cycle, featureset, split) in iterations:
        seed = start_seed + cycle
        print(f"Start cycle {cycle:02d}: {featureset:>6}, split by {split:>8}... ",
              end="",flush=True)
        feats = feat_sets_frames[featureset]
        trainIDs, testIDs = _tt_split(feats, split, test_frac, seed)
        portfolio = _makePortfolio(portfolio_size, timings.loc[timings.ID.isin(trainIDs.ID)])
        print("T/T: {}, {}; portf: {}".format( len(trainIDs),len(testIDs),str(portfolio) ) )

        # set up the hyperparameter space
        hp = dict(
            eta = np.linspace(0.01,0.8,20).round(3).astype(float).tolist(),
            gamma = np.linspace(0,1,10).astype(float).tolist(),
            max_depth = list(range(2,15,2)),
            n_estimators = list(range(5,101,10)),
        )

        # prepare tiebreaker for equal votes - take the config with best mean over training set
        _t = timings.loc[timings.Encs.isin(portfolio)&timings.ID.isin(trainIDs.ID)]
        config_mean = _t.groupby(['Encs'])['OverallTime'].mean().to_dict()
        # pair up the configs
        pairwise_models = {}

        # train the classifiers
        for A,B in it.combinations(portfolio,2):
            print(f"{A}/{B},",end="",flush=True)
            trainX, trainY, trainIDs = _makeXY(feats, timings, trainIDs, [A,B])
            model = RandomizedSearchCV(
                xgb.XGBClassifier(objective='binary:hinge', n_jobs = 1, random_state = seed),
                hp, cv = 3, n_iter = hp_iterations, n_jobs = cpus, random_state = seed,
                scoring = make_scorer(
                    _vb_loss, greater_is_better=False,IDs=trainIDs, timings=timings
                ) if cust_score else None,
            )
            _sample_weight = _sampleWeights(trainIDs,timings) if use_weight else None
            model.fit(trainX, trainY, sample_weight = _sample_weight)
            pairwise_models[(A,B)] = model
        print("TRAINED")

        breaktie = lambda x : sorted(x.dropna().tolist(),key=lambda c:config_mean[c]).pop(0)
        pairwise_choices = testIDs.sort_values(['ID']).copy()
        for ((A,B), model) in pairwise_models.items():
            testX, testY, testIDs = _makeXY(feats, timings, testIDs, [A,B])
            xgb_preds = model.predict(testX)
            pairwise_choices['_v_'.join((A,B))] = xgb_preds
        df_pred = pairwise_choices.loc[:,['ID']]
        _modes = pairwise_choices.drop(columns=['ID']).mode(axis=1)
        df_pred['pred_encs'] = _modes.apply(breaktie, axis=1)
        df_pred['split'] = split
        df_pred['featureset'] = featureset
        df_pred['cycle'] = cycle
        pred_frames.append(df_pred)

    results = {'predictions':pd.concat(pred_frames, ignore_index=True)}
    return results


def _pairwisePredict(models, feats, testIDs, timings, training_config_means):
    """Use the pairwise models to vote on the best encoding"""

    breaktie = lambda x : sorted(
        x.dropna().tolist(),
        key=lambda config:training_config_means[config]
    ).pop(0)

    pairwise_choices = testIDs.sort_values(['ID']).copy()
    for ((A,B), model) in models.items():
        testX, testY, testIDs = _makeXY(feats, timings, testIDs, [A,B])
        pairwise_choices['_v_'.join((A,B))] = model.predict(testX)
    df_pred = pairwise_choices.loc[:,['ID']]
    _modes = pairwise_choices.drop(columns=['ID']).mode(axis=1)
    df_pred['pred_encs'] = _modes.apply(breaktie, axis=1)
    return df_pred


def _permutationImportance(models, feats, testIDs, timings, config_mean,
                           repeat=5, random_state=1):
    """Permute one feature at a time and return the loss in performance for each"""
    print("Extracting Permutation Feature Importance ",end="",flush=True)
    _fs = []
    _is = []
    np.random.seed(random_state)
    for _ in range(repeat):
        not_feat_cols = ['ID', 'ProblemDir', 'ParamFile']
        full_preds = _pairwisePredict(models, feats, testIDs, timings, config_mean)
        for f in feats.columns:
            if f in not_feat_cols:
                continue
            feats_w_perm = feats.copy()
            feats_w_perm[f] = np.random.permutation(feats_w_perm[f])
            preds = _pairwisePredict(models, feats_w_perm, testIDs, timings, config_mean)
            loss = _vb_loss(full_preds['pred_encs'], preds['pred_encs'], testIDs, timings)
            _fs.append(f)
            _is.append(loss)
        print(".",end="",flush=True)
    alltrials = pd.DataFrame(dict(feature=_fs,importance=_is))
    means = alltrials.groupby(['feature'])['importance'].mean().reset_index()
    print("DONE")
    return means


def trainPredictSingleCombined(
        timings, feat_sets_frames, start_seed = 0, test_frac = 0.25,
        portfolio_size = 2, cycles = 1, hp_iterations = 10, cpus = -1,
        cust_score=False, use_weight=False, single={}, diagnostics=False):
    """"Train and predict using a single random forest classifier

    Repeat split/train/predict a few times. Just use a single classifier

    Pass in single to just do the split/train/pred once, e.g.
    single = {'cycle' : 3, 'split' :'class', 'featureset' : 'f2f'}
    """

    if diagnostics:
        assert False, "diagnostics not implemented (yet) for alternative engines"

    if single:
        iterations = [(single['cycle'], single['featureset'], single['split'])]
    else:
        c_range = range(1,cycles+1)
        f_names = feat_sets_frames.keys()
        splits  = ('class', 'instance')
        iterations = it.product(c_range, f_names, splits)
    pred_frames = []
    for (cycle, featureset, split) in iterations:
        seed = start_seed + cycle
        print(f"Start cycle {cycle:02d}: {featureset:>6}, split by {split:>8}... ",
              end="",flush=True)
        feats = feat_sets_frames[featureset]
        trainIDs, testIDs = _tt_split(feats, split, test_frac, seed)
        portfolio = _makePortfolio(portfolio_size, timings.loc[timings.ID.isin(trainIDs.ID)])
        print("T/T: {}, {}; portf: {}".format( len(trainIDs),len(testIDs),str(portfolio) ) )

        # set up the hyperparameter space
        len_f_col = len(feats.columns) - 3 # without the instance stuff
        hp = _hpRanges(len_f_col)

        # if we're doing a single classifier, let it have the equivalent tuning
        # cycles that the pairwise approach would have
        more_it = hp_iterations * len(list(it.combinations(portfolio,2)))

        # format data and tune model
        trainX, trainY, trainIDs = _makeXY(feats, timings, trainIDs, portfolio)
        model = RandomizedSearchCV(
            RandomForestClassifier(n_jobs = cpus, random_state = seed), hp,
            n_iter = more_it, n_jobs = cpus, random_state = seed,
            scoring = make_scorer(
                _vb_loss, greater_is_better=False,IDs=trainIDs, timings=timings
            ) if cust_score else None,
        )
        model.fit(
            trainX, trainY,
            sample_weight=_sampleWeights(trainIDs,timings) if use_weight else None
        )
        print("TRAINED")

        # make some predictions
        testX, testY, testIDs = _makeXY(feats, timings, testIDs, portfolio)
        df_pred = testIDs.copy()
        df_pred['pred_encs'] = model.predict(testX)
        df_pred['split'] = split
        df_pred['featureset'] = featureset
        df_pred['cycle'] = cycle
        pred_frames.append(df_pred)
    return {'predictions': pd.concat(pred_frames, ignore_index=True)}


def trainPredictPairwiseSeparate(
        timings, feat_sets_frames, start_seed = 0, test_frac = 0.25,
        portfolio_size = 2, cycles = 1, hp_iterations = 10, cpus = -1,
        cust_score=False, use_weight=False, single={}, diagnostics = False):
    """Train and predict using a pairwise approach, but address the sum and pb
    encodings separately

    Repeat split/train/predict a few times. Just use a single classifier

    Pass in single to just do the split/train/pred once, e.g.
    single = {'cycle' : 3, 'split' :'class', 'featureset' : 'f2f'}

    """
    if diagnostics:
        assert False, "Not implementing diagnostics for alternative engines yet"
    if single:
        iterations = [(single['cycle'], single['featureset'], single['split'])]
    else:
        c_range = range(1,cycles+1)
        f_names = feat_sets_frames.keys()
        splits  = ('class', 'instance')
        iterations = it.product(c_range, f_names, splits)
    pred_frames = []
    for (cycle, featureset, split) in iterations:
        seed = start_seed + cycle
        print(f"Start cycle {cycle:02d}: {featureset:>6}, split by {split:>8}... ",
              end="",flush=True)
        feats = feat_sets_frames[featureset]
        trainIDs, testIDs = _tt_split(feats, split, test_frac, seed)

        # HORRIBLE HACK!!!!
        original_timings = timings.copy()
        preds_by_con = []
        for conidx, con in enumerate(['sum', 'pb']):
            sep_times = original_timings.copy()

            # if we're doing one independently, we need to fix the other one for
            # the purpose of scoring, weighitng, etc - choose the single best
            # from the training set
            sep_times['OtherEnc'] = sep_times.Encs.apply(lambda x: x.split("_")[1-conidx])
            train_set_times = sep_times.loc[sep_times.ID.isin(trainIDs.ID)]
            other_grouped = train_set_times.groupby(['OtherEnc'])['OverallTime'].sum().reset_index()
            other_sb = other_grouped.sort_values(['OverallTime']).iloc[0]['OtherEnc']
            sep_times = sep_times.loc[sep_times.OtherEnc==other_sb].drop(columns='OtherEnc')
            print(f"Working on {con}. Setting other constraint to SB of {other_sb}.")

            # now focus on the constraint encoding we're learning
            sep_times['Encs'] = sep_times.Encs.apply(lambda x: x.split('_')[conidx])
            _psize = min(portfolio_size, len(sep_times.Encs.unique()))
            portfolio = _makePortfolio(_psize, sep_times.loc[sep_times.ID.isin(trainIDs.ID)])
            print("T/T: {}, {}; portf: {}".format( len(trainIDs),len(testIDs),str(portfolio) ) )

            # set up the hyperparameter space
            len_f_col = len(feats.columns) - 3 # without the instance stuff
            hp = _hpRanges(len_f_col)

            # pair up the configs
            pairwise_models = {}

            # train the classifiers
            for A,B in it.combinations(portfolio,2):
                print(f"{A}/{B},",end="",flush=True)
                trainX, trainY, trainIDs = _makeXY(feats, sep_times, trainIDs, [A,B])
                model = RandomizedSearchCV(
                    RandomForestClassifier(n_jobs = cpus, random_state = seed), hp,
                    n_iter = hp_iterations, n_jobs = cpus, random_state = seed,
                    scoring = make_scorer(
                        _vb_loss, greater_is_better=False,IDs=trainIDs, timings=sep_times
                    ) if cust_score else None,
                )
                model.fit(
                    trainX, trainY,
                    sample_weight=_sampleWeights(trainIDs,sep_times) if use_weight else None
                )
                pairwise_models[(A,B)] = model
            print("TRAINED")

            # prepare tiebreaker for equal votes - take the config with best mean over training set
            _t = sep_times.loc[sep_times.Encs.isin(portfolio)&sep_times.ID.isin(trainIDs.ID)]
            config_mean = _t.groupby(['Encs'])['OverallTime'].mean().to_dict()
            breaktie = lambda x : sorted(
                x.dropna().tolist(),
                key=lambda config:config_mean[config]
            ).pop(0)

            # make some predictions
            pairwise_choices = testIDs.sort_values(['ID']).copy()
            for ((A,B), model) in pairwise_models.items():
                testX, testY, testIDs = _makeXY(feats, sep_times, testIDs, [A,B])
                pairwise_choices['_v_'.join((A,B))] = model.predict(testX)

            _modes = pairwise_choices.drop(columns=['ID']).mode(axis=1)
            preds_by_con.append(_modes.apply(breaktie, axis=1))
        df_pred = pairwise_choices.loc[:,['ID']]
        df_pred['pred_encs'] = np.array(['_'.join([a,b]) for (a,b) in zip(*preds_by_con)])
        df_pred['split'] = split
        df_pred['featureset'] = featureset
        df_pred['cycle'] = cycle
        pred_frames.append(df_pred)
    return {'predictions' : pd.concat(pred_frames, ignore_index=True)}


def _tt_split(frame, split_policy, test_frac, seed, pick_frac=0.5):
    if split_policy=='class':
        d = frame.groupby(['ProblemDir'])['ParamFile'].count().reset_index(name='n_instances')
        d = d.sort_values(['n_instances','ProblemDir'], ascending=False)
        classes = frame.ProblemDir.unique()
        te_tot = tr_tot = 0
        te_probs = []
        tr_probs = []
        while len(d)>0:
            pickfrom = min(len(d), int(pick_frac * len(classes)))
            row = d[:pickfrom].sample(n=1, random_state=seed).iloc[0]
            prob = row.ProblemDir
            n = row.n_instances
            if te_tot < (test_frac * (te_tot+tr_tot)):
                te_probs.append(prob)
                te_tot += n
            else:
                tr_probs.append(prob)
                tr_tot += n
            d.drop(row.name, axis=0, inplace=True)
        test_data = frame.loc[ frame.ProblemDir.isin(te_probs) , ['ID']]
        train_data = frame.loc[ frame.ProblemDir.isin(tr_probs) , ['ID']]
        assert len(test_data)+len(train_data)==len(frame)
        return train_data, test_data

    elif split_policy=='instance':
        test_data = frame.sort_values(['ID']).sample(frac=test_frac,random_state=seed)
        train_data = frame.loc[~frame.index.isin(test_data.index)]
        return train_data[ ['ID'] ], test_data[ ['ID'] ]


def _hpRanges(nfeats):
    """Set up the hyperparameter search space"""
    return dict(
        criterion = ['gini', 'entropy'],
        n_estimators = [200],
        max_depth = list(range(2,40)),
        max_features = np.linspace(np.log(nfeats), np.sqrt(nfeats), 10).astype(int),
        max_samples = np.linspace(0.1,0.999,20).round(3).astype(float).tolist(),
    )
    
def _makePortfolio(size, times):
    """Make a portfolio with `size` encoding configs from the performance `times` given.
    
    Tries starting with each encs config in turn and then greedily adding whichever other config
    would give the smallest new vb time.  Returns the very best result.

    """
    
    best_portfolio = None
    best_time = None
    all_configs = times.Encs.unique().tolist()
    assert len(all_configs) >= size
    pf_cache = {}
    
    # start with each encs at front of queue in turn
    for start_idx in range(len(all_configs)):
        portfolio = []
        queue = all_configs[start_idx:] + all_configs[:start_idx]
        portfolio.append(queue.pop())
        while (len(queue) > 0) and (len(portfolio) < size):
            vb_now = _portfolioVB(times,portfolio,pf_cache)
            vbs_poss = [(_portfolioVB(times,portfolio+[c],pf_cache),c) for c in queue]
            vbs_poss.sort()
            new_vb, best_extra_encs = vbs_poss.pop(0) # take out the recent winner from the queue
            queue = [x[1] for x in vbs_poss] 
            if new_vb < vb_now:
                portfolio.append(best_extra_encs)
        portfolio_vb_time = _portfolioVB(times,portfolio,pf_cache)
        if (best_portfolio is None) or (portfolio_vb_time < best_time):
            best_portfolio = portfolio
            best_time = portfolio_vb_time
    return best_portfolio


def _portfolioVB(times, configs, cache={}):
    """What's the virtual best time with the given models?"""
    key = tuple(sorted(configs))
    vb = cache.get(key,None)
    if vb is None:
        vb = times.loc[times.Encs.isin(configs)].groupby(['ID'])['OverallTime'].min().sum()
        cache[key] = vb
    return vb


def _makeXY(features, timings, IDs, portfolio):
    """Return the features array (X), target column (Y), and instance IDs"""

    # just consider the encodings in the portfolio and the instances in our training set
    timings = timings.loc[timings.Encs.isin(portfolio)]
    timings = timings.loc[timings['ID'].isin(IDs['ID'])]

    # filter to just keep the row with the lowest median time
    timings = timings.loc[timings.groupby(['ID'])['OverallTime'].idxmin()]

    # this will be the label to predict
    timings = timings.rename(columns=dict(Encs='best_encs'))

    # bring in the features for each instance
    df = timings[['ID','best_encs']].merge(features, on=['ID'])

    # sort by instance ID
    df = df.sort_values(['ID']).reset_index(drop=True)

    # fill in columns that are completely empty - this can happen after
    # splitting when columns are partially NaN; if we don't, then the imputer
    # can change the shape which is bad :-(
    df.loc[:,df.isna().all()] = 0.0

    IDs = df[['ID']]
    X = df.drop(columns=['ProblemDir', 'ParamFile', 'ID', 'best_encs'])
    X = SimpleImputer(strategy='median').fit_transform(X)
    Y = df['best_encs']
    return X, Y, IDs



def _vb_loss(vals_i,vals_j,IDs,timings,**kw):
    """prepare a custom loss function, pre-loaded with the features and timings"""

    tcols = ['ID','OverallTime','Encs']
    data = IDs.loc[vals_i.index]
    data['truevals'] = vals_i
    data['predvals'] = vals_j
    data = data.merge(timings[tcols], left_on=['ID','truevals'], right_on=['ID','Encs'])
    data = data.rename(columns={'OverallTime':'best_time'}).drop(columns=['Encs'])
    data = data.merge(timings[tcols], left_on=['ID','predvals'], right_on=['ID','Encs'])
    data = data.rename(columns={'OverallTime':'pred_time'}).drop(columns=['Encs'])
    score = np.sum(data['pred_time'] - data['best_time'])
    return score


def _sampleWeights(IDs,timings):
    """Assign weightings to samples according to their VB time"""

    calc_ratio = lambda x:x['OverallTime'].max()/x['OverallTime'].min()
    ratios = timings.groupby('ID').apply(calc_ratio).reset_index(name="ratio")
    vbtimes = timings.groupby('ID')['OverallTime'].min().reset_index(name="vbtime")
    wtinfo = IDs.merge(ratios, on=['ID']).merge(vbtimes, on=['ID'])
    weights = np.log10(10+(wtinfo.vbtime*wtinfo.ratio)).astype(int)

    # vbtimes = timings.groupby(['ID'])['OverallTime'].min().reset_index()
    # with_vb = IDs.merge(vbtimes, on=['ID'])
    # weights = np.log10(10*(1+with_vb.OverallTime)).astype(int)
    return weights


        
def evaluatePredictions(predictions, timings, out_dir=".", prefix="", costs_path=None,
                        featuresets=[]):
    """Plots and tables summarising the prediction performance
    """
    all_times = _calculatePredictionTimings(predictions, timings, costs_path)
    meantimes = _means(all_times,featuresets).round(2)
    meantimes.to_csv(os.path.join(out_dir,f"{prefix}-means.csv"),index=False)
    _relativeToVB(meantimes).to_csv(os.path.join(out_dir,f"{prefix}-relmeans.csv"),index=False)
    all_times.to_csv(os.path.join(out_dir,f"{prefix}-alltimes.csv.gz"),index=False)
    

def _calculatePredictionTimings(predictions, timings, costs_path=None):
    # grab the PAR timings for the predictions made
    ptimes = predictions.merge(timings[['ID','Encs','OverallTime']],
                               left_on=['ID','pred_encs'], right_on=['ID','Encs'])
    ptimes = ptimes.rename(columns={'OverallTime':'runtime'})
    ptimes = ptimes.drop(columns=['Encs','pred_encs'])
    if costs_path:
        costs = pd.read_csv(costs_path)
        costs['ID'] = costs.ProblemDir + '/' + costs.ParamFile
        costs = costs.loc[:,['ID','featureset','time']]
        costs = costs.rename(columns={'time':'fetime'})
        ptimes = ptimes.merge(costs, on=['ID','featureset'])
        ptimes['runtime_fe'] = ptimes.runtime + ptimes.fetime

    featuresets = predictions.featureset.unique().tolist()
    result_frames = []
    for cycle in predictions.cycle.unique():
        cycle_test_ids = ptimes.loc[ptimes.cycle==cycle,'ID']
        t = timings.loc[timings.ID.isin(cycle_test_ids)]
        sbc = _findSingleBest(cycle_test_ids,timings)
        w_sbc = t.loc[t.Encs==sbc,['ID','OverallTime']]
        w_sbc = w_sbc.rename(columns={'OverallTime':'time_sbc'})
        w_vbc = t.groupby(['ID'])['OverallTime'].min().reset_index(name="time_vbc")
        w_vwc = t.groupby(['ID'])['OverallTime'].max().reset_index(name="time_vwc")
        w_def = t.loc[t.Encs=='tree_tree',['ID','OverallTime']]
        w_def = w_def.rename(columns={'OverallTime':'time_def'})
        ref_t = w_vbc.merge(w_vwc, on=['ID']).merge(w_sbc, on=['ID']).merge(w_def, on=['ID'])
        ref_t['cycle'] = cycle

        wide = None
        for f in featuresets:
            p = ptimes.loc[
                (ptimes.featureset==f) & (ptimes.cycle==cycle),
                ['ID','split','cycle','runtime','runtime_fe']
            ]
            p = p.rename(columns={'runtime':f'time_{f}','runtime_fe':f'time_{f}_fe'})
            wide = p if wide is None else wide.merge(p,on=['ID','cycle','split'])
        wide_ready = wide.merge(ref_t,on=['ID','cycle'])
        result_frames.append(wide_ready)
    return pd.concat(result_frames, ignore_index=True)



def _findSingleBest(testIDs:pd.Series, all_timings:pd.DataFrame):
    """Figure out the single best configuration over the training instances

    Look at the configurations and timings of instances NOT IN the testIDs
    frame, return the name of the single best configuration
    """

    training_times = all_timings.loc[~all_timings.ID.isin(testIDs)]
    means_per_conf = training_times.groupby(['Encs'])['OverallTime'].mean().reset_index()
    sbc = means_per_conf.sort_values('OverallTime').iloc[0]['Encs']
    return sbc
    

def _means(pred_times, featuresets):
    means = pred_times.groupby(['split']).mean().reset_index().drop(columns=['cycle'])
    allcols = means.columns.to_list()
    refcols = [f'time_{x}' for x in ('vbc','sbc','def','vwc')]
    rawcols = [f'time_{x}' for x in featuresets]
    fexcols = [f'time_{x}_fe' for x in featuresets]
    means = means.reindex(columns=['split']+refcols+rawcols+fexcols)
    return means


def _relativeToVB(mean_times):
    tcols = [c for c in mean_times.columns.to_list() if c.startswith("time_")]
    relmeans = mean_times.copy()
    for c in tcols:
        if c=='time_vbc':
            continue
        relmeans[c] = relmeans[c] / relmeans['time_vbc']
    relmeans['time_vbc'] = 1
    relmeans.columns = [c.replace("time_","") for c in relmeans.columns]
    relmeans = relmeans.round(2)
    return relmeans

                  
if __name__ == '__main__':
    main()
