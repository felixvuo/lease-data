import argparse
import logging
import numpy as np
import os
import pandas as pd
import pickle
import re
import yaml

from fullcycle import prepareFeatureFrames as prepFFs
from autofolio.autofolio import AutoFolio
from autofolio.facade.af_csv_facade import AFCsvFacade

ap = argparse.ArgumentParser(
    description="Run AutoFolio on one batch of data")
ap.add_argument('config_path', type=str, 
    help="The yaml file with all the configuration for running the ML")
ap.add_argument('predictions_for_split', type=str,
    help="Use this predictions file to create train/test files separately " +
    "according to the splits in the prediction file")
ap.add_argument('batch_cfs', type=str,
    help="Just do a single run.  Pass in: cycle,featureset,split")
ap.add_argument('wallclock_limit', type=int,
    help="wallclock_limit for AutoFolio")

def main():
    # collect config and arguments
    args = ap.parse_args()
    cfs_str = args.batch_cfs.strip().split(',')
    cycle = int(cfs_str[0])
    fset = cfs_str[1]
    split = cfs_str[2]
    conf = yaml.load(open(args.config_path,"rt"), yaml.SafeLoader)

    # prepare preformance and feature data
    timings_path = os.path.join(
        conf['timings_dir_path'],
        f"times_par{conf['penalty']}_to{conf['total_timeout']}.csv.gz"
    )
    t = pd.read_csv(timings_path)

    feat_dfs_ = prepFFs(conf['featuresets'], conf['feature_dir_path'], t)
    df_feat = feat_dfs_[fset]
    df_feat['instance'] = df_feat.ProblemDir.str.cat(
        df_feat.ParamFile.str.replace(re.compile(r'\.param$'),''),
        sep="/"
    )
    df_feat = df_feat.drop(columns=['ProblemDir','ParamFile','ID'])
    df_feat = df_feat[ ['instance'] + df_feat.columns.tolist()[:-1] ]

    t = t.loc[t.ID.isin(list(feat_dfs_.values())[0].ID.unique())]
    t['instance'] = t.ID.str.replace(re.compile(r'\.param'),'')
    t = t.drop(columns=['ParamFile', 'ProblemDir', 'ID'])
    df_perf = t.pivot(index='instance', columns='Encs', values='OverallTime')

    # determine which instances are needed in the train and test sets
    df_pred = pd.read_csv(args.predictions_for_split)
    df_pred = df_pred.loc[
        (df_pred.cycle==cycle) & (df_pred.featureset==fset) & (df_pred.split==split)
    ]
    test_inst = df_pred.ID.str.replace(re.compile(r'\.param'),'')
    train_inst = df_perf.loc[~df_perf.index.isin(test_inst)].index

    # Prepare CSVs and run autofolio
    doAF(train_inst, test_inst, df_perf, df_feat,
         args.wallclock_limit, conf['total_timeout'], args.batch_cfs)


def doAF(train_inst, test_inst, df_perf, df_feat,
         wallclock_limit, alg_timeout, batch):
    """Train Autofolio and make predictions"""

    # temporary filenames
    fn_train_perf = f"_train_perf_{batch}.csv"
    fn_train_feat = f"_train_feat_{batch}.csv"
    fn_model = f"_afmodel_{batch}.pkl"
    fn_preds = f"af-preds-{batch}.csv"
    
    # prepare csv with just training instances
    df_perf.loc[df_perf.index.isin(train_inst)].to_csv(fn_train_perf, index=True)
    df_feat.loc[df_feat.instance.isin(train_inst)].to_csv(fn_train_feat, index=False)

    # set up an AF instance
    af_fac = AFCsvFacade(
        perf_fn = fn_train_perf, feat_fn = fn_train_feat,
        objective = 'runtime', runtime_cutoff = alg_timeout,
        maximize = False, seed = 1
    )

    # must do this to set up hyperparams before tuning
    af_fac.fit()

    # tune HPs and re-fit
    config = af_fac.tune(wallclock_limit = wallclock_limit, runcount_limit = int(1e6))
    af_fac.fit(config = config, save_fn = fn_model)

    # re-load the saved model
    with open(fn_model,'rb') as f:
        scenario, feature_pre_pipeline, pre_solver, selector, config = pickle.load(f)
    for fpp in feature_pre_pipeline:
        fpp.logger = logging.getLogger("Feature Preprocessing")
    af = AutoFolio()

    # make predictions on test set
    test_feats = df_feat.loc[df_feat.instance.isin(test_inst)]
    preds = pd.DataFrame(columns=['ID','pred_encs','split','featureset','cycle'])
    cycle, fset, split = batch.split(',')
    for i in range(len(test_feats)):
        feature_vec = np.array([test_feats.iloc[i,1:]])
        scenario.feature_data = pd.DataFrame(
            feature_vec, index=["pseudo_instance"], columns=scenario.features)
        scenario.instances = ["pseudo_instance"]
        
        pred = af.predict(
            scenario=scenario, config=config, feature_pre_pipeline=feature_pre_pipeline,
            pre_solver=None, selector=selector)
        preds.loc[len(preds)] = [test_feats.iloc[i,0]+".param", pred['pseudo_instance'][0][0], split, fset, cycle]

    # write out predictions
    preds.to_csv(fn_preds, index=False)

    # clean up temp files
    os.remove(fn_train_perf)
    os.remove(fn_train_feat)
    os.remove(fn_model)


if __name__ == '__main__':
    main()

