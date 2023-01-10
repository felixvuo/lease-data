#!/usr/bin/env python3
#
# Go through outputs in the problems/ directory tree to collate
# timing information and instance features

import argparse
import csv
import os
import pandas as pd
import re
import utils

RE_INFO_FILE = re.compile(
    '''[^/]+/([^/]+)/([^\.]+\.param)\.sum-([\w]+)\.pb-(\w+)\.(\d+)\.info'''
)
RE_F2F_FILE = re.compile(
    '''[^/]+/([^/]+)/([^\.]+\.param)\.f2f\.csv'''
)
RE_SAROFE_FILE = re.compile(
    '''[^/]+/([^/]+)/([^\.]+\.param)\.sarofe\.csv'''
)
RE_F2F_TIMES = re.compile(
    '''[^/]+/([^/]+)/([^\.]+\.param)\.featuretimes'''
)
RE_SAROFE_TIME = re.compile(
    '''[^/]+/([^/]+)/([^\.]+\.param)\.sarofe\.time'''
)

ap = argparse.ArgumentParser(
    description="Look through the outputs in the problems/ directory and create CSVs")
ap.add_argument('--do-infos', '-i', action='store_true', help="collate info files")
ap.add_argument('--do-f2f', '-f', action='store_true', help="collate fzn2feat features")
ap.add_argument('--do-sarofe', '-s', action='store_true', help="collate sarofe features")
ap.add_argument('--do-fe-timings', '-t', action='store_true', help="collate fe times")
ap.add_argument('--infos-out', type=str, default="infos.csv.gz",
    help="Extract timings from .info files and store them in given csv file")
ap.add_argument('--f2f-out',  type=str, default="features-f2f.csv.gz",
    help="Save features from .f2f.csv files into given csv file" )
ap.add_argument('--sarofe-all-out', type=str, default="features-sarofe-all.csv.gz",
    help="Save features from .sarofe.csv files into given csv file" )
ap.add_argument('--f2fsr-out', type=str, default="features-sarofe-f2fsr.csv.gz",
    help="Save f2f-inspired features from .sarofe.csv files into given csv file" )
ap.add_argument('--sumpb-out', type=str, default="features-sarofe-sumpb.csv.gz",
    help="Save sum & pb only features from .sarofe.csv files into given csv file" )
ap.add_argument('--fe-timings-out', type=str, default="fe-times.csv.gz",
    help="Save the timings for feature extraction in this csv file" )
args = ap.parse_args()


def main():
    if args.do_infos:
        print("Collating .info files")
        collateInfos()
    if args.do_f2f:
        print("Collating fzn2feat feature files")
        collateFzn2Feat()
    if args.do_sarofe:
        print("Collating sarofe features")
        collateSarofe()
    if args.do_fe_timings:
        print("Collating feature extraction timings")
        collateFeTimings()


def collateInfos():    
    filepaths = [os.path.join(d,f)
                 for (d, dirs, files) in os.walk('problems')
                 for f in (dirs + files) if f.endswith(".info")]
    frames = []
    for i, path in enumerate(filepaths):
        if (i%1000)==0:
            print(f"Doing file #{i}")
        with open(path, 'rt') as infofile:
            g = RE_INFO_FILE.match(path).groups()
            entry = {
                'ProblemDir' : g[0],
                'ParamFile' : g[1],
                'EncSum' : g[2],
                'EncPB' : g[3],
                'Encs' : "%s_%s" % tuple(g[2:4]),
                'Run' : g[4]
                }
            for line in [x.strip() for x in infofile.readlines()]:
                if not (":" in line):
                    print("missing colon in",path)
                    continue
                k,v = tuple(line.split(":"))
                if k in utils.TIMING_COLS:
                    entry[k]=v
            frames.append(
                pd.DataFrame.from_dict({k:[v] for k,v in entry.items()}))
    df = pd.concat(frames, ignore_index=True)          
    df.to_csv(args.infos_out, index=False)

                
def collateFzn2Feat():
    with open(args.f2f_out, "wt") as f:
        feat_field_names = utils.FZN_FEATURES + utils.ID_COLS[:2]
        filepaths = [os.path.join(d,f)
                     for (d, dirs, files) in os.walk('problems')
                     for f in (dirs + files) if f.endswith(".f2f.csv")]
        out = csv.writer(f)
        out.writerow(feat_field_names)
        for fpath in filepaths:
            with open(fpath,"r") as featfile:
                # first line should have the comma-separated values
                feat_lines = featfile.readlines()
                if len(feat_lines) < 1:
                    print("emptyfile,%s" % fpath)
                    continue
                elif "inf,inf,inf" in feat_lines[0]:
                    print("nofeatures,%s" % fpath)
                    continue
                feat_vals = feat_lines[0].strip().split(",")
                dir_and_param = list(RE_F2F_FILE.match(fpath).groups()[:2])
                out.writerow(feat_vals + dir_and_param)


def collateSarofe():
    filepaths = [os.path.join(d,f)
                 for (d, dirs, files) in os.walk('problems')
                 for f in (dirs + files) if f.endswith(".sarofe.csv")]
    frames = []
    for fpath in filepaths:
        dir_par = list(RE_SAROFE_FILE.match(fpath).groups()[:2])
        try:
            df_instance = pd.read_csv(fpath)
        except:
            print("Found nothing in "+fpath+", skipping")
            continue
        df_instance[utils.ID_COLS[0]] = dir_par[0]
        df_instance[utils.ID_COLS[1]] = dir_par[1]
        frames.append(df_instance)
    df_all = pd.concat(frames, ignore_index=True)
    df_all.to_csv(args.sarofe_all_out, index=False)

    # now just the f2fsr features (they should not contain sarofe_prefix)
    f2fsr_cols = [c for c in df_all.columns if not c.startswith('sarofe_')]
    df_all[ f2fsr_cols ].to_csv(args.f2fsr_out, index=False)

    # and now just the sum/pb specific features
    sumpb_cols = utils.ID_COLS + \
                 [c for c in df_all.columns if c.startswith('sarofe_')]
    df_all[ sumpb_cols ].to_csv(args.sumpb_out, index=False)


        
def collateFeTimings():
    timings = pd.DataFrame(columns=utils.ID_COLS+['feat_source','time'])

    # first the sarofe times, which just have the seconds
    filepaths = [os.path.join(d,f)
                 for (d, dirs, files) in os.walk('problems')
                 for f in (dirs + files) if f.endswith(".sarofe.time")]
    for fpath in filepaths:
        pdir, pfile = tuple(RE_SAROFE_TIME.match(fpath).groups()[:2])
        try:
            fe_time_sec = float(open(fpath,'rt').read().strip())
            timings = timings.append(
                {'ProblemDir'  : pdir,
                 'ParamFile'   : pfile,
                 'feat_source' : 'sarofe',
                 'time'        : fe_time_sec
                },
                ignore_index=True
            )
        except:
            print("Could not extract time from "+fpath)
            continue

    # now let's try for the f2f feature timings
    # first the sarofe times, which just have the seconds
    filepaths = [os.path.join(d,f)
                 for (d, dirs, files) in os.walk('problems')
                 for f in (dirs + files) if f.endswith(".featuretimes")]
    for fpath in filepaths:
        pdir, pfile = tuple(RE_F2F_TIMES.match(fpath).groups()[:2])
        try:
            with open(fpath, 'rt') as ftimes:
                # fzn creation, then f2f extraction
                two_times = [float(l.strip().split(':')[-1]) for l in ftimes.readlines()]
                assert(len(two_times)==2)
                fe_time_sec = sum(two_times)
                timings = timings.append(
                    {'ProblemDir'  : pdir,
                     'ParamFile'   : pfile,
                     'feat_source' : 'f2f',
                     'time'        : fe_time_sec
                    },
                    ignore_index=True
                )
        except:
            print("Could not extract time from "+fpath)
            continue
    timings.to_csv(args.fe_timings_out, index=False)

        
if __name__ == '__main__':
    main()
    
            
