import argparse
import sys
from graphs import *
from workflows import *

txt = r"""
  ____                           ____        _        
 / ___|___ _ __  ___ _   _ ___  |  _ \  __ _| |_ __ _ 
| |   / _ \ '_ \/ __| | | / __| | | | |/ _` | __/ _` |
| |__|  __/ | | \__ \ |_| \__ \ | |_| | (_| | || (_| |
 \____\___|_| |_|___/\__,_|___/ |____/ \__,_|\__\__,_|
|  \/  (_)_ __ (_)_ __   __ _                         
| |\/| | | '_ \| | '_ \ / _` |                        
| |  | | | | | | | | | | (_| |                        
|_|  |_|_|_| |_|_|_| |_|\__, |                        
                        |___/                         
By:   Nolan Busch  ** Andrew Doubrava
    Brooke DeVaney ** Jackson Fletcher
    """


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-x", "--xgb", action="store_true", help="Runs the XGB workflow")
    parser.add_argument("-k", "--knn", action="store_true", help="Runs the KNN workflow")
    parser.add_argument("-d", "--dtc", action="store_true", help="Runs the DTC workflow")
    parser.add_argument("-e", "--etc", action="store_true", help="Runs the ETC workflow")

    parser.add_argument("-f", "--fold", action="store_true", help="Adds the kfold workflow")

    parser.add_argument("-a", "--all", action="store_true", help="Runs all workflows")

    return parser

def run_args(parser):
    missingBarChart()
    missingHeatmap()
    capitalBoxPlots()
    entropyTreePlot()
    infoGainChart()
    args = parser.parse_args()
    print(f"{txt}\n")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    missingBarChart()
    if args.xgb or args.all:
        XGB_workflow()
        if args.fold:
            run_xgboost_kfold_workflow()
    if args.knn or args.all:
        KNN_workflow()
        if args.fold:
            run_knn_kfold_workflow()
    if args.dtc or args.all:
        DTC_workflow()
        if args.fold:
            run_dtc_kfold_workflow()
    if args.etc or args.all:
        ETC_workflow()
        if args.fold:
            run_etc_kfold_workflow()
    
