import argparse
import sys
from graphs.before_processing import missingBarChart, missingHeatmap, capitalBoxPlots, histo
from graphs.info_gain import infoGainChart, entropyTreePlot
from graphs.model_graphs import build_full_data, train_models, confusionMatrices, rocCurve, precisionRecallCurves, kfoldBoxPlots, cleanVsImputed
from workflows import (XGB_workflow, KNN_workflow, DTC_workflow, ETC_workflow, RFC_workflow, SVM_workflow,
                       run_xgboost_kfold_workflow, run_knn_kfold_workflow, run_dtc_kfold_workflow, run_etc_kfold_workflow, run_rfc_kfold_workflow, run_svm_kfold_workflow)

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

# Creates the parser for flags
def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-x", "--xgb", action="store_true", help="Runs the XGB workflow")
    parser.add_argument("-k", "--knn", action="store_true", help="Runs the KNN workflow")
    parser.add_argument("-d", "--dtc", action="store_true", help="Runs the DTC workflow")
    parser.add_argument("-e", "--etc", action="store_true", help="Runs the ETC workflow")
    parser.add_argument("-r", "--rfc", action="store_true", help="Runs the Random Forest workflow")
    parser.add_argument("-s", "--svm", action="store_true", help="Runs the SVM workflow")
   
    parser.add_argument("-f", "--fold", action="store_true", help="Adds the kfold workflow")

    parser.add_argument("-a", "--all", action="store_true", help="Runs all workflows")
    
    parser.add_argument("-g", "--graph", action="store_true", help="Runs all graphs")
    parser.add_argument('-m', '--models', action='store_true', help='Runs all model graphs')
    return parser

# Collects the parser and runs the functions based on what the paser stores as True
def run_args(parser):
    print(f"{txt}\n")
    args = parser.parse_args()
    if args.models:
        df_encoded = build_full_data()
        models = train_models(df_encoded)
        confusionMatrices(models)
        rocCurve(models)
        precisionRecallCurves(models)
        kfoldBoxPlots(models, df_encoded)
        cleanVsImputed(models)
    if args.graph:
        histo()
        missingBarChart()
        missingHeatmap()
        capitalBoxPlots()
        entropyTreePlot()
        infoGainChart()
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
    if args.rfc or args.all:
        RFC_workflow()
        if args.fold:
            run_rfc_kfold_workflow()
    if args.svm or args.all:
        SVM_workflow()
        if args.fold:
            run_svm_kfold_workflow()
    
