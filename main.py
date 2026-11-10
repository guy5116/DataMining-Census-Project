from workflows import *
    
if __name__ == "__main__":
    XGB_workflow()
    run_xgboost_kfold_workflow()
    KNN_workflow()
    run_knn_kfold_workflow()
    DTC_workflow()
    run_dtc_kfold_workflow()
    ETC_workflow()
    run_etc_kfold_workflow()
    
