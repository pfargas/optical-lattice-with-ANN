#Search params
def dcv_search_params(kf5, kf4, model, df, y, labels, features, max_iter):
    warnings.filterwarnings('ignore',category=ConvergenceWarning)
    search_params = {'LogisticRegression': ({'C': np.logspace(-3, 3, 7), 'penalty': ['l2']}),
                     'LDA': ({'solver': ['svd', 'lsqr']}),
                     'SVClineal': ({'C': [0.1, 0.5, 1, 2, 5, 10, 100, 1000], 'gamma': [5, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]}),
                     'SVCrbf': ({'C': [0.1, 0.5, 1, 2, 5, 10, 100, 1000], 'gamma': [5, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]}),
                     'RandomForest': ({'bootstrap': [True],
                                       #'bootstrap': [True, False],
                                       'max_depth': [5,20],
                                       #'max_depth': [5, 10, 15, 20, None],
                                       'max_features': [None],
                                       'min_samples_leaf': [1,4],
                                       'min_samples_split': [2,10],
                                       'n_estimators': [100,1000]}),
                                       #'min_samples_leaf': [1, 2, 4],
                                       #'min_samples_split': [2, 5, 10],
                                       #'n_estimators': [100, 200, 1000, 2000]}),
                                       
                     'MLP': ({'HSizes': [2,20,200],
                                        'activation': ['relu', 'tanh'],
                                        'Solver': ['sgd','adam'],
                                        'alpha': [0.1,0.0001],
                                        'LRate_init': [0.001,0.1]}),
                     'Dummy': ({'strategy': ['stratified'], 'random_state': [42]})}
    param_grid = search_params[model]
    keys, values = zip(*param_grid.items())
    params_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    auc_val_score_old = 0
    best_params = []
    #print(len(params_combinations))
    for pram_combination in params_combinations:
        auc_val_score, std_auc_val, mean_auc_test, std_auc_test = dcv_evaluate_model(kf5, kf4, df, y, labels, model, features, pram_combination, max_iter)
        if auc_val_score > auc_val_score_old:
            auc_val_score_old = auc_val_score
            best_params = pram_combination
            
            auc_val_std=std_auc_val
            auc_test=mean_auc_test
            auc_test_std=std_auc_test

    return best_params, auc_val_score_old, auc_val_std, auc_test, auc_test_std