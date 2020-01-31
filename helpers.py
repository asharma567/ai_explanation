import matplotlib.pyplot as plt #for custom graphs at the end 
import numpy as np

def rf_feature_importance_plot(feature_names, model, top_x=None):
    '''
    Ploting top x most important features
    
    I: feature name (list), model (fitted tree-based model), top x (int)
    Plots: bar plot of features importance with standard error among trees.
    
    *importances weights (normalized by sum) paired with names
    '''

    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    
    if top_x:
        indices = np.argsort(importances)[::-1][:top_x]
    else:
        indices = np.argsort(importances)[::-1]
    
    _plot_barchart_with_error_bars_rf(importances[indices], std[indices], feature_names[indices])
        
    return None

def _plot_barchart_with_error_bars_rf(importance_weights, errors, feature_names):
    '''
    I: importance_weights (np array), errors (np array), feature_names
    O: None
    does the actual plotting of feature weights
    goes with feature_importance_plot
    '''
    
    #if we don't reverse everything it'll plot it upside-down
    importance_weights_reversed = importance_weights[::-1] 
    errors_reversed = errors[::-1]
    feature_names_reversed = feature_names[::-1] 

    number_of_features = range(len(importance_weights_reversed))

    plt.figure(figsize=(12,10))
    plt.title("Feature importances")
    plt.barh(
        number_of_features, 
        importance_weights_reversed,
        color="r", 
        xerr=errors_reversed, 
        align="center"
    )
    plt.yticks(number_of_features, feature_names_reversed)
    plt.show()


def gbm_feature_importance_plot(feature_names, model, top_x=None):
    '''
    Ploting top x most important features
    
    I: feature name (list), model (fitted tree-based model), top x (int)
    Plots: bar plot of features importance with standard error among trees.
    
    *importances weights (normalized by sum) paired with names
    '''

    importances = model.feature_importances_
#     std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    std =None
    if top_x:
        indices = np.argsort(importances)[::-1][:top_x]
    else:
        indices = np.argsort(importances)[::-1]
    
    _plot_barchart_with_error_bars_gbm(importances[indices], std, feature_names[indices])
        
    return None

def _plot_barchart_with_error_bars_gbm(importance_weights, errors, feature_names):
    '''
    I: importance_weights (np array), errors (np array), feature_names
    O: None
    does the actual plotting of feature weights
    goes with feature_importance_plot
    '''
    
    #if we don't reverse everything it'll plot it upside-down
    importance_weights_reversed = importance_weights[::-1] 
#     errors_reversed = errors[::-1]
    feature_names_reversed = feature_names[::-1] 

    number_of_features = range(len(importance_weights_reversed))

    plt.figure(figsize=(12,10))
    plt.title("Feature importances")
    plt.barh(
        number_of_features, 
        importance_weights_reversed,
        color="r", 
#         xerr=errors_reversed, 
        align="center"
    )
    plt.yticks(number_of_features, feature_names_reversed)
    plt.show()