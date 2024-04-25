from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def prediction_matrices(model, X_test, X_train, y_test, y_train, model_name = " "):
    """
    Generate prediction matrices and evaluation metrics for a given model.

    Parameters:
    model (object): The trained classification model.
    X_test (array-like): Test features.
    X_train (array-like): Train features.
    y_test (array-like): True labels for the test set.
    y_train (array-like): True labels for the train set.
    model_name (str, optional): Name of the model. Default is an empty string.

    Prints:
    - Classification report on the train set.
    - Accuracy and ROC score on the test set.
    - Classification report on the test set.
    - Confusion matrix plots for both test and train data.

    Example:
    prediction_matrices(model, X_test, X_train, y_test, y_train, model_name="Logistic Regression")
    """

    # Make predictions for train set
    y_pred_train = model.predict(X_train)

    # Make predictions for train set
    y_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    
    ## Print classification report on the train data
    print("Report on train set : \n", 
    classification_report(y_train, y_pred_train)) 
    print("--------"*10)

    # Print accuracy of our model
    print("Accuracy on test set:", round(accuracy_score(y_test, y_pred), 2))
    print("ROC on test set:", round(roc_auc_score(y_test, y_proba[:,1]), 2))
    print("Report on test set : \n", 
    classification_report(y_test, y_pred)) 

    #confusion matrix train
    conf_matrix = confusion_matrix(y_train, y_pred_train)

    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", cbar=True)
    plt.xlabel('Prediction')
    plt.ylabel('True')

    try:
        plt.title('Confusion Matrix Train Data ' + model_name)
        
    except:
        print("Model_name not a string")
        plt.title('Confusion Matrix Train Data') 
    plt.show()

    #confusion matrix test
    conf_matrix = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", cbar=True)
    plt.xlabel('Prediction')
    plt.ylabel('True')
    try:
        plt.title('Confusion Matrix Test Data ' + model_name)
    except:
        plt.title('Confusion Matrix Test Data')
        print("Model_name not a string")
    plt.show()

    pass



def plot_categorical_features(df, features, hue):
    """
    Plots categorical features in a DataFrame.

    This function creates two subplots for each categorical feature: 
    one showing the absolute numbers of classification outcomes, 
    and the other showing the relative view for each categorical value.

    Parameters:
    df (pandas.DataFrame): The DataFrame to plot.
    features (list): A list of strings representing the names of the categorical columns to plot.
    hue (str): The name of the column in df to plot against.
    """
    # Check if the hue column exists in the DataFrame
    if hue not in df.columns:
        print(f"Error: The DataFrame does not contain a '{hue}' column.")
        return

    # Define the order of the colors
    hue_order = df[hue].value_counts().index

    # Loop over each categorical column
    for plot in features:
        # Create subplots for absolute and relative views
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Absolute view: Countplot
        sns.countplot(df, x=plot, hue=hue, ax=axes[0], hue_order=hue_order)
        axes[0].set_title(f'Absolute numbers: {hue}')

        # Rotate x-axis labels if there are more than 3 unique values or if the labels are long
        if df[plot].nunique() > 3 or df[plot].nunique() > 2 and len(df[plot].unique()[0]) > 10:
            axes[0].tick_params(axis='x', rotation=45)
            axes[0].set_xticklabels(axes[0].get_xticklabels(), ha='right')

        # Relative view: Barplot
        df_prop = df.groupby(plot)[hue].value_counts(normalize=True).rename('proportion').reset_index()
        sns.barplot(data=df_prop, x = plot, y='proportion', hue = hue, ax = axes[1], hue_order=hue_order)
        axes[1].set_title(f'Relative view for each categorical value: {hue}')

        # Rotate x-axis labels if there are more than 3 unique values or if the labels are long
        if df[plot].nunique() > 3 or df[plot].nunique() > 2 and len(df[plot].unique()[0]) > 10:
            axes[1].tick_params(axis='x', rotation=45)
            axes[1].set_xticklabels(axes[1].get_xticklabels(), ha='right')

        # Adjust layout
        plt.tight_layout()

        # Show plot
        plt.show()
    
    return None

def save_metrics(model_name,accuracy=None,precision=None,recall=None,F1=None,ROC_AUC=None):
    '''
    Save model metrics to ./data/df_metrics.pkl.

    Parameters:
    - model_name (str): Name of the model.
    - accuracy (float): Accuracy score of the model.
    - precision (float): Precision score of the model.
    - recall (float): Recall score of the model.
    - F1 (float): F1 score of the model.
    - ROC_AUC (float): ROC AUC score of the model.

    Returns:
    - Prints a message indicating successful saving of metrics.
    '''
    # check if df_metrics already exists, create it if not
    columns = ['model', 'accuracy', 'precision','recall', 'F1', 'ROC_AUC']
    try:
            df_metrics = pd.read_pickle('./data/df_metrics.pkl')
            print('df loaded')
    except:
            df_metrics = pd.DataFrame(columns=columns)
            df_metrics.to_pickle('./data/df_metrics.pkl')
            print('df created')

    # add model's metrics to df_metrics
    new_row = {'model': model_name, 
            'accuracy': accuracy, 
            'precision': precision, 
            'recall': recall, 
            'F1': F1, 
            'ROC_AUC': ROC_AUC,
            }
    
    # save new row to df_metrics
    index = new_row['model']
    df_metrics.loc[index] = new_row

    # save df_metrics.pkl
    df_metrics.to_pickle('./data/df_metrics.pkl')
    
    return print(f'Metrics for {model_name} saved successfully!')