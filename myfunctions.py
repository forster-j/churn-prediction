from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score
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



def plot_categorical_features(df, features, hue=None):
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
