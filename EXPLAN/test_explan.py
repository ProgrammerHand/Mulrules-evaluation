import explan
from utils import *
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

def main():
    # Reading and preparing data set
    path_data = 'datasets/'
    dataset_name = 'adult.csv'
    dataset = prepare_adult_dataset(dataset_name, path_data)

    # Splitting the data set into train and test sets
    X, y = dataset['X'], dataset['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Creating black-box model
    blackbox = GradientBoostingClassifier(random_state=42)
    blackbox.fit(X_train, y_train)

    # Selecting instance to explain
    index = 10
    instance2explain = X_test[index]

    # EXPLAN hyper-parameters
    N_samples = 3000
    tau = 250
    amount = 2
    explanations = []
    info = []
    while len(explanations) < amount:
    # Explaining instance x using EXPLAN
        exp_EXPLAN, info_EXPLAN = explan.Explainer(instance2explain,
                                               blackbox,
                                               dataset,
                                               N_samples=N_samples,
                                               tau=tau)
        if all(old_exp != exp_EXPLAN for old_exp in explanations):
            explanations.append(exp_EXPLAN)
            info.append(info_EXPLAN)

    # Reporting the results
    for exp, inf in zip(explanations, info):
        dfX2E = build_df2explain(blackbox, X_test, dataset).to_dict('records')
        dfx = dfX2E[index]
        # print('x = %s' % dfx)
        print('e = %s' % exp[1])
        # print('Black-box prediction = %s' % inf['y_x_bb'])
        # print('C4.5 Tree prediction = %s' % inf['y_x_dt'])

if __name__ == "__main__":
    main()
