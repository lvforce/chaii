from sklearn.model_selection import StratifiedKFold


def create_folds(train_data, n_splits, seed):
    train_data['Fold'] = -1
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True,
    random_state=seed)

    for k, (train_idx, valid_idx) in enumerate(kfold.split(X=train_data, y=train_data['language'])):
        train_data.loc[valid_idx, 'Fold'] = k

    return train_data

def convert_answers(data):
    return {
        'answer_start': [data[0]],
        'text': [data[1]]
    }

