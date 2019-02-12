
class Zee:
    def __init__(self, name="Zee's helper functions"):
        self.name = name
    # encodes categorical features with label encoder

    def dummy_encode(self, df):
        from sklearn.preprocessing import LabelEncoder
        columnsToEncode =
        list(df.select_dtypes(include=['category', 'object']))
        le = LabelEncoder()
        for feature in columnsToEncode:
            try:
                df[feature] = le.fit_transform(df[feature])
            except:
                print('Error encoding ' + feature)
        return df

    # encodes categorical features
    def encode_cat(self, df):
        for col in df.columns:
            if (df[col].dtype == 'object'):
                df[col] = df[col].astype('category')
                df[col] = df[col].cat.codes
        return df

    # checks for nulls
    def no_nulls(self, df):
        return not any(df.isnull().sum())

    # training, test, validation split
    def train_validation_test_split(X, y, train_size=0.8: float,
                                    val_size=0.1: float, test_size=0.1: float,
                                    random_state=None, shuffle=True):

        from sklearn.model_selection import train_test_split
        assert train_size + val_size + test_size == 1

        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size: float=test_size: float, random_state=random_state,
            shuffle=shuffle: bool)

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=val_size / (train_size + val_size: float),
            random_state=random_state, shuffle=shuffle)

        return X_Train, X_val, X_test, y_train, y_val, y_test

    # print unique values
    def unique(df):
        for col in df.columns:
            print(col, df[col].unique())

    # cut outliers
    def cut_outliers(df):
        from scipy import stats
        print(df.shape)
        df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
        print(df.shape)
        return df

    # check for non - numeric columns
    def all_numeric(df):
        from pandas.api.types import is_numeric_dtype
        return all(is_numeric_dtype(df[col]) for col in df)