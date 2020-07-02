import numpy as np
import os, sys




def convert_to_binary(s):
    possible_characters = 'ARNDCQEGHILKMFPSTWYV'
    c_to_int={}
    for i in range(len(possible_characters)):
        c_to_int[possible_characters[i]] = i
    res = []
    iden = np.identity(len(possible_characters))
    for i in range(len(s)):
        res.append(iden[c_to_int[s[i]]])
    # return a 160-d feature vector per sample
    return np.array(res).flatten()

def data_loader(data_dir, i):
    data_files = os.listdir(data_dir)
    y = []
    X = []
    sum_ = 0
    data_files.sort()
    for data_file in data_files:  # make it deterministic
        with open(os.path.join(data_dir, data_file), 'r') as f:
            lines = f.readlines()
            for sample in lines:
                sample = sample.strip().split(",")
                if sample[1] == '-1':
                    y.append(0)
                else:
                    y.append(1)
                    sum_ += 1
                X.append(convert_to_binary(sample[0]))


    X, y = np.array(X), np.array(y)
    #intercept = np.ones((X.shape[0], 1))
    #X = np.concatenate((X, intercept), axis=1)

    # randomly shuffle data
    np.random.seed(i)
    perm = np.random.permutation(len(y))

    return X[perm], y[perm]

def data_loader_adult(data_dir, i):


    training_dir = data_dir + "/adult.train"
    testing_dir = data_dir + "/adult.test"

    inputs = (
        ("age", ("continuous",)),
        ("workclass", ("Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked")),
        ("fnlwgt", ("continuous",)),
        ("education", ("Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool")),
        ("education-num", ("continuous",)),
        ("marital-status", ("Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse")),
        ("occupation", ("Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces")),
        ("relationship", ("Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried")),
        ("race", ("White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black")),
        ("sex", ("Female", "Male")),
        ("capital-gain", ("continuous",)),
        ("capital-loss", ("continuous",)),
        ("hours-per-week", ("continuous",)),
        ("native-country", ("United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"))
    )

    def isFloat(string):
        try:
            float(string)
            return True
        except ValueError:
            return False


    def prepare_data(raw_data):

        X = raw_data[:, :-1]
        y = raw_data[:, -1:]

        # X:
        def flatten_persons_inputs_for_model(person_inputs):
            input_shape = [1, 8, 1, 16, 1, 7, 14, 6, 5, 2, 1, 1, 1, 41]
            float_inputs = []

            for i in range(len(input_shape)):
                features_of_this_type = input_shape[i]
                is_feature_continuous = features_of_this_type == 1

                if is_feature_continuous:
                    # only train with categorical features
                    '''
                    mean = means[i]
                    if isFloat(person_inputs[i]):
                        scale_factor = 1/(2*mean)  # we prefer inputs mainly scaled from -1 to 1.
                        float_inputs.append(float(person_inputs[i])*scale_factor)
                    else:
                        float_inputs.append(mean)
                    '''
                    pass
                else:
                    for j in range(features_of_this_type):
                        feature_name = inputs[i][1][j]
                        if feature_name == person_inputs[i]:
                            float_inputs.append(1.)
                        else:
                            float_inputs.append(0)
            return float_inputs

        new_X = []
        for person in range(len(X)):
            formatted_X = flatten_persons_inputs_for_model(X[person])
            new_X.append(formatted_X)
        new_X = np.array(new_X)

        # y:
        new_y = []
        for i in range(len(y)):
            if y[i] == ">50K" or y[i] == ">50K.":
                new_y.append(1)
            else:
                new_y.append(0)

        new_y = np.array(new_y)

        return (new_X, new_y)


    def generate_dataset(file_path, i):
        data = np.genfromtxt(file_path, delimiter=', ', dtype=str, autostrip=True)
        print("Data {} count: {}".format(file_path, len(data)))
        print(len(data[0]))

        X, y = prepare_data(data)
        print(X[0].shape)
        percent = sum([i for i in y]) * 1.0 /len(y)
        print("Data percentage {} that is >50k: {}%".format(file_path, percent*100))

        np.random.seed(i)
        perm = np.random.permutation(len(y))

        return X[perm], y[perm]

    X_train, y_train = generate_dataset(training_dir, i)
    subset_phd_id = np.where(X_train[:, 21] == 1)[0]
    X_train = np.delete(X_train, subset_phd_id[:200], axis=0)
    y_train = np.delete(y_train, subset_phd_id[:200])
    print("phd:", len(subset_phd_id), "non-phd:", len(X_train)-len(subset_phd_id))

    X_test, y_test = generate_dataset(testing_dir, i)
    subset_phd_id = np.where(X_test[:,21] == 1)[0]
    print("phd:", len(subset_phd_id), "non-phd:", len(X_test)-len(subset_phd_id))


    return (X_train, y_train), (X_test, y_test)



#data_loader_adult("data/adult/raw", 0)