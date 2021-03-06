import os
import h5py
import numpy as np
from matplotlib import pyplot as plt

from tensorflow.keras.models import load_model, Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Activation, Dense, Dropout, ZeroPadding2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization, ELU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import KFold, StratifiedShuffleSplit

class Model:
    def __init__(self, config):
        self.config = config
        self.threshold = self.config["threshold"]
        self.exp = 'multicam_lr{}_batchs{}_batchnorm{}_w0_{}'.format(self.config["learning_rate"], 
            self.config["mini_batch_size"], self.config["batch_norm"], self.config["weight_0"])

        # initialize VGG16 feature extractor model
        self.model = Sequential()
        self.model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 20)))
        self.model.add(Conv2D(64, (3, 3), activation='relu', name='conv1_1'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(64, (3, 3), activation='relu', name='conv1_2'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(128, (3, 3), activation='relu', name='conv2_1'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(128, (3, 3), activation='relu', name='conv2_2'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_1'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_2'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_3'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_1'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_2'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_3'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_1'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_2'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_3'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        
        self.model.add(Flatten())
        self.model.add(Dense(self.config["num_features"], name='fc6', kernel_initializer='glorot_uniform'))

        # weight intiialization
        layerscaffe = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1',
            'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3',
            'conv5_1', 'conv5_2', 'conv5_3', 'fc6', 'fc7', 'fc8']
        h5 = h5py.File(self.config["vgg_16_weights"], 'r')

        layer_dict = dict([(layer.name, layer) for layer in self.model.layers])

        # copy the weights stored in the 'vgg_16_weights' file to the feature extractor part of the VGG16
        for layer in layerscaffe[:-3]:
            w2, b2 = h5['data'][layer]['0'], h5['data'][layer]['1']
            w2 = np.transpose(np.asarray(w2), (2, 3, 1, 0))
            w2 = w2[::-1, ::-1, :, :]
            b2 = np.asarray(b2)
            layer_dict[layer].set_weights((w2, b2))

        # copy the weights of the first fully-connected layer (fc6)
        layer = layerscaffe[-3]
        w2, b2 = h5['data'][layer]['0'], h5['data'][layer]['1']
        w2 = np.transpose(np.asarray(w2), (1, 0))
        b2 = np.asarray(b2)
        layer_dict[layer].set_weights((w2, b2))
        
        # initialize classifier
        adam = Adam(lr=self.config["learning_rate"], beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0005)
        self.model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

        extracted_features = Input(shape=(self.config["num_features"],),
                    dtype='float32', name='input')
        
        if self.config["batch_norm"]:
            x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(extracted_features)
            x = Activation('relu')(x)
        else:
            x = ELU(alpha=1.0)(extracted_features)
        
        x = Dropout(0.9)(x)
        x = Dense(4096, name='fc2', init='glorot_uniform')(x)
        
        if self.config["batch_norm"]:
            x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
            x = Activation('relu')(x)
        else:
            x = ELU(alpha=1.0)(x)

        x = Dropout(0.8)(x)
        x = Dense(1, name='predictions', init='glorot_uniform')(x)
        x = Activation('sigmoid')(x)

        self.classifier = Model(input=extracted_features, output=x, name='classifier')
        self.classifier.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

        # load model from checkpoints
        print("loading checkpoints...")
        self.classifier = load_model(self.config["model_checkpoints_path"])
        print("Checkpoints loaded.")

    def predict(self, input_features):
        """
        method to predict a single batch of input features
        """
        predicted = self.classifier.predict(input_features)
        for i in range(len(predicted)):
            if predicted[i] < self.config["threshold"]:
                predicted[i] = 0
            else:
                predicted[i] = 1

        predicted = np.asarray(predicted).astype(int)
        if predicted[0] == 0:
            return True
        else:
            return False

    def sample_from_dataset(self, X, y, zeroes, ones):
        """
        Samples from X and y using the indices obtained from the arrays
        all0 and all1 taking slices that depend on the fold, the slice_size
        the mode.
        Input:
        * X: array of features
        * y: array of labels
        * all0: indices of sampled labelled as class 0 in y
        * all1: indices of sampled labelled as class 1 in y
        * fold: integer, fold number (from the cross-validation)
        * slice_size: integer, half of the size of a fold
        * mode: 'train' or 'test', used to choose how to slice
        
        if mode == 'train':
            s, t = 0, fold*slice_size
            s2, t2 = (fold+1)*slice_size, None
            temp = np.concatenate((
                np.hstack((all0[s:t], all0[s2:t2])),
                np.hstack((all1[s:t], all1[s2:t2]))
            ))
        elif mode == 'test':
            s, t = fold*slice_size, (fold+1)*slice_size
            temp = np.concatenate((all0[s:t], all1[s:t])) 
        """

        indices = np.concatenate([zeroes, ones], axis=0)
        sampled_X = X[indices]
        sampled_y = y[indices]
        return sampled_X, sampled_y

    def divide_train_val(self, zeroes, ones, val_size):
        rand0 = np.random.permutation(len(zeroes))
        train_indices_0 = zeroes[rand0[val_size//2:]]
        val_indices_0 = zeroes[rand0[:val_size//2]]
        rand1 = np.random.permutation(len(ones))
        train_indices_1 = ones[rand1[val_size//2:]]
        val_indices_1 = ones[rand1[:val_size//2]]
        return (train_indices_0, train_indices_1, val_indices_0, val_indices_1)

    def plot_training_info(self, case, metrics, save, history):
        """
        Function to create plots for train and validation loss and accuracy
        Input:
        * case: name for the plot, an 'accuracy.png' or 'loss.png' 
        will be concatenated after the name.
        * metrics: list of metrics to store: 'loss' and/or 'accuracy'
        * save: boolean to store the plots or only show them.
        * history: History object returned by the Keras fit function.
        """
        val = False
        if 'val_acc' in history and 'val_loss' in history:
            val = True
        
        plt.ioff()
        if 'accuracy' in metrics:     
            fig = plt.figure()
            plt.plot(history['acc'])
            if val: plt.plot(history['val_acc'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            if val: 
                plt.legend(['train', 'val'], loc='upper left')
            else:
                plt.legend(['train'], loc='upper left')
            if save == True:
                plt.savefig(case + 'accuracy.png')
                plt.gcf().clear()
            else:
                plt.show()
            plt.close(fig)

        # summarize history for loss
        if 'loss' in metrics:
            fig = plt.figure()
            plt.plot(history['loss'])
            if val: plt.plot(history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            #plt.ylim(1e-3, 1e-2)
            plt.yscale("log")
            if val: 
                plt.legend(['train', 'val'], loc='upper left')
            else:
                plt.legend(['train'], loc='upper left')
            if save == True:
                plt.savefig(case + 'loss.png')
                plt.gcf().clear()
            else:
                plt.show()
            plt.close(fig)

    def train_model_combined(self):
        """
        method to train fall detection model with all datasets
        """
        e = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=0, mode='auto')

        # Load features and labels per dataset
        h5features_multicam = h5py.File(self.config["features_file_multicam"], 'r')
        h5labels_multicam = h5py.File(self.config["labels_file_multicam"], 'r')
        h5features_urfd = h5py.File(self.config["features_file_urfd"], 'r')
        h5labels_urfd = h5py.File(self.config["labels_file_urfd"], 'r')
        h5features_fdd = h5py.File(self.config["features_file_fdd"], 'r')
        h5labels_fdd = h5py.File(self.config["labels_file_fdd"], 'r')

        # Load Multicam data in a single array
        # Load Multicam data in a single array
        stages = []
        for i in range(1,25):
            stages.append('chute{:02}'.format(i))
        
        _x = []
        _y = []
        for nb_stage, stage in enumerate(stages):   
            for nb_cam, cam in enumerate(h5features_multicam[stage].keys()):
                for key in h5features_multicam[stage][cam].keys():
                    _x.extend([x for x in h5features_multicam[stage][cam][key]])
                    _y.extend([x for x in h5labels_multicam[stage][cam][key]])
                    # _x.append(np.asarray(h5features_multicam[stage][cam][key]))
                    # _y.append(np.asarray(h5labels_multicam[stage][cam][key]))

        # Load all the datasets into numpy arrays
        X_multicam = np.asarray(_x)
        y_multicam = np.asarray(_y)
        X_urfd = np.asarray(h5features_urfd['features'])
        y_urfd = np.asarray(h5labels_urfd['labels'])
        X_fdd = np.asarray(h5features_fdd['features'])
        y_fdd = np.asarray(h5labels_fdd['labels'])

        # Get the number of samples per class on the smallest dataset: URFD
        size_0 = np.asarray(np.where(y_urfd==0)[0]).shape[0]
        size_1 = np.asarray(np.where(y_urfd==1)[0]).shape[0]

        # Undersample the FDD and Multicam: take 0s and 1s per dataset and
        # undersample each of them separately by random sampling without replacement
        # Step 1
        all0_multicam = np.asarray(np.where(y_multicam==0)[0])
        all1_multicam = np.asarray(np.where(y_multicam==1)[0])
        all0_urfd = np.asarray(np.where(y_urfd==0)[0])
        all1_urfd = np.asarray(np.where(y_urfd==1)[0])
        all0_fdd = np.asarray(np.where(y_fdd==0)[0])
        all1_fdd = np.asarray(np.where(y_fdd==1)[0])

        # Step 2
        all0_multicam = np.random.choice(all0_multicam, size_0, replace=False)
        all1_multicam = np.random.choice(all1_multicam, size_0, replace=False)
        all0_urfd = np.random.choice(all0_urfd, size_0, replace=False)
        all1_urfd = np.random.choice(all1_urfd, size_0, replace=False)
        all0_fdd = np.random.choice(all0_fdd, size_0, replace=False)
        all1_fdd = np.random.choice(all1_fdd, size_0, replace=False)

        # Arrays to save the results
        sensitivities = { 'combined': [], 'multicam': [], 'urfd': [], 'fdd': [] }
        specificities = { 'combined': [], 'multicam': [], 'urfd': [], 'fdd': [] }

        # Use a 5 fold cross-validation
        kfold = KFold(n_splits=5, shuffle=True)
        kfold0_multicam = kfold.split(all0_multicam)
        kfold1_multicam = kfold.split(all1_multicam)
        kfold0_urfd = kfold.split(all0_urfd)
        kfold1_urfd = kfold.split(all1_urfd)
        kfold0_fdd = kfold.split(all0_fdd)
        kfold1_fdd = kfold.split(all1_fdd)

        # CROSS-VALIDATION: Stratified partition of the dataset into train/test sets
        for fold in range(5):
            # Get the train and test indices, then get the actual indices
            _train0_multicam, _test0_multicam = kfold0_multicam.next()
            _train1_multicam, _test1_multicam = kfold1_multicam.next()
            train0_multicam = all0_multicam[_train0_multicam]
            train1_multicam = all1_multicam[_train1_multicam]
            test0_multicam = all0_multicam[_test0_multicam]
            test1_multicam = all1_multicam[_test1_multicam]

            _train0_urfd, _test0_urfd = kfold0_urfd.next()
            _train1_urfd, _test1_urfd = kfold1_urfd.next()
            train0_urfd = all0_urfd[_train0_urfd]
            train1_urfd = all1_urfd[_train1_urfd]
            test0_urfd = all0_urfd[_test0_urfd]
            test1_urfd = all1_urfd[_test1_urfd]

            _train0_fdd, _test0_fdd = kfold0_fdd.next()
            _train1_fdd, _test1_fdd = kfold1_fdd.next()
            train0_fdd = all0_fdd[_train0_fdd]
            train1_fdd = all1_fdd[_train1_fdd]
            test0_fdd = all0_fdd[_test0_fdd]
            test1_fdd = all1_fdd[_test1_fdd]

            if self.config["use_validation"]:
                # Multicam
                (train0_multicam, train1_multicam,
                val0_multicam, val1_multicam) = self.divide_train_val(
                    train0_multicam, train1_multicam, self.config["validation_size"]//3)
                temp = np.concatenate((val0_multicam, val1_multicam))
                X_val_multicam = X_multicam[temp]
                y_val_multicam = y_multicam[temp]

                # URFD
                (train0_urfd, train1_urfd,
                val0_urfd, val1_urfd) = self.divide_train_val(
                    train0_urfd, train1_urfd, self.config["validation_size"]//3)
                temp = np.concatenate((val0_urfd, val1_urfd))
                X_val_urfd = X_urfd[temp]
                y_val_urfd = y_urfd[temp]

                # FDD
                (train0_fdd, train1_fdd,
                val0_fdd, val1_fdd) = self.divide_train_val(
                    train0_fdd, train1_fdd, self.config["validation_size"]//3)
                temp = np.concatenate((val0_fdd, val1_fdd))
                X_val_fdd = X_fdd[temp]
                y_val_fdd = y_fdd[temp]

                # Join all the datasets
                X_val = np.concatenate((X_val_multicam, X_val_urfd, X_val_fdd), axis=0)
                y_val = np.concatenate((y_val_multicam, y_val_urfd, y_val_fdd), axis=0)

            # Sampling
            X_train_multicam, y_train_multicam = self.sample_from_dataset(
                X_multicam, y_multicam, train0_multicam, train1_multicam)
            X_train_urfd, y_train_urfd = self.sample_from_dataset(
                X_urfd, y_urfd, train0_urfd, train1_urfd)
            X_train_fdd, y_train_fdd = self.sample_from_dataset(
                X_fdd, y_fdd, train0_fdd, train1_fdd)

            # Create the evaluation folds for each dataset
            X_test_multicam, y_test_multicam = self.sample_from_dataset(
                X_multicam, y_multicam, test0_multicam, test1_multicam)
            X_test_urfd, y_test_urfd = self.sample_from_dataset(
                X_urfd, y_urfd, test0_urfd, test1_urfd)
            X_test_fdd, y_test_fdd = self.sample_from_dataset(
                X_fdd, y_fdd, test0_fdd, test1_fdd)
        
            # Join all the datasets
            X_train = np.concatenate((X_train_multicam, X_train_urfd, X_train_fdd), axis=0)
            y_train = np.concatenate((y_train_multicam, y_train_urfd, y_train_fdd), axis=0)
            X_test = np.concatenate((X_test_multicam, X_test_urfd, X_test_fdd), axis=0)
            y_test = np.concatenate((y_test_multicam, y_test_urfd, y_test_fdd), axis=0)

            # ==================== TRAINING ========================     
            class_weight = {0:self.config["weight_0"], 1: 1}
            callbacks = None

            if self.config["use_validation"]:
                # callback definition
                metric = 'val_loss'
                e = EarlyStopping(monitor=metric, min_delta=0,patience=100, mode='auto')
                c = ModelCheckpoint(self.config["model_checkpoints_path"], monitor=metric, 
                    save_best_only=True, save_weights_only=False, mode='auto')
                callbacks = [e, c]

            validation_data = None
            if self.config["use_validation"]:
                validation_data = (X_val,y_val)

            _mini_batch_size = self.config["mini_batch_size"]
            if self.config["mini_batch_size"] == 0:
                _mini_batch_size = X_train.shape[0]

            history = self.classifier.fit(X_train, y_train, validation_data=validation_data,
                batch_size=_mini_batch_size, nb_epoch=self.config["epochs"], shuffle='batch',
                class_weight=class_weight, callbacks=callbacks)

            if not self.config["use_validation"]:
                self.classifier.save(self.config["model_checkpoints_path"])

            self.plot_training_info(self.config["plots_folder"] + self.exp, ['accuracy', 'loss'], 
                self.config["save_plots"], history.history)

            if self.config["use_validation"] and self.config["use_validation_for_training"]:
                classifier = load_model(self.config["model_checkpoints_path"])

                # Use full training set (training + validation)
                X_train = np.concatenate((X_train, X_val), axis=0)
                y_train = np.concatenate((y_train, y_val), axis=0)

                history = self.classifier.fit(X_train, y_train, validation_data=validation_data,
                    batch_size=_mini_batch_size, nb_epoch=self.config["epochs"], shuffle='batch',
                    class_weight=class_weight, callbacks=callbacks)

                self.classifier.save(self.config["model_checkpoints_path"])

            # ==================== EVALUATION ========================
            print("loading checkpoints...")
            self.classifier = load_model(self.config["model_checkpoints_path"])
            print("Checkpoints loaded.")

            # Evaluate for the combined test set
            predicted = self.classifier.predict(X_test)
            for i in range(len(predicted)):
                if predicted[i] < self.config["threshold"]:
                    predicted[i] = 0
                else:
                    predicted[i] = 1
            predicted = np.asarray(predicted).astype(int)
            cm = confusion_matrix(y_test, predicted, labels=[0,1])
            tp = cm[0][0]
            fn = cm[0][1]
            fp = cm[1][0]
            tn = cm[1][1]
            tpr = tp/float(tp+fn)
            fpr = fp/float(fp+tn)
            fnr = fn/float(fn+tp)
            tnr = tn/float(tn+fp)
            print('Combined test set')
            print('-'*10)
            print('TP: {}, TN: {}, FP: {}, FN: {}'.format(tp,tn,fp,fn))
            print('TPR: {}, TNR: {}, FPR: {}, FNR: {}'.format(tpr,tnr,fpr,fnr))
            print('Sensitivity/Recall: {}'.format(tp/float(tp+fn)))
            print('Specificity: {}'.format(tn/float(tn+fp)))  
            print('Accuracy: {}'.format(accuracy_score(y_test, predicted)))
            sensitivities['combined'].append(tp/float(tp+fn))
            specificities['combined'].append(tn/float(tn+fp))
                
            # Evaluate for the URFD test set
            predicted = self.classifier.predict(X_test_urfd)
            for i in range(len(predicted)):
                if predicted[i] < self.config["threshold"]:
                    predicted[i] = 0
                else:
                    predicted[i] = 1
            predicted = np.asarray(predicted).astype(int)
            cm = confusion_matrix(y_test_urfd, predicted, labels=[0,1])
            tp = cm[0][0]
            fn = cm[0][1]
            fp = cm[1][0]
            tn = cm[1][1]
            tpr = tp/float(tp+fn)
            fpr = fp/float(fp+tn)
            fnr = fn/float(fn+tp)
            tnr = tn/float(tn+fp)
            print('URFD test set')
            print('-'*10)
            print('TP: {}, TN: {}, FP: {}, FN: {}'.format(tp,tn,fp,fn))
            print('TPR: {}, TNR: {}, FPR: {}, FNR: {}'.format(tpr,tnr,fpr,fnr))
            print('Sensitivity/Recall: {}'.format(tp/float(tp+fn)))
            print('Specificity: {}'.format(tn/float(tn+fp)))
            print('Accuracy: {}'.format(accuracy_score(y_test_urfd, predicted)))
            sensitivities['urfd'].append(tp/float(tp+fn))
            specificities['urfd'].append(tn/float(tn+fp))
            
            # Evaluate for the Multicam test set
            predicted = self.classifier.predict(X_test_multicam)
            for i in range(len(predicted)):
                if predicted[i] < self.config["threshold"]:
                    predicted[i] = 0
                else:
                    predicted[i] = 1
            predicted = np.asarray(predicted).astype(int)
            cm = confusion_matrix(y_test_multicam, predicted, labels=[0,1])
            tp = cm[0][0]
            fn = cm[0][1]
            fp = cm[1][0]
            tn = cm[1][1]
            tpr = tp/float(tp+fn)
            fpr = fp/float(fp+tn)
            fnr = fn/float(fn+tp)
            tnr = tn/float(tn+fp)
            print('Multicam test set')
            print('-'*10)
            print('TP: {}, TN: {}, FP: {}, FN: {}'.format(tp,tn,fp,fn))
            print('TPR: {}, TNR: {}, FPR: {}, FNR: {}'.format(tpr,tnr,fpr,fnr))
            print('Sensitivity/Recall: {}'.format(tp/float(tp+fn)))
            print('Specificity: {}'.format(tn/float(tn+fp)))
            print('Accuracy: {}'.format(accuracy_score(y_test_multicam, predicted)))
            sensitivities['multicam'].append(tp/float(tp+fn))
            specificities['multicam'].append(tn/float(tn+fp))
            
            # Evaluate for the FDD test set
            predicted = self.classifier.predict(X_test_fdd)
            for i in range(len(predicted)):
                if predicted[i] < self.config["threshold"]:
                    predicted[i] = 0
                else:
                    predicted[i] = 1
            predicted = np.asarray(predicted).astype(int)
            cm = confusion_matrix(y_test_fdd, predicted,labels=[0,1])
            tp = cm[0][0]
            fn = cm[0][1]
            fp = cm[1][0]
            tn = cm[1][1]
            tpr = tp/float(tp+fn)
            fpr = fp/float(fp+tn)
            fnr = fn/float(fn+tp)
            tnr = tn/float(tn+fp)
            print('FDD test set')
            print('-'*10)
            print('TP: {}, TN: {}, FP: {}, FN: {}'.format(tp,tn,fp,fn))
            print('TPR: {}, TNR: {}, FPR: {}, FNR: {}'.format(tpr,tnr,fpr,fnr))
            print('Sensitivity/Recall: {}'.format(tp/float(tp+fn)))
            print('Specificity: {}'.format(tn/float(tn+fp)))
            print('Accuracy: {}'.format(accuracy_score(y_test_fdd, predicted)))
            sensitivities['fdd'].append(tp/float(tp+fn))
            specificities['fdd'].append(tn/float(tn+fp))

        # End of the Cross-Validation
        print('CROSS-VALIDATION RESULTS ===================')
        print("Sensitivity Combined: {:.2f}% (+/- {:.2f}%)".format(
            np.mean(sensitivities['combined'])*100., np.std(sensitivities['combined'])*100.))
        print("Specificity Combined: {:.2f}% (+/- {:.2f}%)\n".format(
            np.mean(specificities['combined'])*100., np.std(specificities['combined'])*100.))
        print("Sensitivity URFD: {:.2f}% (+/- {:.2f}%)".format(
            np.mean(sensitivities['urfd'])*100., np.std(sensitivities['urfd'])*100.))
        print("Specificity URFD: {:.2f}% (+/- {:.2f}%)\n".format(
            np.mean(specificities['urfd'])*100., np.std(specificities['urfd'])*100.))
        print("Sensitivity Multicam: {:.2f}% (+/- {:.2f}%)".format(
            np.mean(sensitivities['multicam'])*100., np.std(sensitivities['multicam'])*100.))
        print("Specificity Multicam: {:.2f}% (+/- {:.2f}%)\n".format(
            np.mean(specificities['multicam'])*100., np.std(specificities['multicam'])*100.))
        print("Sensitivity Multicam: {:.2f}% (+/- {:.2f}%)".format(
            np.mean(sensitivities['fdd'])*100., np.std(sensitivities['fdd'])*100.))
        print("Specificity FDDs: {:.2f}% (+/- {:.2f}%)".format(
            np.mean(specificities['fdd'])*100., np.std(specificities['fdd'])*100.))

    def train_ufrd(self):
        """
        method to train fall detection model with UFRD dataset
        """
        h5features = h5py.File(self.config["features_file_urfd"], 'r')
        h5labels = h5py.File(self.config["labels_file_urfd"], 'r')

        # X_full will contain all the feature vectors extracted
        # from optical flow images
        X_full = h5features["features"]
        _y_full = np.asarray(h5labels["labels"])

        zeroes_full = np.asarray(np.where(_y_full==0)[0])
        ones_full = np.asarray(np.where(_y_full==1)[0])
        zeroes_full.sort()
        ones_full.sort()
        
        # Use a 5 fold cross-validation
        kf_falls = KFold(n_splits=5, shuffle=True)
        kf_falls.get_n_splits(X_full[zeroes_full, ...])
        
        kf_nofalls = KFold(n_splits=5, shuffle=True)
        kf_nofalls.get_n_splits(X_full[ones_full, ...])        

        sensitivities = []
        specificities = []
        fars = []
        mdrs = []
        accuracies = []
        
        fold_number = 1

        # CROSS-VALIDATION: Stratified partition of the dataset into train/test sets
        for ((train_index_falls, test_index_falls), (train_index_nofalls, test_index_nofalls)) in zip(
            kf_falls.split(X_full[zeroes_full, ...]), kf_nofalls.split(X_full[ones_full, ...])):
            train_index_falls = np.asarray(train_index_falls)
            test_index_falls = np.asarray(test_index_falls)
            train_index_nofalls = np.asarray(train_index_nofalls)
            test_index_nofalls = np.asarray(test_index_nofalls)

            X = np.concatenate((X_full[zeroes_full, ...][train_index_falls, ...], 
                X_full[ones_full, ...][train_index_nofalls, ...]))
            _y = np.concatenate((_y_full[zeroes_full, ...][train_index_falls, ...], 
                _y_full[ones_full, ...][train_index_nofalls, ...]))
            X_test = np.concatenate((X_full[zeroes_full, ...][test_index_falls, ...],
                X_full[ones_full, ...][test_index_nofalls, ...]))
            y_test = np.concatenate((_y_full[zeroes_full, ...][test_index_falls, ...],
                _y_full[ones_full, ...][test_index_nofalls, ...]))   

            if self.config["use_validation"]:
                # Create a validation subset from the training set
                zeroes = np.asarray(np.where(_y==0)[0])
                ones = np.asarray(np.where(_y==1)[0])
                
                zeroes.sort()
                ones.sort()

                trainval_split_0 = StratifiedShuffleSplit(n_splits=1, test_size=self.config["validation_size"]/2, random_state=7)
                indices_0 = trainval_split_0.split(X[zeroes,...], np.argmax(_y[zeroes,...], 1))
                trainval_split_1 = StratifiedShuffleSplit(n_splits=1, test_size=self.config["validation_size"]/2, random_state=7)
                indices_1 = trainval_split_1.split(X[ones,...], np.argmax(_y[ones,...], 1))
                train_indices_0, val_indices_0 = indices_0.next()
                train_indices_1, val_indices_1 = indices_1.next()

                X_train = np.concatenate([X[zeroes,...][train_indices_0,...], X[ones,...][train_indices_1,...]], axis=0)
                y_train = np.concatenate([_y[zeroes,...][train_indices_0,...], _y[ones,...][train_indices_1,...]], axis=0)
                X_val = np.concatenate([X[zeroes,...][val_indices_0,...], X[ones,...][val_indices_1,...]], axis=0)
                y_val = np.concatenate([_y[zeroes,...][val_indices_0,...], _y[ones,...][val_indices_1,...]], axis=0)
            else:
                X_train = X
                y_train = _y
        
            # Balance the number of positive and negative samples so that
            # there is the same amount of each of them
            all0 = np.asarray(np.where(y_train==0)[0])
            all1 = np.asarray(np.where(y_train==1)[0])  

            if len(all0) < len(all1):
                all1 = np.random.choice(all1, len(all0), replace=False)
            else:
                all0 = np.random.choice(all0, len(all1), replace=False)
            allin = np.concatenate((all0.flatten(),all1.flatten()))
            allin.sort()
            X_train = X_train[allin,...]
            y_train = y_train[allin]

            # ==================== TRAINING ========================     
            class_weight = {0:self.config["weight_0"], 1: 1}
            callbacks = None

            if self.config["use_validation"]:
                # callback definition
                metric = 'val_loss'
                e = EarlyStopping(monitor=metric, min_delta=0,patience=100, mode='auto')
                c = ModelCheckpoint(self.config["model_checkpoints_path"], monitor=metric, 
                    save_best_only=True, save_weights_only=False, mode='auto')
                callbacks = [e, c]

            validation_data = None
            if self.config["use_validation"]:
                validation_data = (X_val,y_val)

            _mini_batch_size = self.config["mini_batch_size"]
            if self.config["mini_batch_size"] == 0:
                _mini_batch_size = X_train.shape[0]

            history = self.classifier.fit(X_train, y_train, validation_data=validation_data,
                batch_size=_mini_batch_size, nb_epoch=self.config["epochs"], shuffle=True,
                class_weight=class_weight, callbacks=callbacks)

            if not self.config["use_validation"]:
                self.classifier.save(self.config["model_checkpoints_path"])

            self.plot_training_info(self.config["plots_folder"] + self.exp, ['accuracy', 'loss'], 
                self.config["save_plots"], history.history)

            if self.config["use_validation"] and self.config["use_validation_for_training"]:
                self.classifier = load_model(self.config["model_checkpoints_path"])

                # Use full training set (training + validation)
                X_train = np.concatenate((X_train, X_val), axis=0)
                y_train = np.concatenate((y_train, y_val), axis=0)

                history = self.classifier.fit(X_train, y_train, validation_data=validation_data,
                    batch_size=_mini_batch_size, nb_epoch=self.config["epochs"], shuffle='batch',
                    class_weight=class_weight, callbacks=callbacks)

                self.classifier.save(self.config["model_checkpoints_path"])

            # ==================== EVALUATION ========================
            print("loading checkpoints...")
            self.classifier = load_model(self.config["model_checkpoints_path"])
            print("Checkpoints loaded.")

            # Evaluate for the combined test set
            predicted = self.classifier.predict(np.asarray(X_test))
            for i in range(len(predicted)):
                if predicted[i] < self.config["threshold"]:
                    predicted[i] = 0
                else:
                    predicted[i] = 1

            predicted = np.asarray(predicted).astype(int)
            cm = confusion_matrix(y_test, predicted,labels=[0,1])
            tp = cm[0][0]
            fn = cm[0][1]
            fp = cm[1][0]
            tn = cm[1][1]
            tpr = tp/float(tp+fn)
            fpr = fp/float(fp+tn)
            fnr = fn/float(fn+tp)
            tnr = tn/float(tn+fp)
            precision = tp/float(tp+fp)
            recall = tp/float(tp+fn)
            specificity = tn/float(tn+fp)
            f1 = 2*float(precision*recall)/float(precision+recall)
            accuracy = accuracy_score(y_test, predicted)

            print('FOLD {} results:'.format(fold_number))
            print('TP: {}, TN: {}, FP: {}, FN: {}'.format(tp,tn,fp,fn))
            print('TPR: {}, TNR: {}, FPR: {}, FNR: {}'.format(
                            tpr,tnr,fpr,fnr))   
            print('Sensitivity/Recall: {}'.format(recall))
            print('Specificity: {}'.format(specificity))
            print('Precision: {}'.format(precision))
            print('F1-measure: {}'.format(f1))
            print('Accuracy: {}'.format(accuracy))
            
            # Store the metrics for this epoch
            sensitivities.append(tp/float(tp+fn))
            specificities.append(tn/float(tn+fp))
            fars.append(fpr)
            mdrs.append(fnr)
            accuracies.append(accuracy)
            fold_number += 1

        print('5-FOLD CROSS-VALIDATION RESULTS ===================')
        print("Sensitivity: %.2f%% (+/- %.2f%%)" % (np.mean(sensitivities)*100.,
                            np.std(sensitivities)*100.))
        print("Specificity: %.2f%% (+/- %.2f%%)" % (np.mean(specificities)*100.,
                            np.std(specificities)*100.))
        print("FAR: %.2f%% (+/- %.2f%%)" % (np.mean(fars)*100.,
                        np.std(fars)*100.))
        print("MDR: %.2f%% (+/- %.2f%%)" % (np.mean(mdrs)*100.,
                        np.std(mdrs)*100.))
        print("Accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(accuracies)*100.,
                            np.std(accuracies)*100.))