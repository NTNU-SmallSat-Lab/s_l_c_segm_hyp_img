



"""

************************* TERMS OF USE *************************

Users within the open community are fully permitted and encouraged to access, download, analyze, and use this software code
as long as proper credit is given to the authors in the citations below. The present material is released under the Attribution 4.0 International (CC BY 4.0) license.
    © 2023 Norwegian University of Science and Technology (NTNU) in Trondheim, Norway. All rights reserved.
    - by Jon Alvarez Justo, Joseph Landon Garrett, Mariana-Iuliana Georgescu, Jesus Gonzalez-Llorente, Radu Tudor Ionescu, and Tor Arne Johansen. 


    
Models citation (BibTeX)
@article{justo2023sea,
         title={Sea-Land-Cloud Segmentation in Satellite Hyperspectral Imagery by Deep Learning},
         author={Justo, Jon Alvarez and Garrett, Joseph Landon and Georgescu, Mariana-Iuliana and Gonzalez-Llorente, Jesus and Ionescu, Radu Tudor and Johansen, Tor Arne},
         journal={arXiv preprint arXiv:2310.16210},
         year={2023}
}
Article at https://arxiv.org/abs/2310.16210 - download full supplementary materials including further software codes from https://github.com/jonalvjusto/s_l_c_segm_hyp_img.



Dataset citation (BibTeX)
  @article{justo2023open,
           title={An Open Hyperspectral Dataset with Sea-Land-Cloud Ground-Truth from the HYPSO-1 Satellite},
           author={Justo, Jon A and Garrett, Joseph and Langer, Dennis D and Henriksen, Marie B and Ionescu, Radu T and Johansen, Tor A},
           journal={arXiv preprint arXiv:2308.13679},
        year={2023}
}
Article at https://arxiv.org/abs/2308.13679 - download dataset from https://ntnu-smallsat-lab.github.io/hypso1_sea_land_clouds_dataset.

****************************************************************

"""



from tensorflow import keras        
from keras import Model 
from keras.layers import Input, Conv1D, Conv2D, Activation, MaxPooling1D, Flatten, Dense, UpSampling2D,  concatenate, SeparableConv2D, MaxPooling2D , Conv2DTranspose
from keras.layers import BatchNormalization






def train_1D_ML_SGD(DATA_POINTS_USED_FOR_TRAINING, RESPECTIVE_ANNOTATIONS_FOR_THE_DATA_POINTS_USED_FOR_TRAINING,\
                                                       EPOCHS, TOLERANCE_FOR_EARLY_STOP, INITIAL_LEARNING_RATE, LEARNING_RATE_SCHEDULE, VERBOSE):
        """
            SGD: Stochastic Gradient Descent 
            Example use of the method is found below. 
            Informative Note: 
                1) The hyperparameters below are initialized as in the experiments of the paper: "Sea-Land-Cloud Segmentation in Satellite Hyperspectral Imagery by Deep Learning" by Justo et al.
                2) The data points used for training must be of size NUMBER_OF_PIXELS x CHANNEL_FEATURES
                3) The ground-truth annotations must be categorical and of size NUMBER_OF_PIXELS


            print('Training model...')
            EPOCHS=20
            TOLERANCE_FOR_EARLY_STOP=0.0001
            INITIAL_LEARNING_RATE=0.01
            LEARNING_RATE_SCHEDULE='optimal'
            
            model_classifier=train_1D_ML_SGD(
                                                    DATA_POINTS_USED_FOR_TRAINING, RESPECTIVE_ANNOTATIONS_FOR_THE_DATA_POINTS_USED_FOR_TRAINING,\
                                                    EPOCHS=EPOCHS,\
                                                    TOLERANCE_FOR_EARLY_STOP=TOLERANCE_FOR_EARLY_STOP,\
                                                    INITIAL_LEARNING_RATE=INITIAL_LEARNING_RATE,\
                                                    LEARNING_RATE_SCHEDULE=LEARNING_RATE_SCHEDULE,\
                                                    VERBOSE=1) 
            print('Training completed...')

                    

                    
            BibTeX Citation:         
            @article{justo2023sea,
                     title={Sea-Land-Cloud Segmentation in Satellite Hyperspectral Imagery by Deep Learning},
                     author={Justo, Jon Alvarez and Garrett, Joseph Landon and Georgescu, Mariana-Iuliana and Gonzalez-Llorente, Jesus and Ionescu, Radu Tudor and Johansen, Tor Arne},
                     journal={arXiv preprint arXiv:2310.16210},
                     year={2023}
            }
        """

        import sklearn.linear_model as linmod
        SGD_classifier_model=linmod.SGDClassifier(max_iter=EPOCHS,\
                                                tol=TOLERANCE_FOR_EARLY_STOP,\
                                                eta0=INITIAL_LEARNING_RATE,\
                                                learning_rate=LEARNING_RATE_SCHEDULE, \
                                                verbose=VERBOSE)
        SGD_classifier_model.fit(DATA_POINTS_USED_FOR_TRAINING, RESPECTIVE_ANNOTATIONS_FOR_THE_DATA_POINTS_USED_FOR_TRAINING)
        return SGD_classifier_model
        



def train_1D_ML_NB(DATA_POINTS_USED_FOR_TRAINING, RESPECTIVE_ANNOTATIONS_FOR_THE_DATA_POINTS_USED_FOR_TRAINING, class_priors): 
    """
        NB: Naive Bayes
        Example use of the method is found below. 
        Informative Note: 
            1) The hyperparameters below are initialized as in the experiments of the paper: "Sea-Land-Cloud Segmentation in Satellite Hyperspectral Imagery by Deep Learning" by Justo et al.
            2) The data points used for training must be of size NUMBER_OF_PIXELS x CHANNEL_FEATURES
            3) The ground-truth annotations must be categorical and of size NUMBER_OF_PIXELS

        
        print('Training model...')
        class_priors=[0.3701, 0.4014, 0.2285]
        model_classifier=train_1D_ML_NB(DATA_POINTS_USED_FOR_TRAINING, RESPECTIVE_ANNOTATIONS_FOR_THE_DATA_POINTS_USED_FOR_TRAINING, class_priors)
        print('Training completed...')

        BibTeX Citation:     
        @article{justo2023sea,
                title={Sea-Land-Cloud Segmentation in Satellite Hyperspectral Imagery by Deep Learning},
                author={Justo, Jon Alvarez and Garrett, Joseph Landon and Georgescu, Mariana-Iuliana and Gonzalez-Llorente, Jesus and Ionescu, Radu Tudor and Johansen, Tor Arne},
                journal={arXiv preprint arXiv:2310.16210},
                year={2023}
        }
    """
    from sklearn.naive_bayes import GaussianNB
    nb_classifier_model = GaussianNB(priors=class_priors)
    nb_classifier_model.fit(DATA_POINTS_USED_FOR_TRAINING, RESPECTIVE_ANNOTATIONS_FOR_THE_DATA_POINTS_USED_FOR_TRAINING)
    return nb_classifier_model



def train_1D_ML_LDA(DATA_POINTS_USED_FOR_TRAINING, RESPECTIVE_ANNOTATIONS_FOR_THE_DATA_POINTS_USED_FOR_TRAINING, \
                                          solver, shrinkage):
    """ 
        LDA: Linear Discriminant Analysis
        Example use of the method is found below. 
        Informative Note: 
            1) The hyperparameters below are initialized as in the experiments of the paper: "Sea-Land-Cloud Segmentation in Satellite Hyperspectral Imagery by Deep Learning" by Justo et al.
            2) The data points used for training must be of size NUMBER_OF_PIXELS x CHANNEL_FEATURES
            3) The ground-truth annotations must be categorical and of size NUMBER_OF_PIXELS


        print('Training model...')
        solver='svd' 
        shrinkage=None 
        model_classifier=train_1D_ML_LDA(DATA_POINTS_USED_FOR_TRAINING, RESPECTIVE_ANNOTATIONS_FOR_THE_DATA_POINTS_USED_FOR_TRAINING, \
                                                    solver=solver, shrinkage=shrinkage )
        print('Training completed...')

        BibTeX Citation: 
        @article{justo2023sea,
                title={Sea-Land-Cloud Segmentation in Satellite Hyperspectral Imagery by Deep Learning},
                author={Justo, Jon Alvarez and Garrett, Joseph Landon and Georgescu, Mariana-Iuliana and Gonzalez-Llorente, Jesus and Ionescu, Radu Tudor and Johansen, Tor Arne},
                journal={arXiv preprint arXiv:2310.16210},
                year={2023}
        }
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda_classifier_model = LinearDiscriminantAnalysis(solver=solver, \
                                                      shrinkage=shrinkage)
    lda_classifier_model.fit(DATA_POINTS_USED_FOR_TRAINING, RESPECTIVE_ANNOTATIONS_FOR_THE_DATA_POINTS_USED_FOR_TRAINING)
    return lda_classifier_model



def train_1D_ML_QDA(DATA_POINTS_USED_FOR_TRAINING, RESPECTIVE_ANNOTATIONS_FOR_THE_DATA_POINTS_USED_FOR_TRAINING, \
                    reg_param, tolerance): 
    """
        QDA: Quadratic Discriminant Analysis
        Example use of the method is found below. 
        Informative Note: 
            1) The hyperparameters below are initialized as in the experiments of the paper: "Sea-Land-Cloud Segmentation in Satellite Hyperspectral Imagery by Deep Learning" by Justo et al.
            2) The data points used for training must be of size NUMBER_OF_PIXELS x CHANNEL_FEATURES
            3) The ground-truth annotations must be categorical and of size NUMBER_OF_PIXELS


        print('Training model...')
        reg_param=0.0
        tolerance=0.0001
        model_classifier=train_1D_ML_QDA(DATA_POINTS_USED_FOR_TRAINING, RESPECTIVE_ANNOTATIONS_FOR_THE_DATA_POINTS_USED_FOR_TRAINING, \
                                                    reg_param=reg_param, tolerance=tolerance)
        print('Training completed...')

        BibTeX Citation: 
        @article{justo2023sea,
                 title={Sea-Land-Cloud Segmentation in Satellite Hyperspectral Imagery by Deep Learning},
                 author={Justo, Jon Alvarez and Garrett, Joseph Landon and Georgescu, Mariana-Iuliana and Gonzalez-Llorente, Jesus and Ionescu, Radu Tudor and Johansen, Tor Arne},
                 journal={arXiv preprint arXiv:2310.16210},
                 year={2023}
        }
    """
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    qda_classifier_model = QuadraticDiscriminantAnalysis(reg_param=reg_param, \
                                                         tol=tolerance) 
    qda_classifier_model.fit(DATA_POINTS_USED_FOR_TRAINING, RESPECTIVE_ANNOTATIONS_FOR_THE_DATA_POINTS_USED_FOR_TRAINING)
    return qda_classifier_model



def model_1D_Justo_LiuNet(NUMBER_OF_FEATURES, NUMBER_OF_CLASSES, KERNEL_SIZE, STARTING_NUMBER_OF_KERNELS_FOR_CONVS):
    """
        1D-Justo-LiuNet
        Example use of the method is found below. 
        Informative Note: 
            1) The hyperparameters below are initialized as in the experiments of the paper: "Sea-Land-Cloud Segmentation in Satellite Hyperspectral Imagery by Deep Learning" by Justo et al.
            2) The data points used for training must be of size NUMBER_OF_PIXELS x CHANNEL_FEATURES
            3) The ground-truth annotations must be encoded. For instance, one-hot encode categorical annotations and obtain NUMBER_OF_PIXELS x NUMBER_OF_CLASSES
    

        NUMBER_OF_CHANNELS=112
        NB: Dropout not used
        model_classifier=model_1D_Justo_LiuNet( NUMBER_OF_FEATURES=NUMBER_OF_CHANNELS,\
                                                NUMBER_OF_CLASSES=3,\
                                                KERNEL_SIZE=6,\
                                                STARTING_NUMBER_OF_KERNELS_FOR_CONVS=6)

        BibTeX Citation: 
        @article{justo2023sea,
                 title={Sea-Land-Cloud Segmentation in Satellite Hyperspectral Imagery by Deep Learning},
                 author={Justo, Jon Alvarez and Garrett, Joseph Landon and Georgescu, Mariana-Iuliana and Gonzalez-Llorente, Jesus and Ionescu, Radu Tudor and Johansen, Tor Arne},
                 journal={arXiv preprint arXiv:2310.16210},
                 year={2023}
        }
    """

    inp = Input(shape=(NUMBER_OF_FEATURES, 1))

    x = Conv1D(filters=STARTING_NUMBER_OF_KERNELS_FOR_CONVS, kernel_size=KERNEL_SIZE, activation='relu')(inp)
    x = MaxPooling1D(2)(x)

    x = Conv1D(filters=STARTING_NUMBER_OF_KERNELS_FOR_CONVS*2, kernel_size=KERNEL_SIZE, activation='relu')(x)
    x = MaxPooling1D(2)(x)

    x = Conv1D(filters=STARTING_NUMBER_OF_KERNELS_FOR_CONVS*3, kernel_size=KERNEL_SIZE, activation='relu')(x)
    x = MaxPooling1D(2)(x)

    x = Conv1D(filters=STARTING_NUMBER_OF_KERNELS_FOR_CONVS*4, kernel_size=KERNEL_SIZE, activation='relu')(x)
    x = MaxPooling1D(2)(x)

    x = Flatten()(x)
    x = Dense(NUMBER_OF_CLASSES, activation='softmax')(x)

    model=Model(inputs=inp, outputs=x)


    print('Model architecture DEFINED!')
    print('Defining now its compiler, loss function and metrics to prepare for training...')
    model.compile(optimizer='adam', \
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    print('Defined compiler, loss function and metrics!')

    return model




def model_1D_Justo_HuNet(NUMBER_OF_FEATURES, NUMBER_OF_CLASSES, KERNEL_SIZE, NUMBER_OF_KERNELS, ACTIVATION_FUNCTION_IN_CONV_LAYER, NEURONS_IN_DENSE_LAYER, ACTIVATION_FUNCTION_IN_DENSE_LAYER):
    """
        1D-Justo-HuNet
        Example use of the method is found below. 
        Informative Note: 
            1) The hyperparameters below are initialized as in the experiments of the paper: "Sea-Land-Cloud Segmentation in Satellite Hyperspectral Imagery by Deep Learning" by Justo et al.
            2) The data points used for training must be of size NUMBER_OF_PIXELS x CHANNEL_FEATURES
            3) The ground-truth annotations must be encoded. For instance, one-hot encode categorical annotations and obtain NUMBER_OF_PIXELS x NUMBER_OF_CLASSES
    

        NUMBER_OF_CHANNELS=112
        model_classifier=model_1D_Justo_HuNet(
                                        NUMBER_OF_FEATURES=NUMBER_OF_CHANNELS,\
                                        NUMBER_OF_CLASSES=3,
                                        KERNEL_SIZE=9,\
                                        NUMBER_OF_KERNELS=6, \
                                        ACTIVATION_FUNCTION_IN_CONV_LAYER='tanh', \
                                        NEURONS_IN_DENSE_LAYER=30,\
                                        ACTIVATION_FUNCTION_IN_DENSE_LAYER='relu')


        BibTeX Citation: 
        @article{justo2023sea,
                 title={Sea-Land-Cloud Segmentation in Satellite Hyperspectral Imagery by Deep Learning},
                 author={Justo, Jon Alvarez and Garrett, Joseph Landon and Georgescu, Mariana-Iuliana and Gonzalez-Llorente, Jesus and Ionescu, Radu Tudor and Johansen, Tor Arne},
                 journal={arXiv preprint arXiv:2310.16210},
                 year={2023}
        }
    """
    pool_size = int((NUMBER_OF_FEATURES - KERNEL_SIZE + 1) / 35) 
                # Ref: "Deep convolutional neural networks for hyperspectral image classification" by W. Hu et al.
                # Ref: "Soil texture classification with 1D convolutional neural networks based on hyperspectral data" by F.M. Riese et al.

    input = Input(shape=(NUMBER_OF_FEATURES, 1))

    x = Conv1D(filters=NUMBER_OF_KERNELS, kernel_size=KERNEL_SIZE, activation=ACTIVATION_FUNCTION_IN_CONV_LAYER)(input)
    x = MaxPooling1D(pool_size)(x)
    
    x = Flatten()(x)
    x = Dense(units=NEURONS_IN_DENSE_LAYER, activation=ACTIVATION_FUNCTION_IN_DENSE_LAYER)(x)
    x = Dense(NUMBER_OF_CLASSES, activation='softmax')(x) 

    model=Model(inputs=input, outputs=x)

    print('Model architecture DEFINED!')
    print('Defining now its compiler, loss function and metrics to prepare for training...')
    model.compile(optimizer='adam', \
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    print('Defined compiler, loss function and metrics!')

    return model





def model_1D_Justo_LucasCNN(NUMBER_OF_FEATURES, NUMBER_OF_CLASSES, KERNEL_SIZE, NUMBER_OF_KERNELS_FOR_CONV_LAYER, ACTIVATION_CONV_LAYER, ACTIVATION_DENSE_LAYERS, \
                            NUMBER_OF_NEURONS_IN_FIRST_DENSE_LAYER, NUMBER_OF_NUERONS_IN_SECOND_DENSE_LAYER):
    """
        1D-Justo-LucasCNN
        Example use of the method is found below. 
        Informative Note: 
            1) The hyperparameters below are initialized as in the experiments of the paper: "Sea-Land-Cloud Segmentation in Satellite Hyperspectral Imagery by Deep Learning" by Justo et al.
            2) The data points used for training must be of size NUMBER_OF_PIXELS x CHANNEL_FEATURES
            3) The ground-truth annotations must be encoded. For instance, one-hot encode categorical annotations and obtain NUMBER_OF_PIXELS x NUMBER_OF_CLASSES

    
        NUMBER_OF_CHANNELS=112
        model_classifier=model_1D_Justo_LucasCNN(
                                        NUMBER_OF_FEATURES=NUMBER_OF_CHANNELS,\
                                        NUMBER_OF_CLASSES=3,\
                                        KERNEL_SIZE=9, \
                                        NUMBER_OF_KERNELS_FOR_CONV_LAYER=16, \
                                        ACTIVATION_CONV_LAYER='tanh', \
                                        ACTIVATION_DENSE_LAYERS='tanh', \
                                        NUMBER_OF_NEURONS_IN_FIRST_DENSE_LAYER=30, \
                                        NUMBER_OF_NUERONS_IN_SECOND_DENSE_LAYER=5)



        BibTeX Citation: 
        @article{justo2023sea,
                 title={Sea-Land-Cloud Segmentation in Satellite Hyperspectral Imagery by Deep Learning},
                 author={Justo, Jon Alvarez and Garrett, Joseph Landon and Georgescu, Mariana-Iuliana and Gonzalez-Llorente, Jesus and Ionescu, Radu Tudor and Johansen, Tor Arne},
                 journal={arXiv preprint arXiv:2310.16210},
                 year={2023}
        }
    """

    input = Input(shape=(NUMBER_OF_FEATURES, 1))

    x = Conv1D(filters=NUMBER_OF_KERNELS_FOR_CONV_LAYER,
               kernel_size=KERNEL_SIZE,
               activation=ACTIVATION_CONV_LAYER,
               padding='valid')(input)
    x = MaxPooling1D(2)(x)

    x = Flatten()(x)
    x = Dense(NUMBER_OF_NEURONS_IN_FIRST_DENSE_LAYER, activation=ACTIVATION_DENSE_LAYERS)(x)
    x = Dense(NUMBER_OF_NUERONS_IN_SECOND_DENSE_LAYER, activation=ACTIVATION_DENSE_LAYERS)(x)
    x = Dense(NUMBER_OF_CLASSES, activation='softmax')(x)


    model=Model(inputs=input, outputs=x)

    print('Model architecture DEFINED!')
    print('Defining now its compiler, loss function and metrics to prepare for training...')
    model.compile(optimizer='adam', \
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    print('Defined compiler, loss function and metrics!')

    return model




def model_2D_Justo_UNet_Simple(input_size, num_classes):
    """
        2D-Justo-UNet-Simple
        Example use of the method is found below. 
        Informative Note: 
            1) The hyperparameters below are initialized as in the experiments of the paper: "Sea-Land-Cloud Segmentation in Satellite Hyperspectral Imagery by Deep Learning" by Justo et al.
            2) The input size must be of size: NUMBER_OF_PATCHES x PATCH_SIZE x PATCH_SIZE x CHANNEL_FEATURES
            3) The ground-truth annotations must be encoded. For instance, one-hot encode categorical annotations and obtain: NUMBER_OF_PATCHES x PATCH_SIZE x PATCH_SIZE x NUMBER_OF_CLASSES

        PATCH_SIZE=48
        NUMBER_OF_CHANNELS=112
        model_classifier=model_2D_Justo_UNet_Simple(
                                        input_size=(PATCH_SIZE, PATCH_SIZE, NUMBER_OF_CHANNELS),\
                                        num_classes=3)

            
        BibTeX Citation: 
        @article{justo2023sea,
                 title={Sea-Land-Cloud Segmentation in Satellite Hyperspectral Imagery by Deep Learning},
                 author={Justo, Jon Alvarez and Garrett, Joseph Landon and Georgescu, Mariana-Iuliana and Gonzalez-Llorente, Jesus and Ionescu, Radu Tudor and Johansen, Tor Arne},
                 journal={arXiv preprint arXiv:2310.16210},
                 year={2023}
        }
    """
    KERNEL_SIZE=(3,3)
    input = Input(shape=input_size)
                
    x = Conv2D(filters=6, kernel_size=KERNEL_SIZE, strides=(1,1), padding='same')(input) # No activation is applied since the activation is not defined
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid')(x)

    x = Conv2D(filters=12, kernel_size=KERNEL_SIZE, strides=(1,1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid')(x)

    x = UpSampling2D(size=(2,2))(x)
    x = Conv2D(filters=6, kernel_size=KERNEL_SIZE, strides=(1,1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = UpSampling2D(size=(2,2))(x)
    x = Conv2D(filters=num_classes, kernel_size=KERNEL_SIZE, strides=(1,1), padding='same')(x)
    x = BatchNormalization()(x) 
    x = Activation('softmax')(x)

    model = Model(inputs=input, outputs=x)

    print('Model architecture DEFINED!')
    print('Defining now its compiler, loss function, and metrics to prepare for training...')
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    print('Defined compiler, loss function, and metrics!')

    return model



def model_2D_NU_Net_mod(input_size, num_classes):
    """
        2D-Justo-NU-Net-mod
        Example use of the method is found below. 
        Informative Note: 
            1) The hyperparameters below are initialized as in the experiments of the paper: "Sea-Land-Cloud Segmentation in Satellite Hyperspectral Imagery by Deep Learning" by Justo et al.
            2) The input size must be of size: NUMBER_OF_PATCHES x PATCH_SIZE x PATCH_SIZE x CHANNEL_FEATURES
            3) The ground-truth annotations must be encoded. For instance, one-hot encode categorical annotations and obtain: NUMBER_OF_PATCHES x PATCH_SIZE x PATCH_SIZE x NUMBER_OF_CLASSES


        PATCH_SIZE=48
        NUMBER_OF_CHANNELS=112
        model_classifier=NN_0_NU_Net_mod(input_size=(PATCH_SIZE, PATCH_SIZE, NUMBER_OF_CHANNELS),\
                                            num_classes=3)
            
    
        Baseline Model Sourced in (slight adjustements):
        @article{salazar2022cloud,
                 title={Cloud Detection Autonomous System Based on Machine Learning and COTS Components On-Board Small Satellites},
                 author={Salazar, Carlos and Gonzalez-Llorente, Jesus and Cardenas, Lorena and Mendez, Javier and Rincon, Sonia and Rodriguez-Ferreira, Julian and Acero, Ignacio F},
                 journal={Remote Sensing},
                 volume={14},
                 number={21},
                 pages={5597},
                 year={2022},
                 publisher={MDPI}
        }
        Further BibTeX Citation - "Sea-Land-Cloud Segmentation in Satellite Hyperspectral Imagery by Deep Learning" by Justo et al.
        @article{justo2023sea,
                 title={Sea-Land-Cloud Segmentation in Satellite Hyperspectral Imagery by Deep Learning},
                 author={Justo, Jon Alvarez and Garrett, Joseph Landon and Georgescu, Mariana-Iuliana and Gonzalez-Llorente, Jesus and Ionescu, Radu Tudor and Johansen, Tor Arne},
                 journal={arXiv preprint arXiv:2310.16210},
                 year={2023}
        }
    """


    inputs = Input(input_size)
    
    conv1 = Conv2D(1, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)

    conv2 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv4 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    up5 = Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv4))    
    merge5 = concatenate([conv3,up5], axis = 3)

    conv5 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge5)
    conv5 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    up6 = Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5)) 
    merge6 = concatenate([conv2,up6], axis = 3)

    conv6 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    poolmult1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    convmult2 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(poolmult1)
    convmult2 = BatchNormalization()(convmult2)
    convmult2 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(convmult2)
    convmult2 = BatchNormalization()(convmult2)
    poolmult2 = MaxPooling2D(pool_size=(2, 2))(convmult2) 
    
    convmult3 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(poolmult2)
    convmult3 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(convmult3)
    upmult3 = Conv2D(8, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(convmult3)) 
    mergemult3 = concatenate([convmult2,upmult3], axis =3)
    
    convmult4 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(mergemult3)
    convmult4 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(convmult4)
    upmult4 = Conv2D(8, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(convmult4))
    merge7 = concatenate([conv6,upmult4], axis = 3)
    
    conv7 = Conv2D(num_classes, 1, activation = 'softmax')(merge7)

    model = Model(inputs = inputs, outputs = conv7)

    print('Model architecture DEFINED!')
    print('Defining now its compiler, loss function and metrics to prepare for training...')
    model.compile(optimizer='adam', \
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    print('Defined compiler, loss function and metrics!')

    return model




def model_2D_NU_Net(input_size, num_classes):
    """
        2D-Justo-NU-Net
        Example use of the method is found below. 
        Informative Note: 
            1) The hyperparameters below are initialized as in the experiments of the paper: "Sea-Land-Cloud Segmentation in Satellite Hyperspectral Imagery by Deep Learning" by Justo et al.
            2) The input size must be of size: NUMBER_OF_PATCHES x PATCH_SIZE x PATCH_SIZE x CHANNEL_FEATURES
            3) The ground-truth annotations must be encoded. For instance, one-hot encode categorical annotations and obtain: NUMBER_OF_PATCHES x PATCH_SIZE x PATCH_SIZE x NUMBER_OF_CLASSES


        PATCH_SIZE=48
        NUMBER_OF_CHANNELS=112
        model_classifier=model_2D_NU_Net(input_size=(PATCH_SIZE, PATCH_SIZE, NUMBER_OF_CHANNELS),\
                                         num_classes=3)


        Baseline Model Sourced in (slight adjustements):
        @article{salazar2022cloud,
                 title={Cloud Detection Autonomous System Based on Machine Learning and COTS Components On-Board Small Satellites},
                 author={Salazar, Carlos and Gonzalez-Llorente, Jesus and Cardenas, Lorena and Mendez, Javier and Rincon, Sonia and Rodriguez-Ferreira, Julian and Acero, Ignacio F},
                 journal={Remote Sensing},
                 volume={14},
                 number={21},
                 pages={5597},
                 year={2022},
                 publisher={MDPI}
        }
        Further BibTeX Citation - "Sea-Land-Cloud Segmentation in Satellite Hyperspectral Imagery by Deep Learning" by Justo et al.
        @article{justo2023sea,
                 title={Sea-Land-Cloud Segmentation in Satellite Hyperspectral Imagery by Deep Learning},
                 author={Justo, Jon Alvarez and Garrett, Joseph Landon and Georgescu, Mariana-Iuliana and Gonzalez-Llorente, Jesus and Ionescu, Radu Tudor and Johansen, Tor Arne},
                 journal={arXiv preprint arXiv:2310.16210},
                 year={2023}
        }
    """


    inputs = Input(input_size)
    
    conv1 = Conv2D(1, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    
    conv2 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)
    

    
    conv3 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv4 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    
    up5 = Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv4))    
    merge5 = concatenate([conv3,up5], axis = 3)
    conv5 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge5)
    conv5 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    
    up6 = Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5)) 
    merge6 = concatenate([conv2,up6], axis = 3)
    conv6 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    poolmult1 = MaxPooling2D(pool_size=(2, 2))(inputs)
    
    convmult2 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(poolmult1)
    convmult2 = BatchNormalization()(convmult2)
    convmult2 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(convmult2)
    convmult2 = BatchNormalization()(convmult2)

    poolmult2 = MaxPooling2D(pool_size=(2, 2))(convmult2) 
    
    convmult3 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(poolmult2)
    convmult3 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(convmult3)

    upmult3 = Conv2D(8, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(convmult3)) 

    mergemult3 = concatenate([convmult2,upmult3], axis =3)
    
    convmult4 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(mergemult3)
    convmult4 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(convmult4)
    
    upmult4 = Conv2D(8, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(convmult4))
    
    merge7 = concatenate([conv6,upmult4], axis = 3)
    
    conv7 = Conv2D(num_classes, 1, activation = 'softmax')(merge7)

    model = Model(inputs = inputs, outputs = conv7)


    print('Model architecture DEFINED!')
    print('Defining now its compiler, loss function and metrics to prepare for training...')
    model.compile(optimizer='adam', \
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    print('Defined compiler, loss function and metrics!')

    return model




def model_2D_CUNet(in_shape, num_classes, use_padding=True):
    """
        2D-CUNet
        Example use of the method is found below. 
        Informative Note: 
            1) The hyperparameters below are initialized as in the experiments of the paper: "Sea-Land-Cloud Segmentation in Satellite Hyperspectral Imagery by Deep Learning" by Justo et al.
            2) The input size must be of size: NUMBER_OF_PATCHES x PATCH_SIZE x PATCH_SIZE x CHANNEL_FEATURES
            3) The ground-truth annotations must be encoded. For instance, one-hot encode categorical annotations and obtain: NUMBER_OF_PATCHES x PATCH_SIZE x PATCH_SIZE x NUMBER_OF_CLASSES

        PATCH_SIZE=48
        NUMBER_OF_CHANNELS=112
        model_classifier=model_2D_CUNet(in_shape=(PATCH_SIZE, PATCH_SIZE, NUMBER_OF_CHANNELS),\
                                        num_classes=3, \
                                        use_padding=True)

    

        Baseline Model Sourced in (slight adjustements):
        @mastersthesis{tagestad2021hardware,
                       title={Hardware acceleration of a compact CNN model for semantic segmentation of hyperspectral satellite images},
                       author={Tagestad, Sondre},
                       year={2021},
                       school={NTNU}
        }
        Further BibTeX Citation - "Sea-Land-Cloud Segmentation in Satellite Hyperspectral Imagery by Deep Learning" by Justo et al.
        @article{justo2023sea,
                 title={Sea-Land-Cloud Segmentation in Satellite Hyperspectral Imagery by Deep Learning},
                 author={Justo, Jon Alvarez and Garrett, Joseph Landon and Georgescu, Mariana-Iuliana and Gonzalez-Llorente, Jesus and Ionescu, Radu Tudor and Johansen, Tor Arne},
                 journal={arXiv preprint arXiv:2310.16210},
                 year={2023}
        }
    """

    pad = 'same' if use_padding else 'valid'
    model = keras.Sequential()

    model.add(Conv2D(input_shape=in_shape, kernel_size=3, filters=8, activation='relu', padding=pad))
    model.add(Conv2D(filters=8, kernel_size=3,  activation='relu', padding=pad))
    model.add(SeparableConv2D(filters=8, kernel_size=3, activation='relu', padding=pad))
    model.add(MaxPooling2D(pool_size=2, padding=pad))

    model.add(Conv2D(filters=16, kernel_size=3, activation='relu', padding=pad))
    model.add(Conv2D(filters=16, kernel_size=3, activation='relu', padding=pad))
    model.add(SeparableConv2D(filters=16, kernel_size=3, activation='relu', padding=pad))
    model.add(MaxPooling2D(pool_size=2, padding=pad))

    model.add(Conv2D(filters=32, kernel_size=3, activation='relu', padding=pad))
    model.add(Conv2D(filters=32, kernel_size=3, activation='relu', padding=pad))
    model.add(SeparableConv2D(filters=32, kernel_size=3, activation='relu', padding=pad))
    model.add(MaxPooling2D(pool_size=2, padding=pad))

    model.add(Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding=pad))
    model.add(Conv2D(filters=32, kernel_size=3, activation='relu', padding=pad))
    model.add(Conv2D(filters=32, kernel_size=3, activation='relu', padding=pad))

    model.add(Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding=pad))
    model.add(Conv2D(filters=16, kernel_size=3, activation='relu', padding=pad))
    model.add(Conv2D(filters=16, kernel_size=3, activation='relu', padding=pad))

    model.add(Conv2DTranspose(filters=8, kernel_size=3, strides=2, padding=pad))
    model.add(Conv2D(filters=8, kernel_size=3, activation='relu', padding=pad))
    model.add(Conv2D(filters=8, kernel_size=3, activation='relu', padding=pad))

    model.add(Conv2D(filters=num_classes, kernel_size=1, activation='softmax', padding=pad))

    print('Model architecture DEFINED!')
    print('Defining now its compiler, loss function and metrics to prepare for training...')
    model.compile(optimizer='adam', \
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    print('Defined compiler, loss function and metrics!')

    return model



def model_2D_CUNet_PLUS_PLUS(in_shape, num_classes):
    """
        2D-CUNet ++
        Example use of the method is found below. 
        Informative Note: 
            1) The hyperparameters below are initialized as in the experiments of the paper: "Sea-Land-Cloud Segmentation in Satellite Hyperspectral Imagery by Deep Learning" by Justo et al.
            2) The input size must be of size: NUMBER_OF_PATCHES x PATCH_SIZE x PATCH_SIZE x CHANNEL_FEATURES
            3) The ground-truth annotations must be encoded. For instance, one-hot encode categorical annotations and obtain: NUMBER_OF_PATCHES x PATCH_SIZE x PATCH_SIZE x NUMBER_OF_CLASSES


        PATCH_SIZE=48
        NUMBER_OF_CHANNELS=112
        model_classifier=model_2D_CUNet_PLUS_PLUS(in_shape=(PATCH_SIZE, PATCH_SIZE, NUMBER_OF_CHANNELS),\
                                                  num_classes=3)
        

        Baseline Model Sourced in (slight adjustements):
        @mastersthesis{tagestad2021hardware,
                       title={Hardware acceleration of a compact CNN model for semantic segmentation of hyperspectral satellite images},
                       author={Tagestad, Sondre},
                       year={2021},
                       school={NTNU}
        }
        Further BibTeX Citation - "Sea-Land-Cloud Segmentation in Satellite Hyperspectral Imagery by Deep Learning" by Justo et al.
        @article{justo2023sea,
                 title={Sea-Land-Cloud Segmentation in Satellite Hyperspectral Imagery by Deep Learning},
                 author={Justo, Jon Alvarez and Garrett, Joseph Landon and Georgescu, Mariana-Iuliana and Gonzalez-Llorente, Jesus and Ionescu, Radu Tudor and Johansen, Tor Arne},
                 journal={arXiv preprint arXiv:2310.16210},
                 year={2023}
        }
    """

    model = keras.Sequential([
        Input(in_shape),
        Conv2D(kernel_size=3, filters=8, activation='relu', padding='same'),

        MaxPooling2D(pool_size=2),
        SeparableConv2D(filters=16, kernel_size=3, activation='relu', padding='same'),

        MaxPooling2D(pool_size=2),
        SeparableConv2D(filters=32, kernel_size=3, activation='relu', padding='same'),

        MaxPooling2D(pool_size=2),

        Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same'),
        Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding='same'),
        Conv2DTranspose(filters=8, kernel_size=3, strides=2, padding='same'),
        Conv2D(filters=8, kernel_size=3, activation='relu', padding='same'),

        Conv2D(filters=num_classes, kernel_size=1, activation='softmax', padding='same'),
    ])


    print('Model architecture DEFINED!')
    print('Defining now its compiler, loss function and metrics to prepare for training...')
    model.compile(optimizer='adam', \
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    print('Defined compiler, loss function and metrics!')

    return model





def model_2D_CUNet_Reduced(in_shape, num_classes, use_padding=True):
    """
        2D-CUNet Reduced
        Example use of the method is found below. 
        Informative Note: 
            1) The hyperparameters below are initialized as in the experiments of the paper: "Sea-Land-Cloud Segmentation in Satellite Hyperspectral Imagery by Deep Learning" by Justo et al.
            2) The input size must be of size: NUMBER_OF_PATCHES x PATCH_SIZE x PATCH_SIZE x CHANNEL_FEATURES
            3) The ground-truth annotations must be encoded. For instance, one-hot encode categorical annotations and obtain: NUMBER_OF_PATCHES x PATCH_SIZE x PATCH_SIZE x NUMBER_OF_CLASSES


        PATCH_SIZE=48
        NUMBER_OF_CHANNELS=112
        model_classifier=model_2D_CUNet_Reduced(in_shape=(PATCH_SIZE, PATCH_SIZE, NUMBER_OF_CHANNELS),\
                                                num_classes=3)


        Baseline Model Sourced in (slight adjustements):
        @mastersthesis{tagestad2021hardware,
                       title={Hardware acceleration of a compact CNN model for semantic segmentation of hyperspectral satellite images},
                       author={Tagestad, Sondre},
                       year={2021},
                       school={NTNU}
        }
        Further BibTeX Citation - "Sea-Land-Cloud Segmentation in Satellite Hyperspectral Imagery by Deep Learning" by Justo et al.
        @article{justo2023sea,
                 title={Sea-Land-Cloud Segmentation in Satellite Hyperspectral Imagery by Deep Learning},
                 author={Justo, Jon Alvarez and Garrett, Joseph Landon and Georgescu, Mariana-Iuliana and Gonzalez-Llorente, Jesus and Ionescu, Radu Tudor and Johansen, Tor Arne},
                 journal={arXiv preprint arXiv:2310.16210},
                 year={2023}
        }
    """
    pad = 'same' if use_padding else 'valid'
    model = keras.Sequential()

    model.add(Conv2D(input_shape=in_shape, kernel_size=3, filters=8, activation='relu', padding=pad))
    model.add(Conv2D(filters=8, kernel_size=3,  activation='relu', padding=pad))
    model.add(SeparableConv2D(filters=8, kernel_size=3, activation='relu', padding=pad))
    model.add(MaxPooling2D(pool_size=2, padding=pad))

    model.add(Conv2D(filters=16, kernel_size=3, activation='relu', padding=pad))
    model.add(Conv2D(filters=16, kernel_size=3, activation='relu', padding=pad))
    model.add(SeparableConv2D(filters=16, kernel_size=3, activation='relu', padding=pad))
    model.add(MaxPooling2D(pool_size=2, padding=pad))

    model.add(Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding=pad))
    model.add(Conv2D(filters=16, kernel_size=3, activation='relu', padding=pad))
    model.add(Conv2D(filters=16, kernel_size=3, activation='relu', padding=pad))

    model.add(Conv2DTranspose(filters=8, kernel_size=3, strides=2, padding=pad))
    model.add(Conv2D(filters=8, kernel_size=3, activation='relu', padding=pad))
    model.add(Conv2D(filters=8, kernel_size=3, activation='relu', padding=pad))

    model.add(Conv2D(filters=num_classes, kernel_size=1, activation='softmax', padding=pad))

    print('Model architecture DEFINED!')
    print('Defining now its compiler, loss function and metrics to prepare for training...')
    model.compile(optimizer='adam', \
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    print('Defined compiler, loss function and metrics!')

    return model




def model_2D_CUNet_PLUS_PLUS_Reduced (in_shape, num_classes, use_padding=True):
    """
        2D-CUNet++ Reduced
        Example use of the method is found below. 
        Informative Note: 
            1) The hyperparameters below are initialized as in the experiments of the paper: "Sea-Land-Cloud Segmentation in Satellite Hyperspectral Imagery by Deep Learning" by Justo et al.
            2) The input size must be of size: NUMBER_OF_PATCHES x PATCH_SIZE x PATCH_SIZE x CHANNEL_FEATURES
            3) The ground-truth annotations must be encoded. For instance, one-hot encode categorical annotations and obtain: NUMBER_OF_PATCHES x PATCH_SIZE x PATCH_SIZE x NUMBER_OF_CLASSES

        PATCH_SIZE=48
        NUMBER_OF_CHANNELS=112
        model_classifier=model_2D_CUNet_PLUS_PLUS_Reduced(
                                            in_shape=(PATCH_SIZE, PATCH_SIZE, NUMBER_OF_CHANNELS),\
                                            num_classes=3)
                                            
        Baseline Model Sourced in (slight adjustements):
        @mastersthesis{tagestad2021hardware,
                       title={Hardware acceleration of a compact CNN model for semantic segmentation of hyperspectral satellite images},
                       author={Tagestad, Sondre},
                       year={2021},
                       school={NTNU}
        }
        Further BibTeX Citation - "Sea-Land-Cloud Segmentation in Satellite Hyperspectral Imagery by Deep Learning" by Justo et al.
        @article{justo2023sea,
                 title={Sea-Land-Cloud Segmentation in Satellite Hyperspectral Imagery by Deep Learning},
                 author={Justo, Jon Alvarez and Garrett, Joseph Landon and Georgescu, Mariana-Iuliana and Gonzalez-Llorente, Jesus and Ionescu, Radu Tudor and Johansen, Tor Arne},
                 journal={arXiv preprint arXiv:2310.16210},
                 year={2023}
        }
    """

    pad = 'same' if use_padding else 'valid'
    model = keras.Sequential()

    model.add(Conv2D(input_shape=(in_shape), kernel_size=3, filters=8, activation='relu', padding=pad))
    model.add(MaxPooling2D(pool_size=2, padding=pad))
    model.add(SeparableConv2D(filters=16, kernel_size=3, activation='relu', padding=pad))
    model.add(MaxPooling2D(pool_size=2, padding=pad))

    model.add(Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding=pad))
    model.add(Conv2DTranspose(filters=8, kernel_size=3, strides=2, padding=pad))
    model.add(Conv2D(filters=8, kernel_size=3, activation='relu', padding=pad))

    model.add(Conv2D(filters=num_classes, kernel_size=1, activation='softmax', padding=pad))

    print('Model architecture DEFINED!')
    print('Defining now its compiler, loss function and metrics to prepare for training...')
    model.compile(optimizer='adam', \
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    print('Defined compiler, loss function and metrics!')

    return model




def model_2D_UNet_FAUBAI(input_size, N_CLASSES):
    """
        2D-UNet FAUBAI
        Example use of the method is found below. 
        Informative Note: 
            1) The hyperparameters below are initialized as in the experiments of the paper: "Sea-Land-Cloud Segmentation in Satellite Hyperspectral Imagery by Deep Learning" by Justo et al.
            2) The input size must be of size: NUMBER_OF_PATCHES x PATCH_SIZE x PATCH_SIZE x CHANNEL_FEATURES
            3) The ground-truth annotations must be encoded. For instance, one-hot encode categorical annotations and obtain: NUMBER_OF_PATCHES x PATCH_SIZE x PATCH_SIZE x NUMBER_OF_CLASSES

        PATCH_SIZE=48
        NUMBER_OF_CHANNELS=112
        model_classifier=model_2D_UNet_FAUBAI(input_size=(PATCH_SIZE, PATCH_SIZE, NUMBER_OF_CHANNELS),\
                                              N_CLASSES=3)


        Baseline Model Sourced in (slight adjustements):
        @mastersthesis{netteland2022exploration,
                       title={Exploration and Implementation of Large CNN Models for Image Segmentation in Hyperspectral CubeSat Missions},
                       author={Netteland, Simen},
                       year={2022},
                       school={NTNU}
        }
        Further BibTeX Citation - "Sea-Land-Cloud Segmentation in Satellite Hyperspectral Imagery by Deep Learning" by Justo et al.
        @article{justo2023sea,
                 title={Sea-Land-Cloud Segmentation in Satellite Hyperspectral Imagery by Deep Learning},
                 author={Justo, Jon Alvarez and Garrett, Joseph Landon and Georgescu, Mariana-Iuliana and Gonzalez-Llorente, Jesus and Ionescu, Radu Tudor and Johansen, Tor Arne},
                 journal={arXiv preprint arXiv:2310.16210},
                 year={2023}
        }
    """

    inputs = Input(input_size)

    c1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)

    u5 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c4)
    u5 = concatenate([u5, c3])
    c5 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u5)
    c5 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c2])
    c6 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c1])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    outputs = Conv2D(N_CLASSES, (1, 1), activation='softmax')(c7)

    model = Model(inputs=inputs, outputs=outputs)

    print('Model architecture DEFINED!')
    print('Defining now its compiler, loss function and metrics to prepare for training...')
    model.compile(optimizer='adam', \
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    print('Defined compiler, loss function and metrics!')

    return model




def model_2D_UNet_FAUBAI_Reduced(inp_size, N_CLASSES):
    """
        2D-UNet FAUBAI Reduced
        Example use of the method is found below. 
        Informative Note: 
            1) The hyperparameters below are initialized as in the experiments of the paper: "Sea-Land-Cloud Segmentation in Satellite Hyperspectral Imagery by Deep Learning" by Justo et al.
            2) The input size must be of size: NUMBER_OF_PATCHES x PATCH_SIZE x PATCH_SIZE x CHANNEL_FEATURES
            3) The ground-truth annotations must be encoded. For instance, one-hot encode categorical annotations and obtain: NUMBER_OF_PATCHES x PATCH_SIZE x PATCH_SIZE x NUMBER_OF_CLASSES

    
        PATCH_SIZE=48
        NUMBER_OF_CHANNELS=112
        model_classifier=model_2D_UNet_FAUBAI_Reduced(inp_size=(PATCH_SIZE, PATCH_SIZE, NUMBER_OF_CHANNELS),\
                                                      N_CLASSES=3)

    
        Baseline Model Sourced in (slight adjustements):
        @mastersthesis{netteland2022exploration,
                       title={Exploration and Implementation of Large CNN Models for Image Segmentation in Hyperspectral CubeSat Missions},
                       author={Netteland, Simen},
                       year={2022},
                       school={NTNU}
        }
        Further BibTeX Citation - "Sea-Land-Cloud Segmentation in Satellite Hyperspectral Imagery by Deep Learning" by Justo et al.
        @article{justo2023sea,
                 title={Sea-Land-Cloud Segmentation in Satellite Hyperspectral Imagery by Deep Learning},
                 author={Justo, Jon Alvarez and Garrett, Joseph Landon and Georgescu, Mariana-Iuliana and Gonzalez-Llorente, Jesus and Ionescu, Radu Tudor and Johansen, Tor Arne},
                 journal={arXiv preprint arXiv:2310.16210},
                 year={2023}
        }
    """

    inputs = Input(inp_size, name='input')

    c1 = Conv2D(16, (3, 3), name='conv2d_c1_1', activation='relu', kernel_initializer='he_normal', padding='same')(inputs) 
    c1 = Conv2D(16, (3, 3), name='conv2d_c1_2', activation='relu', kernel_initializer='he_normal', padding='same')(c1) 
    p1 = MaxPooling2D((2, 2), name='maxpool_p1')(c1) 

    c2 = Conv2D(32, (3,3), name='conv2d_c2_1', activation='relu', kernel_initializer='he_normal', padding='same')(p1) 
    c2 = Conv2D(32, (3, 3), name='conv2d_c2_2', activation='relu', kernel_initializer='he_normal', padding='same')(c2) 
    p2 = MaxPooling2D((2, 2), name='maxpool_p2')(c2) 

    c3 = Conv2D(64, (3,3), name='conv2d_c3_1', activation='relu', kernel_initializer='he_normal', padding='same')(p2) 
    c3 = Conv2D(64, (3,3), name='conv2d_c3_2', activation='relu', kernel_initializer='he_normal', padding='same')(c3) 
    p3 = MaxPooling2D((2,2), name='maxpool_p3')(c3) 

    c4 = Conv2D(128, (3,3), name='conv2d_c4_1', activation='relu', kernel_initializer='he_normal', padding='same')(p3) 
    c4 = Conv2D(128, (3,3), name='conv2d_c4_2', activation='relu', kernel_initializer='he_normal', padding='same')(c4) 
    p4 = MaxPooling2D((2,2), name='maxpool_p4')(c4) 

    c5 = Conv2D(filters=256, kernel_size=3, name='conv2d_c5_1', activation='relu', kernel_initializer='he_normal', padding='same')(p4) 
    c5 = Conv2D(filters=256, kernel_size=3, name='conv2d_c5_2', activation='relu', kernel_initializer='he_normal', padding='same')(c5) 
    u6 = Conv2DTranspose(128, (2,2), name='conv2d_tran_u6', strides=(2,2), padding='same')(c5) 

    u6 = concatenate([u6, c4], name='concat_u6') 
    c6 = Conv2D(128, (3,3), name='conv2d_c6_1', activation='relu', kernel_initializer='he_normal', padding='same')(u6) 
    c6 = Conv2D(128, (3,3), name='conv2d_c6_2', activation='relu', kernel_initializer='he_normal', padding='same')(c6) 
    u7 = Conv2DTranspose(64, (2,2), name='conv2d_tran_u7', strides=(2,2), padding='same')(c6) 

    u7 = concatenate([u7, c3], name='concat_u7') 
    c7 = Conv2D(64, (3,3), name='conv2d_c7_1', activation='relu', kernel_initializer='he_normal', padding='same')(u7) 
    c7 = Conv2D(64, (3,3), name='conv2d_c7_2', activation='relu', kernel_initializer='he_normal', padding='same')(c7) 
    u8 = Conv2DTranspose(32, (2,2), name='conv2d_tran_u8', strides=(2,2), padding='same')(c7) 

    u8 = concatenate([u8, c2], name='concat_u8') 
    c8 = Conv2D(32, (3,3), name='conv2d_c8_1', activation='relu', kernel_initializer='he_normal', padding='same')(u8) 
    c8 = Conv2D(32, (3,3), name='conv2d_c8_2', activation='relu', kernel_initializer='he_normal', padding='same')(c8) 
    u9 = Conv2DTranspose(16, (2,2), name='conv2d_tran_u9', strides=(2,2), padding='same')(c8) 
    
    u9 = concatenate([u9, c1], name='concat_u9', axis=3) 
    c9 = Conv2D(16, (3,3), name='conv2d_c9_1', activation='relu', kernel_initializer='he_normal', padding='same')(u9) 
    c9 = Conv2D(16, (3,3), name='conv2d_c9_2', activation='relu', kernel_initializer='he_normal', padding='same')(c9) 

    outputs = Conv2D(N_CLASSES, (1,1), name='conv2d_outputs', activation='softmax')(c9) 

    model = Model(inputs=inputs, outputs=outputs)

    print('Model architecture DEFINED!')
    print('Defining now its compiler, loss function and metrics to prepare for training...')
    model.compile(optimizer='adam', \
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    print('Defined compiler, loss function and metrics!')

    return model

