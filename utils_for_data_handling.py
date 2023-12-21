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

from patchify import patchify
import numpy as np



class utils_for_data_handling: 
    def __init__(self):
        print('Utils object instanced!')

        


    def cropping_or_padding_of_spatial_resolution(self, DATA_POINTS_PROCESSING_MODE_DURING_INFERENCE, ENABLE_SHOULD_PAD, DATA, LABELS, PATCH_SIZE, PADDING_TECHNIQUE):
        """
        Further details in the call to the method in the Python noteobok.
        """
        if DATA_POINTS_PROCESSING_MODE_DURING_INFERENCE=='1D-PROCESSING':
            print('The segmentation mode during inference is: ', DATA_POINTS_PROCESSING_MODE_DURING_INFERENCE, ' and hence it does not make sense to make any cropping or padding as that is only done when the data needs to be patched.')
            return 
        elif DATA_POINTS_PROCESSING_MODE_DURING_INFERENCE=='3D-PROCESSING':
            if not ENABLE_SHOULD_PAD: 
                print('The segmentation mode is: ', DATA_POINTS_PROCESSING_MODE_DURING_INFERENCE, '. Before patching the images and labels, the spatial dimensions will be cropped (not padded) as the flag for padding is set to: ', ENABLE_SHOULD_PAD)
            else: 
                print('The segmentation mode is: ', DATA_POINTS_PROCESSING_MODE_DURING_INFERENCE, '. Before patching the images and labels, the spatial dimensions will be padded (not cropped) as the flag for padding is set to: ', ENABLE_SHOULD_PAD)
 


        NUMBER_OF_LINES=DATA.shape[1]
        NUMBER_OF_SAMPLES=DATA.shape[2]
        # Analyze the SAMPLES dimension
        SAMPLES_DIMENSION_NEED_ADJUSTMENT=False
        NUMBER_OF_PATCHES_IN_SAMPLES=NUMBER_OF_SAMPLES/PATCH_SIZE # This may give a non-integer number of patches for the samples dimension
        if NUMBER_OF_PATCHES_IN_SAMPLES != int(NUMBER_OF_PATCHES_IN_SAMPLES): # Not integer number of patches
            print('The samples dimension does not give an integer number of patches. Number of samples: ', DATA.shape[2], ', patch_size: ', PATCH_SIZE)
            SAMPLES_DIMENSION_NEED_ADJUSTMENT=True

        # Analyze the LINES dimension
        LINES_DIMENSION_NEED_ADJUSTMENT=False
        NUMBER_OF_PATCHES_IN_LINES=NUMBER_OF_LINES/PATCH_SIZE # This may give a non-integer number of patches for the lines dimension
        if NUMBER_OF_PATCHES_IN_LINES != int(NUMBER_OF_PATCHES_IN_LINES): # Not integer number of patches
            print('The lines dimension does not give an integer number of patches. Number of lines: ', DATA.shape[1], ', patch_size: ', PATCH_SIZE)
            LINES_DIMENSION_NEED_ADJUSTMENT=True
            



        # The following is to calculate the number of lines/samples that may be cropped/padded
        if not ENABLE_SHOULD_PAD: # We need to apply cropping - WE DON'T DO IT YET, WE ONLY COMPUTE THE NEEDED DIMENSIONS
            if SAMPLES_DIMENSION_NEED_ADJUSTMENT:
                NUMBER_OF_SAMPLES_ADJUSTED=int(NUMBER_OF_SAMPLES - (NUMBER_OF_PATCHES_IN_SAMPLES-np.floor(NUMBER_OF_PATCHES_IN_SAMPLES))*PATCH_SIZE)
                NUMBER_OF_PATCHES_IN_SAMPLES_DIMENSION_ADJUSTED=np.floor(NUMBER_OF_PATCHES_IN_SAMPLES)
                NUMBER_OF_SAMPLES_TO_CROP_OR_PAD= - int(round( (NUMBER_OF_PATCHES_IN_SAMPLES-np.floor(NUMBER_OF_PATCHES_IN_SAMPLES))*PATCH_SIZE ) ) # The minus indicates the samples are to crop
                                # The reason why we need to apply a round is because due to decimal precision, it could happen for instance that we get instead of -4, we get e.g.:
                                #               -3.999999, then when applying int() we would get -3, which is wrong, so this is why we need the round here
                print('Number of samples after cropping adjustment will be: ', NUMBER_OF_SAMPLES_ADJUSTED)
                print('Number of patches in samples dimension (adjusted) will be: ', NUMBER_OF_PATCHES_IN_SAMPLES_DIMENSION_ADJUSTED)
                print('Number of samples to crop will be: ', NUMBER_OF_SAMPLES_TO_CROP_OR_PAD)
            if LINES_DIMENSION_NEED_ADJUSTMENT:
                NUMBER_OF_LINES_ADJUSTED=int(NUMBER_OF_LINES - (NUMBER_OF_PATCHES_IN_LINES-np.floor(NUMBER_OF_PATCHES_IN_LINES))*PATCH_SIZE)
                NUMBER_OF_PATCHES_IN_LINES_DIMENSION_ADJUSTED=np.floor(NUMBER_OF_PATCHES_IN_LINES)   
                NUMBER_OF_LINES_TO_CROP_OR_PAD= - int( round( (NUMBER_OF_PATCHES_IN_LINES-np.floor(NUMBER_OF_PATCHES_IN_LINES))*PATCH_SIZE ) )  # The minus indicates the lines are to crop
                print('Number of lines after cropping adjustment will be: ', NUMBER_OF_LINES_ADJUSTED)
                print('Number of patches in lines dimension (adjusted) will be: ', NUMBER_OF_PATCHES_IN_LINES_DIMENSION_ADJUSTED)     
                print('Number of lines to crop will be: ', NUMBER_OF_LINES_TO_CROP_OR_PAD)
        else: # We need to apply padding - WE DON'T DO IT YET, WE ONLY COMPUTE THE NEEDED DIMENSIONS
            if SAMPLES_DIMENSION_NEED_ADJUSTMENT:
                NUMBER_OF_SAMPLES_ADJUSTED=NUMBER_OF_SAMPLES + (np.ceil(NUMBER_OF_PATCHES_IN_SAMPLES) - NUMBER_OF_PATCHES_IN_SAMPLES)*PATCH_SIZE
                NUMBER_OF_PATCHES_IN_SAMPLES_DIMENSION_ADJUSTED=np.ceil(NUMBER_OF_PATCHES_IN_SAMPLES)
                NUMBER_OF_SAMPLES_TO_CROP_OR_PAD= + int(round( (np.ceil(NUMBER_OF_PATCHES_IN_SAMPLES) - NUMBER_OF_PATCHES_IN_SAMPLES)*PATCH_SIZE )) # The positive sign indicates the samples are to pad
                print('Number of samples after padding adjustment will be: ', NUMBER_OF_SAMPLES_ADJUSTED)
                print('Number of patches in samples dimension (adjusted) will be: ', NUMBER_OF_PATCHES_IN_SAMPLES_DIMENSION_ADJUSTED)
                print('Number of samples to padd will be: ', NUMBER_OF_SAMPLES_TO_CROP_OR_PAD)

            if LINES_DIMENSION_NEED_ADJUSTMENT: 
                NUMBER_OF_LINES_ADJUSTED=NUMBER_OF_LINES + (np.ceil(NUMBER_OF_PATCHES_IN_LINES) - NUMBER_OF_PATCHES_IN_LINES)*PATCH_SIZE
                NUMBER_OF_PATCHES_IN_LINES_DIMENSION_ADJUSTED=np.ceil(NUMBER_OF_PATCHES_IN_LINES)
                NUMBER_OF_LINES_TO_CROP_OR_PAD= + int(round((np.ceil(NUMBER_OF_PATCHES_IN_LINES) - NUMBER_OF_PATCHES_IN_LINES)*PATCH_SIZE)) # The positive sign indicates the lines are to pad
                                            # For instance, if 956 lines and PATCH_SIZE=100, the result inside the int would be: 43.999, and with int() it becomes 43, so the round is needed
                                            # to ensure that it goes to 44 instead
            
                print('Number of lines after padding adjustment will be: ', NUMBER_OF_LINES_ADJUSTED)
                print('Number of patches in lines dimension (adjusted) will be: ', NUMBER_OF_PATCHES_IN_LINES_DIMENSION_ADJUSTED) 
                print('Number of lines to pad will be: ', NUMBER_OF_LINES_TO_CROP_OR_PAD)



        if not SAMPLES_DIMENSION_NEED_ADJUSTMENT: # The number of samples adjusted will just contain the same number of samples
                                                  # Also, the number of patches in the samples dimension remain the same as no cropping or padding is needed
            print('The number of samples will not be adjusted by cropping/padding since it seems we can obtain an integer number of patches in this dimension.')
            NUMBER_OF_SAMPLES_ADJUSTED=NUMBER_OF_SAMPLES
            NUMBER_OF_PATCHES_IN_SAMPLES_DIMENSION_ADJUSTED=int(NUMBER_OF_PATCHES_IN_SAMPLES)
            NUMBER_OF_SAMPLES_TO_CROP_OR_PAD=int(0)

        if not LINES_DIMENSION_NEED_ADJUSTMENT:   # The number of lines adjusted will just contain the same number of lines
                                                  # Also, the number of patches in the lines dimension remain the same as no cropping or padding is needed
            print('The number of lines will not be adjusted by cropping/padding since it seems we can obtain an integer number of patches in this dimension.')
            NUMBER_OF_LINES_ADJUSTED=NUMBER_OF_LINES
            NUMBER_OF_PATCHES_IN_LINES_DIMENSION_ADJUSTED=int(NUMBER_OF_PATCHES_IN_LINES)
            NUMBER_OF_LINES_TO_CROP_OR_PAD=int(0)
        
        print('DATA before CROPPING OR PADDING: ', DATA.shape, ', and dtype: ', DATA.dtype)
        print('ANNOTATIONS before CROPPING OR PADDING: ', LABELS.shape, ', and dtype: ', LABELS.dtype)

        if not ENABLE_SHOULD_PAD: # It will crop  
            # The number of pixels to crop is even
            if NUMBER_OF_SAMPLES_TO_CROP_OR_PAD/2==int(NUMBER_OF_SAMPLES_TO_CROP_OR_PAD/2) and NUMBER_OF_LINES_TO_CROP_OR_PAD/2==int(NUMBER_OF_LINES_TO_CROP_OR_PAD/2):
                
                DATA=DATA[:, int(-NUMBER_OF_LINES_TO_CROP_OR_PAD/2):, int(-NUMBER_OF_SAMPLES_TO_CROP_OR_PAD/2):, :]
                DATA=DATA[:, :(DATA.shape[1]-(int(-NUMBER_OF_LINES_TO_CROP_OR_PAD/2))), :(DATA.shape[2]-(int(-NUMBER_OF_SAMPLES_TO_CROP_OR_PAD/2))), :]

                LABELS=LABELS[:, int(-NUMBER_OF_LINES_TO_CROP_OR_PAD/2):, int(-NUMBER_OF_SAMPLES_TO_CROP_OR_PAD/2):]
                LABELS=LABELS[:, :(LABELS.shape[1]-(int(-NUMBER_OF_LINES_TO_CROP_OR_PAD/2))), :(LABELS.shape[2]-(int(-NUMBER_OF_SAMPLES_TO_CROP_OR_PAD/2)))]
            else: # Any of the number of pixels in lines or samples (or both) are not even, we need to check below which exactly

                if NUMBER_OF_SAMPLES_TO_CROP_OR_PAD/2==int(NUMBER_OF_SAMPLES_TO_CROP_OR_PAD/2):
                    # The number of pixels to remove in samples direction is integer
                    DATA=DATA[:, :, int( - NUMBER_OF_SAMPLES_TO_CROP_OR_PAD/2):, :]
                    DATA=DATA[:, :, :(DATA.shape[2]-int( - NUMBER_OF_SAMPLES_TO_CROP_OR_PAD/2)), :]
                    LABELS=LABELS[:, :, int( - NUMBER_OF_SAMPLES_TO_CROP_OR_PAD/2):]
                    LABELS=LABELS[:, :, :(LABELS.shape[2]-int( - NUMBER_OF_SAMPLES_TO_CROP_OR_PAD/2)) ]           
                else: 
                    # Number of pixels to remove in samples direction is not integer
                    DATA=DATA[:, :, int(np.floor( - NUMBER_OF_SAMPLES_TO_CROP_OR_PAD/2)):, :]
                    DATA=DATA[:, :, :(DATA.shape[2]-int(np.ceil(-NUMBER_OF_SAMPLES_TO_CROP_OR_PAD/2))), :] # +1 extra pixel compared to first samples
                    LABELS=LABELS[:, :, int(np.floor( - NUMBER_OF_SAMPLES_TO_CROP_OR_PAD/2)):]
                    LABELS=LABELS[:, :, :(LABELS.shape[2]-int(np.ceil( - NUMBER_OF_SAMPLES_TO_CROP_OR_PAD/2))) ] # +1 extra pixel compared to first samples
                
                
                if NUMBER_OF_LINES_TO_CROP_OR_PAD/2==int(NUMBER_OF_LINES_TO_CROP_OR_PAD/2):
                    # Number of pixels to remove in lines direction is integer
                    DATA=DATA[:, int( - NUMBER_OF_LINES_TO_CROP_OR_PAD/2):, :, :]
                    DATA=DATA[:, :(DATA.shape[1]-int( - NUMBER_OF_LINES_TO_CROP_OR_PAD/2)), :, :]
                    LABELS=LABELS[:, int( - NUMBER_OF_LINES_TO_CROP_OR_PAD/2):, :]
                    LABELS=LABELS[:, :(LABELS.shape[1]-int( - NUMBER_OF_LINES_TO_CROP_OR_PAD/2)), :]
                else:
                    # Number of pixels to remove in lines direction is not integer
                    DATA=DATA[:, int(np.floor( - NUMBER_OF_LINES_TO_CROP_OR_PAD/2)):, :, :] 
                    DATA=DATA[:, :(DATA.shape[1]-int(np.ceil( - NUMBER_OF_LINES_TO_CROP_OR_PAD/2))), :, :] # +1 extra pixel
                    LABELS=LABELS[:, int(np.floor( - NUMBER_OF_LINES_TO_CROP_OR_PAD/2)):, :] 
                    LABELS=LABELS[:, :(LABELS.shape[1]-int(np.ceil( - NUMBER_OF_LINES_TO_CROP_OR_PAD/2))), :] # extra pixel here
        
        else: # If not cropping, then it has to be padding
            if PADDING_TECHNIQUE=='CONSTANT_PADDING_EXTENDING_EDGES':

                if NUMBER_OF_SAMPLES_TO_CROP_OR_PAD/2==int(NUMBER_OF_SAMPLES_TO_CROP_OR_PAD/2) and NUMBER_OF_LINES_TO_CROP_OR_PAD/2==int(NUMBER_OF_LINES_TO_CROP_OR_PAD/2):
                    
                    DATA=np.pad(DATA, \
                                pad_width=((0, 0), \
                                            (int(NUMBER_OF_LINES_TO_CROP_OR_PAD/2), int(NUMBER_OF_LINES_TO_CROP_OR_PAD/2)), \
                                            (int(NUMBER_OF_SAMPLES_TO_CROP_OR_PAD/2), int(NUMBER_OF_SAMPLES_TO_CROP_OR_PAD/2)),
                                            (0,0)   
                                          ),\
                                mode='edge') 
                    LABELS= np.pad(LABELS,\
                                    pad_width=( (0, 0), \
                                                (int(NUMBER_OF_LINES_TO_CROP_OR_PAD/2), int(NUMBER_OF_LINES_TO_CROP_OR_PAD/2)), \
                                                (int(NUMBER_OF_SAMPLES_TO_CROP_OR_PAD/2), int(NUMBER_OF_SAMPLES_TO_CROP_OR_PAD/2))
                                              ),\
                                    mode='edge') 
                else: # The number of samples or lines to add is not even

                    if NUMBER_OF_SAMPLES_TO_CROP_OR_PAD/2==int(NUMBER_OF_SAMPLES_TO_CROP_OR_PAD/2):
                        DATA = np.pad(DATA, \
                                        pad_width=((0, 0), \
                                                    (0,0), \
                                                    (int(NUMBER_OF_SAMPLES_TO_CROP_OR_PAD/2), int(NUMBER_OF_SAMPLES_TO_CROP_OR_PAD/2)), 
                                                    (0,0)
                                                  ),\
                                        mode='edge') 

                        LABELS= np.pad(LABELS,\
                                        pad_width=( (0, 0), \
                                                    (0, 0), \
                                                    (int(NUMBER_OF_SAMPLES_TO_CROP_OR_PAD/2), int(NUMBER_OF_SAMPLES_TO_CROP_OR_PAD/2))),\
                                        mode='edge')   
                    else: 

                        DATA = np.pad(DATA,\
                                        pad_width=((0, 0), \
                                                    (0,0), \
                                                    (int(np.floor(NUMBER_OF_SAMPLES_TO_CROP_OR_PAD/2)), int(np.ceil(NUMBER_OF_SAMPLES_TO_CROP_OR_PAD/2))), 
                                                    (0,0)
                                                  ),\
                                        mode='edge') 
                        LABELS = np.pad(LABELS,\
                                        pad_width=((0, 0), \
                                                    (0, 0), \
                                                    (int(np.floor(NUMBER_OF_SAMPLES_TO_CROP_OR_PAD/2)), int(np.ceil(NUMBER_OF_SAMPLES_TO_CROP_OR_PAD/2)))),\
                                        mode='edge')    
                    if NUMBER_OF_LINES_TO_CROP_OR_PAD/2==int(NUMBER_OF_LINES_TO_CROP_OR_PAD/2):

                        DATA = np.pad(DATA,\
                                    pad_width=((0, 0), \
                                                (int(NUMBER_OF_LINES_TO_CROP_OR_PAD/2), int(NUMBER_OF_LINES_TO_CROP_OR_PAD/2)), \
                                                (0,0),
                                                (0, 0)
                                              ),\
                                    mode='edge') 

                        LABELS = np.pad(LABELS,\
                                        pad_width=((0, 0), \
                                                    (int(NUMBER_OF_LINES_TO_CROP_OR_PAD/2), int(NUMBER_OF_LINES_TO_CROP_OR_PAD/2)), \
                                                    (0, 0)
                                                  ),\
                                        mode='edge')
                    else:

                        DATA = np.pad(DATA,\
                                        pad_width=((0, 0), \
                                                    (int(np.floor(NUMBER_OF_LINES_TO_CROP_OR_PAD/2)), int(np.ceil(NUMBER_OF_LINES_TO_CROP_OR_PAD/2))), \
                                                    (0, 0), 
                                                    (0, 0)
                                                  ),\
                                        mode='edge') 
                        
                        LABELS = np.pad(LABELS,\
                                        pad_width=((0, 0), \
                                                    (int(np.floor(NUMBER_OF_LINES_TO_CROP_OR_PAD/2)), int(np.ceil(NUMBER_OF_LINES_TO_CROP_OR_PAD/2))), \
                                                    (0, 0)),\
                                        mode='edge')     



            print('DATA AFTER CROPPING OR PADDING: ', DATA.shape, ', and dtype: ', DATA.dtype)
            print('LABELS AFTER CROPPING OR PADDING: ', LABELS.shape, ', and dtype: ', LABELS.dtype)
            return DATA, LABELS, NUMBER_OF_PATCHES_IN_SAMPLES_DIMENSION_ADJUSTED, NUMBER_OF_PATCHES_IN_LINES_DIMENSION_ADJUSTED




    def patch_data_and_annotations(self, DATA, ANNOTATIONS, PATCH_SIZE, NUMBER_OF_PATCHES_IN_LINES_DIMENSION_ADJUSTED, NUMBER_OF_PATCHES_IN_SAMPLES_DIMENSION_ADJUSTED):
        """
        Further details in the call to the method in the Python noteobok.
        """
        print('Starting patching.....')
        print('DATA BEFORE PATCHING: ', DATA.shape, ', and dtype: ', DATA.dtype)
        print('ANNOTATIONS BEFORE PATCHING: ', ANNOTATIONS.shape, ', and dtype: ', ANNOTATIONS.dtype)
        DATA_patched=\
                    np.zeros((DATA.shape[0],\
                              NUMBER_OF_PATCHES_IN_LINES_DIMENSION_ADJUSTED,\
                              NUMBER_OF_PATCHES_IN_SAMPLES_DIMENSION_ADJUSTED,\
                              PATCH_SIZE, PATCH_SIZE, DATA.shape[3]), \
                              dtype=DATA.dtype) 
                    # Dimensions are: 
                    #   IMAGES x PATCHES_IN_LINES_DIRECTION x PATCHES_IN_SAMPLES_DIRECTION x PATCH_SIZE x PATCH_SIZE x CHANNEL_FEATURES
                    #   NB: The patch map for an image is given by PATCHES_IN_LINES_DIRECTION x PATCHES_IN_SAMPLES_DIRECTION
        ANNOTATIONS_patched=\
                    np.zeros((ANNOTATIONS.shape[0],\
                              NUMBER_OF_PATCHES_IN_LINES_DIMENSION_ADJUSTED,\
                              NUMBER_OF_PATCHES_IN_SAMPLES_DIMENSION_ADJUSTED,\
                              PATCH_SIZE, PATCH_SIZE), \
                              dtype=ANNOTATIONS.dtype) # NB: No 1 needs to be included since patchify method does not produce an extra dimension in this case
                    # Dimensions are: 
                    #   IMAGES x PATCHES_IN_LINES_DIRECTION x PATCHES_IN_SAMPLES_DIRECTION x PATCH_SIZE x PATCH_SIZE
            
        for iterator_images in range(DATA.shape[0]): # It seems patchify does not support, at the time of writing, patching all images at once
            try:
                DATA_patched_intermediate=patchify( DATA[iterator_images, :, :, :],\
                                                    (PATCH_SIZE, PATCH_SIZE, DATA.shape[3]), \
                                                    step=PATCH_SIZE) 
                        # Patchify will give these dimensions: 
                        #       PATCHES_IN_LINES_DIRECTION x PATCHES_IN_SAMPLES_DIRECTION x 1 x PATCH_SIZE x PATCH_SIZE x CHANNEL_FEATURES
                DATA_patched_intermediate=np.squeeze(DATA_patched_intermediate, axis=2) 
                        # The following is only needed for the data:
                        #   Patchify produces an extra dimension after the dimensions for the patches in lines and samples, for instance:
                        #   29x21x1, i.e., 29 patches in lines and 21 patches in samples, and the x 1 is the extra dimension
                        #   With squeeze we remove the x 1 
                DATA_patched[iterator_images, :, :, :, :, :] = DATA_patched_intermediate
                
                ANNOTATIONS_patched[iterator_images, :, :, :, :] = \
                        patchify(ANNOTATIONS[iterator_images, :, :],\
                                (PATCH_SIZE, PATCH_SIZE), \
                                step=PATCH_SIZE) 
                        # Patchify produces PATCHES_IN_LINES_DIRECTION x PATCHES_IN_SAMPLES_DIRECTION x PATCH_SIZE x PATCH_SIZE
            except: 
                print('ERROR when patching the DATA or ANNOTATIONS.')
                exit(0)

        
        print('DATA AFTER PATCHING: ', DATA_patched.shape, ', and dtype: ', DATA_patched.dtype)
        print('ANNOTATIONS AFTER PATCHING: ', ANNOTATIONS_patched.shape, ', and dtype: ', ANNOTATIONS_patched.dtype)

        return DATA_patched, ANNOTATIONS_patched




    def unpatch_predictions(self, PREDICTIONS_patched, TARGET_shape, PATCH_SIZE):
            """
            Unpatch the predictions back to their original shape.

            PREDICTIONS_patched has dimension: 
                NUMBER_OF_IMAGES x NUMBER_OF_PATCHES_IN_LINES_DIRECTION x NUMBER_OF_PATCHES_IN_SAMPLES_DIRECTION x PATCH_SIZE x PATCH_SIZE
            TARGET_shape should be -> TARGET_shape=(NUMBER_OF_IMAGES, LINES, SAMPLES)
                Important! Make sure to pass the right number of LINES and SAMPLES in the likely case of having used padding/cropping
            
            """
            print('Starting unpatching.....')
            print('Predictions before unpatching: ', PREDICTIONS_patched.shape, ', and dtype: ', PREDICTIONS_patched.dtype)

            PREDICTIONS_unpatched=np.zeros(TARGET_shape, dtype=PREDICTIONS_patched.dtype)

            for iterator_images in range(PREDICTIONS_patched.shape[0]):        # Iterate images
                for i_LINES_DIR in range(PREDICTIONS_patched.shape[1]):        # Iterate for PATCHES_IN_LINES_DIRECTION
                    for j_SAMPLES_DIR in range(PREDICTIONS_patched.shape[2]):  # Iterate for PATCHES_IN_SAMPLES_DIRECTION
                        PREDICTIONS_unpatched[iterator_images,\
                                            i_LINES_DIR*PATCH_SIZE:(i_LINES_DIR+1)*PATCH_SIZE,\
                                            j_SAMPLES_DIR*PATCH_SIZE:(j_SAMPLES_DIR+1)*PATCH_SIZE]=\
                                                    PREDICTIONS_patched[iterator_images, i_LINES_DIR, j_SAMPLES_DIR, :, :]

            print('Predictions after unpatching:: ', PREDICTIONS_unpatched.shape, ', and dtype: ', PREDICTIONS_unpatched.dtype)

            return PREDICTIONS_unpatched


    