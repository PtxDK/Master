

# Techniques
- general
  - look at a unet as an encoder decoder
  - only 2 classes
    - bg and heart
  - represent each pixel as classification problem with spare CCE
- 2d unet
  - output per pixel class for image
  - convolute data down to encode the data
  - deconvolute data along with concatenating the results from the other convolutions
  - here we only look at the scans from a single direction
- mpu
  - uses a 2d unet along with a fusion model
  - unet to generate segmentations in different directions
  - can be any number of vectors to look along
  - chosen 6
  - vectors are randomly generated
  - trained in 2 steps
    - first unet
    - then fusion model
  - fusion model gets all representations with the vectors that they are given in
  - finds weights to combine the different slices to be a single pixel 
- 3d unet
  - pass all slices through to show the scan in a 3d setting

# Data augmentation
- Elastic
  - 2d and 3d
  - stretch and squish areas
  - used to "shake the box"
  - apply randomly to the data
  - dont use the same data for each epoch
  - every time it is applied can be seen as shaking the box
# Current models
- mpu
  - base
    - this is our jumping off point for the project
    - is meant to be light weight for a 3d area
    - look at as a 2.5d unet
  - without augmentations
    - see the results for the effect of data augmentations
- 3d unet
  - all have batch size 1 due to memory
  - change the dim argument in unet to change the size of the input data
  - pad
    - 112
    - add a single all zero slice to the information
  - shrink
    - 96
    - reduce the info to not affect data by adding information
    - found to perform better
  - augmentation
    - elastic3d
    - found to improve the model by a few percent
    - need to not repeat the tf dataset to reduce overfitting
- 2d unet
  - can achieve the same effect as 3d unet by running all slices through the model and concatting together after
  - base
    - test can this be represented as looking at a scan as a number of slices
    - same techinique done in transunet
  - match 3d
    - increae the batch size to match that of 3d unet 111.
    - took too much memory to have the correct batch size of 111
  - augmentations
    - elastic2d
    - need to test for the results
# Training

# Results
- compare the dsc scores
- look at recall and precision
- look at some examples from the various epochs

# Evaluation
- looking at the results it appears to be that the best model is the 3d unet
- the worst was the 2d unet
  - this shows that there is more to the problem than just the color of the pixel
- looking at some further improvments that are done on sota models use a 2d unet
- this might be that there is an encoder problem
- newer models tend to focus on improving the encoder rather than the decoder
- for example transunet paper 
- mpu performed slightly worse than 3d unet
- data aumentation gives prelimincary better results so far
  - further testing could improve this further
- can improve the number of parameters
- what if we use a 3d unet with mpunet combiner
- can further test if there are some simpler methods for 



# Data augmentations
- what is the best method to apply data augmentation?
  - apply randomly each time an example is taken
    - reapeat the dataset multiple times
      - can cause the problem of overfitting
  - build dataset without augmentations once
    - some core examples might not be generated
  - have 

# Further Work
- Data augmentation
  - the effect of different methods
- SPN
  - basis/benefits
    - remove the need for fixed space
    - would help with the 3d unet not needing uniform space
    - can use all the data in each example
  - Other
    - need to implement 3d version of SPN
- Transformer(potential: unlikely/stretch)
  - Basis/benefits
    - have shown to work well in NLP and later vision tasks
  - Drawbacks
  - Other