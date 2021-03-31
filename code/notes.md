# Current models
- mpu
  - base
    - this is our jumping off point for the project
    - is meant to be light weight for a 3d area
  - without augmentations
    - see the results for the effect of data augmentations
- 3d unet
  - pad
    - add a single layer to the information
  - shrink
  - augmentation
    - elastic3d
- 2d unet
  - base
  - match 3d
  - augmentations
    - elastic2d

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
- Transformer
  - Basis/benefits
    - have shown to work well in NLP and later vision tasks
  - Drawbacks
  - Other