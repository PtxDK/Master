# Notes for Meeting 7-9-20
- want to write this as an article
- with potential to do a ph.d
- focus on the application rather than the method it self
- will look at some pet images

## Heart Project (Segmentation)
- scan using the sugars found in the cell with pet images since cancer cells need large amounts of sugars
- we want to determine of the viability of the heart by looking ath the hearts blood flow to the muscles
  - robinium 82
    - 70 second half life
- look at infarctions
- look at segmenting the heart
  - pet images 
  - CT images
- the left ventricle is the most interesting
- have 2 states
  - calm
  - stressed, medically induced
- can use amonia
  - 20 min half life
- can use radioactive water as contrast
  - 2 min halflife
- have 2000 cases (pet and CT)
  - x2 (calm and stressed)
- dynamic undersogelse
- 2x gated
  - get images of the gates
  - split into 8 or 16 pieces
  - make image from the pieces for each heart beat
- time for CT images
  - few seconds
- can be rigid and not rigid
- would like a mask for each ventricle and one for the left ventricle
- can use about 100 images for segmentation
- will annotate self
- can get help with it

- 3 methods to segement
  - pixel by pixel
  - registration based method
    - deform the elements and work from there
  - shape model 

- Priority
  - Left ventricle
  - each part of heart
  - viability (yes or no)
- texture analysis
- There are methods for explainable AI
- 
- where can we get books on cardiology **?**
- which framework
  - torch
  - tensorflow
- places for segmentation errors
  - CT scans

write email about what was talked about


kaggle challenge
mikai challenge
    segmentation decalthalon

### Email subjects
dataset findings (unsuccessful)
text type (article)
reading material about the heart
next meeting time monday 8.40