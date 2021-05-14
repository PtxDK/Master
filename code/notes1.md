# Notes
- how does the memory use of the model work
- how are the parameters affected by this
- how does this affect the data needed to train the model
  - unet might need fewer
- Why is it that we are looking at this
- create an academic discussion about this
- compare the runtimes for the different models

the errors that we have now what is it that is creating those problems
what if someone is sick
is based on how strongly we difined earlier
start with metrics
move more in depth

technical validation 
- is it realisitc to use this in the clinic
- what would be needed for this to be used in a clinical setting
- can move the goal line in the discussion
- adjust what we want to do in the introduction



Questions
Elastic deformation
  have no info
- need reply from perslev
how should we frame giving the information
- write for just be fore the thesis
- explain some of the basics and build on top of it

# hyper parameter
- could not do full search due to time
  - The main problem in our ability to complete the hyper parameter tuning is time. Due to this we must be abit smarter in what we choose to search on for the hypper parameters. the reason for the time contrainat is due to the exponential nature of searching. we work around this problem by spliting the options and searching along subsections of them.
  - break it into small sections
    - search for some then search for the rest
    - use a form of cheat
  - show the math for all the options that were used for searching
    - instead of 3^7 do -> 3^4 + 3^3
    - the first set is
      - dim, alpha, apply prob,
      - we focus on these due to these having the most intuitive effect.
    - second set is
      - simga lower, sigma higher, depth, complexity factor
  - look at what params have memory problems
    - depth
      - found 5 depth to be too large also created problems with parameter numbers
        - too large memory to run validation set
    - complexity factor
      - look at 1,2,3
        - found that there were no problems here
      - 