- With increasing requirements for optimization and automation, we present a model for segmenting the left ventricle of the left ventricle of the heart with an accuracy of xx\% that uses multi-planar u-net.
- Talk about the type of scans
talk about the type of patients
talk about why multi-planar u-net would be suitable for solving our problem (We have used the more generalized multi planar u-net)
explain the environment that currently exists
”What”. Vis en skanning i Intro. Er det svært at segmentere hjertet? Hvor er den venstre ventrikel?
”Why”. Der skal mere fundament bag. Hvorfor vil man segmentere hjertet? Hvilke sygdomme er det relevant for? Hvorfor har man brug for en segmentering – hvad giver den som man ikke kan se direkte på skanningen? Er det ”kun” et spørgsmål om at gøre noget som nu gøres manuelt? Eller ville det give ekstra, vigtig viden i det kliniske workflow at have segmenteringen?
Litterature review (søg google og  find andre relevante metoder for at lave en... kontekst som vores spaciale kan eksistere i)
Training. Det er super vigtigt med en overbevisende metode til hyper-parameter tuning. Samt en analyse af om træningen er stabil (får man cirka samme resultat hvis man gentræner?).… Indsæt: Parametre er i stor grad taget fra mpunet.
Vi skal validere MPUnet's hyper parametre med en kørse eller to
Vi skal validere vores egne hyper parametre med en kørsel eller to
Ændre på results table at der er med trænings/validerings sættet
Lig vægt på at træning og eksperimentering skete med udelukkende trænings sættet, og de endelige resultater blev kørt helt og alderes fuldstændig seperat, til aller sidst.
Noter der skal sættes ind:
MPUnet har demsonstreret performance i en lang række opgaver, derfor er den et ekstremt godt udgangspunkt for dette research.
Vi valgte at prøve data augmentation, det vi har prøvet virkede ikke godt, det gjorde resultater dårligere. det betyder ikke at data augmentation er dårligt, det kan stadig være relevant, men vi har ikke fundet noget der forbedrede performance.
Things to explain
what are the use cases
    use in other models to segment the left ventricle without the need for humans
what is the environment that the system must exist in
    is there a max response time for the hospital for diagnosis
selection for testing
    what is our baseline model: mpunet
    what are we testing: run-time vs accuracy
    2d unet for run time
    3d unet for accuracy
what is the goal of the research
    check if feasible and we can do it
    feasible: have 50% of hearts to be 80%
    solved: 85% of hearts have 95% correct annotations
look in the literature for what others have been able to achieve
where have we contributed and what have we coded
consider what can we add on top of the other methods