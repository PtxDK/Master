
vi kan nu representere 128,128 ting i modellerne

vi har lavet mini version af 2d unet

Start med at køre uden data augmentation; evt. tilføj til sidst.

Vigtig: Hvor skal vi ligge dybden i projektet? eg. data augmentation, hyper parametre, 2d vs 3d unet, forskellige modeller, venstre ventrikkel eller flere kamre.

- Listen over problem sources vil blive forholdsvist lang, og selvindtegnet data er bare end nu et punkt.

##### VIGTIG: Skriv en mini rapport færdig, med introduction, pointer, metoder, conclusions hvor vi starter med at have en basis version af hele projektet. Sådan at vi bagefter kan få lov til at lege med alt muligt forskelligt.

- Vi skal overveje måske kun gøre data augmentation i 2d frem for 3d..... pas på at lave for mange forskelle mellem 2d og 3d unet, da det introducerer flere variable der gør dem mere og mere usammenlignelige.

- Der skal laves overvejelser over hyper parametre; man har 10 dimensionelle hyper parameter rum,vi skal have en ad-hoc tilgang til potentielt at finde de bedste parametre, men vi kan ikke køre i det 10 dimensionelle rum; vi skal overveje, hvad gør man så??? alle ender med at gøre noget random og prøve sig frem, overvej at tage nogle bedre start parametre et andet sted fra; incrementer/decrementer faktor 10 på dine parametre.

Der kan laves grid search, hvor alle kombinationer kan prøves af.
- Evt. prøv at lave en sekventiel fremgangsmåde, hvor hvert parameter prøves en enkelt gang, og der tilføjes en mere i hver iteration. Lav meget eager early stopping, der kun skal give foreløbige resultater som hyper parametre vil basere sig på.

Når hyper parametre testes, så husk at forøg given variabel med faktor 2/3 stykker

