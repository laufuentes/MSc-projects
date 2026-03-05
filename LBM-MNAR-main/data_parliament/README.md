This dataset is the records from the lower house of the French Parliament (Assemblée Nationale).

It gathers the results of the 1256 ballots of year 2018 of the 577 French members of parliament (MPs) for the procedural motions and amendments for the 15th legislature (June 2017).

For each text, the vote of each MP is recorded asa 4-level categorical response: “yes”, ‘no”, “abstained” or “absent”.

The original votes  from  the  French  National  Assembly  are  available  from http://data.assemblee-nationale.fr/travaux-parlementaires/votes

We gather the data in a matrix where each row represents an MP and each column represents a text.
This matrix is available in the votes.txt files and is reduced to 3 levels:
    1 : for "yes"
    0 : for "missing or MPs not present"
    -1 : for "No"

Meta-data concerning MPs is available in deputes.json
The row indice of the vote matrix corresponds to the i_th MP in the deputes.json file.

Meta-data concerning texts is available in texts.json
The column indice of the vote matrix corresponds to the j_th text in the texts.json file.
