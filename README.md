# denovo-eval
Script for comparing multiple de novo tools


## Example command

python evaluate-de-novo.py summary --database_search XTandem-MS2Rescore/F06.pout Casanovo/F06.mztab PointNovo/F06_features.csv.deepnovo_denovo InstaNovo/Base-model\(instanovo.pt\)/F06.csv --instanovo_ipc ../analysis-2023/001-Datasets/F06/F06.ipc

## Example output
Currently it outputs all results without using any thresholding approach.
Tool_name
Peptide_Recall, AminoAcid_Recall, AminoAcid_Precision

Casanovo
['0.0'] ['0.3556395098004921'] ['0.41068038294262593']
Pointnovo
['0.0'] ['0.3332158332683047'] ['0.427743574923058']
Instanovo
['0.0'] ['0.9587467138102042'] ['1.1165522154510186']