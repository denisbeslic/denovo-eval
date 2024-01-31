# denovo-eval
Script for comparing multiple de novo tools

## Example command

python evaluate-de-novo.py summary --database_search ../XTandem-MS2Rescore/F06.pout ../Casanovo/F06.mztab ../PointNovo/F06_features.csv.deepnovo_denovo ../InstaNovo/Base-model\(instanovo.pt\)/F06.csv --instanovo_ipc ../../analysis-2023/001-Datasets/F06/F06.ipc --out ../F06-summary.tsv

## Example output
Currently it outputs all results without using any thresholding approach.
Tool_name
Peptide_Recall, AminoAcid_Recall, AminoAcid_Precision