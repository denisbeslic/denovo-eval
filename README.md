# denovo-eval
Script for comparing multiple de novo tools

## Example command

python evaluate-de-novo.py summary --database_search ../XTandem-MS2Rescore/S07.pout ../Casanovo/S07.mztab ../PointNovo/S07_features.csv.deepnovo_denovo ../InstaNovo/Base-model_instanovo_pt/S07.csv --instanovo_ipc ../../analysis-2023/001-Datasets/S07/S07.ipc --out ../S07-summary.tsv --pointnovo_mgf ../../analysis-2023/001-Datasets/S07/S07_msconvert_reformatted_deepnovo.mgf --mgf_in ../../analysis-2023/001-Datasets/S07/S07.mgf

## Example output
Currently it outputs all results without using any thresholding approach.
Tool_name
Peptide_Recall, AminoAcid_Recall, AminoAcid_Precision