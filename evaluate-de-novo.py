#!/usr/bin/env python

import logging
import rich_click as click
import sys
import polars as pl
import numpy as np
import re
from pyteomics import mgf
logger = logging.getLogger("denovo-eval")

_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_START_VOCAB = [_PAD, _GO, _EOS]
mass_H = 1.0078
mass_H2O = 18.0106
mass_NH3 = 17.0265
mass_N_terminus = 1.0078
mass_C_terminus = 17.0027
mass_CO = 27.9949
mass_Phosphorylation = 79.96633
vocab_reverse = [
    "A",
    "R",
    "N",
    "n",
    "D",
    "C",
    "E",
    "Q",
    "q",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "m",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]
mass_AA = {
    "_PAD": 0.0,
    "_GO": mass_N_terminus - mass_H,
    "_EOS": mass_C_terminus + mass_H,
    "A": 71.03711,  # 0
    "R": 156.10111,  # 1
    "N": 114.04293,  # 2
    "n": 115.02695,
    "D": 115.02694,  # 3
    "C": 160.03065,  # 103.00919,  # 4
    "E": 129.04259,  # 5
    "Q": 128.05858,  # 6
    "q": 129.0426,
    "G": 57.02146,  # 7
    "H": 137.05891,  # 8
    "I": 113.08406,  # 9
    "L": 113.08406,  # 10
    "K": 128.09496,  # 11
    "M": 131.04049,  # 12
    "m": 147.0354,
    "F": 147.06841,  # 13
    "P": 97.05276,  # 14
    "S": 87.03203,  # 15
    "T": 101.04768,  # 16
    "W": 186.07931,  # 17
    "Y": 163.06333,  # 18
    "V": 99.06841,  # 19
}
vocab_reverse = _START_VOCAB + vocab_reverse
vocab = dict([(x, y) for (y, x) in enumerate(vocab_reverse)])
vocab_size = len(vocab_reverse)
mass_ID = [mass_AA[vocab_reverse[x]] for x in range(vocab_size)]

@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
def main():
    """
    # denovo-eval

    """

@main.command()
@click.argument(
    "denovo",
    required=True,
    nargs=-1,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--database_search",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to database search file",
)
@click.option(
    "--instanovo_ipc",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to instanovo ipc file. Necessary to get correct ID",
)
@click.option(
    "--pointnovo_mgf",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to pointnovo mgf. Necessary to get correct ID",
)
@click.option(
    "--mgf_in",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to defeault mgf. Necessary to get correct ID",
)
@click.option(
    "--out",
    required=True,
    type=click.Path(exists=False, dir_okay=False),
    help="Path where summary file will be exported to. Needs to be tsv or csv.",
)
def summary(
    denovo,
    database_search,
    instanovo_ipc,
    pointnovo_mgf,
    mgf_in,
    out
):
    """
    Summarize different de novo files into a csv table

    denovo are different files from the de novo tools.

    """
    setup_logging("info")

    # read the mgf file and extract the scan IDs
    reader_pointnovo_mgf = mgf.read(pointnovo_mgf)
    reader_mgf = mgf.read(mgf_in)
    scan_pointnovo = [int(i['params']['scans']) for i in reader_pointnovo_mgf]
    scan_mgf = [int(i['params']['scans']) for i in reader_mgf]

    for denovo_file in denovo:
        if denovo_file.endswith(".mztab"):
            logger.info(f"Casnovo file detected: {denovo_file}")
            casanovo_df = pl.read_csv(denovo_file, separator="\t", truncate_ragged_lines=True, comment_prefix="MTD")

            # This is necessary because Casanovo re-indexes the mgf file
            # index and scan will start at 0 which is not common to the original mgf file

            # Replace PSM_ID by spectra_refs scan_id to get correct matching
            scan_id = casanovo_df['spectra_ref'].to_list()
            scan_id = [int(i.split("index=")[-1])+1 for i in scan_id]
            casanovo_df = casanovo_df.with_columns(pl.Series(name="PSM_ID", values=scan_id))
            
            # get the number of total scans in the dataset
            last_scan = int(scan_id[-1])
            new_scan_list = range(1, last_scan + 1)
            scan_df = pl.DataFrame({"scan_ID_mgf": scan_mgf, "scan_ID_casanovo": new_scan_list})


            casanovo_df = casanovo_df.join(scan_df, left_on="PSM_ID", right_on="scan_ID_casanovo", how="outer_coalesce")
            casanovo_df = casanovo_df.with_columns([pl.col("scan_ID_mgf").cast(pl.Int64, strict=False).alias("PSM_ID")])

            # Modify peptide sequence, uniform modifications, remove all other mods besides m q n
            casanovo_peptide = casanovo_df['sequence'].to_list()
            casanovo_peptide = [str(i).replace('M+15.995', 'm').replace('Q+0.984', 'q').replace('N+0.984', 'n').replace(' ',
                                        '').replace('C+57.021', 'C').replace('+43.006', '').replace('-17.027', '').replace('+42.011','') for i in casanovo_peptide]
            casanovo_df = casanovo_df.with_columns(pl.Series(name="Casanovo_Peptide", values=casanovo_peptide)) 
            casanovo_df = casanovo_df.drop_nulls()

            new_score = [i * 100 for i in casanovo_df['search_engine_score[1]'].to_list()]
            casanovo_df = casanovo_df.with_columns(pl.Series(name="Casanovo_Score", values=new_score)) 

            casanovo_df = casanovo_df.select(["PSM_ID", "Casanovo_Peptide", "Casanovo_Score"])
        elif denovo_file.endswith(".deepnovo_denovo"):
            logger.info(f"Pointnovo file detected: {denovo_file}")
            pointnovo_df = pl.read_csv(denovo_file, separator="\t", truncate_ragged_lines=True)
            pointnovo_df = pointnovo_df.drop_nulls()
            pointnovo_peptide = pointnovo_df['predicted_sequence'].to_list()
            for i in range(len(pointnovo_peptide)):
                pointnovo_peptide[i] = str(pointnovo_peptide[i])
                pointnovo_peptide[i] = pointnovo_peptide[i].replace(",", "").replace("N(Deamidation)",
                "n").replace("Q(Deamidation)", "q").replace("C(Carbamidomethylation)", "C").replace("M(Oxidation)", "m")
            pointnovo_df = pointnovo_df.with_columns(pl.Series(name="Pointnovo_Peptide", values=pointnovo_peptide)) 

            pointnovo_score = pointnovo_df['predicted_score'].to_list()
            new_score =  [np.exp(i) * 100 for i in pointnovo_score]
            pointnovo_df = pointnovo_df.with_columns(pl.Series(name="Pointnovo_Score", values=new_score)) 
            
            pointnovo_df = pointnovo_df.select(["feature_id", "Pointnovo_Peptide", "Pointnovo_Score"])
            pointnovo_df = pointnovo_df.rename({"feature_id":"PSM_ID"})

            # This is necessary because I used a reformatted file for de novo predicitng for de novo
            # To infer the original we need to compare both mgf files since the pointnovo_mgf has been reindexed
            # This part can be removed in case we use the same mgf file for Pointnovo and the other tools

            scan_df = pl.DataFrame({"scan_ID_pointnovo": scan_pointnovo, "scan_ID_mgf": scan_mgf})
            # Here again, increase for correct match between scan and PSM_ID from de novo tool
            
            pointnovo_df = pointnovo_df.join(scan_df, left_on="PSM_ID", right_on="scan_ID_pointnovo", how="outer_coalesce")
            pointnovo_df = pointnovo_df.with_columns([pl.col("scan_ID_mgf").cast(pl.Int64, strict=False).alias("PSM_ID")])
            # TODO Filter

        elif denovo_file.endswith(".csv"):
            logger.info(f"Instanovo file detected: {denovo_file}")
            instanovo_df = pl.read_csv(denovo_file, separator=",", truncate_ragged_lines=True).with_row_index(name="PSM_ID")
            instanovo_df = instanovo_df.drop_nulls()
            instanovo_df = instanovo_df.with_columns(pl.col('PSM_ID').cast(pl.Int64, strict=False).alias('PSM_ID'))
            
            instanovo_df = instanovo_df.rename({"preds":"Instanovo_Peptide", "log_probs":"Instanovo_Score"})
            instanovo_peptide = instanovo_df['Instanovo_Peptide'].to_list()
            for i in range(len(instanovo_peptide)):
                instanovo_peptide[i] = str(instanovo_peptide[i])
                instanovo_peptide[i] = instanovo_peptide[i].replace(",", "").replace("M(ox)", "m").replace("None","")
            instanovo_df = instanovo_df.with_columns(pl.Series(name="Instanovo_Peptide", values=instanovo_peptide)) 

            if instanovo_ipc == None:
                logger.warning("Instanovo IPC was not added. Correct ID cannot be detemined. This file will be skipped")
                continue
            
            # Again as before increase to get correct match
            scan_id = instanovo_df['PSM_ID'].to_list()
            scan_id = [i + 1 for i in scan_id]
            instanovo_df = instanovo_df.with_columns(pl.Series(name="PSM_ID", values=scan_id))


            # Here we use the IPC file to retrieve the original scan ID / index!
            in_ipc = pl.read_ipc(instanovo_ipc)
            in_ipc = in_ipc.select(["scan_number", "scans"])
            instanovo_df = instanovo_df.join(in_ipc, left_on="PSM_ID", right_on="scan_number", how="inner")
            instanovo_df = instanovo_df.with_columns([pl.col("scans").alias("PSM_ID")])
            instanovo_df = instanovo_df.with_columns(pl.col('PSM_ID').cast(pl.Int64, strict=True))
            instanovo_df = instanovo_df.select(["PSM_ID", "Instanovo_Peptide", "Instanovo_Score"])
        else:
            logger.info(f"File {denovo_file} could not be identified.")

    denovo_df = casanovo_df.join(pointnovo_df, left_on="PSM_ID", right_on="PSM_ID", how="outer_coalesce")
    denovo_df = denovo_df.join(instanovo_df, left_on="PSM_ID", right_on="PSM_ID", how="outer_coalesce")
    database_df = pl.read_csv(database_search, separator="\t", truncate_ragged_lines=True)

    # Take the last part for PSM_ID, f.e. xyz564; scan=123, take 123
    psm_id = database_df["PSMId"].to_list()
    psm_id = [int(i.split('scan=', 1)[-1]) for i in psm_id]
    database_df = database_df.with_columns(pl.Series(name="PSMId", values=psm_id)) 

    

    db_peptide = database_df['peptide'].to_list()
    # Replace Modfications and numbers
    db_peptide = [str(i).replace('[57.02147]', '').replace('M[15.99492]', 'm').replace('.', '')
                  .replace('-', '').replace('[','').replace(']','').replace('X', '')[1:-1]
                   for i in db_peptide]
    # Remove all other numbers in the peptide sequence for comparison
    db_peptide = [re.sub(r'[0-9]', '', i) for i in db_peptide] 
    # TODO Instead of just removing the numbers in the peptide seq, we should remove all rows containing this kind of modifications
    # TODO Also remove rows containing - symbols
    database_df = database_df.with_columns(pl.Series(name="peptide", values=db_peptide)) 

    # Merge Database and De novo df
    merged_df = database_df.join(denovo_df, left_on="PSMId", right_on="PSM_ID", how="outer_coalesce")
    merged_df.write_csv(out, separator="\t")
    logger.info(f"Merged dataframe with de novo sequencing results and database reference was exported to {out}")

    evaluation(merged_df, out)
    logger.info("DONE!")


def _match_AA_novor(target, predicted):
    """
    This was taken from DeepNovo. How AA Precision and Recall is calculated in their publications. 
    """
    num_match = 0
    target_len = len(target)
    predicted_len = len(predicted)
    target_mass = [mass_ID[x] for x in target]
    target_mass_cum = np.cumsum(target_mass)
    predicted_mass = [mass_ID[x] for x in predicted]
    predicted_mass_cum = np.cumsum(predicted_mass)

    i = 0
    j = 0
    while i < target_len and j < predicted_len:
        if abs(target_mass_cum[i] - predicted_mass_cum[j]) < 0.5:
            if abs(target_mass[i] - predicted_mass[j]) < 0.1:
                num_match += 1
            i += 1
            j += 1
        elif target_mass_cum[i] < predicted_mass_cum[j]:
            i += 1
        else:
            j += 1

    return num_match
    
def precision_recall_with_threshold(peptides_truth, peptides_predicted, peptides_predicted_confidence, threshold):
    """
    Calculate precision and recall for the given confidence score threshold
    Parameters
    ----------
    peptides_truth : list 
        List of confidence scores for correct amino acids predictions
    peptides_predicted : list
        List of confidence scores for all amino acids prediction
    num_original_aa : list
        Number of amino acids in the predicted peptide sequences
    threshold : float
        confidence score threshold
           
    Returns
    -------
    aa_precision: float
        Number of correct aa predictions divided by all predicted aa
    aa_recall: float
        Number of correct aa predictions divided by all ground truth aa   
    peptide_recall: float
        Number of correct peptide preiditions divided by number of ground truth peptides  
    """  
    length_of_predictedAA = 0
    length_of_realAA = 0
    number_peptides = 0
    sum_peptidematch = 0
    sum_AAmatches = 0
    for i, (predicted_peptide, true_peptide) in enumerate(zip(peptides_predicted, peptides_truth)):
        length_of_realAA += len(true_peptide)
        number_peptides += 1
        if (type(predicted_peptide) is str and type(true_peptide) is str and peptides_predicted_confidence[i] >= threshold):
            length_of_predictedAA += len(predicted_peptide)
            predicted_AA_id = [vocab[x] for x in predicted_peptide]
            target_AA_id = [vocab[x] for x in true_peptide]
            recall_AA = _match_AA_novor(target_AA_id, predicted_AA_id)
            sum_AAmatches += recall_AA
            if recall_AA == len(true_peptide):
                sum_peptidematch += 1
        else:
            sum_AAmatches += 0
    return length_of_predictedAA, length_of_realAA, number_peptides, sum_peptidematch, sum_AAmatches

def evaluation(df, out):
    df = df.filter(~pl.all_horizontal(pl.col('score').is_null()))
    recall = []
    AA_recall = []
    AA_prec = []
    tool_list = ["Casanovo", "Pointnovo", "Instanovo"]
    for tool in tool_list:
        tool_AArecall = []
        tool_accuracy = []
        tool_AAprecision = []
        tool_scorecutoff = []
        true_list = df['peptide'].to_list()
        to_test = df[tool + '_Peptide'].to_list()
        to_test_score = df[tool + '_Score'].to_list()
        length_of_predictedAA, length_of_realAA, number_peptides, sum_peptidematch, sum_AAmatches = precision_recall_with_threshold(true_list, to_test, to_test_score, -1000)
        if length_of_predictedAA != 0:
                tool_accuracy.append(str(sum_peptidematch * 100 / number_peptides))
                tool_AAprecision.append(str(sum_AAmatches * 100 / length_of_predictedAA))
                tool_AArecall.append(str(sum_AAmatches * 100 / length_of_realAA))
                #tool_scorecutoff.append(score_cutoff)
        logger.info(f"Performance for {tool}")
        logger.info(f"Peptide recall (%): {tool_accuracy[0]}")
        logger.info(f"AA recall (%): {tool_AArecall[0]}")
        logger.info(f"AA precision (%): {tool_AAprecision[0]}")
        recall.append(tool_accuracy[0])
        AA_recall.append(tool_AArecall[0])
        AA_prec.append(tool_AAprecision[0])
    
    eval_df = pl.DataFrame({"tools": tool_list, "peptide_recall": recall, "AA_recall": AA_recall, "AA_precision": AA_prec})
    out = out.replace(".tsv", "-stats.tsv")
    eval_df.write_csv(out, separator="\t")

def setup_logging(verbosity):
    logging_levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }

    # Configure logging.
    logging.captureWarnings(True)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    warnings_logger = logging.getLogger("py.warnings")

    # Formatters for file vs console:
    console_formatter = logging.Formatter(
        "{name} {levelname} {asctime}: {message}", style="{", datefmt="%H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging_levels[verbosity.lower()])
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    warnings_logger.addHandler(console_handler)

    logging.getLogger("fsspec").setLevel(logging.WARNING)
    logging.getLogger("github").setLevel(logging.WARNING)
    logging.getLogger("h5py").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)



if __name__ == "__main__":
    main()