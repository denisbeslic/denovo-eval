#!/usr/bin/env python

import logging
import pathlib
import os
import rich_click as click
import sys
import polars as pl
import numpy as np

logger = logging.getLogger("denovo-eval")

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
def summary(
    denovo,
    database_search,
    instanovo_ipc,
):
    """
    Summarize different de novo files into a csv table

    denovo are different files from the de novo tools.

    """
    setup_logging("info")

    for denovo_file in denovo:
        print(denovo_file)
        if denovo_file.endswith(".mztab"):
            logger.info(f"Casnovo file detected: {denovo_file}")
            casanovo_df = pl.read_csv(denovo_file, separator="\t", truncate_ragged_lines=True, comment_prefix="MTD")
            casanovo_df = casanovo_df.drop_nulls()

            casanovo_peptide = casanovo_df['sequence'].to_list()
            casanovo_peptide = [str(i).replace('M(+15.99)', 'm').replace('Q(+.98)', 'q').replace('N(+.98)', 'n').replace(' ',
                                        '').replace('C(+57.02)', 'C').replace('+43.006', '') for i in casanovo_peptide]
            casanovo_df = casanovo_df.with_columns(pl.Series(name="Casanovo_Peptide", values=casanovo_peptide)) 

            new_score = [i * 100 for i in casanovo_df['search_engine_score[1]'].to_list()]
            casanovo_df = casanovo_df.with_columns(pl.Series(name="Casanovo_Score", values=new_score)) 

            casanovo_df = casanovo_df.select(["PSM_ID", "Casanovo_Peptide", "Casanovo_Score"])
        elif denovo_file.endswith(".deepnovo_denovo"):
            logger.info(f"PointNovo file detected: {denovo_file}")
            pointnovo_df = pl.read_csv(denovo_file, separator="\t", truncate_ragged_lines=True)
            pointnovo_df = pointnovo_df.drop_nulls()
            pointnovo_peptide = pointnovo_df['predicted_sequence'].to_list()
            for i in range(len(pointnovo_peptide)):
                pointnovo_peptide[i] = str(pointnovo_peptide[i])
                pointnovo_peptide[i] = pointnovo_peptide[i].replace(",", "").replace("I", "L").replace("N(Deamidation)",
                "n").replace("Q(Deamidation)", "q").replace("C(Carbamidomethylation)", "C").replace("M(Oxidation)", "m")
            pointnovo_df = pointnovo_df.with_columns(pl.Series(name="PointNovo_Peptide", values=pointnovo_peptide)) 
            pointnovo_score = pointnovo_df['predicted_score'].to_list()
            new_score =  [np.exp(i) * 100 for i in pointnovo_score]
            pointnovo_df = pointnovo_df.with_columns(pl.Series(name="PointNovo_Score", values=new_score)) 
            pointnovo_df = pointnovo_df.select(["feature_id", "PointNovo_Peptide", "PointNovo_Score"])
            pointnovo_df = pointnovo_df.rename({"feature_id":"PSM_ID"})
        elif denovo_file.endswith(".csv"):
            logger.info(f"Instanovo file detected: {denovo_file}")
            instanovo_df = pl.read_csv(denovo_file, separator=",", truncate_ragged_lines=True).with_row_count(name="PSM_ID")
            
            instanovo_df = instanovo_df.with_columns(pl.col('PSM_ID').cast(pl.Int64, strict=False).alias('PSM_ID'))
            
            instanovo_df = instanovo_df.rename({"preds":"Instanovo_Peptide", "log_probs":"Instanovo_Score"})

            if instanovo_ipc == None:
                logger.warning("Instanovo IPC was not added. Correct ID cannot be detemined. This file will be skipped")
                continue
            
            in_ipc = pl.read_ipc(instanovo_ipc)
            #print(in_ipc)
            #print(instanovo_df)
            # TODO We need another parameter for the initial file 
        else:
            logger.info(f"File {denovo_file} could not be identified.")

    #with pl.Config(tbl_cols=instanovo_df.width):
    #    print(instanovo_df)


    denovo_df = casanovo_df.join(pointnovo_df, left_on="PSM_ID", right_on="PSM_ID", how="outer_coalesce")
    print(denovo_df)
    denovo_df = denovo_df.join(instanovo_df, left_on="PSM_ID", right_on="PSM_ID", how="outer_coalesce")
    print(denovo_df)
    exit()


    # print(denovo_df)
    # TODO Merge with Instanovo
    exit()



    database_df = pl.read_csv(database_search, separator="\t", truncate_ragged_lines=True)
    print(database_df)

    logger.info("DONE!")



@main.command()
@click.argument(
    "tsv",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
)
def evaluate(
    test
):
    """
    Evaluate the different tools by calculating summary stats

    """
    setup_logging("info")
    logger.info("DONE!")




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