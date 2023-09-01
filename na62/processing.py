import traceback
from pathlib import Path
from typing import Callable, Dict, Generator, List, Tuple, Union

import pandas as pd

from na62 import histo
from na62.prepare import import_root_files


class AnalysisObject:
    """
    Object holding the output of an analysis function
    """

    def __init__(self, name):
        self.name = name
        self.df = None
        self.histograms = []
        self.numbers = {}

    def merge(self, other):
        for k in self.numbers:
            self.numbers[k] += other.numbers[k]
        merged_histo = [t.merge(o) for t, o in zip(
            self.histograms, other.histograms)]
        self.histograms = merged_histo

        return self


def load_yield(input_files: List[Union[str, Path]], chunk_size: int) -> Generator[Tuple[pd.DataFrame, int], None, None]:
    """
    Generator chunking the input data. Each chunk is yielded as a dataframe and a normalization number

    :param input_files: List of input files to chunk
    :param chunk_size: Maximum size of each chunk
    :return: None
    :yield: Tuple with the chunk dataframe, and the chunk normalization
    """
    start = 0
    while len(input_files) > 0:
        # Import the next chunk
        data, normalization = import_root_files(
            input_files, total_limit=chunk_size, skip_entries=start)

        # If no data read, we have completed
        if data is None or len(data) == 0:
            break

        # Update the state variables: remove the files already read and the position in the current file
        input_files = input_files[input_files.index(data.attrs["last_file"]):]
        start = data.attrs["last_entry"]
        yield data, normalization


def run_analysis_on_sample(input_files: List[Union[str, Path]], functions: List[Callable], chunk_size: int, isMC: bool) -> Dict[str, AnalysisObject]:
    """
    Run an analysis on the list of input files. The analysis is run on chunks of data
    and the outputs of each chunks are merged in the output.

    :param input_files: List of input files
    :param functions: List of analysis functions to run
    :param chunk_size: Maximum number of events in each chunk
    :param isMC: True if the input files are MC
    :return: Dictionary of AnalysisObject (one for each analysis function + origin)
    """
    output_object = None

    # Loop over the dataframes provided by the generator
    for data, normalization in load_yield(input_files, chunk_size):
        # Prepare the origin input
        input = {"origin": AnalysisObject("origin")}
        input["origin"].df = data
        input["origin"].numbers["normalization"] = normalization

        # Run all analysis functions and update the dictionary of AnalysisObject
        for f in functions:
            new_output = f(input, isMC)
            input[new_output.name] = new_output

        # Merge the AnalysisObject with the previous one (if already exists)
        if output_object is None:
            output_object = input
        else:
            output_object = {n: output_object[n].merge(
                input[n]) for n in output_object}

    return output_object


def run_complete_analysis(data_files: List[Union[str, Path]], mc_dict_files: Dict[str, List[Union[str, Path]]],
                          functions: List[Callable], chunk_size: int) -> Tuple[Dict[str, AnalysisObject], Dict[str, Dict[str, AnalysisObject]]]:
    """
    Run an analysis on all provided samples (data + MC). The analysis is run on chunks of data
    and the outputs of each chunks are merged in the output.

    :param data_files: List of input data files
    :param mc_dict_files: Dictionary of lists of input MC files. The keys are the MC sample names.
    :param functions: List of analysis functions to run (in order)
    :param chunk_size: Maximum number of events in each chunk
    :return: Dictionaries of AnalysisObject merged from all chunks.
        The first element is the dictionary for data, the second element is a dictionary for each MC samples
    """

    histo.disable_plotting = True

    try:
        # Run the analysis on data
        data_result = run_analysis_on_sample(
            data_files, functions, chunk_size, False)
        mc_result_dict = {}
        for mc_sample in mc_dict_files:
            # Run the analysis on each MC sample
            mc_result_dict[mc_sample] = run_analysis_on_sample(
                mc_dict_files[mc_sample], functions, chunk_size, True)

        return data_result, mc_result_dict
    except:
        traceback.print_exc()
        histo.disable_plotting = False
    return None, None


def plot_prepared_histo_scale(data_result, mc_results, object_name, ihisto, **kwargs):
    ndata = histo._hist_data(data_result[object_name].histograms[ihisto], **kwargs)
    nmc = histo._stack_mc_scale([mc_results[mc][object_name].histograms[ihisto][0]
                          for mc in mc_results], labels=[mc for mc in mc_results], ndata=ndata, **kwargs)
    return ndata, nmc


def plot_prepared_histo_flux(data_result, mc_results, normalization_dict, object_name, kaon_flux, ihisto, **kwargs):
    ndata = histo._hist_data(data_result[object_name].histograms[ihisto], **kwargs)
    nmc = histo._stack_mc_flux([mc_results[mc][object_name].histograms[ihisto] for mc in mc_results],
                         normalization_dict, labels=[mc for mc in mc_results], kaon_flux=kaon_flux, **kwargs)
    return ndata, nmc
