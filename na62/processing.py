import traceback
from pathlib import Path
from typing import Callable, Dict, Generator, List, Tuple, Union

import pandas as pd
from tqdm.autonotebook import tqdm

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
        def merge_or_list(el, other):
            if isinstance(el, list):
                return [merge_or_list(t, o) for t, o in zip(el, other)]
            else:
                return el.merge(other)

        for k in self.numbers:
            self.numbers[k] += other.numbers[k]
        merged_histo = [merge_or_list(t, o) for t, o in zip(
            self.histograms, other.histograms)]
        self.histograms = merged_histo

        return self


def load_yield(input_files: List[Union[str, Path]], chunk_size: int, sample_name: str) -> Generator[Tuple[pd.DataFrame, int], None, None]:
    """
    Generator chunking the input data. Each chunk is yielded as a dataframe and a normalization number

    :param input_files: List of input files to chunk
    :param chunk_size: Maximum size of each chunk
    :return: None
    :yield: Tuple with the chunk dataframe, and the chunk normalization
    """
    start = 0
    total = len(input_files)
    with tqdm(total=total, desc=f"{sample_name} files processed", leave=True) as t:
        while len(input_files) > 0:
            # Import the next chunk
            data, normalization = import_root_files(
                input_files, total_limit=chunk_size, skip_entries=start)

            # If no data read, we have completed
            if data is None or len(data) == 0:
                break

            # Update the state variables: remove the files already read and the position in the current file
            processed_files = input_files.index(data.attrs["last_file"])
            input_files = input_files[processed_files:]
            start = data.attrs["last_entry"]
            yield data, normalization
            t.update(processed_files)
        t.update(total)


def run_analysis_on_sample(input_files: List[Union[str, Path]], functions: List[Callable], chunk_size: int, isMC: bool, sample_name: str) -> Dict[str, AnalysisObject]:
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
    for data, normalization in load_yield(input_files, chunk_size, sample_name):
        # Prepare the origin input
        input = {"origin": AnalysisObject("origin")}
        input["origin"].df = data
        input["origin"].numbers["normalization"] = normalization

        # Run all analysis functions and update the dictionary of AnalysisObject
        for f in functions:
            new_output = f(input, isMC)
            input[new_output.name] = new_output

        # Merge the AnalysisObject with the previous one (if already exists)
        # Also remove the df as it will not be necessary anymore
        for step in input:
            del input[step].df
            input[step].df = None
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
            data_files, functions, chunk_size, False, "Data")
        mc_result_dict = {}
        for mc_sample in mc_dict_files:
            # Run the analysis on each MC sample
            mc_result_dict[mc_sample] = run_analysis_on_sample(
                mc_dict_files[mc_sample], functions, chunk_size, True, mc_sample)

        return data_result, mc_result_dict
    except:
        traceback.print_exc()
        histo.disable_plotting = False
    return None, None


def plot_prepared_histo_scale(data_result: AnalysisObject, mc_results: Dict[str, AnalysisObject],
                              object_name: str, ihisto: int, **kwargs) -> Tuple[int]:
    ndata = histo._hist_data(
        data_result[object_name].histograms[ihisto], **kwargs)
    nmc = histo._stack_mc_scale([mc_results[mc][object_name].histograms[ihisto][0]
                                 for mc in mc_results], labels=[mc for mc in mc_results], ndata=ndata, **kwargs)
    return ndata, nmc


def plot_prepared_histo_flux(data_result: AnalysisObject, mc_results: Dict[str, AnalysisObject],
                             normalization_dict: Dict[str, float], object_name: str,
                             kaon_flux: float, ihisto: int, **kwargs) -> Tuple[int]:
    ndata = histo._hist_data(
        data_result[object_name].histograms[ihisto], **kwargs)
    nmc = histo._stack_mc_flux([mc_results[mc][object_name].histograms[ihisto] for mc in mc_results],
                               normalization_dict, labels=[mc for mc in mc_results], kaon_flux=kaon_flux, **kwargs)
    return ndata, nmc
