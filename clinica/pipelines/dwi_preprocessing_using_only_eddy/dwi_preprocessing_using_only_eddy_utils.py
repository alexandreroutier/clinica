# coding: utf8


def rename_into_caps(in_bids_dwi,
                     fname_dwi, fname_bval, fname_bvec, fname_brainmask):
    """
    Rename the outputs of the pipelines into CAPS format namely:
    <source_file>_space-b0_preproc{.nii.gz|bval|bvec}
    <source_file>_space-b0_brainmask.nii.gz
    <source_file>_space-b0_magnitude1.nii.gz
    <source_file>_space-b0[_fwhm-<fwhm>]_fmap.nii.gz

    Args:
        in_bids_dwi (str): Input BIDS DWI to extract the <source_file>
        fname_dwi (str): Preprocessed DWI file.
        fname_bval (str): Preprocessed bval.
        fname_bvec (str): Preprocessed bvec.
        fname_brainmask (str): B0 mask.

    Returns:
        The different outputs in CAPS format
    """
    from nipype.utils.filemanip import split_filename
    from nipype.interfaces.utility import Rename
    import os

    # Extract <source_file> in format sub-CLNC01_ses-M00_[acq-label]_dwi
    _, source_file_dwi, _ = split_filename(in_bids_dwi)

    # Extract base path from fname:
    base_dir_dwi, _, _ = split_filename(fname_dwi)
    base_dir_bval, _, _ = split_filename(fname_bval)
    base_dir_bvec, _, _ = split_filename(fname_bvec)
    base_dir_brainmask, _, _ = split_filename(fname_brainmask)

    # Rename into CAPS DWI:
    rename_dwi = Rename()
    rename_dwi.inputs.in_file = fname_dwi
    rename_dwi.inputs.format_string = os.path.join(
        base_dir_dwi, source_file_dwi + "_space-b0_preproc.nii.gz")
    out_caps_dwi = rename_dwi.run()

    # Rename into CAPS bval:
    rename_bval = Rename()
    rename_bval.inputs.in_file = fname_bval
    rename_bval.inputs.format_string = os.path.join(
        base_dir_bval, source_file_dwi + "_space-b0_preproc.bval")
    out_caps_bval = rename_bval.run()

    # Rename into CAPS bvec:
    rename_bvec = Rename()
    rename_bvec.inputs.in_file = fname_bvec
    rename_bvec.inputs.format_string = os.path.join(
        base_dir_bvec, source_file_dwi + "_space-b0_preproc.bvec")
    out_caps_bvec = rename_bvec.run()

    # Rename into CAPS brainmask:
    rename_brainmask = Rename()
    rename_brainmask.inputs.in_file = fname_brainmask
    rename_brainmask.inputs.format_string = os.path.join(
        base_dir_brainmask, source_file_dwi + "_space-b0_brainmask.nii.gz")
    out_caps_brainmask = rename_brainmask.run()

    return (out_caps_dwi.outputs.out_file,
            out_caps_bval.outputs.out_file,
            out_caps_bvec.outputs.out_file,
            out_caps_brainmask.outputs.out_file,)


def get_grad_fsl(bvec, bval):
    grad_fsl = (bvec, bval)
    return grad_fsl


def init_input_node(dwi, bvec, bval, dwi_json):
    """Initialize pipeline (read JSON, check files and print begin message)."""
    import datetime
    import nibabel as nib
    from colorama import Fore
    from clinica.utils.filemanip import get_subject_id, extract_metadata_from_json
    from clinica.utils.dwi import bids_dir_to_fsl_dir, check_dwi_volume
    from clinica.utils.stream import cprint
    from clinica.utils.ux import print_begin_image

    # Extract image ID
    image_id = get_subject_id(dwi)

    # Check that the number of DWI, bvec & bval are the same
    try:
        check_dwi_volume(dwi, bvec, bval)
    except ValueError as e:
        now = datetime.datetime.now().strftime('%H:%M:%S')
        error_msg = '%s[%s] Error: Number of DWIs, b-vals and b-vecs mismatch for  %s%s' %  (
            Fore.RED, now, image_id.replace('_', ' | '), Fore.RESET)
        cprint(error_msg)
        raise

    # Read metadata from DWI JSON file:
    [total_readout_time, phase_encoding_direction] = \
        extract_metadata_from_json(dwi_json, ['TotalReadoutTime', 'PhaseEncodingDirection'])
    phase_encoding_direction = bids_dir_to_fsl_dir(phase_encoding_direction)

    # Print begin message
    print_begin_image(image_id,
                      ['TotalReadoutTime', 'PhaseEncodingDirection'],
                      [str(total_readout_time), phase_encoding_direction])

    return (image_id, dwi, bvec, bval, total_readout_time, phase_encoding_direction)


def print_end_pipeline(image_id, final_file):
    """Display end message for `image_id` when `final_file` is connected."""
    from clinica.utils.ux import print_end_image
    print_end_image(image_id)
