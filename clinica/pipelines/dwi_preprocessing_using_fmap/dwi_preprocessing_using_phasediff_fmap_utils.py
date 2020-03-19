# coding: utf8


def rename_into_caps(in_bids_dwi,
                     fname_dwi, fname_bval, fname_bvec, fname_brainmask,
                     fname_magnitude, fname_fmap, fname_smoothed_fmap):
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
        fname_smoothed_fmap (str): Smoothed (calibrated) fmap on b0 space.
        fname_fmap (str): Calibrated fmap on b0 space.
        fname_magnitude (str): Magnitude image on b0 space.

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
    base_dir_smoothed_fmap, _, _ = split_filename(fname_smoothed_fmap)
    base_dir_calibrated_fmap, _, _ = split_filename(fname_fmap)
    base_dir_magnitude, _, _ = split_filename(fname_magnitude)

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

    # Rename into CAPS magnitude:
    rename_magnitude = Rename()
    rename_magnitude.inputs.in_file = fname_magnitude
    rename_magnitude.inputs.format_string = os.path.join(
        base_dir_magnitude, source_file_dwi + "_space-b0_magnitude1.nii.gz")
    out_caps_magnitude = rename_magnitude.run()

    # Rename into CAPS fmap:
    rename_calibrated_fmap = Rename()
    rename_calibrated_fmap.inputs.in_file = fname_fmap
    rename_calibrated_fmap.inputs.format_string = os.path.join(
        base_dir_calibrated_fmap, source_file_dwi + "_space-b0_fmap.nii.gz")
    out_caps_fmap = rename_calibrated_fmap.run()

    # Rename into CAPS smoothed fmap:
    rename_smoothed_fmap = Rename()
    rename_smoothed_fmap.inputs.in_file = fname_smoothed_fmap
    rename_smoothed_fmap.inputs.format_string = os.path.join(
        base_dir_smoothed_fmap, source_file_dwi + "_space-b0_fwhm-4_fmap.nii.gz")
    out_caps_smoothed_fmap = rename_smoothed_fmap.run()

    return (out_caps_dwi.outputs.out_file,
            out_caps_bval.outputs.out_file,
            out_caps_bvec.outputs.out_file,
            out_caps_brainmask.outputs.out_file,
            out_caps_magnitude.outputs.out_file,
            out_caps_fmap.outputs.out_file,
            out_caps_smoothed_fmap.outputs.out_file,)


def get_grad_fsl(bvec, bval):
    grad_fsl = (bvec, bval)
    return grad_fsl


def init_input_node(dwi, bvec, bval, dwi_json,
                    fmap_magnitude, fmap_phasediff, fmap_phasediff_json):
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

    # Check that PhaseDiff and magnitude1 have the same header
    # Otherwise, FSL in FugueExtrapolationFromMask will crash
    img_phasediff = nib.load(fmap_phasediff)
    img_magnitude = nib.load(fmap_magnitude)
    if img_phasediff.shape != img_magnitude.shape:
        now = datetime.datetime.now().strftime('%H:%M:%S')
        error_msg = '%s[%s] Error: Headers of PhaseDiff and Magnitude1 are not the same for %s (%s vs %s)%s' % (
            Fore.RED, now, image_id.replace('_', ' | '), img_phasediff.shape, img_magnitude.shape, Fore.RESET)
        cprint(error_msg)
        raise NotImplementedError(error_msg)

    # Read metadata from DWI JSON file:
    [total_readout_time, phase_encoding_direction] = \
        extract_metadata_from_json(dwi_json, ['TotalReadoutTime', 'PhaseEncodingDirection'])
    phase_encoding_direction = bids_dir_to_fsl_dir(phase_encoding_direction)

    # Read metadata from PhaseDiff JSON file:
    [echo_time_1, echo_time_2] = extract_metadata_from_json(fmap_phasediff_json, ['EchoTime1', 'EchoTime2'])
    delta_echo_time = abs(echo_time_2 - echo_time_1)

    # Print begin message
    print_begin_image(image_id,
                      ['TotalReadoutTime', 'PhaseEncodingDirection', 'DeltaEchoTime'],
                      [str(total_readout_time), phase_encoding_direction, str(delta_echo_time)])

    return (image_id, dwi, bvec, bval, total_readout_time, phase_encoding_direction,
            fmap_magnitude, fmap_phasediff, delta_echo_time)


def print_end_pipeline(image_id, final_file):
    """Display end message for `image_id` when `final_file` is connected."""
    from clinica.utils.ux import print_end_image
    print_end_image(image_id)


def rads2hz(in_file, delta_te, out_file=None):
    """Converts input phase difference map to Hz."""
    import math
    import os.path as op
    import numpy as np
    import nibabel as nb
    from nipype.utils import NUMPY_MMAP

    if out_file is None:
        fname, fext = op.splitext(op.basename(in_file))
        if fext == '.gz':
            fname, _ = op.splitext(fname)
        out_file = op.abspath('./%s_radsec.nii.gz' % fname)

    im = nb.load(in_file, mmap=NUMPY_MMAP)
    data = im.get_data().astype(np.float32) * (1.0 / (delta_te * 2 * math.pi))
    nb.Nifti1Image(data, im.affine, im.header).to_filename(out_file)
    return out_file


def resample_fmap_to_b0(in_fmap, in_b0, out_file=None):
    """
    Resample fieldmap onto the b0 image.

    Warnings:
        The fieldmap should already be aligned on the b0.

    Args:
        in_fmap(str): Fieldmap image.
        in_b0(str): B0 image.
        out_file(optional[str]): Filename (default: <in_fmap>_space-b0.nii.gz

    Returns:
        Fieldmap image resampled on b0.
    """
    import os.path as op
    import nibabel
    from nilearn.image import resample_to_img

    if out_file is None:
        fname, ext = op.splitext(op.basename(in_fmap))
        if ext == ".gz":
            fname, ext2 = op.splitext(fname)
            ext = ext2 + ext
        out_resampled_fmap = op.abspath("%s_space-b0%s" % (fname, ext))
    else:
        out_resampled_fmap = out_file

    resampled_fmap = resample_to_img(
        source_img=in_fmap, target_img=in_b0,
        interpolation='continuous')

    nibabel.nifti1.save(resampled_fmap, out_resampled_fmap)

    return out_resampled_fmap


def remove_filename_extension(in_file):
    """Remove extension from `in_file`. This is needed for FSL eddy --field option."""
    import os
    from nipype.utils.filemanip import split_filename

    path, source_file_dwi, _ = split_filename(in_file)
    file_without_extension = os.path.join(path, source_file_dwi)

    return file_without_extension