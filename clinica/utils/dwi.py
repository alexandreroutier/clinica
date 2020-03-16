# coding: utf8


"""This module contains utilities for DWI handling."""


def merge_volumes_tdim(in_file1, in_file2):
    """
    Merge 'in_file1' and 'in_file2' in the t dimension.

    Args:
        in_file1 (str): First set of volumes.
        in_file2 (str): Second set of volumes.

    Returns:
        out_file (str): The two sets of volumes merged.
    """
    import os.path as op
    import os

    out_file = op.abspath('merged_files.nii.gz')
    cmd = 'fslmerge -t %s %s %s ' % (out_file, in_file1, in_file2)
    os.system(cmd)
    return out_file


def count_b0s(in_bval, low_bval=5.0):
    """
    Count the number of volumes where b<=low_bval.

    Args:
        in_bval (str): bval file.
        low_bval (Optional[int]): Define the b0 volumes as all volume
            bval <= lowbval. (Default=5.0)

    Returns:
        num_b0s: Number of b0s.
    """
    import numpy as np

    bvals = np.loadtxt(in_bval)
    num_b0s = len(np.where(bvals <= low_bval)[0])

    return num_b0s


def b0_average(in_file, out_file=None):
    """
    Average the b0 volumes.

    Args:
        in_file (str): The b0 volumes already registered.
        out_file (optional[str]): Name of the file.

    Returns:
        The mean of the b0 volumes.

    Warnings:
        The b0 volumes must be registered.
    """
    import numpy as np
    import nibabel as nb
    import os.path as op

    if out_file is None:
        fname, ext = op.splitext(op.basename(in_file))
        if ext == ".gz":
            fname, ext2 = op.splitext(fname)
            ext = ext2 + ext
        out_file = op.abspath("%s_avg_b0%s" % (fname, ext))

    imgs = np.array(nb.four_to_three(nb.load(in_file)))
    b0s = [im.get_data().astype(np.float32)
           for im in imgs]
    b0 = np.average(np.array(b0s), axis=0)

    hdr = imgs[0].get_header().copy()
    hdr.set_data_shape(b0.shape)
    hdr.set_xyzt_units('mm')
    hdr.set_data_dtype(np.float32)
    nb.Nifti1Image(b0, imgs[0].get_affine(), hdr).to_filename(out_file)

    return out_file


def b0_dwi_split(in_dwi, in_bval, in_bvec, low_bval=5.0):
    """
    Split the DWI volumes into two datasets :
     - the first dataset contains the set of b<=low_bval volumes.
     - the second dataset contains the set of DWI volumes.

    Args:
        in_dwi (str): DWI dataset.
        in_bval (str): File describing the b-values of the DWI dataset.
        in_bvec (str): File describing the directions of the DWI dataset.
        low_bval (Optional[int]): Define the b0 volumes as all volume
            bval <= lowbval. (Default=5.0)

    Returns:
        out_b0 (str): The set of b<=low_bval volumes.
        out_dwi (str): Output. The set of b>low_bval volumes.
        out_bvals (str): The b-values corresponding to the out_dwi.
        out_bvecs (str): The b-vecs corresponding to the out_dwi.
    """
    import numpy as np
    import nibabel as nib
    import os.path as op
    import warnings

    assert(op.isfile(in_dwi))
    assert(op.isfile(in_bval))
    assert(op.isfile(in_bvec))
    assert(low_bval >= 0)

    im = nib.load(in_dwi)
    data = im.get_data()
    hdr = im.get_header().copy()
    bvals = np.loadtxt(in_bval)
    bvecs = np.loadtxt(in_bvec)

    if bvals.shape[0] == bvecs.shape[0]:
        warnings.warn('Warning: The b-vectors file should be column-wise. '
                      + 'The b-vectors will be transposed',
                      UserWarning)
        bvecs = bvecs.T

    lowbs = np.where(bvals <= low_bval)[0]

    fname_b0, ext_b0 = op.splitext(op.basename(in_dwi))
    if ext_b0 == ".gz":
        fname_b0, ext2 = op.splitext(fname_b0)
        ext_b0 = ext2 + ext_b0
    out_b0 = op.abspath("%s_b0%s" % (fname_b0, ext_b0))
    # out_b0 = op.abspath('b0.nii.gz')
    b0data = data[..., lowbs]
    hdr.set_data_shape(b0data.shape)
    nib.Nifti1Image(b0data, im.get_affine(), hdr).to_filename(out_b0)

    dwi_bvals = np.where(bvals > low_bval)[0]
    out_dwi = op.abspath('dwi.nii.gz')
    dwi_data = data[..., dwi_bvals]
    hdr.set_data_shape(dwi_data.shape)
    nib.Nifti1Image(dwi_data, im.get_affine(), hdr).to_filename(out_dwi)

    bvals_dwi = bvals[dwi_bvals]
    out_bvals = op.abspath('bvals')
    np.savetxt(out_bvals, bvals_dwi, fmt='%d', delimiter=' ')

    bvecs_dwi = np.array([bvecs[0][dwi_bvals].tolist(),
                          bvecs[1][dwi_bvals].tolist(),
                          bvecs[2][dwi_bvals].tolist()])
    out_bvecs = op.abspath('bvecs')
    np.savetxt(out_bvecs, bvecs_dwi, fmt='%10.5f', delimiter=' ')

    return out_b0, out_dwi, out_bvals, out_bvecs


def compute_average_b0(in_dwi, in_bval, low_bval=5.0):
    """Compute average b0 volume from DWI dataset."""
    import numpy as np
    import nibabel
    import os.path as op

    assert (op.isfile(in_dwi))
    assert (op.isfile(in_bval))
    assert (low_bval >= 0)

    imgs = np.array(nibabel.four_to_three(nibabel.load(in_dwi)))
    bvals = np.loadtxt(in_bval)
    low_bvals = np.where(bvals <= low_bval)[0]

    fname_b0, ext_b0 = op.splitext(op.basename(in_dwi))
    if ext_b0 == ".gz":
        fname_b0, ext2 = op.splitext(fname_b0)
        ext_b0 = ext2 + ext_b0
    out_b0_average = op.abspath("%s_avg_b0%s" % (fname_b0, ext_b0))

    b0s_data = [imgs[i].get_data() for i in low_bvals]
    avg_b0_data = np.average(np.array(b0s_data), axis=0)

    hdr = imgs[0].get_header().copy()
    hdr.set_data_shape(avg_b0_data.shape)
    nibabel.Nifti1Image(avg_b0_data, imgs[0].get_affine(), hdr).to_filename(out_b0_average)

    return out_b0_average


def insert_b0_into_dwi(in_b0, in_dwi, in_bval, in_bvec):
    """
    This function inserts a b0 volume into the dwi dataset as the first volume
    and updates the bvals and bvecs files.

    Args:
        in_b0 (str): One b=0 volume (could be the average of a b0 dataset).
        in_dwi (str): The set of DWI volumes.
        in_bval (str): File describing the b-values of the DWI dataset.
        in_bvec (str): File describing the directions of the DWI dataset.

    Returns:
        out_dwi (str): Diffusion dataset : b0 volume + dwi volumes.
        out_bval (str): B-values update.
        out_bvec (str): Directions of diffusion update.
    """
    from clinica.utils.dwi import merge_volumes_tdim
    import os.path as op
    import numpy as np

    assert(op.isfile(in_b0))
    assert(op.isfile(in_dwi))
    assert(op.isfile(in_bval))
    assert(op.isfile(in_bvec))

    out_dwi = merge_volumes_tdim(in_b0, in_dwi)

    lst = np.loadtxt(in_bval).tolist()
    lst.insert(0, 0)
    out_bvals = op.abspath('bvals')
    np.savetxt(out_bvals, np.matrix(lst), fmt='%d', delimiter=' ')

    bvecs = np.loadtxt(in_bvec)
    bvecs_0 = bvecs[0].tolist()
    bvecs_0.insert(0, 0.0)
    bvecs_1 = bvecs[1].tolist()
    bvecs_1.insert(0, 0.0)
    bvecs_2 = bvecs[2].tolist()
    bvecs_2.insert(0, 0.0)
    bvecs_dwi = np.array([bvecs_0, bvecs_1, bvecs_2])
    out_bvecs = op.abspath('bvecs')
    np.savetxt(out_bvecs, bvecs_dwi, fmt='%10.5f', delimiter=' ')

    return out_dwi, out_bvals, out_bvecs


def check_dwi_volume(in_dwi, in_bvec, in_bval):
    """
    Check that # DWI = # B-val = # B-vec.

    Raises
        ValueError if # DWI, # B-val and # B-vec mismatch
    """
    import nibabel as nib
    import numpy as np

    bvals = np.loadtxt(in_bval)
    num_b_vals = len(bvals)

    bvecs = np.loadtxt(in_bvec)
    _, num_b_vecs = bvecs.shape

    img = nib.load(in_dwi)
    _, _, _, num_dwis = img.shape

    if not (num_b_vals == num_b_vecs == num_dwis):
        raise ValueError('Number of DWIs, b-vals and b-vecs mismatch (# DWI = %s, # B-vec = %s, #B-val = %s) ' %
                         (num_dwis, num_b_vecs, num_b_vals))


def find_b0_indices(in_bval, low_bval=5.0):
    """
    Returns indices of bval <= low_bval.

    Args:
        in_bval (str): Bval file.
        low_bval (Optional[int]): Define the b0 volumes as all volume
            bval <= low_bval. (Default=5.0)

    Returns:
        out_indices (int): Indices of bval <= low_bval.
    """
    import numpy as np
    import os.path as op

    assert(op.isfile(in_bval))
    bvals = np.loadtxt(in_bval)

    idx_low_bvals = np.where(bvals <= low_bval)
    out_indices = idx_low_bvals[0].tolist()

    return out_indices


def generate_index_file(in_bval, low_bval=5.0, image_id=None):
    """
    Generate [`image_id`]_index.txt file for FSL eddy command.

    Args:
        in_bval: Bval file.
        low_bval: Define the b0 volumes as all volume
            bval <= low_bval. (Default=5.0)
        image_id: Optional prefix.

    Returns:
        out_index: [`image_id`]_index.txt or index.txt file.
    """
    import os
    import numpy as np
    import os.path as op

    assert(op.isfile(in_bval))
    bvals = np.loadtxt(in_bval)
    idx_low_bvals = np.where(bvals <= low_bval)
    b0_index = idx_low_bvals[0].tolist()

    if not b0_index:
        raise ValueError("Could not find b-value <= %s in bval file (%s). Found values: %s"
                         % (low_bval, in_bval, bvals))

    if image_id:
        out_index = os.path.abspath(image_id + '_index.txt')
    else:
        out_index = os.path.abspath('index.txt')

    vols = len(bvals)
    index_list = []
    for i in range(0, len(b0_index)):
        if i == (len(b0_index) - 1):
            index_list.extend([i+1] * (vols - b0_index[i]))
        else:
            index_list.extend([i+1] * (b0_index[i+1] - b0_index[i]))
    index_array = np.asarray(index_list)
    try:
        len(index_list) == vols
    except ValueError:
        raise ValueError("It seems that you do not define the index file for FSL eddy correctly!")
    np.savetxt(out_index, index_array.T)

    return out_index


def generate_acq_file(in_dwi, fsl_phase_encoding_direction, total_readout_time, image_id=None):
    """
    Generate [`image_id`]_acq.txt file for FSL eddy command.

    Args:
        in_dwi: DWI file.
        fsl_phase_encoding_direction: PhaseEncodingDirection from BIDS specifications
            in FSL format (i.e. x/y/z instead of i/j/k).
        total_readout_time: TotalReadoutTime from BIDS specifications.
        image_id: Optional prefix.
    Returns:
        out_acq: [`image_id`]_acq.txt or acq.txt file.
    """
    import numpy as np
    import os
    import nibabel as nb

    if image_id:
        out_acq = out_acq = os.path.abspath(image_id + '_acq.txt')
    else:
        out_acq = os.path.abspath('acq.txt')
    vols = nb.load(in_dwi).get_data().shape[-1]
    arr = np.ones([vols, 4])
    for i in range(vols):
        if fsl_phase_encoding_direction == 'y-':
            arr[i, :] = np.array((0, -1, 0, total_readout_time))
        elif fsl_phase_encoding_direction == 'y':
            arr[i, :] = np.array((0, 1, 0, total_readout_time))
        elif fsl_phase_encoding_direction == 'x':
            arr[i, :] = np.array((1, 0, 0, total_readout_time))
        elif fsl_phase_encoding_direction == 'x-':
            arr[i, :] = np.array((-1, 0, 0, total_readout_time))
        elif fsl_phase_encoding_direction == 'z':
            arr[i, :] = np.array((0, 1, 0, total_readout_time))
        elif fsl_phase_encoding_direction == 'z-':
            arr[i, :] = np.array((0, 0, -1, total_readout_time))
        else:
            raise RuntimeError("FSL PhaseEncodingDirection (found value: %s) is unknown,"
                               "it should be a value in (x, y, z, x-, y-, z-)" % fsl_phase_encoding_direction)

    np.savetxt(out_acq, arr, fmt="%d "*3 + "%f")

    return out_acq


def bids_dir_to_fsl_dir(bids_dir):
    """Converts BIDS PhaseEncodingDirection parameters (i,j,k,i-,j-,k-) to FSL direction (x,y,z,x-,y-,z-)."""
    fsl_dir = bids_dir.lower()
    if 'i' not in fsl_dir and 'j' not in fsl_dir and 'k' not in fsl_dir:
        raise ValueError("Unknown PhaseEncodingDirection %s: it should be a value in (i, j, k, i-, j-, k-)" % fsl_dir)

    return fsl_dir.replace('i', 'x').replace('j', 'y').replace('k', 'z')
