# coding: utf8


def ants_bias_correction(low_bval=5.0, name='ants_bias_correction'):
    """This workflow reproduces dwibiascorrect script from MRtrix with ANTs algorithm."""
    import nipype.pipeline.engine as npe
    import nipype.interfaces.utility as niu
    import nipype.interfaces.fsl as fsl
    import nipype.interfaces.ants as ants
    from clinica.utils.dwi import compute_average_b0

    input_node = npe.Node(niu.IdentityInterface(
        fields=['dwi', 'bval', 'bvec', 'mask']),
        name='input_node')

    output_node = npe.Node(niu.IdentityInterface(
        fields=['bias_corrected_dwi']),
        name='output_node')

    # Compute average b0
    compute_average_b0 = npe.Node(niu.Function(input_names=['in_dwi', 'in_bval'],
                                               output_names=['out_b0_average'],
                                               function=compute_average_b0),
                                  name='ComputeB0Average')
    compute_average_b0.inputs.low_bval = low_bval

    # Estimate bias field
    n4 = npe.Node(ants.N4BiasFieldCorrection(
        dimension=3, save_bias=True,
        shrink_factor=4,
        bspline_fitting_distance=100, bspline_order=3,
        n_iterations=[1000], convergence_threshold=0.0
    ), name='BiasB0')

    # Split DWI dataset
    split = npe.Node(fsl.Split(dimension='t'), name='SplitDWIs')

    # Remove bias to each DWI volume
    rm_bias = npe.MapNode(fsl.MultiImageMaths(op_string='-div %s'),
                          iterfield=['in_file'],
                          name='RemoveBiasOfDWIs')

    # Remove negative values
    rm_negative = npe.MapNode(fsl.Threshold(thresh=0.0),
                              iterfield=['in_file'],
                              name='RemoveNegative')

    # Merge corrected DWIs
    merge = npe.Node(fsl.utils.Merge(dimension='t'), name='MergeDWIs')

    wf = npe.Workflow(name=name)
    wf.connect([
        # Compute average b0
        (input_node, compute_average_b0, [('dwi', 'in_dwi'),
                                          ('bval', 'in_bval')]),
        # Estimate bias field
        # Note from MRtrix developers:
        # Use the brain mask as a weights image rather than a mask; means that voxels at the edge of the mask
        #   will have a smoothly-varying bias field correction applied, rather than multiplying by 1.0 outside the mask
        (input_node, n4, [('mask', 'weight_image')]),
        (compute_average_b0, n4, [('out_b0_average', 'input_image')]),
        # Split DWI dataset
        (input_node, split, [('dwi', 'in_file')]),
        # Remove bias to each DWI volume
        (n4,    rm_bias, [('bias_image', 'operand_files')]),
        (split, rm_bias, [('out_files', 'in_file')]),
        # Remove negative values
        (rm_bias, rm_negative, [('out_file', 'in_file')]),
        # Merge corrected DWIs
        (rm_negative, merge, [('out_file', 'in_files')]),
        # Output node
        (merge,   output_node, [('merged_file', 'bias_corrected_dwi')]),
    ])
    return wf
