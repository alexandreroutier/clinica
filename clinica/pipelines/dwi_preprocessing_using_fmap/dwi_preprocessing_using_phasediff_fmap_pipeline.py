# coding: utf8

from nipype import config

import clinica.pipelines.engine as cpe

# Use hash instead of parameters for iterables folder names
# Otherwise path will be too long and generate OSError
cfg = dict(execution={'parameterize_dirs': False})
config.update_config(cfg)


class DwiPreprocessingUsingPhaseDiffFMap(cpe.Pipeline):
    """DWI Preprocessing using phase difference fieldmap.

    Ideas for improvement:
        - Use promising sdcflows workflows and/or dMRIprep

    Note:
        Some reading regarding the reproducibility of FSL eddy command:
        https://www.jiscmail.ac.uk/cgi-bin/webadmin?A2=fsl;1ccf038f.1608

    Returns:
        A clinica pipeline object containing the DwiPreprocessingUsingPhaseDiffFMap pipeline.
    """
    @staticmethod
    def get_processed_images(caps_directory, subjects, sessions):
        import os
        from clinica.utils.inputs import clinica_file_reader
        from clinica.utils.input_files import DWI_PREPROC_NII
        from clinica.utils.filemanip import extract_image_ids
        image_ids = []
        if os.path.isdir(caps_directory):
            preproc_files = clinica_file_reader(
                subjects, sessions,
                caps_directory, DWI_PREPROC_NII, False
            )
            image_ids = extract_image_ids(preproc_files)
        return image_ids


    def check_pipeline_parameters(self):
        """Check pipeline parameters."""
        from colorama import Fore
        from clinica.utils.exceptions import ClinicaException
        from clinica.utils.stream import cprint

        if self.parameters['low_bval'] < 0:
            raise ValueError('%sThe low_bval is equals to %s: it should be zero or close to zero.%s' %
                             (Fore.RED, self.parameters['low_bval'], Fore.RESET))

        if self.parameters['low_bval'] > 100:
            cprint('%sWarning: The low_bval parameter is %s: it should be close to zero.%s' %
                   (Fore.YELLOW, self.parameters['low_bval'], Fore.RESET))

        if 'use_cuda_8_0' not in self.parameters.keys():
            self.parameters['use_cuda_8_0'] = False
        if 'use_cuda_9_1' not in self.parameters.keys():
            self.parameters['use_cuda_9_1'] = False
        if self.parameters['use_cuda_8_0'] and self.parameters['use_cuda_9_1']:
            raise ClinicaException('\n%s[Error] You must choose between CUDA 8.0 or CUDA 9.1, not both.%s' %
                                   (Fore.RED, Fore.RESET))

        if self.parameters['initrand'] not in self.parameters.keys():
            self.parameters['initrand'] = None

    def check_custom_dependencies(self):
        """Check dependencies that can not be listed in the `info.json` file."""
        from colorama import Fore
        from clinica.utils.check_dependency import is_binary_present
        from clinica.utils.exceptions import ClinicaMissingDependencyError

        if self.parameters['use_cuda_8_0']:
            if not is_binary_present('eddy_cuda8.0'):
                raise ClinicaMissingDependencyError(
                    '%s\n[Error] FSL eddy with CUDA 8.0 was set but Clinica could not find '
                    'eddy_cuda8.0 in your PATH environment.%s' % (Fore.RED, Fore.RESET))

        if self.parameters['use_cuda_9_1']:
            if not is_binary_present('eddy_cuda9.1'):
                raise ClinicaMissingDependencyError(
                    '%s\n[Error] FSL eddy with CUDA 9.1 was set but Clinica could not find '
                    'eddy_cuda9.1 in your PATH environment.%s' % (Fore.RED, Fore.RESET))

    def get_input_fields(self):
        """Specify the list of possible inputs of this pipeline.

        Note:
            The list of inputs of the DwiPreprocessingUsingPhaseDiffFieldmap pipeline is:
                * dwi (str): Path of the diffusion weighted image in BIDS format
                * bvec (str): Path of the bvec file in BIDS format
                * bval (str): Path of the bval file in BIDS format
                * dwi_json (str): Path of the DWI JSON file in BIDS format and containing
                    TotalReadoutTime and PhaseEncodingDirection metadata (see BIDS specifications)
                * fmap_magnitude (str): Path of the magnitude (1st) image in BIDS format
                * fmap_phasediff (str): Path of the phase difference image in BIDS format
                * fmap_phasediff_json (str): Path of the phase difference JSON file in BIDS format
                    and containing EchoTime1 & EchoTime2 metadata (see BIDS specifications)

        Returns:
            A list of (string) input fields name.
        """
        input_list = ['dwi', 'bvec', 'bval', 'dwi_json',
                      'fmap_magnitude', 'fmap_phasediff', 'fmap_phasediff_json']

        return input_list

    def get_output_fields(self):
        """Specify the list of possible outputs of this pipeline.

        Note:
            The list of outputs of the DwiPreprocessingUsingPhaseDiffFieldmap pipeline is:
                * preproc_dwi (str): Path of the preprocessed DWI
                * preproc_bvec (str): Path of the preprocessed bvec
                * preproc_bval (str): Path of the preprocessed bval
                * b0_mask (str): Path of the b0 brainmask
                * magnitude_on_b0 (str): Path of the smoothed calibrated fmap on b0 space
                * calibrated_fmap_on_b0 (str): Path of the calibrated fmap on b0 space
                * smoothed_fmap_on_b0 (str): Path of the magnitude fmap on b0 space

        Returns:
            A list of (string) output fields name.
        """
        output_list = ['preproc_dwi', 'preproc_bvec', 'preproc_bval', 'b0_mask',
                       'magnitude_on_b0', 'calibrated_fmap_on_b0', 'smoothed_fmap_on_b0']

        return output_list

    def build_input_node(self):
        """Build and connect an input node to the pipeline."""
        import os

        import nipype.interfaces.utility as nutil
        import nipype.pipeline.engine as npe

        from clinica.utils.filemanip import save_participants_sessions
        from clinica.utils.inputs import clinica_list_of_files_reader
        from clinica.utils.input_files import (DWI_NII, DWI_BVAL, DWI_BVEC, DWI_JSON,
                                               FMAP_MAGNITUDE1_NII, FMAP_PHASEDIFF_NII, FMAP_PHASEDIFF_JSON)
        from clinica.utils.stream import cprint
        from clinica.utils.ux import print_images_to_process

        list_bids_files = clinica_list_of_files_reader(
            self.subjects,
            self.sessions,
            self.bids_directory,
            [DWI_NII, DWI_BVEC, DWI_BVAL, DWI_JSON, FMAP_MAGNITUDE1_NII, FMAP_PHASEDIFF_NII, FMAP_PHASEDIFF_JSON],
            raise_exception=True)

        # Save subjects to process in <WD>/<Pipeline.name>/participants.tsv
        folder_participants_tsv = os.path.join(self.base_dir, self.name)
        save_participants_sessions(self.subjects, self.sessions, folder_participants_tsv)

        if len(self.subjects):
            print_images_to_process(self.subjects, self.sessions)
            cprint('List available in %s' % os.path.join(folder_participants_tsv, 'participants.tsv'))
            cprint('Computational time will depend of the number of volumes in your DWI dataset and the use of CUDA.')

        read_node = npe.Node(name="ReadingFiles",
                             iterables=[
                                 ('dwi', list_bids_files[0]),
                                 ('bvec', list_bids_files[1]),
                                 ('bval', list_bids_files[2]),
                                 ('dwi_json', list_bids_files[3]),
                                 ('fmap_magnitude', list_bids_files[4]),
                                 ('fmap_phasediff', list_bids_files[5]),
                                 ('fmap_phasediff_json', list_bids_files[6]),
                             ],
                             synchronize=True,
                             interface=nutil.IdentityInterface(
                                 fields=self.get_input_fields()))
        self.connect([
            (read_node, self.input_node, [('dwi', 'dwi'),
                                          ('bvec', 'bvec'),
                                          ('bval', 'bval'),
                                          ('dwi_json', 'dwi_json'),
                                          ('fmap_magnitude', 'fmap_magnitude'),
                                          ('fmap_phasediff', 'fmap_phasediff'),
                                          ('fmap_phasediff_json', 'fmap_phasediff_json')]),
        ])

    def build_output_node(self):
        """Build and connect an output node to the pipeline."""
        import nipype.interfaces.utility as nutil
        import nipype.pipeline.engine as npe
        import nipype.interfaces.io as nio
        from clinica.utils.nipype import fix_join, container_from_filename
        from .dwi_preprocessing_using_phasediff_fmap_utils import rename_into_caps

        # Find container path from DWI filename
        # =====================================
        container_path = npe.Node(nutil.Function(
            input_names=['bids_or_caps_filename'],
            output_names=['container'],
            function=container_from_filename),
            name='container_path')

        rename_into_caps = npe.Node(nutil.Function(
            input_names=['in_bids_dwi',
                         'fname_dwi', 'fname_bval', 'fname_bvec', 'fname_brainmask',
                         'fname_magnitude', 'fname_fmap', 'fname_smoothed_fmap'],
            output_names=['out_caps_dwi', 'out_caps_bval', 'out_caps_bvec', 'out_caps_brainmask',
                          'out_caps_magnitude', 'out_caps_fmap', 'out_caps_smoothed_fmap'],
            function=rename_into_caps),
            name='rename_into_caps')

        # Writing results into CAPS
        # =========================
        write_results = npe.Node(name='write_results',
                                 interface=nio.DataSink())
        write_results.inputs.base_directory = self.caps_directory
        write_results.inputs.parameterization = False

        self.connect([
            (self.input_node, container_path,    [('dwi', 'bids_or_caps_filename')]),
            (self.input_node,  rename_into_caps, [('dwi', 'in_bids_dwi')]),
            (self.output_node, rename_into_caps, [('preproc_dwi',  'fname_dwi'),
                                                  ('preproc_bval', 'fname_bval'),
                                                  ('preproc_bvec', 'fname_bvec'),
                                                  ('b0_mask',      'fname_brainmask'),
                                                  ('magnitude_on_b0',       'fname_magnitude'),
                                                  ('calibrated_fmap_on_b0', 'fname_fmap'),
                                                  ('smoothed_fmap_on_b0',   'fname_smoothed_fmap')]),
            (container_path, write_results,      [(('container', fix_join, 'dwi'), 'container')]),
            (rename_into_caps, write_results,    [('out_caps_dwi',           'preprocessing.@preproc_dwi'),
                                                  ('out_caps_bval',          'preprocessing.@preproc_bval'),
                                                  ('out_caps_bvec',          'preprocessing.@preproc_bvec'),
                                                  ('out_caps_brainmask',     'preprocessing.@b0_mask'),
                                                  ('out_caps_magnitude',     'preprocessing.@magnitude'),
                                                  ('out_caps_fmap',          'preprocessing.@fmap'),
                                                  ('out_caps_smoothed_fmap', 'preprocessing.@smoothed_fmap')]),
        ])

    def build_core_nodes(self):
        """Build and connect the core nodes of the pipeline."""
        import nipype.interfaces.utility as nutil
        import nipype.pipeline.engine as npe
        import nipype.interfaces.utility as niu
        import nipype.interfaces.fsl as fsl
        import nipype.interfaces.ants as ants
        import nipype.interfaces.mrtrix3 as mrtrix3

        from clinica.lib.nipype.interfaces.fsl.epi import Eddy

        from clinica.utils.dwi import generate_acq_file, generate_index_file, compute_average_b0

        from .dwi_preprocessing_using_phasediff_fmap_workflows import prepare_phasediff_fmap, ants_bias_correction
        from .dwi_preprocessing_using_phasediff_fmap_utils import (init_input_node,
                                                                   get_grad_fsl,
                                                                   print_end_pipeline,
                                                                   remove_filename_extension)

        # Step 0: Initialization
        # ======================
        # Initialize input parameters and print begin message
        init_node = npe.Node(interface=nutil.Function(
            input_names=self.get_input_fields(),
            output_names=['image_id',
                          'dwi', 'bvec', 'bval', 'total_readout_time', 'phase_encoding_direction',
                          'fmap_magnitude', 'fmap_phasediff', 'delta_echo_time'],
            function=init_input_node),
            name='0-InitNode')

        # Generate (bvec, bval) tuple for MRtrix interfaces
        get_grad_fsl = npe.Node(nutil.Function(
            input_names=['bval', 'bvec'],
            output_names=['grad_fsl'],
            function=get_grad_fsl),
            name='0-GetFslGrad')

        # Generate <image_id>_acq.txt for eddy
        gen_acq_txt = npe.Node(nutil.Function(
            input_names=['in_dwi', 'fsl_phase_encoding_direction', 'total_readout_time', 'image_id'],
            output_names=['out_acq'],
            function=generate_acq_file),
            name='0-GenerateAcqFile')

        # Generate <image_id>_index.txt for eddy
        gen_index_txt = npe.Node(nutil.Function(
            input_names=['in_bval', 'low_bval', 'image_id'],
            output_names=['out_index'],
            function=generate_index_file),
            name='0-GenerateIndexFile')
        gen_index_txt.inputs.low_bval = self.parameters['low_bval']

        # Step 1: Computation of the reference b0 (i.e. average b0 but with EPI distortions)
        # =======================================
        # Compute whole brain mask
        pre_mask_b0 = npe.Node(mrtrix3.BrainMask(),
                               name='1a-PreMaskB0')
        pre_mask_b0.inputs.out_file = 'brainmask.nii.gz'  # On default, .mif file is generated

        # Run eddy without calibrated fmap
        pre_eddy = npe.Node(name='1b-PreEddy',
                            interface=Eddy())
        pre_eddy.inputs.repol = True
        if self.parameters['use_cuda_8_0']:
            pre_eddy.inputs.use_cuda8_0 = self.parameters['use_cuda_8_0']
        if self.parameters['use_cuda_9_1']:
            pre_eddy.inputs.use_cuda9_1 = self.parameters['use_cuda_9_1']
        if self.parameters['initrand']:
            pre_eddy.inputs.initrand = self.parameters['initrand']

        # Compute the reference b0
        compute_ref_b0 = npe.Node(niu.Function(input_names=['in_dwi', 'in_bval'],
                                               output_names=['out_b0_average'],
                                               function=compute_average_b0),
                                  name='1c-ComputeReferenceB0')
        compute_ref_b0.inputs.low_bval = self.parameters['low_bval']

        # Compute brain mask from reference b0
        mask_ref_b0 = npe.Node(fsl.BET(mask=True, robust=True),
                               name='1d-MaskReferenceB0')

        # Step 2: Calibrate and register FMap
        # ===================================
        # Bias field correction of the magnitude image
        bias_mag_fmap = npe.Node(ants.N4BiasFieldCorrection(dimension=3),
                                 name='2a-N4MagnitudeFmap')
        # Brain extraction of the magnitude image
        bet_mag_fmap = npe.Node(fsl.BET(frac=0.4, mask=True),
                                name='2b-BetN4MagnitudeFmap')

        # Calibrate FMap
        calibrate_fmap = prepare_phasediff_fmap(name='2c-CalibrateFMap')

        # Register the BET magnitude fmap onto the BET b0
        bet_mag_fmap2b0 = npe.Node(interface=fsl.FLIRT(),
                                   name="2d-RegistrationBetMagToB0")
        bet_mag_fmap2b0.inputs.dof = 6
        bet_mag_fmap2b0.inputs.output_type = "NIFTI_GZ"

        # Apply the transformation on the calibrated fmap
        fmap2b0 = npe.Node(interface=fsl.ApplyXFM(),
                           name="2e-1-FMapToB0")
        fmap2b0.inputs.output_type = "NIFTI_GZ"

        # Apply the transformation on the magnitude image
        mag_fmap2b0 = fmap2b0.clone('2e-2-MagFMapToB0')

        # Smooth the registered (calibrated) fmap
        smoothing = npe.Node(interface=fsl.maths.IsotropicSmooth(),
                             name='2f-Smoothing')
        smoothing.inputs.sigma = 4.0

        # Remove ".nii.gz" from fieldmap filename for eddy --field
        rm_extension = npe.Node(interface=nutil.Function(
            input_names=['in_file'],
            output_names=['file_without_extension'],
            function=remove_filename_extension),
            name="2h-RemoveFNameExtension")

        # Step 3: Run FSL eddy
        # ====================
        eddy = pre_eddy.clone('3-Eddy')

        # Step 4: Bias correction
        # =======================
        bias = ants_bias_correction(name='4-RemoveBias')

        # Step 5: Final brainmask
        # =======================
        # Compute average b0 on corrected dataset (for brain mask extraction)
        compute_avg_b0 = compute_ref_b0.clone('5a-ComputeB0Average')

        # Compute b0 mask on corrected avg b0
        mask_avg_b0 = mask_ref_b0.clone('5b-MaskB0')

        # Print end message
        print_end_message = npe.Node(
            interface=nutil.Function(
                input_names=['image_id', 'final_file'],
                function=print_end_pipeline),
            name='99-WriteEndMessage')

        # Connection
        # ==========
        self.connect([
            # Step 0: Initialization
            # ======================
            # Initialize input parameters and print begin message
            (self.input_node, init_node, [('dwi', 'dwi'),
                                          ('bvec', 'bvec'),
                                          ('bval', 'bval'),
                                          ('dwi_json', 'dwi_json'),
                                          ('fmap_magnitude', 'fmap_magnitude'),
                                          ('fmap_phasediff', 'fmap_phasediff'),
                                          ('fmap_phasediff_json', 'fmap_phasediff_json')]),
            # Generate (bvec, bval) tuple for MRtrix interfaces
            (init_node, get_grad_fsl, [('bval', 'bval'),
                                       ('bvec', 'bvec')]),
            # Generate <image_id>_acq.txt for eddy
            (init_node, gen_acq_txt, [('dwi', 'in_dwi'),
                                      ('total_readout_time', 'total_readout_time'),
                                      ('phase_encoding_direction', 'fsl_phase_encoding_direction'),
                                      ('image_id', 'image_id')]),
            # Generate <image_id>_index.txt for eddy
            (init_node, gen_index_txt, [('bval', 'in_bval'),
                                        ('image_id', 'image_id')]),

            # Step 1: Computation of the reference b0 (i.e. average b0 but with EPI distortions)
            # =======================================
            # Compute whole brain mask
            (get_grad_fsl, pre_mask_b0, [('grad_fsl', 'grad_fsl')]),
            (init_node,    pre_mask_b0, [('dwi', 'in_file')]),
            # Run eddy without calibrated fmap
            (init_node,     pre_eddy, [('dwi', 'in_file'),
                                       ('bval', 'in_bval'),
                                       ('bvec', 'in_bvec'),
                                       ('image_id', 'out_base')]),
            (gen_acq_txt,   pre_eddy, [('out_acq', 'in_acqp')]),
            (gen_index_txt, pre_eddy, [('out_index', 'in_index')]),
            (pre_mask_b0,   pre_eddy, [('out_file', 'in_mask')]),
            # Compute the reference b0
            (init_node, compute_ref_b0, [('bval', 'in_bval')]),
            (pre_eddy,  compute_ref_b0, [('out_corrected', 'in_dwi')]),
            # Compute brain mask from reference b0
            (compute_ref_b0, mask_ref_b0, [('out_b0_average', 'in_file')]),

            # Step 2: Calibrate and register FMap
            # ===================================
            # Bias field correction of the magnitude image
            (init_node, bias_mag_fmap, [('fmap_magnitude', 'input_image')]),
            # Brain extraction of the magnitude image
            (bias_mag_fmap, bet_mag_fmap, [('output_image', 'in_file')]),
            # Calibration of the FMap
            (bet_mag_fmap, calibrate_fmap, [('mask_file', 'input_node.fmap_mask'),
                                            ('out_file', 'input_node.fmap_magnitude')]),
            (init_node, calibrate_fmap, [('fmap_phasediff', 'input_node.fmap_phasediff'),
                                         ('delta_echo_time', 'input_node.delta_echo_time')]),
            # Register the BET magnitude fmap onto the BET b0
            (bet_mag_fmap, bet_mag_fmap2b0, [('out_file', 'in_file')]),
            (mask_ref_b0,  bet_mag_fmap2b0, [('out_file', 'reference')]),
            # Apply the transformation on the magnitude image
            (bet_mag_fmap2b0, mag_fmap2b0, [('out_matrix_file', 'in_matrix_file')]),
            (bias_mag_fmap,   mag_fmap2b0, [('output_image', 'in_file')]),
            (mask_ref_b0,     mag_fmap2b0, [('out_file', 'reference')]),
            # Apply the transformation on the calibrated fmap
            (bet_mag_fmap2b0, fmap2b0, [('out_matrix_file', 'in_matrix_file')]),
            (calibrate_fmap,  fmap2b0, [('output_node.calibrated_fmap', 'in_file')]),
            (mask_ref_b0,     fmap2b0, [('out_file', 'reference')]),
            # # Smooth the registered (calibrated) fmap
            (fmap2b0, smoothing, [('out_file', 'in_file')]),
            # Remove ".nii.gz" from fieldmap filename for eddy --field
            (smoothing, rm_extension, [('out_file', 'in_file')]),

            # Step 3: Run FSL eddy
            # ====================
            (init_node,     eddy, [('dwi', 'in_file'),
                                   ('bval', 'in_bval'),
                                   ('bvec', 'in_bvec'),
                                   ('image_id', 'out_base')]),
            (gen_acq_txt,   eddy, [('out_acq', 'in_acqp')]),
            (gen_index_txt, eddy, [('out_index', 'in_index')]),
            (rm_extension,  eddy, [('file_without_extension', 'field')]),
            (pre_mask_b0,   eddy, [('out_file', 'in_mask')]),

            # Step 4: Bias correction
            # =======================
            (pre_mask_b0, bias, [('out_file',          'input_node.mask')]),
            (eddy,        bias, [('out_corrected',     'input_node.dwi'),
                                 ('out_rotated_bvecs', 'input_node.bvec')]),
            (init_node,   bias, [('bval',              'input_node.bval')]),

            # Step 5: Final brainmask
            # =======================
            # Compute average b0 on corrected dataset (for brain mask extraction)
            (init_node, compute_avg_b0, [('bval', 'in_bval')]),
            (bias,      compute_avg_b0, [('output_node.bias_corrected_dwi', 'in_dwi')]),
            # Compute b0 mask on corrected avg b0
            (compute_avg_b0, mask_avg_b0, [('out_b0_average', 'in_file')]),

            # Print end message
            (init_node,   print_end_message, [('image_id', 'image_id')]),
            (mask_avg_b0, print_end_message, [('mask_file', 'final_file')]),

            # Output node
            (init_node,   self.output_node, [('bval', 'preproc_bval')]),
            (eddy,        self.output_node, [('out_rotated_bvecs', 'preproc_bvec')]),
            (bias,        self.output_node, [('output_node.bias_corrected_dwi', 'preproc_dwi')]),
            (mask_avg_b0, self.output_node, [('mask_file', 'b0_mask')]),
            (bet_mag_fmap2b0, self.output_node, [('out_file', 'magnitude_on_b0')]),
            (fmap2b0,     self.output_node, [('out_file', 'calibrated_fmap_on_b0')]),
            (smoothing,   self.output_node, [('out_file', 'smoothed_fmap_on_b0')]),
        ])