# `statistics-surface` - Surface-based mass-univariate analysis with SurfStat

This command performs statistical analysis (e.g. group comparison, correlation) on surface-based features using the general linear model (GLM). To that aim, the pipeline relies on the Matlab toolbox [SurfStat](http://www.math.mcgill.ca/keith/surfstat/) designed for statistical analyses of univariate and multivariate surface and volumetric data using the GLM [[Worsley et al., 2009](http://dx.doi.org/10.1016/S1053-8119(09)70882-1)].

Surface-based measurements are analyzed on the FsAverage surface template (from FreeSurfer).

Currently, this pipeline can handle cortical thickness measurements from T1 images [`t1-freesurfer` pipeline](../T1_FreeSurfer) or map of activity from PET data using [`pet-surface` pipeline](../PET_Surface).

!!! note
    We are aware that the [SurfStat](http://www.math.mcgill.ca/keith/surfstat/) toolbox is not maintained anymore. The reasons why we rely on it are: 1) its great flexibility; 2) our profound admiration for the late Keith Worsley.

## Prerequisites
You need to process your data with the [`t1-freesurfer` pipeline](../T1_FreeSurfer) for measurements of cortical thickness measurements from T1 images or [`pet-surface` pipeline](../PET_Surface) for measurements of activity map from PET.

Do not hesitate to have a look at the paragraph **Specifying what surface data to use** if you want to use your own surface feature

## Dependencies
<!--If you installed the docker image of Clinica, nothing is required.-->

If you only installed the core of Clinica, this pipeline needs the installation of **Matlab** and **FreeSurfer 6.0** on your computer. You can find how to install these software packages on the [third-party](../../Third-party) page. Note that the Matlab `Statistics and Machine Learning Toolbox` is required.

!!! bug "Compatibility issue with Matlab R2019 / R2020"
    It has been reported that newer versions of Matlab (see details on [GitHub](https://github.com/aramis-lab/clinica/issues/90)) were not compatible with this pipeline. For the moment, we advise you to use at best R2018b version. Matlab versions between 2015 and 2018 are known to work.


## Running the pipeline

The pipeline can be run with the following command line:
```Text
clinica run statistics-surface  <caps_directory> <subject_visits_with_covariates_tsv> <design_matrix> <contrast> <string_format> <group_label> <glm_type>
```
where:

- `caps_directory` is the folder containing the results of the [`t1-freesurfer`](../T1_FreeSurfer) or [`pet-surface`](../PET_Surface) pipeline and the output of the present command, both in a [CAPS hierarchy](../../CAPS/Introduction).
- `subject_visits_with_covariates_tsv` is a TSV file containing a list of subjects with their sessions and all the covariates and factors in your model (the content of the file is explained in the [Example](../Stats_Surface/#comparison-analysis) subsection).
- `design_matrix` is a string defining the model that fits into the GLM, e.g. `1 + group + sex + age` where `group`, `sex` and `age` correspond to the names of columns in the TSV file provided.
- `contrast` is a string defining the contrast matrix or the variable of interest for the GLM, e.g. `group` or `age`.
- `string_format` is a string defining the format of the columns in the TSV file. For example, the columns contain a string, a string and a number (e.g. `participant_id`, `session_id` and `age`), then you will need to replace `string_format` by `%s %s %f`, meaning that the columns of your TSV file contain a `s`tring, a `s`tring and a `f`loat.
- `group_label` is a string defining the group label for the current analysis which helps you keep track of different analyses.
- `glm_type` is a string defining the type of analysis of your model, choose one between `group_comparison` and `correlation`.

By default, the pipeline will try to run the analysis using the cortical thickness generated by the `t1-freesurfer` pipeline. Add the `--feature_type pet_fdg_projection` option to run the analyses on PET data generated by the `pet-surface` pipeline.

!!! tip
    Check the [Example](../Stats_Surface/#comparison-analysis) subsection for further clarification.

## Outputs

### Group comparison analysis
Results are stored in the following folder of the [CAPS hierarchy](../../CAPS/Specifications/#group-comparison): `groups/group-<group_label>/statistics/surfstat_group_comparison/`.

The main outputs for the group comparison are:

- `group-<group_label>_<group_1>-lt-<group_2>_measure-<label>_fwhm-<label>_correctedPValue.jpg`: contains both the cluster level and the vertex level corrected p-value maps, based on random field theory.
- `group-<group_label>_<group_1>-lt-<group_2>_measure-<label>_fwhm-<label>_FDR.jpg`: contains corrected p-value maps, based on the false discovery rate (FDR).
- `group-<group_label>_participants.tsv` is a copy of the `subject_visits_with_covariates_tsv` parameter.
- `group-<group_label>_glm.json` is a JSON file containing all the model information of the analysis (i.e. what you wrote on the command line).

The `<group_1>-lt-<group_2>` means that the tested hypothesis is: "the measurement of `<group_1>` is lower than (lt) the measurement of `<group_2>`". The pipeline includes both contrasts so `*<group_2>-lt-<group_1>*` files are also saved.

The value for FWHM corresponds to the size of the surface-based smoothing in mm and can be 5, 10, 15, 20.

Analysis with cortical thickness (respectively FDG-PET data) will be saved under the `_measure-ct` keyword (respectively the `_measure-fdg` keyword).

!!! tip
    See the Example subsection for further clarification.


### Correlations analysis
Results are stored in the following folder of the [CAPS hierarchy](../../CAPS/Specifications/#correlation-analysis): `groups/group-<group_label>/statistics/surfstat_correlation/`.

The main outputs for the correlation are:

- `group-<group_label>_correlation-<label>_contrast-<label>_measure-<label>_fwhm-<label>_correctedPValue.jpg`: contains p-value maps corrected at both the cluster and vertex levels, based on random field theory.
- `group-<group_label>_correlation-<label>_contrast-<label>_measure-<label>_fwhm-<label>_FDR.jpg`: contains corrected p-value maps, based on the false discovery rate (FDR).
- `group-<group_label>_correlation-<label>_contrast-<label>_measure-<label>_fwhm-<label>_T-statistics.jpg`: contains the maps of T statistics.
- `group-<group_label>_correlation-<label>_contrast-<label>_measure-<label>_fwhm-<label>_Uncorrected p-value.jpg`: contains the maps of uncorrected p-values.
- `group-<group_label>_participants.tsv` is a copy of `subject_visits_with_covariates_tsv`.
- `group-<group_label>_glm.json` is a JSON file summarizing the parameters of the analysis.

The `correlation-<label>` describes the factor of the model, which can be for example `age`. The `contrast-<label>` is the sign of your factor which can be `negative` or `positive`.

Analysis with cortical thickness (respectively FDG-PET data) will be saved under the `_measure-ct` keyword (respectively the `_measure-fdg` keyword).


!!! note
    The full list of output files can be found in the [ClinicA Processed Structure (CAPS) specifications](../../CAPS/Specifications/#statistics-surface-surface-based-mass-univariate-analysis-with-surfstat).

<!--### GLM-->
<!--@TODO-->


## Example
### Comparison analysis
Let's assume that you want to perform a group comparison between patients with Alzheimer’s disease (`group_1` will be called `AD`) and healthy subjects (`group_2` will be called `HC`). `ADvsHC` will define the `group_label`.

The TSV file containing the participants and covariates will look like this:
```
participant_id	session_id 	sex   	group 	age
sub-CLNC0001  	ses-M00    	Female	CN    	71.1
sub-CLNC0002  	ses-M00    	Male  	CN    	81.3
sub-CLNC0003  	ses-M00    	Male  	CN    	75.4
sub-CLNC0004  	ses-M00    	Female	CN    	73.9
sub-CLNC0005  	ses-M00    	Female	AD    	64.1
sub-CLNC0006  	ses-M00    	Male  	AD    	80.1
sub-CLNC0007  	ses-M00    	Male  	AD    	78.3
sub-CLNC0008  	ses-M00    	Female	AD    	73.2
```
Note that to make the display clearer, the rows contain successive tabs, which should not happen in an actual BIDS TSV file.

We call this file `ADvsHC_participants.tsv`. The columns of the TSV file contains consecutively `s`trings, `s`trings, `s`trings, `s`trings and `f`loat (i.e. numbers). The `string_format` is therefore `%s %s %s %s %f`.

Our linear model formula will be: `CorticalThickness = 1 + age + sex + group`. In this linear model, the `age` and `sex` are the covariates, and `group` is the contrast. Please note that all these variables should correspond to the names of the columns in the `ADvsHC_participants.tsv` file.

Finally, the command line is:
```Text
clinica run statistics-surface caps_directory ADvsHC_participants.tsv "1 + age + sex + group" "group" "%s %s %s %s %f" group_comparison
```

The parameters of the command line are stored in the `group-ADvsHC_glm.json` file:
```javascript
{
"DesignMatrix": "1 + age + sex + group"
"StringFormatTSV": "%s %f %f"
"Contrast": "group"
"ClusterThreshold": 0.001
}
```

The results of the group comparison between AD and HC are given by the `group-ADvsHC_AD-lt-HC_measure-ct_fwhm-20_correctedPValue.jpg` file and is illustrated as follows:
![](../../img/StatsSurfStat_images/ContrastNegative-CorrectedPValue.jpg)
*<center>Visualization of corrected p-value map.</center>*

The blue area corresponds to the vertex-based corrected p-value and the yellow area represents the cluster-based corrected p-value.

### Correlation analysis
Let's now assume that you are interested in knowing whether cortical thickness is correlated with age using the same population as above, namely `ADvsHC_participants.tsv`. The string format of the TSV file does not change (i.e. `"%s %s %s %s %f"`). The contrast will become `age` and we will choose `correlation` instead of `group_comparison`.

Finally, the command line is simply:
```Text
clinica run statistics-surface caps_directory ADvsHC_participants.tsv "1 + age + sex + group" "age" "%s %s %s %s %f" correlation
```
## Describing this pipeline in your paper

!!! cite "Example of paragraph:"
    Theses results have been obtained using the `statistics-surface` command of Clinica [[Routier et al](https://hal.inria.fr/hal-02308126/)]. More precisely, a point-wise, vertex-to-vertex model based on the Matlab SurfStat toolbox (http://www.math.mcgill.ca/keith/surfstat/) was used to conduct a group comparison of whole brain cortical thickness. The data were smoothed using a Gaussian kernel with a full width at half maximum (FWHM) set to `<FWHM>` mm. The general linear model was used to control for the effect of `<covariate_1>`, ... and  `<covariate_N>`. Statistics were corrected for multiple comparisons using the random field theory for non-isotropic images [[Worsley et al., 1999](http://dx.doi.org/10.1002/(SICI)1097-0193(1999)8:2/3<98::AID-HBM5>3.0.CO;2-F)]. A statistical threshold of P < `<ClusterThreshold>` was first applied (height threshold). An extent threshold of P < 0.05 corrected for multiple comparisons was then applied at the cluster level..

!!! tip
    Easily access the papers cited on this page on [Zotero](https://www.zotero.org/groups/2240070/clinica_aramislab/items/collectionKey/U2APQD82).


## Support

-   You can use the [Clinica Google Group](https://groups.google.com/forum/#!forum/clinica-user) to ask for help!
-   Report an issue on [GitHub](https://github.com/aramis-lab/clinica/issues).


## (Advanced) Specifying what surface data to use

If you run the help command line `clinica run statistics-surface -h`, you will find 2 optional flags that we will describe :

- `--feature_type FEATURE_TYPE` allows you to decide what feature type to take for your analysis. If it is `cortical_thickness` (default value), the thickness file for each hemisphere and each subject and session of the tsv file will be used. Keep in mind that those thickness files are generated using the `t1-freesurfer` pipeline, so be sure to have run it before using it! Other directly-implemented solutions are present but they are not yet released.
- The other flag `--custom_file CUSTOM_FILE` allows to specify yourself what file should be taken in the `CAPS/subjects` directory. `CUSTOM_FILE` is a string describing the folder hierarchy to find the file. For instance, let's say we want to manually indicate to use the cortical thickness. Here is the generic link to the surface data files.

`CAPS/subjects/sub-*/ses-M*/t1/freesurfer_cross_sectional/sub-*_ses-M*/surf/*h.thickness.fwhm*.fsaverage.mgh`

(Example: `CAPS/subjects/sub-ADNI011S4075/ses-M00/t1/freesurfer_cross_sectional/sub-ADNI011S4075_ses-M00/surf/lh.thickness.fwhm15.fsaverage.mgh`)

Note that the file must be in the `CAPS/subjects` directory. So my `CUSTOM_STRING` must only describe the path starting after the `subjects` folder. So now, we just need to replace the `*` by the correct keywords, in order for the pipeline to catch the correct filenames. `@subject` is the subject, `@session` the session, `@hemi` the hemisphere, `@fwhm` the full width at half maximum. All those variables are already known, you just need to indicate where they are in the filename!

As a result, we will get for `CUSTOM_FILE` of cortical thickness :
`@subject/@session/t1/freesurfer_cross_sectional/@subject_@session/surf/@hemi.thickness.fwhm@fwhm.fsaverage.mgh`

You will finally need to define the name your surface feature `--feature_label FEATURE_LABEL`. It will appear in the `_measure-<FEATURE_LABEL>` of the output files once the pipeline has run.

Note that `--custom_file` and `--feature_type` cannot be combined.

## Appendix
* For more information about **SurfStat**, please check [here](http://www.math.mcgill.ca/keith/surfstat/).
* For more information about the **GLM**, please check [here](https://en.wikipedia.org/wiki/Generalized_linear_model).
* The cortical thickness map is obtained from the FreeSurfer segmentation. More precisely, it corresponds to the subject’s map normalized onto FSAverage and smoothed using a Gaussian kernel FWHM of `<fwhm>` mm (the `surf/?h.thickness.fwhm<fwhm>.fsaverage.mgh` files).
