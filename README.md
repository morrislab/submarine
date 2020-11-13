# SubMARine

SubMARine is a polynomial-time algorithm that reconstructs cancer evolutionary histories by building partial clone trees.
It comes in a basic and extended version.
In the basic version, SubMARine takes a subclonal frequency matrix as input and builds the subMAR and a possible parent matrix.
The subMAR is a partial clone tree that represents all valid clone trees fitting the input frequencies.
It is an approximation to the MAR: the maximally-constraint ancestral reconstruction.
The possible parent matrix indicates the possible parents of each subclone.
Additionally, SSMs and clonal CNAs can be provided as input.
The extended version can also work with subclonal CNAs.
Here, SubMARine needs as input a subclonal frequency matrix, CNAs as copy number changes assigned to subclones, genome segments and parental alleles, SSMs assigned to genome segments and subclones, and an impact matrix, which indicates which CNAs change the mutant copy numbers of which SSMs.
SubMARine than reconstructs the extended subMAR, a partial clone tree with SSM phasing that represents all valid and equivalent clone trees fitting the input data, together with a possible parent matrix.

SubMARine assume precisely measured subclonal frequencies.
In order to deal with subclonal frequencies inferred from noisy mutational frequencies, a noise-buffered version is offered.

## Input files

<!---`parents_file`:
This file gives the parent of each cancerous lineage.
* `lineage`: index of the cancerous lineage. Given `K` lineages in total, the normal or healthy lineage has index `0`, the cancerous lineages have index `1` to `K-1`. All cancerous lineages need to appear in a row of this file.
* `parent`: index of parental lineage.--->

`freq_file`:
This file gives the subclonal frequencies across `N` tumor samples of the same patient.
* `subclone_ID`: ID of subclone, has to be a number<br>
If two subclones have the same average subclonal frequencies across all samples, they are sorted according to their IDs.
* `frequencies`: tab-delimited frequencies across `N` tumor samples

Note that the germline is not part of this frequency file.

`cna_file`:
This file shows the assignment of CNAs to segments, subclones and phases. Furthermore, it indicates the type of the CNA.
* `CNA_index`: index of CNA, must be consecutive, starting at `0`
* `seg_index`: segment index of CNA
* `subclone_ID`: ID of subclone the CNA is assigned to
* `phase`: allele the CNA is assigned to, can either be `A` or `B`
* `change`: copy number change of CNA, a loss has `-1`, a gain has a value greater than or equal to `1`

`ssm_file`:
This file shows the assignment of SSMs to segments and subclones. 
* `SSM_index`: index of SSM, must be consecutive, starting at `0`
* `seg_index`: segment index of SSM
* `subclone_ID`: ID of subclone the SSM is assigned to

`impact_file`:
This file shows which CNAs impact the mutant copy numbers of which SSMs. 
* `SSM_index`: index of SSM that is impacted by CNA in same line
* `CNA_index`: index of CNA that impacts SSM in same line


Furthermore, the user can provide additional information which ancestral relationships or SSM phases are required and are not allowed to be changed by SubMARine. This information can be provided in the following two tab-delimited text files.

`userZ_file`:
This file can be provided by the user and indicates which ancestor-descendant relationships between subclones are not allowed to be changed by SubMARine.
* `ancestor_ID`: ID of first subclone
* `descendant_ID`: ID of second subclone
* `relationship`: whether the first subclone should be an ancestor or the second subclone, shown as `1`, or not, shown as `0`

`userSSM_file`:
This file can be provided by the user and indicates for which SSMs the phasing cannot be changed by SubMARine.
* `SSM_index`: index of SSMs whose phase should not be changed
* `phase`: phase of SSMs that should not be changed, can either be `A` or `B`

## Output files

SubMARine produces the following four output files.

`<my_file_name>.log`:
A log file.
Amongst other information, it contains a mapping from the subclone IDs to the subclone indices used internally by SubMARine and in all output files.

`<my_file_name>.zmatrix`:
This file contains the ancestry matrix `Z` in a `json` list. If `Z[k][k'] = 1`, subclone `k` is an ancestor of subclone `k'`, if `Z[k][k'] = 0` subclone `k` is not an ancestor of subclone `k'`, and if `Z[k][k'] = -1`, subclone `k` could be an ancestor of subclone `k'`.

`<my_file_name>.pospars`:
This file contains the possible parent matrix `\tau` as a comma-separated text file. If subclone `k` is a possible parent of subclone `k'`, then `\tau[k'][k] = 1`, otherwise `\tau[k'][k] = 0`.

`<my_file_name>.lineage.json`:
This file contains the sorted subclones of the built partial clone tree in `json` format. Each subclone contains a list of SSMs assigned to allele `A` (`ssms_a`) or `B` (`ssms_b`) or being unphased (`ssms`), a list of indices of all descendant subclones (`sublins`), a list with CNAs assigned to allele `A` (`cnvs_a`) or `B` (`cnvs_b`), and a list with the frequencies of the current subclone in all samples (`freq`).
<br>
For each present SSM, the following information is given: the subclone it is assigned to (`lineage`), its phase (`phase`, with phase `A` being `0`, phase `B` being `1` and unphased being `2`), its segment index (`seg_index`), and its own index (`index`). The other information (`infl_cnv_same_lin`, `pos`, `chr`, `ref_count`, `variant_count`) are not needed in the context of SubMARine and are thus set to `-1`.
<br>
For each present CNA, the following information is given: the subclone it is assigned to (`lineage`), its phase (`phase`, with phase `A` being `0` and phase `B` being `1`), its segment index (`seg_index`), its own index (`index`) and its relative copy number change (`change`). The other information (`start`, `chr`, `end`) are not needed in the context of SubMARine and are thus set to `-1`.

If SubMARine is applied in extended mode, it also produces the following output file.

`<my_file_name>.ssm_phasing`:
This file contains the phasing information for each SSM. The first column gives the SSM index and the second the phase. Note that `0` means unphased. 

If SubMARine is applied and instructed to account for noise in the subclonal frequencies, it also produces the following output file.

`<my_file_name>.noisebuffer`: 
This file contains the noise buffer set used to find a valid partial clone tree.
The noise buffer of subclone `k'` in sample `n` is added to the frequency of sample `n` of possible ancestors and parents when using the crossing rule and the Subpoplar algorithm.

## Output files of depth-first search

We provide a depth-first search that enumerates all valid and equivalent clone trees completing a partial clone tree. It produces the following three output files.

`<my_file_name>.dfs.log`: A log file containing information about how many trees were considered and how many are valid.

`<my_file_name>.valid_count.txt`: A text file containing the number of valid trees.

`<my_file_name>.ambiguity.txt`: A text file with an analysis of undefined ancestral relationships in the partial clone tree. If the partial clone tree contains no undefined relationships, the text file reads `All ancestral relationships are defined.`. Otherwise, if the MAR was provided as input or if the provided subMAR equals its MAR, the text file reads `True` because all undefined relationships are truely ambiguous. Otherwise the text file provides an analysis of all undefined relationships that take only one defined value in all valid and equivalent clone trees in the following format: `False \t subclone k \t subclone k' \t whether k is an ancestor of k' (with 1 for yes and 0 for no) \t whether k is not an ancestor of k' (with 1 for yes and 0 for no)`

## Output files of modified depth-first search to find subclone- and sample-specific noise buffer set

If the subclone- and sample-specific noise buffer set cannot be found in polynomial time, it can be found with a modified version of the depth-first search.
This version also produces the two files `<my_file_name>.dfs.log` and `<my_file_name>.valid_count.txt`.
The file `<my_file_name>.ambiguity.txt`is not generated because in order to find the subclone- and sample-specific noise buffer set, the MAR is produced and hence all uncertain ancestral relationships are ambiguous.
The MAR and other information are contained in the following four output files:

`<my_file_name>.zmatrix.MAR`: This file contains the ancestral matrix `Z` of the MAR in a comma-separated format.

`<my_file_name>.pospars.MAR`: This file contains the possible parent matrix `\tau` belonging to the MAR in the same format as `<my_file_name>.pospars`.

`<my_file_name>.noisebuffer.MAR`: This file contains the subclone- and sample-specific noise buffer set in the same format as `<my_file_name>.noisebuffer`.

`<my_file_name>.negfreqs.MAR`: This file contains the amount of negative available frequencies used by the MAR-completing clone trees.

## Running SubMARine

To run SubMARine on the provided test files in basic version, type:
```
python3 submarine.py --basic_version --freq_file submarine_example/frequencies2.csv 
  --userZ_file submarine_example/userZ.csv --output_prefix submarine_example/my_test_basic 
```

To run SubMARine on the provided test files in basic version with SSMs and clonal CNAs, type:
```
python3 submarine.py --basic_version --freq_file submarine_example/frequencies2.csv 
  --userZ_file submarine_example/userZ.csv --cna_file submarine_example/cnas4.csv 
  --ssm_file submarine_example/ssms4.csv --output_prefix submarine_example/my_test_basic_clonalCNAs
```

To run SubMARine on the provided test files in extended version, type:
```
python3 submarine.py --extended_version --freq_file submarine_example/frequencies3.csv 
  --cna_file submarine_example/cnas3.csv --ssm_file submarine_example/ssms3.csv 
  --impact_file submarine_example/impact3.csv --userZ_file submarine_example/userZ_3.csv 
  --userSSM_file submarine_example/userSSM3.csv --output_prefix submarine_example/my_test_extended
```

To run SubMARine accounting for noise in the subclonal frequencies, add `--allow_noise` to your command, e.g.:
```
python3 submarine.py --basic_version --freq_file submarine_example/frequencies4.csv 
  --allow_noise --output_prefix submarine_example/my_test_noise
```
The log file informs whether a subclone-and sample-specific noise buffer set could be found in polynomial time. For the above sample this is the case. Here is another example where this is not the case:
```
python3 submarine.py --basic_version --freq_file submarine_example/frequencies5.csv 
  --allow_noise --output_prefix submarine_example/my_test_noise_2
```
In order to find the subclone-and sample-specific noise buffer set, the modified depth-first search can be used:
```
python3 submarine.py --find_best_noise_buffer 
  --possible_parent_file submarine_example/my_test_noise_2.pospars 
  --z_matrix_file submarine_example/my_test_noise_2.zmatrix 
  --lineage_file submarine_example/my_test_noise_2.lineage.json 
  --noise_buffer_file submarine_example/my_test_noise_2.noisebuffer 
  --output_prefix submarine_example/my_test_noise_2 
```
When CNAs and SSMs are given, their files need to be added with `--cna_file` and the `--ssm_file` options.

To start the depth-first search to find the number of valid completing clone trees and see whether all uncertain entries in the subMAR are truely ambiguos, type:
```
python3 submarine.py --dfs --possible_parent_file submarine_example/my_test_extended.pospars 
  --z_matrix_file submarine_example/my_test_extended.zmatrix 
  --lineage_file submarine_example/my_test_extended.lineage.json 
  --cna_file submarine_example/cnas3.csv --ssm_file submarine_example/ssms3.csv 
  --output_prefix submarine_example/my_test_extended 
```
When no CNAs and SSMs are given, simply do not use the `--cna_file` and the `--ssm_file` options.

In order to compute the upper bound on the number of valid (sub)MAR-completing clone trees type:
```
python3 submarine.py --upper_bound --possible_parent_file submarine_example/my_test_noise_2.pospars
```
If no possible parent file should be present for a dataset, the ancestry matrix `Z` can also be used:
```
python3 submarine.py --upper_bound --z_matrix_file submarine_example/my_test_noise_2.zmatrix
```
