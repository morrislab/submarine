# SubMARine

SubMARine is a polynomial-time algorithm that buils partial clone trees and comes in a basic and extended version.
In the basic version, SubMARine takes a subclonal frequency matrix as input and builds the subMAR and a possible parent matrix.
The subMAR is a partial clone tree that represents all valid clone trees fitting the input frequencies.
The possible parent matrix indicates the possible parents of each clone subclone.
Additionally, SSMs and clonal CNAs can be provided as input.
The extended version can also work with subclonal CNAs.
Here, SubMARine needs a subclonal frequency matrix, CNAs as copy number changes assigned to subclones, segments and parental alleles, SSMs assigned to segments and subclones, and an impact matrix, which indicates which CNAs change the mutant copy numbers of which SSMs, as input.
SubMARine than reconstructs the extended subMAR, a partial clone tree with SSM phasing that represents all valid and equivalent clone trees fitting the input data, together with a possible parent matrix.

## Input files

<!---`parents_file`:
This file gives the parent of each cancerous lineage.
* `lineage`: index of the cancerous lineage. Given `K` lineages in total, the normal or healthy lineage has index `0`, the cancerous lineages have index `1` to `K-1`. All cancerous lineages need to appear in a row of this file.
* `parent`: index of parental lineage.--->

`freq_file`:
This file gives the subclonal frequencies across `N` tumor samples of the same patient.
* `lineage`: index of subclone.
* `frequencies`: tab-delimited frequencies across `N` tumor samples.
Note that the order of subclones implied by their indices has to be consistent with their frequencies. The lineages have to be sorted in decreasing order of their frequencies in the first sample, with frequency ties being broken by the subsequent samples.

<!---`cna_file`:
This file shows the assignment of CNAs to segments, lineages and phases. Furthermore, it indicates the type of the CNA and can also show the chromosome, start and end position.
* `seg_index`: segment index of CNA.
* `chr`: chromosome of CNA.
* `start`: start position of CNA.
* `end`: end position of CNA.
* `lineage`: index of lineage the CNA is assigned to.
* `phase`: allele the CNA is assigned to, can either be `A` or `B`.
* `change`: copy number change of CNA. A loss has `-1`, a gain has a value greater or equal than `1`.

`ssm_file`:
This file shows the assignment of SSMs to segments, lineages and eventually phases. Furthermore, it provides information whether an SSM is influenced by a copy number gain in the same lineage it is assigned to. The chromosome and position of the SSM can also be given.
* `seg_index`: segment index of SSM.
* `chr`: chromosome of SSM.
* `pos`: position of SSM.
* `lineage`: index of lineage the SSM is assigned to.
* `phase`: allele the SSM is assigned to, can be `A` or `B`. If the SSM is not phased, this is indicated by the value `0`.
* `cna_infl_same_lineage`: whether the SSM is influenced by a copy number gain that is assigned to the same lineage, segment and phase as the SSM. `1` if there exist such influence, `0` if not.

`seg_file`:
This file gives the number of segments as a single number.--->

Furthermore, the user can provide additional information which ancestral relationships <!---or SSM phases---> are required and are not allowed to be changed by SubMARine. <!---This information can be provided in the following two tab-delimited text files.--->

`userZ_file`:
This file can be provided by the user and indicates which ancestor-descendant relationships between subclones are not allowed to be changed by SubMARine.
* `ancestor`: index of subclone with lower index, thus higher frequency in the first sample.
* `descendant`: index of subclone with higher index, thus lower frequency in the first sample.
* `relationship`: whether subclone with lower index should be an ancestor of lineage with higher index, shown as `1`, or not, shown as `0`.

<!---`userSSM_file`:
This file can be provided by the user and indicates for which SSMs the phasing cannot be changed by SubMARine. Note that SSMs of one segment and phase of a lineage can be addressed only as whole and not individually.
* `seg_index`: segment index of SSMs whose phase should not be changed.
* `phase`: phase of SSMs that should not be changed.
* `lineage`: lineage index SSMs are assigned to whose phase should not be changed.--->

## Output files

SubMARine produces the following four output files.

`<my_file_name>.zmatrix`:
This file contains the ancestry matrix `Z` in a `json` list. If `Z[k][k'] = 1`, subclone `k` is an ancestor of subclone `k'`, if `Z[k][k'] = 0` subclone `k` is not an ancestor of subclone `k'`, and if `Z[k][k'] = -1`, subclone `k` could be an ancestor of subclone `k'`.

`<my_file_name>.pospars`:
This file contains the possible parent matrix `\tau` as a comma-separated text file. If subclone `k` is a possible parent of subclone `k'`, then `\tau[k'][k] = 1`, otherwise `\tau[k'][k] = 0`.

<!---`<my_file_name>_ssms.csv`:
This file contains the eventually updated phasing information for each SSM in the same format as the required input file `ssm_file`.--->

`<my_file_name>.lineage.json`:
This file contains the sorted subclones of the built partial clone tree in `json` format. Each subclone contains a list of SSMs assigned to allele `A` (`ssms_a`) or `B` (`ssms_b`) or being unphased (`ssms`), a list of indices of all descendant subclone (`sublins`), a list with CNAs assigned to allele `A` (`cnvs_a`) or `B` (`cnvs_b`), and a list with the frequencies of the current subclone in all samples (`freq`).

<!---For each SSM, the following information is given: the lineage it is assigned to (`lineage`), whether it is influenced by a copy number gain in the same lineage (`infl_cnv_same_lin`), its position on the chromosome (`pos`), its reference count (`ref_count`, we don't need this information in the context of SubMARine, thus the value is `-1`), its chromosome (`chr`), its variant count (`variant_count`, we don't need this information in the context of SubMARine, thus the value is `-1`), its phase (`phase`, with phase `A` being `0`, phase `B` being `1` and unphased being `2`), and its segment index (`seg_index`).

For each CNA, the following information is given: the lineage it is assigned to (`lineage`), its start position (`start`), its chromosome (`chr`), its end position (`end`), its phase (`phase`, with phase `A` being `0` and phase `B` being `1`), its segment index (`seg_index`), and its relative copy number change (`change`).-->

## Running SubMARine

To run SubMARine on the provided test files type:
```
python3 submarine.py --parents_file submarine_example/ex1_parents.csv 
  --freq_file submarine_example/ex1_frequencies.csv 
  --cna_file submarine_example/ex1_cnas.csv --ssm_file submarine_example/ex1_ssms.csv 
  --seg_file submarine_example/ex1_seg_num.csv --output_prefix my_test 
  --userZ_file submarine_example/ex1_userZ.csv 
  --userSSM_file submarine_example/ex1_userSSMs.csv --overwrite
```

To start the depth-first search type:
```
python3 submarine.py --dfs --lineage_file my_test.lineage.json 
  --seg_file submarine_example/ex1_seg_num.csv --z_matrix_file my_test.zmatrix 
  --output_prefix my_test--write_trees_to_file
```
