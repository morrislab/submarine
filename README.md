# SubMARine

SubMARine is a polynomial-time algorithm that given a subclonal reconstruction finds its maximally ambiguous subclonal reconstruction (MAR). A subclonal reconstruction describes the evolutionary history of a cancer. Lineage frequencies and relationships are inferred and simple somatic mutations (SSMs; single nucleotide variants and small insertions and deletions) as well as copy number aberrations (CNAs) are assigned to the lineages. The MAR shows which lineage relationships are certain and which can be ambiguous in the solution set of phylogenetic trees with identical fit to the sequencing data as the input subclonal reconstruction. Furthermore, the MAR informs about certain and ambiguous phasing of SSMs relative to CNAs. When the input subclonal reconstruction contains no CNAs, lineage relationships do not need to be inferred for SubMARine.

## Input files

Different tools exist that build subclonal reconstructions with SSMs and CNAs. To use these subclonal reconstructions as input for SubMARine, their information has to be parsed into the following five tab-delimited text files.

`parents_file`:
This file gives the parent of each cancerous lineage.
* `lineage`: index of the cancerous lineage. Given `K` lineages in total, the normal or healthy lineage has index `0`, the cancerous lineages have index `1` to `K-1`. All cancerous lineages need to appear in a row of this file.
* `parent`: index of parental lineage.

`freq_file`:
This file gives the lineage frequencies across `N` tumor samples of the same patient.
* `lineage`: index of cancerous lineage.
* `frequencies`: tab-delimited frequencies across `N` tumor samples.
Note that the order of lineages implied by their indices has to be consistent with their frequencies. The lineages have to be sorted in decreasing order of their frequencies in the first sample, with frequency ties being broken by the subsequent samples.

`cna_file`:
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
This file gives the number of segments as a single number.

Furthermore, the user can provide additional information which lineage relationships or SSM phases are certain and are not allowed to be changed by SubMARine. This information can be provided in the following two tab-delimited text files.

`userZ_file`:
This file can be provided by the user and indicates which ancestor-descendant relationships between lineages are not allowed to be changed by SubMARine.
* `ancestor`: index of lineage with lower index, thus higher frequency in the first sample.
* `descendant`: index of lineage with higher index, thus lower frequency in the first sample.
* `relationship`: whether lineage with lower index should be an ancestor of lineage with higher index, shown as `1`, or not, shown as `0`.

`userSSM_file`:
This file can be provided by the user and indicates for which SSMs the phasing cannot be changed by SubMARine. Note that SSMs of one segment and phase of a lineage can be addressed only as whole and not individually.
* `seg_index`: segment index of SSMs whose phase should not be changed.
* `phase`: phase of SSMs that should not be changed.
* `lineage`: lineage index SSMs are assigned to whose phase should not be changed.

## Output files

SubMARine produces the following four output files.

`<my_file_name>.zmatrix`:
This file contains the ancestry matrix `Z` in a `json` list. If `Z[k][k'] = 1`, lineage `k` is an ancestor of lineage `k'`, if `Z[k][k'] = 0` lineage `k` is not an ancestor of lineage `k'`, and if `Z[k][k'] = ?`, lineage `k` could be an ancestor of lineage `k'`.

`<my_file_name>.pospars`:
This file contains the possible parent matrix `\tau` as a comma-separated text file. If lineage `k` is a possible parent of lineage `k'`, then `\tau[k'][k] = 1`, otherwise `\tau[k'][k] = 0`.

`<my_file_name>_ssms.csv`:
This file contains the eventually updated phasing information for each SSM in the same format as the required input file `ssm_file`.

`<my_file_name>.lineage.json`:
This file contains the sorted lineages of the built subclonal reconstruction in `json` format. Each lineage contains a list of SSMs assigned to allele `A` (`ssms_a`) or `B` (`ssms_b`) or being unphased (`ssms`), a list of indices of all descendant lineages (`sublins`), a list with CNAs assigned to allele `A` (`cnvs_a`) or `B` (`cnvs_b`), and a list with the frequencies of the current lineage in all samples (`freq`).

For each SSM, the following information is given: the lineage it is assigned to (`lineage`), whether it is influenced by a copy number gain in the same lineage (`infl_cnv_same_lin`), its position on the chromosome (`pos`), its reference count (`ref_count`, we don't need this information in the context of SubMARine, thus the value is `-1`), its chromosome (`chr`), its variant count (`variant_count`, we don't need this information in the context of SubMARine, thus the value is `-1`), its phase (`phase`, with phase `A` being `0`, phase `B` being `1` and unphased being `2`), and its segment index (`seg_index`).

For each CNA, the following information is given: the lineage it is assigned to (`lineage`), its start position (`start`), its chromosome (`chr`), its end position (`end`), its phase (`phase`, with phase `A` being `0` and phase `B` being `1`), its segment index (`seg_index`), and its relative copy number change (`change`).

## Running SubMARine

To run SubMARine on the provided test files type:
```
python2 submarine.py --parents_file submarine_example/ex1_parents.csv 
  --freq_file submarine_example/ex1_frequencies.csv 
  --cna_file submarine_example/ex1_cnas.csv --ssm_file submarine_example/ex1_ssms.csv 
  --seg_file submarine_example/ex1_seg_num.csv --output_prefix my_test 
  --userZ_file submarine_example/ex1_userZ.csv 
  --userSSM_file submarine_example/ex1_userSSMs.csv --overwrite
```

To start the depth-first search type:
```
python2 submarine.py --dfs --lineage_file my_test.lineage.json 
  --seg_file submarine_example/ex1_seg_num.csv --z_matrix_file my_test.zmatrix 
  --output_prefix my_test--write_trees_to_file
```
