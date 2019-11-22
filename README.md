# SubMARine

SubMARine is a polynomial-time algorithm that given a subclonal reconstruction finds its maximally ambiguous subclonal reconstruction (MAR). A subclonal reconstruction describes the evolutionary history of a cancer. Lineage frequencies and relationships are inferred and simple somatic mutations (SSMs; single nucleotide variants and small insertions and deletions) as well as copy number aberrations (CNAs) are assigned to the lineages. The MAR shows which lineage relationships are certain and which can be ambiguous in the solution set of phylogenetic trees with identical fit to the sequencing data as the input subclonal reconstruction. Furthermore, the MAR informs about certain and ambiguous phasing of SSMs relative to CNAs. When the input subclonal reconstruction contains no CNAs, lineage relationships do not need to be inferred for SubMARine.

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
