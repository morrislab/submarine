# SubMARine

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
python submarine.py --dfs --lineage_file my_test.lineage.json 
  --seg_file submarine_example/ex1_seg_num.csv --z_matrix_file my_test.zmatrix 
  --output_prefix my_test--write_trees_to_file
```
