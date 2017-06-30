# NeuroBind--Yet Another Model for Finding Binding Sites Using Neural Networks

Finding binding sites is important because they are associated with cell processes such as transcription, translation, repair, etc. In this research I apply a deep neural network model on Protein Binding Microarrays for two popular transcription factors, i.e., CEH-22 and Oct-1. Spearman rank correlation coefficients of our models are much higher than those of any other models.    

## Glossary
_From http://www.biosyn.com/bioinformatics.aspx_
* Assay: A method for measuring a biological activity. This may be enzyme activity, binding affinity, or protein turnover. Most assays utilize a measurable parameter such as color, fluorescence or radioactivity to correlate with the biological activity.
* DNA (deoxyribonucleic acid): The chemical that forms the basis of the genetic material in virtually all organisms. DNA is composed of the four nitrogenous bases Adenine, Cytosine, Guanine, and Thymine, which are covalently bonded to a backbone of deoxyribose - phosphate to form a DNA strand. Two complementary strands (where all Gs pair with Cs and As with Ts) form a double helical structure which is held together by hydrogen bonding between the cognate bases. 
* DNA microarrays: The deposition of oligonucleotides or cDNAs onto an inert substrate such as glass or silicon. Thousands of molecules may be organized spatially into a high-density matrix. These DNA chips may be probed to allow expression monitoring of many thousands of genes simultaneously. Uses include study of polymorphisms in genes, de novo sequencing or molecular diagnosis of disease.
* DNA polymerase: An enzyme that catalyzes the synthesis of DNA from a DNA template given the deoxyribonucleotide precursors.
* DNA probes: Short single stranded DNA molecules of specific base sequence, labeled either radioactively or immunologically, that are used to detect and identify the complementary base sequence in a gene or genome by hybridizing specifically to that gene or sequence. 
* Gene expression: The conversion of information from gene to protein via transcription and translation. 
* High-throughput screening: The method by which very large numbers of compounds are screened against a putative drug target in either cell-free or whole-cell assays. Typically, these screenings are carried out in 96 well plates using automated, robotic station based technologies or in higher- density array ("chip") formats. 
* Motif: A conserved element of a protein sequence alignment that usually correlates with a particular function. Motifs are generated from a local multiple protein sequence alignment corresponding to a region whose function or structure is known. It is sufficient that it is conserved, and is hence likely to be predictive of any subsequent occurrence of such a structural/functional region in any other novel protein sequence. 
* Mutation: An inheritable alteration to the genome that includes genetic (point or single base) changes, or larger scale alterations such as chromosomal deletions or rearrangements. 
* Nucleotide: A nucleic acid unit composed of a five carbon sugar joined to a phosphate group and a nitrogen base. 
* Primer: A short oligonucleotide that provides a free 3í hydroxyl for DNA or RNA synthesis by the appropriate polymerase (DNA polymerase or RNA polymerase). 
* Probe: Any biochemical that is labeled or tagged in some way so that it can be used to identify or isolate a gene, RNA, or protein. 
* Regulatory gene: A DNA sequence that functions to control the expression of other genes by producing a protein that modulates the synthesis of their products (typically by binding to the gene promoter). (cf. Structural gene). 
* Replication: The synthesis of an informationally identical macromolecule (e.g. DNA) from a template molecule.
* Transcription: The assembly of complementary single-stranded RNA on a DNA template.
* Transcription factors: A group of regulatory proteins that are required for transcription in eukaryotes. Transcription factors bind to the promoter region of a gene and facilitate transcription by RNA polymerase. 

## Model Description
 * Prenet: Two fully connected layers with dropouts
 * Conv1d Banks: Banks of convolution layers of width from 1 to 16
 * Position-wise convolutions
 * Highway nets
 * Final Fully Connected layer

## Data
[The UniProbe PBM data](http://thebrain.bwh.harvard.edu/pbms/webworks_pub/index.php) are used for training and evaluation.
My dataset scheme follows that of [DeeperBind: Enhancing Prediction of Sequence Specificities of DNA Binding Proteins](https://arxiv.org/pdf/1611.05777.pdf).

## Requirements
 * numpy >= 1.11.1
 * TensorFlow >= 1.2
 * scipy == 0.19.0
 * tqdm >= 4.14.0
 * matplotlib >= 1.5.3


## File description

 * `hyperparams.py` includes all hyper parameters that are needed.
 * `data_load.py` loads data and put them in queues.
 * `modules.py` contains building blocks for the network.
 * `train.py` is for training.
 * `validation_checkk.py` is for validation check
 * `test.py` is for the final evaluation on test set.

## Training
  * STEP 0. Make sure you meet the requirements.
  * STEP 1. Download PBM data of [CEH-22](http://thebrain.bwh.harvard.edu/pbms/webworks_pub/academic-license.php?file=NBT06/CEH-22/CEH-22.zip) or [Oct-1](http://thebrain.bwh.harvard.edu/pbms/webworks_pub/academic-license.php?file=NBT06/Oct-1/Oct-1.zip).
  * STEP 2. Adjust hyper parameters in `hyperparams.py` if necessary.
  * STEP 3. Run `train.py` or download the pretrained files for [CEH-22](https://u42868014.dl.dropboxusercontent.com/u/42868014/neurobind/log.zip) or for [Oct-1](https://u42868014.dl.dropboxusercontent.com/u/42868014/neurobind/Oct-1/log.zip).


## Validation Check
  * Run `validation_check.py` to find the best model.

## Test evaluation
  * Run `test.py` to get the test results for the final model.

## Results
I got Spearman rank correlation coefficients of 0.60 and 0.70 on arrays #2 of CEH-22 and Oct-1, respectively.

| TF | PRG | RKM | S&W | KHM | DBD | DEBD | NB (Proposed) |
|--|--|--|--|--|--|--|--|
| CEH-22 | 0.28 | 0.43 | 0.28 | 0.31 | 0.40 | 0.43 | **0.60**|
| Oct-1 | 0.27 | 0.29 | 0.21 | 0.36 | 0.49 | 0.60 | **0.70**|





