library(EWCE) 

setwd('')

ctd_mtg <- EWCE::load_rdata() # INPUT rda file for comparison

bg_prots <- #INPUT background proteins
bg_prots <- as.vector(bg_prots) 

clust_prots <- #INPUT proteins in cluster
clust_prots <- as.vector(clust_prots)

# define number of repetitions in bootstrap and cell expression category annotation level
reps <- 10000
annotLevel <- 1

# perform bootstrap enrichment test
full_results <- EWCE::bootstrap_enrichment_test(sct_data = ctd_mtg,
                                                bg = bg_prots,
                                                sctSpecies = "human",
                                                genelistSpecies = "human",
                                                hits = clust_prots, 
                                                reps = reps,
                                                annotLevel = annotLevel)

# print results
print(full_results$results)


