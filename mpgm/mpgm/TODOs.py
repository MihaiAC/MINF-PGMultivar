#TODO: Eliminate terms which do not depend on the parameters we fit. (e.g: log(Xs!) in SPGM and TPGM? + will fuck up if they get large enough).
#TODO: For SPGM, when the condition is not true, should we calculate the partition instead? (iteratively)
#TODO: For LPGM, they ignore the parameters theta_s (they are assumed to be 0 or unimportant and ignored). Is this a
# correct approach? Can I do this for the other 2 methods too? Change the methods depending on the answer.
# TODO: rewrite lpgm to use only matrix multiplications?
# TODO: how to initialise theta_init?; Try different theta_init strategies.
# TODO: rewrite prox_grad in order to remove node, data and include them into f and grad_f parameters + re-work
#   fit methods.
#TODO: add a specific variable list to each fit method.
#TODO: different theta_inits in lpgm? DUNNO.
#TODO: map must present elements always in the same order.
#TODO: add logging instead of print statements + handle the file requiring user input thing.


# TODO: MAKE SURE THAT THE SEEDS GIVEN TO LPGM FIT ARE DIFFERENT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



#TODO: Test the performance of the TPGM model for different R values.


#TODO: Check what to do with theta st and theta ts not agreeing. Can you do anything?
#TODO: Check what to do with theta ss (node parameters). Include them too after testing?
#TODO: add references to all methods used (towards the end I guess).