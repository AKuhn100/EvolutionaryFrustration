# EvolutionaryFrustration
Structure independant evaluation of local mutational frustration in proteins using evolutionary covariance and statistical potentials.

The Proviz5.0 jupyter notebook contains all of the code used for calculatin evolutionary frustration and generating plots. 

The SerialEvolutionaryFrustration python file is a parallelized version of the script to calculate evolutionary frustration that significantly expidites the calculation.

Proteins are the fundamental functional components of biological systems. Given their importance in human health and catalysis, the relationships between protein sequence, structure, dynamics, and function have become a topic of great interest. One way to extract information from proteins is to compute the local energetic frustration of their native state. Traditionally, energetic frustration calculations require protein structures as a starting point. However, using a single protein structure to evaluate the energetic frustration for a given amino acid sequence assumes that a static structure can adequately represent a dynamic entity, which is not always true. Therefore, we have developed a structure-independent method to evaluate energetic frustration in proteins using direct coupling analysis (DCA) and statistical potentials. Our approach exhibits significant agreement with established structure-based frustration methods for static proteins and consistently outperforms established methods for proteins with large conformational variability.
