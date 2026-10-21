#ifndef ARCH_H
#define ARCH_H

// ============================================================
//  SELECTEUR D'ARCHITECTURE (meme principe que USE_FIXED_POINT dans types.h)
//  -------------------------------------------------------
//  UN SEUL interrupteur : ARCH. Tu changes UNE ligne, tu recompiles,
//  tu obtiens une IP HLS differente -> 2 projets Vivado = 2 bitstreams.
//
//  Les 2 architectures calculent EXACTEMENT le meme reseau (memes poids,
//  meme resultat numerique). Elles ne different QUE par la maniere dont
//  les NUM_BLOCKS transformers sont deployes sur le FPGA :
//
//    ARCH_ITERATIF : UN SEUL transformer materiel, REUTILISE NUM_BLOCKS fois
//                    en changeant les poids a chaque tour (folding du tuteur).
//                    => boucle NON deroulee + INLINE off (1 instance partagee).
//                    + : le plus PETIT (1x le calcul, champion des ressources),
//                        le plus sur a synthetiser.
//                    - : aucun parallelisme entre blocs.
//
//    ARCH_SERIE    : les NUM_BLOCKS transformers CABLES SPATIALEMENT, en serie :
//                    image -> bloc0 -> bloc1 -> bloc2 -> sortie, 1 image a la fois.
//                    => boucle DEROULEE (UNROLL) + INLINE recursive (3 blocs
//                       materiels DISTINCTS). MEME interface que l'iteratif
//                       (1 port poids) => MEME block design / TCL, rien a
//                       changer dans Vivado.
//                    + : simple (pas de DATAFLOW), pas de refonte BD.
//                    - : ~3x les ressources de l'iteratif pour la MEME vitesse
//                        (les blocs sont sequentiels) => sert a montrer l'interet
//                        du folding (iteratif). Peut ne pas rentrer sur le FPGA.
//
//    ARCH_PIPELINE : les NUM_BLOCKS transformers en CHAINE DE MONTAGE (DATAFLOW) :
//                    pendant que T2 traite l'image N, T1 traite deja l'image N+1.
//                    => 3 etages relies par des flux (hls::stream), chaque etage
//                       = 1 transformer cable (INLINE recursive => auto-suffisant).
//                    + : le plus de DEBIT (FPS) EN FLUX d'images.
//                    - : le plus gros (logique 3x + tampons => +RAM), plus dur
//                        a faire passer en synthese.
//
//  IMPORTANT (flux) : le pipeline ne gagne des FPS que si l'accelerateur
//  traite un LOT d'images par appel (argument num_images de vitis_top).
//  Sur UNE seule image, pipeline == iteratif (les blocs sont dependants :
//  T2 attend toute la sortie de T1). D'ou le traitement par lot.
//
//  DEMARCHE : on demarre sur ARCH_PIPELINE. Si la synthese DATAFLOW rale,
//  bascule sur ARCH_ITERATIF (change juste la ligne #define ARCH) : c'est
//  le filet de securite, il passe a coup sur.
// ============================================================

#define ARCH_ITERATIF  1
#define ARCH_PIPELINE  2
#define ARCH_SERIE     3

#define ARCH ARCH_ITERATIF   // CHANGER CETTE LIGNE (ITERATIF / SERIE / PIPELINE)
#define BATCH_MAX 64

#endif
