#ifndef TYPES_H
#define TYPES_H

// ============================================================
//  TYPE DE DONNEES COMMUTABLE (float <-> virgule fixe)
//  -------------------------------------------------------
//  UN SEUL interrupteur pour tout le projet : USE_FIXED_POINT.
//
//    USE_FIXED_POINT 0  ->  data_t = float
//                           => comportement STRICTEMENT identique
//                              a la version d'origine. C'est ton
//                              "filet de securite" : tu ne perds rien.
//
//    USE_FIXED_POINT 1  ->  data_t = ap_fixed<16, 6>
//                           => version quantifiee (virgule fixe 16 bits :
//                              6 bits avant la virgule = portee +/-32,
//                              10 bits apres = resolution ~0.001).
//                           AP_RND  : arrondi (plus precis que la troncature)
//                           AP_SAT  : saturation (evite l'explosion en cas
//                                     de depassement au lieu de "wrap around").
//
//  DEMARCHE : laisse a 0, verifie que tout compile et que la csim passe.
//  Puis passe a 1, relance la csim et REGARDE "erreur max absolue".
//  Tant qu'elle reste petite, tu peux reduire les bits (16 -> 12 -> 10 -> 8)
//  en changeant seulement la ligne ap_fixed ci-dessous.
//
//  IMPORTANT : quand tu passes a 1, REGENERE params.h :
//      python generate_params_h.py
//  (pour que le tableau WEIGHTS[] soit lui aussi en data_t).
// ============================================================

#define USE_FIXED_POINT 1   // <-- 0 = float (defaut sur) | 1 = ap_fixed (quantifie)

#if USE_FIXED_POINT
    #include <ap_fixed.h>
    typedef ap_fixed<16, 6, AP_RND, AP_SAT> data_t;   // <-- rends ce type plus petit pour quantifier plus fort

    // ============================================================
    //  [FIXED-ACC] TYPE ACCUMULATEUR (acc_t)
    //  -------------------------------------------------------
    //  Avant : les accumulateurs (sum de matmul, mean/var de layer_norm...)
    //  etaient forces en "float" => Vitis instanciait quand meme des IP
    //  flottantes (fadd/fmul) DANS les boucles => la quantification ne
    //  changeait rien sur le chemin de calcul, seulement le stockage.
    //
    //  Maintenant : on accumule en VIRGULE FIXE LARGE (acc_t), pour :
    //    - supprimer les IP flottantes des boucles (moins de DSP/LUT,
    //      cosim RTL beaucoup plus rapide) ;
    //    - garder assez de bits pour NE PAS deborder pendant la somme
    //      de K produits (data_t*data_t donne deja 32 bits ; on somme
    //      plusieurs de ces produits).
    //
    //  Choix : ap_fixed<32,12> = 12 bits entiers (portee +/-2048, large
    //  marge avant saturation en additionnant ~128 produits) + 20 bits
    //  fractionnaires (resolution ~1e-6, bien plus fine que data_t).
    //  Si tu debordes malgre tout -> augmente la partie entiere (12 -> 16).
    //  Si tu manques de precision -> augmente la largeur totale (32 -> 40).
    // ============================================================
    typedef ap_fixed<32, 12, AP_RND, AP_SAT> acc_t;   // <-- accumulateur large (virgule fixe)
#else
    typedef float data_t;
    typedef float acc_t;   // [FIXED-ACC] en mode float : acc_t = float => comportement STRICTEMENT identique a l'origine
#endif

#endif
