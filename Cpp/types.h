#ifndef TYPES_H
#define TYPES_H



#define USE_FIXED_POINT 1   // <-- 0 = float (defaut sur) | 1 = ap_fixed (quantifie)

#if USE_FIXED_POINT
    #include <ap_fixed.h>
    typedef ap_fixed<16, 6, AP_RND, AP_SAT> data_t;   // <-- rends ce type plus petit pour quantifier plus fort
    typedef ap_fixed<32, 12, AP_RND, AP_SAT> acc_t;   // <-- accumulateur large (virgule fixe)
#else
    typedef float data_t;
    typedef float acc_t;   // [FIXED-ACC] en mode float : acc_t = float => comportement STRICTEMENT identique a l'origine
#endif

#endif
